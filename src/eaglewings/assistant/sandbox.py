"""Пісочниця для коду моделі.

Код PandasAI виконується в окремому процесі з жорстким лімітом часу й пам'яті, без мережі, без ключів у
змінних середовища й з обмеженням на запис файлів. SQL до таблиць парку виконується в батьківському процесі:
дочірній надсилає запит трубою й отримує таблицю. Перед виконанням код проходить assistant/guard.py.
Пам'ять читаємо з /proc: лише власну, без сторінок, спільних із сервером.
"""
import ctypes
import logging
import multiprocessing
import os
import queue
import resource
import time

from pandasai.exceptions import CodeExecutionError, NoResultFoundError
from pandasai.sandbox import Sandbox

from .guard import POLICY_MARK, GuardError, check_code

log = logging.getLogger(__name__)

SECRET_ENV_MARKERS = ("PASSWORD", "SECRET", "KEY", "TOKEN", "DATABASE_URL", "CLEARML", "AWS")
MAX_WRITE_BYTES = 50 * 1024 * 1024
ISOLATION = os.environ.get("SANDBOX_ISOLATION", "auto")          # auto | require | off
_CLONE_NEWUSER, _CLONE_NEWNET = 0x10000000, 0x40000000
_warned = False
_PAGE = os.sysconf("SC_PAGE_SIZE")


def isolate_network():
    """Процес переходить у власний простір мережі без інтерфейсів: нікуди не достукатись."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNET) == 0
    except (OSError, AttributeError):
        return False


def rss(pid):
    """Власна пам'ять процесу в байтах (Private_Clean + Private_Dirty з /proc/<pid>/smaps_rollup).
    Дочірній процес форкнутий із сервера й ділить із ним таблиці парку (копіювання під час запису), тож повний
    RSS одразу був би розміром усього сервера; ліміт рахуємо лише на те, що код моделі виділив сам."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            return sum(int(l.split()[1]) * 1024 for l in f if l.startswith(("Private_Clean:", "Private_Dirty:")))
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * _PAGE
    except (OSError, ValueError, IndexError):
        return 0


class TimeoutSandbox(Sandbox):
    def __init__(self, timeout_seconds=30, memory_limit_mb=1536, poll_interval=.05):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.poll_interval = poll_interval
        self.policy_hits = []

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def _exec_code(self, code, environment):
        try:
            check_code(code)
        except GuardError as exc:
            self.policy_hits.append(str(exc))
            raise CodeExecutionError(str(exc)) from exc
        compiled = self._compile_code(code)
        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue()
        parent_end, child_end = ctx.Pipe()
        process = ctx.Process(target=self._run, args=(compiled, environment, result_queue, child_end))
        process.start()
        try:
            got = self._watch(process, parent_end, environment.get("execute_sql_query"), result_queue)
            if got is None:
                try:
                    got = result_queue.get(timeout=.5)
                except queue.Empty:
                    raise CodeExecutionError(f"Generated code exited without producing a result (exit code {process.exitcode}).") from None
            status, payload = got
            if status == "error":
                raise CodeExecutionError(payload)
            return payload
        finally:
            result_queue.close()
            parent_end.close()
            if process.is_alive():
                process.kill()
            process.join()

    def _serve_sql(self, parent_end, execute_sql_query):
        _, query = parent_end.recv()
        try:
            if execute_sql_query is None:
                raise RuntimeError("execute_sql_query is not available")
            parent_end.send(("ok", execute_sql_query(query)))
        except Exception as exc:  # noqa: BLE001 — текст помилки йде моделі для наступної спроби
            message = self._describe(exc)
            if POLICY_MARK in message:
                self.policy_hits.append(message)
            parent_end.send(("error", message))

    def _watch(self, process, parent_end, execute_sql_query, result_queue):
        """Обслуговує SQL дочірнього процесу й стежить за лімітами. Результат забирає з черги одразу, щойно він є:
        процес із великим результатом (тисячі рядків) не завершиться, поки хтось не прочитає чергу, тож чекати
        його завершення перед читанням означало б тупик до ліміту часу. → (status, payload) або None."""
        deadline = time.monotonic() + self.timeout_seconds
        last = 0.0
        while process.is_alive() and time.monotonic() < deadline:
            try:
                return result_queue.get_nowait()
            except queue.Empty:
                pass
            if parent_end.poll(self.poll_interval):
                try:
                    self._serve_sql(parent_end, execute_sql_query)
                except EOFError:
                    pass
            if time.monotonic() - last < .2:
                continue
            last = time.monotonic()
            if rss(process.pid) > self.memory_limit_bytes:
                process.kill()
                process.join()
                raise CodeExecutionError(f"Generated code exceeded the {self.memory_limit_bytes // (1024 * 1024)}MB memory limit and was terminated.")
        if process.is_alive():
            process.terminate()
            process.join(5)
            raise CodeExecutionError(f"Generated code exceeded the {self.timeout_seconds}s wall-clock limit and was terminated.")
        return None

    @staticmethod
    def _confine():
        global _warned
        if ISOLATION != "off" and not isolate_network():
            if ISOLATION == "require":
                raise RuntimeError("network isolation of the sandbox is not available (SANDBOX_ISOLATION=require)")
            if not _warned:
                log.warning("sandbox: network isolation unavailable here - running without it")
                _warned = True
        for name in list(os.environ):
            if any(m in name.upper() for m in SECRET_ENV_MARKERS):
                del os.environ[name]
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_WRITE_BYTES, MAX_WRITE_BYTES))

    @staticmethod
    def _run(compiled, environment, result_queue, child_end):
        try:
            TimeoutSandbox._confine()

            def execute_sql_query(sql_query):
                child_end.send(("sql", sql_query))
                status, payload = child_end.recv()
                if status == "error":
                    raise RuntimeError(payload)
                return payload

            environment["execute_sql_query"] = execute_sql_query
            exec(compiled, environment)                                     # noqa: S102 — код уже пройшов guard.check_code
            if "result" not in environment:
                raise NoResultFoundError("No result was returned from the code execution. Please return the result in dictionary format, for example: result = {'type': ..., 'value': ...}")
            result_queue.put(("ok", environment["result"]))
        except Exception as exc:  # noqa: BLE001
            result_queue.put(("error", TimeoutSandbox._describe(exc)))

    @staticmethod
    def _describe(exc):
        def nxt(e):
            return e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        parts, cause = [f"{type(exc).__name__}: {exc}"], nxt(exc)
        while cause is not None and len(parts) < 4:
            parts.append(f"Caused by {type(cause).__name__}: {str(cause).strip()}")
            cause = nxt(cause)
        return "\n".join(parts)
