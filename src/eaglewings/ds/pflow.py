"""Вкладені кроки в Prefect для скриптів, які запускає flows/pipeline.py.

Потік запускає кожен скрипт окремим процесом і передає id своєї задачі в PREFECT_PIPELINE_TASK_RUN. Тут ми
вкладаємо під цю задачу сабфлов: скрипт → кроки → підходи → налаштування. Prefect лише показує структуру,
стан і метрики; розрахунок від цього не залежить.

Без змінної середовища (ноутбук, ручний запуск скрипта) усе тут нічого не робить.

    with pflow.node("FD · що згоріло у 2023"):        # сабфлов, усередині все наступне
        pflow.mark("1. датасет")                      # послідовні кроки: mark закриває попередній і відкриває новий
        ...
        pflow.mark("2. підходи")
        with pflow.node("s2_dnbr"): ...               # вкладений сабфлов
    pflow.metrics({"test_f1": .87})                   # метрики поточного вузла → таблиця-артефакт у Prefect
"""
import atexit
import contextlib
import os
import re
import sys
import uuid
from pathlib import Path

PARENT = os.environ.get("PREFECT_PIPELINE_TASK_RUN")
_stack = []          # відкриті вузли: dict(flow_run, task_run, mark, metrics)
_client = None


def on():
    return bool(PARENT)


def _c():
    global _client
    if _client is None:
        from prefect.client.orchestration import get_client
        _client = get_client(sync_client=True)
    return _client


_FLOWS, _TASKS = {}, {}


def _flow(kind):
    if kind not in _FLOWS:
        from prefect import Flow
        _FLOWS[kind] = Flow(fn=lambda: None, name=kind)
    return _FLOWS[kind]


def _task(name):
    if name not in _TASKS:
        from prefect import Task
        _TASKS[name] = Task(fn=lambda: None, name=name)
    return _TASKS[name]


_TR = dict(zip("абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
               "a b v h g d e ie zh z y i i i k l m n o p r s t u f kh ts ch sh shch  iu ia A B V H G D E Ie Zh Z Y I I I K L M N O P R S T U F Kh Ts Ch Sh Shch  Iu Ia".split(" ")))


def latin(name):
    """Граф Prefect не має кирилиці в шрифті: прибираємо українські пояснення в дужках, решту транслітеруємо."""
    name = re.sub(r"\s*\([^)]*[а-яіїєґА-ЯІЇЄҐ][^)]*\)", "", str(name))
    return "".join(_TR.get(ch, ch) for ch in name).replace("→", "->").replace("·", "-").replace("«", "").replace("»", "").replace("–", "-").replace("—", "-")


def _open(name, kind, mark=False):
    from prefect.states import Running
    name = latin(name)
    c = _c()
    if _stack:
        parent_flow = _stack[-1]["flow_run"]
        tr = c.create_task_run(_task(name[:120]), flow_run_id=parent_flow, dynamic_key=uuid.uuid4().hex, name=name[:200], state=Running())
        parent_task = tr.id
    else:
        tr, parent_task = None, PARENT
    fr = c.create_flow_run(_flow(kind), name=name[:200], parameters={}, parent_task_run_id=parent_task, state=Running())
    _stack.append(dict(flow_run=fr.id, task_run=tr.id if tr else None, mark=mark, metrics={}, name=name))


def _close(ok=True):
    from prefect.states import Completed, Failed
    n = _stack.pop()
    c = _c()
    if n["metrics"]:
        from prefect.client.schemas.actions import ArtifactCreate
        rows = "\n".join(f"| {k} | {v:.4g} |" if isinstance(v, (int, float)) else f"| {k} | {v} |" for k, v in n["metrics"].items())
        try:
            c.create_artifact(ArtifactCreate(type="markdown", flow_run_id=n["flow_run"], description="метрики",
                                             data=f"| метрика | значення |\n|---|---|\n{rows}\n"))
        except Exception:
            pass
    ok = ok and not n.get("failed")
    st = Completed() if ok else Failed(message="error")
    c.set_flow_run_state(n["flow_run"], st, force=True)
    if n["task_run"]:
        c.set_task_run_state(n["task_run"], Completed() if ok else Failed(), force=True)


def _close_mark():
    while _stack and _stack[-1]["mark"]:
        _close()


@contextlib.contextmanager
def node(name, kind="step"):
    """Сабфлов name; усе, що відкривається всередині, вкладається в нього."""
    if not on():
        yield
        return
    _open(name, kind)
    depth = len(_stack)
    try:
        yield
    except BaseException:
        while len(_stack) > depth:
            _close(ok=False)
        _close(ok=False)
        raise
    else:
        while len(_stack) > depth:
            _close()
        _close()


def mark(name):
    """Послідовний крок усередині поточного вузла: закриває попередній mark того самого рівня й відкриває новий."""
    if not on():
        return
    if not _stack:                                   # кроки скрипта завжди всередині вузла самого скрипта
        _open(Path(sys.argv[0]).stem, "script")
    _close_mark()
    _open(name, "step", mark=True)


def failed():
    """Позначити поточний крок як невдалий, не зупиняючи скрипт."""
    if on() and _stack:
        _stack[-1]["failed"] = True


def artifact(md, key=None, description=""):
    """Сторінка-звіт (markdown з таблицями й графіками) у поточному вузлі. Ключ дає їй історію у вкладці Artifacts."""
    if not on():
        return
    from prefect.client.schemas.actions import ArtifactCreate
    where = dict(flow_run_id=_stack[-1]["flow_run"]) if _stack else dict(task_run_id=PARENT)
    key = re.sub(r"[^a-z0-9-]+", "-", key.lower()).strip("-")[:100] if key else None
    try:
        _c().create_artifact(ArtifactCreate(key=key, type="markdown", description=description, data=md, **where))
    except Exception as e:
        print(f"Prefect: звіт {key} не записано: {type(e).__name__}: {str(e)[:200]}", flush=True)


def metrics(d):
    """Метрики поточного вузла; у Prefect з'являються таблицею-артефактом, коли вузол закривається."""
    if on() and _stack:
        _stack[-1]["metrics"].update({k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in d.items()})


@atexit.register
def _finish():
    while _stack:
        try:
            _close()
        except Exception:
            _stack.clear()
