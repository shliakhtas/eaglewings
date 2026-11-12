"""Команда eaglewings: новий проєкт, потік, сайт, асистент. Запускати з теки проєкту (де eaglewings.toml).

    eaglewings init <тека> [--from <проєкт-зразок>]   новий проєкт: конфіг, контент лендінгу, переклад, LIMITATIONS
    eaglewings plan                                    порядок кроків потоку, нічого не запускаючи
    eaglewings run [кроки|гілки] [--from X] [--clearml]  потік Prefect (без аргументів — усе від завантаження даних)
    eaglewings script <назва> [аргументи]              один скрипт бібліотеки (scripts/<назва>.py) у теці проєкту, як його запускає потік
    eaglewings serve [--port 8766]                     сайт проєкту: web/ (лендінг, застосунок, дослідження, /ua/ і /en/)
    eaglewings assistant [--port 8770]                 сервер асистента чату (потрібне середовище з requirements/assistant.txt)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent
# що з проєкту-зразка переноситься в новий проєкт: лише вхідний контент, без даних і зібраних сторінок
SEED = ["eaglewings.toml", "config", "i18n", "docs/LIMITATIONS.md", "docs/DECISIONS.md", "web/landing/blocks", "web/landing/views.json",
        "README.md"]
SEED_GLOBS = ["web/assets/photo_*.jpg"]


def _project():
    from eaglewings.project import ROOT
    return ROOT


def init(args):
    if not args:
        sys.exit("eaglewings init <тека> [--from <проєкт-зразок>]")
    dst = Path(args[0]).resolve()
    src = Path(args[args.index("--from") + 1]).resolve() if "--from" in args else None
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst}: тека не порожня")
    dst.mkdir(parents=True, exist_ok=True)
    if src is None:
        (dst / "eaglewings.toml").write_text("# тека проєкту EagleWings\n")
        (dst / "config").mkdir()
        print(f"{dst}: порожній проєкт. Далі: config/regions.py (REGION), скіл new-study або приклад із demo/greece.")
        return
    for rel in SEED:
        s = src / rel
        if s.is_dir():
            shutil.copytree(s, dst / rel)
        elif s.exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dst / rel)
    for g in SEED_GLOBS:
        for f in src.glob(g):
            (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.relative_to(src))
    if not (dst / "eaglewings.toml").exists():
        (dst / "eaglewings.toml").write_text("# тека проєкту EagleWings\n")
    print(f"{dst}: проєкт із {src.name} — конфіг, контент лендінгу, переклад, LIMITATIONS. Даних і сторінок немає: eaglewings run.")


def run(args):
    root = _project()
    cmd = [sys.executable, str(LIB / "flows" / "pipeline.py"), *args]
    sys.exit(subprocess.call(cmd, cwd=root, env=dict(os.environ, EAGLEWINGS_PROJECT=str(root))))


def script(args):
    if not args:
        sys.exit("eaglewings script <назва> [аргументи]; назви: " + " ".join(sorted(f.stem for f in (LIB / "scripts").glob("*.py"))))
    root = _project()
    path = LIB / "scripts" / (args[0].removesuffix(".py") + ".py")
    if not path.exists():
        sys.exit(f"{path.name}: такого скрипта в бібліотеці немає")
    sys.exit(subprocess.call([sys.executable, str(path), *args[1:]], cwd=root, env=dict(os.environ, EAGLEWINGS_PROJECT=str(root))))


def serve(args):
    root = _project()
    port = args[args.index("--port") + 1] if "--port" in args else "8766"
    os.chdir(root / "web")
    os.execv(sys.executable, [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1"])


def assistant(args):
    root = _project()
    port = args[args.index("--port") + 1] if "--port" in args else "8770"
    env = dict(os.environ, EAGLEWINGS_PROJECT=str(root))
    sys.exit(subprocess.call([sys.executable, "-m", "uvicorn", "eaglewings.assistant.server:app", "--host", "127.0.0.1", "--port", port],
                             cwd=root / "logs" if (root / "logs").exists() else root, env=env))


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else "help"
    from eaglewings import BANNER
    print(BANNER, file=sys.stderr)
    if cmd == "init":
        return init(a[1:])
    if cmd == "plan":
        return run(["--plan"])
    if cmd == "run":
        return run(a[1:])
    if cmd == "script":
        return script(a[1:])
    if cmd == "serve":
        return serve(a[1:])
    if cmd == "assistant":
        return assistant(a[1:])
    print(__doc__)


if __name__ == "__main__":
    main()
