"""Запуск експериментів. За замовчуванням усе лише локально; у ClearML — тільки з прапорцем --clearml.

    .venv/bin/python scripts/05_experiments.py                  # усі задачі, локально
    .venv/bin/python scripts/05_experiments.py t1 t3            # вибрані, локально
    .venv/bin/python scripts/05_experiments.py t1 --clearml     # із записом у ClearML (app.clear.ml, проєкт Dadia)

Результат: logs/runs/ (запуски), data/datasets/ (версії датасетів), data/processed/experiments_summary.json.
Ключі ClearML — у ~/clearml.conf.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")                         # графіки лише у файли: без вікон, без Tk
import json
import sys
import time
from pathlib import Path
import importlib  # noqa: E402
from eaglewings.ds.common import P  # noqa: E402

from eaglewings.ds import branches as BR  # noqa: E402
TASKS = BR.tasks()                                   # задачі з реєстру гілок ds/branches.py

if __name__ == "__main__":
    if "--clearml" in sys.argv:                                      # або DADIA_CLEARML=1 від flows/pipeline.py --clearml
        os.environ["DADIA_CLEARML"] = "1"
    os.environ.setdefault("DADIA_CLEARML", "0")
    pick = [a.lower() for a in sys.argv[1:] if a != "--clearml"] or list(TASKS)
    f = P / "experiments_summary.json"
    for k in pick:
        t = time.time()
        name, _, part = TASKS[k].partition(":")              # «модуль:частина»: одна гілка задачі, яка рахує кілька гілок
        mod = importlib.import_module(f"eaglewings.ds.tasks.{name}")      # ліниво: не кожна задача потребує torch
        run = getattr(mod, f"run_{part}") if part else mod.run
        title = getattr(mod, "NAMES", {}).get(part, mod.EXPERIMENT)
        base = k.split("_")[0]                                   # t19_ld → t19: у підсумку гілки однієї задачі лежать разом
        from types import SimpleNamespace
        from eaglewings.ds import pflow, pages
        with pflow.node(title, "task"):
            res, info = run()
            recs = json.loads(res.drop(columns=[c for c in res.columns if c.startswith("by_fire")], errors="ignore").to_json(orient="records", force_ascii=False))
            if recs:
                pflow.mark("report: leaders and all approaches")
                pages.task(k, SimpleNamespace(EXPERIMENT=title, __doc__=mod.__doc__), recs)
        if not recs:                                             # лише дані (t19): підсумку немає
            print(f"[{k}] {title}: дані зібрано, {time.time() - t:.0f} с · {info}")
            continue
        summary = json.loads(f.read_text()) if f.exists() else {}      # перечитуємо: паралельні запуски пишуть у той самий файл
        keep = {a: b for a, b in info.items() if not hasattr(b, "shape") and not isinstance(b, dict)}
        if base != k and base in summary:                        # частина: замінюємо лише свої рядки (свої part), решту лишаємо
            mine = {r.get("part") for r in recs}
            recs = [r for r in summary[base]["results"] if r.get("part") not in mine] + recs
        summary[base] = dict(experiment=mod.EXPERIMENT, info={a: str(b) for a, b in keep.items()}, results=recs)
        print(f"[{k}] {mod.EXPERIMENT}: {len(res)} рядків, {time.time() - t:.0f} с · {keep}")
        f.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
