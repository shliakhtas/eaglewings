"""Датасети, які збираються лише з сирих даних: кожен своєю функцією збору (ds/datasets.py, BUILDERS).

    eaglewings script datasets_build                    # усі
    eaglewings script datasets_build h3_risk_cells      # вибрані

Якщо датасет уже зібрано з тих самих сирих файлів тим самим кодом, він лише читається. Після кожного йде його EDA.
Задачі беруть ці датасети тим самим викликом datasets.get, тож можуть працювати й окремо від цього кроку.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import datasets, pflow, tracking  # noqa: E402

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(datasets.BUILDERS)
    for n in names:
        pflow.mark(f"build: {n}")
        dv = datasets.get(n)
        if tracking.online():
            tracking.dataset_id(dv)
        print(n, dv.tag, dv.digest, len(dv.df), flush=True)
