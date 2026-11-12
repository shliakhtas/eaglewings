"""EDA сирих наборів даних одразу після завантаження й моків: що в кожному наборі, розподіли, час, мапи.

    eaglewings script eda                 # усі набори
    eaglewings script eda firms hansen    # вибрані
    eaglewings script eda --clearml       # ще й звіти в ClearML Reports і графіки в Plots наборів

Результати лягають у data/processed/eda/sources/ (<набір>.json, <набір>.maps.json) і data/processed/eda/maps/*.png.
Сайт (крок site_data: 11_source_eda.py, 12_source_maps.py) лише перекладає їх у JSON, сам нічого не рахує.
Функції опису ті самі, що й раніше: scripts/11_source_eda.py і scripts/12_source_maps.py.
EDA датасетів робить кожна задача окремим підкроком, коли бере свій датасет (ds/datasets.py, get).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
sys.path.insert(0, str(LIB / "scripts"))
from data_lineage import SRC  # noqa: E402
from eaglewings.ds import pages, pflow, tracking  # noqa: E402,F401  (tracking: --clearml)

if __name__ == "__main__":
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SRC)
    for k in keys:
        pflow.mark(f"EDA: {k}")
        pages.source(k)
        print("EDA", k, "готово", flush=True)
