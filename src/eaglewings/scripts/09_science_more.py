"""Дані для нових блоків сторінки дослідження → web/science/more.json. Запускати після scripts/08_app_extra.py.

    eaglewings script 09_science_more

Нових експериментів немає: лише зведення наявних результатів і описів датасетів.
"""
import collections
import json
from pathlib import Path

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
P = ROOT / "data" / "processed"
S = json.load(open(P / "experiments_summary.json"))
import sys  # noqa: E402
from eaglewings.ds import park  # noqa: E402
X = json.load(open(P / "app_extra.json"))
O = json.load(open(P / "ops.json"))


def ds(n):
    d = ROOT / "data" / "datasets" / n
    v = json.load(open(d / "versions.json"))[-1]
    m = json.load(open(d / f"v{v['version']}" / "manifest.json"))
    return dict(name=n, version=v["version"], digest=m["digest"], rows=m["rows"], unit=m["unit"])


src = collections.Counter()
for n in O["feed"]:
    src[n["source"]] += n.get("count", 1)
res = lambda t: S[t]["results"] if t in S else []          # гілки, вимкнені в config/branches.py, дають порожній список
out = dict(towers=X["towers"], t22=res("t22"), t23=[r for r in res("t23") if r.get("park", park.MAIN) == park.MAIN], t24=res("t24"), cp=X["cp"], t21=res("t21"),
           ar_map=json.load(open(P / "ar_map.json")) if (P / "ar_map.json").exists() else None, t20=res("t20"), t13=res("t13"), t11=res("t11"), t18=res("t18"), t19=res("t19"),
           ds={n: ds(n) for n in ("matching_firms_effis", "cv_benchmark_aerial", "cv_benchmark_ground", "logging_gr_pixels", "logging_gr_cells", "opera_gr_snapshots", "xm_signals", "ar_fires", "cv_camera_zone", "tp_fires", "tp_candidates", "cp_events")
               if (ROOT / "data" / "datasets" / n / "versions.json").exists()},
           fires=X["fires"], cut=X["logging"], forecast=X["forecast"],
           ops=dict(signals=O["signals"], real=O["signals_real"], clusters=O["clusters"], src=sorted(src.items(), key=lambda x: -x[1]),
                    areals=[dict(id=a["id"], kind=a["type_name"], life=a["lifecycle"], ha=a["area_ha"], lat=a["lat"], lon=a["lon"], n=a["clusters"]) for a in O["areals"]]))
(ROOT / "web" / "science" / "more.json").write_text(json.dumps(out, ensure_ascii=False, default=float))
print(len(json.dumps(out)))
