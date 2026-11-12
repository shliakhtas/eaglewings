"""Дані блоку «По парках» сторінки дослідження → web/science/parks.json. Запуск із теки eaglewings:

    eaglewings script 15_parks_page

Нічого не рахує: лише перекладає готові результати задач у вигляд, який читає сторінка.
  parks     про кожен парк: межа, площа, ліс, теплові точки FIRMS, втрати лісу Hansen, гекси
  branches  метрики гілок по парках: для кожного алгоритму значення й 95 % CI разом і окремо в кожному парку
            (колонки результатів *_<парк>, які пишуть задачі); TP — свій переможець у кожному парку
  maps      мапи кожного парку для перемикача: FD переможець проти Hansen (картинка), FG_risk рівні ризику 2026,
            LD кандидати на рубки, LG гекси прогнозу, TP вежі й пожежі
З одним парком у ds/regions.py блок не потрібен: пишеться порожній файл.
"""
import base64
import io
import json
import math
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from PIL import Image

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import branches as BR, common, datasets, park  # noqa: E402

OUT = ROOT / "web" / "science" / "parks.json"
RES = common.P / "results"
S = json.loads((common.P / "experiments_summary.json").read_text()) if (common.P / "experiments_summary.json").exists() else {}
CODES = ["FD", "FG_day", "FG_risk", "AR", "LD", "LM", "LG", "TP"]


def num(v, n=4):
    try:
        v = float(v)
        return round(v, n) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def ci(r, spec, suffix=""):
    """95 % CI рядка: колонка-рядок «lo–hi» або пара колонок (lo, hi); suffix — _<парк>."""
    if not spec:
        return None
    if isinstance(spec, (tuple, list)):
        lo, hi = num(r.get(spec[0] + suffix)), num(r.get(spec[1] + suffix))
        return [lo, hi] if lo is not None and hi is not None else None
    v = r.get(spec + suffix)
    if isinstance(v, str) and "–" in v:
        try:
            a, b = (float(x) for x in v.split("–"))
            return [round(a, 4), round(b, 4)] if math.isfinite(a) and math.isfinite(b) else None
        except ValueError:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return [num(v[0]), num(v[1])]
    return None


# на чому порахована метрика парку в тесті і скільки цього треба, щоб число щось означало (менше — «замало подій», без числа)
BASE = {"FD": ("true_ha_test_{k}", "га", 5), "LD": ("true_ha_{k}", "га", 2), "LM": ("true_ha_{k}", "га", 2), "LG": ("true_ha_{k}", "га", 2),
        "FG_day": ("positives_{k}", "дн.", 10), "FG_risk": ("burned_2023_{k}", "гекс.", 20), "AR": ("n_test_{k}", "пож.", 10), "TP": ("n", "пож.", 10)}
BASE_NOTE = {"FD": "під числом — га згарища в тесті парку", "LD": "під числом — га рубок у тесті парку", "LM": "під числом — га рубок у тесті парку",
             "LG": "під числом — га рубок у тесті парку", "FG_day": "під числом — днів з подією в тесті парку", "FG_risk": "під числом — згорілих гексів 2023 у парку",
             "AR": "під числом — пожеж у тесті парку", "TP": "під числом — пожеж у тесті парку"}


def fmt_n(v, unit):
    return (f"{v:,.0f}".replace(",", " ") if v >= 10 else f"{v:.1f}".replace(".", ",")) + f" {unit}"


def cell(r, k, test, cis, code):
    """Значення парку, CI і основа; якщо основа менша за мінімум — значення не показуємо (few=True)."""
    col, unit, mn = BASE.get(code, (None, "", 0))
    n = r.get(col.format(k=k)) if col else None
    n = None if n is None or n != n else float(n)
    out = dict(v=num(r.get(f"{test}_{k}")), ci=ci(r, cis, f"_{k}"), n=fmt_n(n, unit) if n is not None else None)
    if n is not None and n < mn:
        out.update(v=None, ci=None, few=True)
    return out


def park_info(k):
    B = common.base(k)
    dom = B.park_mask & B.forest
    f = B.firms
    loss = {2000 + y: num(B.PIX_HA[(B.loss == y) & B.park_mask].sum(), 1) for y in range(1, 25)}
    return dict(name=park.ALL[k]["name"], short=park.ALL[k]["short"], area_ha=num(B.park_u.area.iloc[0] / 1e4, 0), forest_ha=num(B.PIX_HA[dom].sum(), 0),
                forest_share=num(B.PIX_HA[dom].sum() / (B.park_u.area.iloc[0] / 1e4), 3), firms=int(len(f)), firms_by_year={int(y): int(n) for y, n in f.groupby("year").size().items()},
                hansen_loss_ha=loss, hansen_loss_total_ha=num(sum(v or 0 for v in loss.values()), 1), h3_cells=int(len(common.h3grid(k).grid)),
                boundary=json.loads(B.park.geometry.simplify(.0005).set_precision(1e-5).to_json()))


def branch(code):
    b = BR.R[code]
    key = b["summary"]
    lead = BR.lead().get(key)
    rows = S.get(key, {}).get("results", [])
    if not lead or not rows:
        return None
    if b.get("part"):
        rows = [r for r in rows if r.get("part") == b["part"]]
    name, test, cis, grp = lead["name"], lead["test"], lead.get("ci"), lead.get("group")
    metric = {"LD": "F1 за площею рубок", "LM": "F1 за площею рубок", "LG": "частка площі рубок у верхніх 11 % гексів прогнозу"}.get(code, lead["metric"])
    out = dict(code=code, metric=metric, how=lead["how"], low=bool(lead.get("low")), parks=list(park.PARKS), groups={}, base=BASE_NOTE.get(code, ""),
               min=f"менше {BASE[code][2]} {BASE[code][1]} — замало, число не показуємо" if code in BASE else "")
    if code == "TP":                                         # вежі ставляться в кожному парку окремо: свій переможець, свій тест
        rows = [r for r in rows if r.get("part") == "test"]
        alg = {}
        for r in rows:
            a = f"{r['approach']}, {int(r['pool'])} місць"
            alg.setdefault(a, dict(approach=a, champion={}, total=None, total_ci=None, by_park={}))
            n = float(r.get("n") or 0)
            alg[a]["by_park"][r.get("park", park.MAIN)] = dict(v=num(r.get(test)), ci=ci(r, cis), n=fmt_n(n, "пож.")) if n >= BASE["TP"][2] else dict(v=None, ci=None, n=fmt_n(n, "пож."), few=True)
            if r.get("champion"):
                alg[a]["champion"][r.get("park", park.MAIN)] = True
        out["groups"][""] = list(alg.values())
        out["note"] = "Вежі ставляться всередині парку, тому в кожного парку свій переможець (обраний на його валідації 2022)."
        return out
    for r in rows:
        g = " · ".join(str(r.get(x, "")) for x in grp) if isinstance(grp, (tuple, list)) else (str(r.get(grp, "")) if grp and grp != "part" else "")
        item = dict(approach=str(r.get(name)), champion=bool(r.get("champion")), total=num(r.get(test)), total_ci=ci(r, cis),
                    by_park={k: cell(r, k, test, cis, code) for k in park.PARKS})
        out["groups"].setdefault(g, []).append(item)
    for g in out["groups"].values():
        g.sort(key=lambda x: (not x["champion"], -(x["total"] or -1e9) if not out["low"] else (x["total"] or 1e9)))
    if code in ("LD", "LM"):
        tr = [r for r in S.get(key, {}).get("results", []) if r.get("part") == f"{code}_transfer"]
        if tr:                                              # окрема перевірка: модель з інших парків на парку, якого не бачила
            out["transfer"] = [dict(approach=str(r.get(name)), total=num(r.get(test)), total_ci=ci(r, cis),
                                    by_park={k: cell(r, k, test, cis, code) for k in park.PARKS}) for r in tr]
    return out


def fd_image(R):
    """Переможець FD проти Hansen у кожній клітинці 4×4 пікселі → PNG: влучив, зайве, пропуск, ліс без згарища."""
    r, c, y, pc = (np.array(R[x]) for x in ("r", "c", "y", "pc"))
    r0, c0 = r.min(), c.min()
    h, w = r.max() - r0 + 1, c.max() - c0 + 1
    img = np.zeros((h, w, 4), np.uint8)
    yy, pp = y >= 50, pc >= 50
    col = np.tile(np.array([233, 236, 239, 90], np.uint8), (len(r), 1))
    for m, c_ in ((yy & pp, [201, 42, 42, 230]), (~yy & pp, [255, 169, 77, 230]), (yy & ~pp, [151, 117, 250, 230])):
        col[m] = c_
    img[r - r0, c - c0] = col
    b = io.BytesIO()
    Image.fromarray(img, "RGBA").save(b, "PNG", optimize=True)
    lat_n, lon_w = R["lat0"] + R["dlat"] * r0, R["lon0"] + R["dlon"] * c0
    return dict(img="data:image/png;base64," + base64.b64encode(b.getvalue()).decode(), bounds=[[lat_n + R["dlat"] * h, lon_w], [lat_n, lon_w + R["dlon"] * w]],
                tp=int((yy & pp).sum()), fp=int((~yy & pp).sum()), fn=int((yy & ~pp).sum()))


def hexpoly(h):
    return [[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(h)]


def maps():
    out = {}
    f = RES / "burned_area.json"
    if f.exists() and "raster_parks" in (K := json.loads(f.read_text())):
        out["a"] = {k: fd_image(v) for k, v in K["raster_parks"].items()}
    out["c"] = {}
    for k in park.PARKS:
        g = park.processed(k) / "risk_h3_2026.geojson"
        if g.exists():
            import geopandas as gpd
            d = gpd.read_file(g)
            out["c"][k] = dict(cells=[[hexpoly(h), lv] for h, lv in zip(d.h3, d.level)],
                               counts=d.level.value_counts().to_dict())
    for pid, fn in (("ld", "logging_ld.json"), ("lg", "logging_lg.json")):
        f = RES / fn
        if f.exists() and "parks" in (K := json.loads(f.read_text())):
            out[pid] = {k: v for k, v in K["parks"].items() if v}
    d = datasets.load("h3_risk_cells").df
    out["tp"] = {}
    for k in park.PARKS:
        f = park.processed(k) / "tp_final.json"
        if f.exists():
            T = json.loads(f.read_text())
            dk = (d[d.park == k] if "park" in d else d).reset_index(drop=True)
            ll = lambda i: [round(x, 5) for x in h3.cell_to_latlng(dk.h3.iat[int(i)])]
            out["tp"][k] = dict(chosen=T["chosen"], towers=[ll(i) for i in T["cells"]], fires=T["fires"])
    return out


def main():
    if len(park.PARKS) < 2:
        OUT.write_text(json.dumps(dict(parks={}), ensure_ascii=False))
        print("parks.json: один парк, блок не потрібен")
        return
    data = dict(parks={k: park_info(k) for k in park.PARKS}, branches={}, maps=maps())
    for code in CODES:
        x = branch(code)
        if x:
            data["branches"][BR.R[code]["page"]] = x
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=lambda o: None))
    print(f"web/science/parks.json: {len(data['parks'])} парки, гілок {len(data['branches'])}, мап {sum(len(v) for v in data['maps'].values())},"
          f" {OUT.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
