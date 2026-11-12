"""Результати досліджень для розділів застосунку → app_extra.json у теці кожного парку. Запускати перед scripts/04_build_page.py.

Нічого не навчає й не підбирає: бере переможців і те, що задачі вже застосували до останніх даних (data/processed/results/).

    eaglewings script 08_app_extra                       # усі парки: основний тут, решта окремими процесами
    GR_PARK=emt eaglewings script 08_app_extra           # один парк

Розділи, прив'язані до місць основного парку (камера-вежа, поширення Еврос-2023, камери на дорогах), є лише в ньому.

Це не нові експерименти: правила й моделі ті самі, що в задачах, застосовані до останніх даних. Метрики на сторінці беруться з experiments_summary.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import geopandas as gpd

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import common, datasets, park  # noqa: E402

R0 = ROOT / "data" / "processed"                       # спільне: підсумок задач і результати, які задачі записали
P = park.processed()                                   # тека парку GR_PARK: app_extra.json, вежі, ризик 2026
KEY, IS_MAIN = park.KEY, park.KEY == park.MAIN
S = json.load(open(R0 / "experiments_summary.json"))
B = common.base()
here = lambda df: df[df.park == KEY].drop(columns="park").reset_index(drop=True) if "park" in df else df   # рядки цього парку
pk = lambda r, m: r.get(f"{m}_{KEY}", r.get(m)) if len(park.PARKS) > 1 else r.get(m)                       # метрика тесту в цьому парку
latlon = lambda poly: [[round(y, 5), round(x, 5)] for x, y in poly]


def kept(task):
    """Результати, які задача записала разом із метриками (ds/tracking.keep): застосунок їх лише показує."""
    return json.loads((R0 / "results" / f"{task}.json").read_text())


def burn():
    """Растр як на сторінці дослідження: збіг, перебільшення, пропуск проти Hansen."""
    from PIL import Image
    r = next(x for x in S["t1"]["results"] if x.get("champion"))
    from eaglewings.ds.tasks import burned_area as BA
    d = here(datasets.load("burned_area_pixels").df)
    p = BA.predict(d, r["family"], r["best_config"])                          # обраний переможець, хоч би який він був
    y = d.y.values.astype(bool)
    img = np.zeros((B.H, B.W, 4), np.uint8)
    for m, col in ((p & y, (217, 89, 38, 235)), (p & ~y, (242, 165, 65, 235)), (~p & y, (154, 165, 155, 235))):
        img[d.row.values[m], d.col.values[m]] = col
    name = "an_burn.png" if IS_MAIN else f"an_burn_{KEY}.png"
    Image.fromarray(img, "RGBA").save(ROOT / "web" / "assets" / name, optimize=True)
    ll = lambda v: round(float(v), 5)
    return dict(bounds=[[ll(B.TR.f + B.TR.e * B.H), ll(B.TR.c)], [ll(B.TR.f), ll(B.TR.c + B.TR.a * B.W)]], ha=round(float(d.pix_ha.values[p].sum())), f1=pk(r, "test_f1"), precision=pk(r, "test_precision"), recall=pk(r, "test_recall"),
                hansen_ha=round(json.load(open(ROOT / "web" / "science" / "parks.json"))["parks"][KEY]["hansen_loss_ha"]["2023"]), img=f"assets/{name}")


def fires():
    """Зведення точок у пожежі параметрами переможця FM (а не своїми)."""
    import re
    ch = next(x for x in S["t13"]["results"] if x.get("champion"))
    km, days = map(float, re.findall(r"([\d.]+)km_([\d.]+)d", ch["approach"])[0])
    f = B.firms.copy()
    f["ev"] = common.st_dbscan(f, km, days)
    g = f.groupby("ev").agg(n=("dt", "size"), first=("dt", "min"), last=("dt", "max"), lat=("latitude", "mean"), lon=("longitude", "mean"), sensors=("sensor", "nunique"))
    top = g.sort_values("n", ascending=False).head(6)
    return dict(points=len(f), fires=len(g), both=int((g.sensors > 1).sum()), f1=ch["bcubed_f1"], approach=ch["approach"],
                top=[dict(n=int(x.n), first=str(x.first)[:10], last=str(x.last)[:10], lat=round(x.lat, 5), lon=round(x.lon, 5)) for x in top.itertuples()])


def day():
    """Щоденна ймовірність переможця FG «день пожежі» за останній тестовий рік, як її порахувала задача."""
    c = kept("fire_day")["champion_series"]["fire_day"]
    if not IS_MAIN:
        c = dict(c["parks"][KEY], approach=c["approach"])
    te = pd.DataFrame(dict(date=pd.to_datetime(c["date"]), p=c["score"], y=c["y"]))
    te = te[te.date.dt.year == te.date.dt.year.max()]
    w = te.set_index("date").resample("7D").agg(p=("p", "max"), y=("y", "max"))
    r = next(x for x in S["t2"]["results"] if x.get("champion") and x.get("target") == "fire_day")
    return dict(x=[str(i)[5:10] for i in w.index], p=[None if v != v else round(float(v), 3) for v in w.p], y=[int(v) for v in w.y], pr_auc=pk(r, "pr_auc"), roc=pk(r, "roc_auc"),
                fires=int(te.y.sum()), approach=c["approach"])


def towers(d, p):
    """Покриття веж із гілки TP, зважене ризиком обраної моделі: огляд 360°, 10 км, рельєф SRTM."""
    rank = pd.Series(p).rank(pct=True).values
    glat, glon = np.array([h3.cell_to_latlng(c) for c in d.h3]).T
    wgt = rank * (d.treecover.values >= 30)
    tp = json.load(open(P / "tp_final.json"))                                  # місця обрала гілка TP
    vis = {j: common.los(glat[j], glon[j], glat, glon, 15) for j in tp["risk_cells"]}
    vis = {j: v & (dist <= 10000) for j, (v, dist) in vis.items()}
    covered = np.zeros(len(d), bool)                                           # камера-вежа з відео є лише в основному парку
    if IS_MAIN:
        V = json.load(open(P / "vision" / "video.json"))
        v, dist = common.los(V["cam_lat"], V["cam_lon"], glat, glon, 15)
        az = common.azimuth(glat, glon, V["cam_lat"], V["cam_lon"])
        covered = v & (dist <= 10000) & (((az - (V["bearing"] - 30)) % 360) <= 60)
    base = wgt[covered].sum() / wgt.sum()
    seen, out = np.zeros(len(d), bool), []
    for j in tp["risk_cells"]:
        covered = covered | vis[j]
        seen |= vis[j]
        out.append(dict(lat=round(float(glat[j]), 5), lon=round(float(glon[j]), 5), elev=round(float(d.elev.iat[j])), cells=int(vis[j].sum()), coverage=round(float(wgt[covered].sum() / wgt.sum()), 3)))
    main = lambda r: r.get("park", park.MAIN) == KEY                          # у TP кожен парк має свого переможця
    q = {r["approach"]: r for r in S["t23"]["results"] if r["part"] == "test" and r.get("pool") == 300 and main(r)}
    ch = next(r for r in S["t23"]["results"] if r.get("champion") and main(r))
    return dict(h3=d.h3.tolist(), rank=np.round(rank, 3).tolist(), seen=seen.astype(int).tolist(), towers=out, base_coverage=round(float(base), 3), candidates=tp["pool"], first=ch["first"], first_lo=ch["first_lo"], first_hi=ch["first_hi"], random=q["random"]["first"], burn=ch["burn"], fires=tp["fires"])


def cameras():
    """Місця камер із гілки CP: дерево під'їздів спрощене, щоб мапа була легкою."""
    if not IS_MAIN:
        return None                                                          # місця лише основного парку
    from shapely.geometry import MultiLineString
    from shapely.ops import linemerge
    c = json.load(open(P / "cp_final.json"))
    m = linemerge(MultiLineString([[(p[1], p[0]) for p in x] for x in c["segs"]])).simplify(.0004)
    geoms = getattr(m, "geoms", [m])
    R = S["t24"]["results"]
    ch, rnd, sens = next(r for r in R if r.get("champion")), next(r for r in R if r["part"] == "test" and r["approach"] == "random"), next(r for r in R if r["part"] == "sens")
    return dict(lines=[[[round(y, 4), round(x, 4)] for x, y in g.coords] for g in geoms if g.length > .002], cameras=c["cameras"], events=c["events"], forest=c["forest"], log=ch["log"], log_lo=ch["log_lo"], log_hi=ch["log_hi"], fire=ch["fire"], random=rnd["log"], sens=sens["log"], gates=c["gates"])


def risk():
    """Ризик ділянок: прогноз 2026, який записала задача FG «ризик» своєю фінальною моделлю."""
    d = here(datasets.load("h3_risk_cells").df)
    g = gpd.read_file(P / "risk_h3_2026.geojson").set_index("h3").reindex(d.h3)
    p = g.risk_2026.values
    top = np.argsort(-p)[:int(len(p) * .2)]
    r = next(x for x in S["t3"]["results"] if x.get("champion"))
    global TOWERS
    TOWERS = safe(lambda: towers(d, p)) if (P / "tp_final.json").exists() else None   # вежі лише з гілкою TP
    return dict(cells=[dict(poly=[[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(d.h3.iat[i])], p=round(float(p[i]), 2)) for i in top], roc=pk(r, "test_roc_auc"))


def spread():
    if not IS_MAIN:
        return None                                                          # місця лише основного парку
    s = json.load(open(ROOT / "web" / "science" / "spread.json"))
    d = max(s["days"], key=lambda x: len(x["y24"]))
    c = lambda ids: [[[round(a, 5), round(b, 5)] for a, b in s["poly"][i]] for i in ids]
    r = next(x for x in S["t5"]["results"] if x.get("champion"))
    return dict(name=s["name"], t=d["t"], burned=c(d["burned"]), pred=c(d[f"p24_{s['champion']}"]), true=c(d["y24"]), f1=r["test_f1_24"], approach=s["champion"])


def logging():
    """Кандидати на рубки 2024 від переможця LD, як їх записала задача. Це кандидати, а не доведені рубки."""
    k = kept("logging_ld")["app"] if IS_MAIN else kept("logging_ld")["parks"].get(KEY)
    q = {x["part"] + x["approach"]: x for x in S["t19"]["results"]}
    ch = next(x for x in S["t19"]["results"] if x["part"] == "LD" and x.get("champion"))
    return dict(year=(k or {}).get("year", 2024), cand=(k or {}).get("cand", []), approach=ch["approach"], f1=pk(ch, "test_metric"), precision=pk(ch, "precision"),
                both_precision=pk(q["LMобидва"], "precision"))


def forecast():
    """Прогноз переможця LG на наступний рік, як його записала задача."""
    k = kept("logging_lg")["app"] if IS_MAIN else kept("logging_lg")["parks"][KEY]
    r = next(x for x in S["t19"]["results"] if x["part"] == "LG" and x.get("champion"))
    return dict(year=k["year"], capture=pk(r, "test_metric"), cells=k["cells"], approach=k["approach"])


def camera():
    """Дані клина камери, ті самі, що на сторінці дослідження."""
    if not IS_MAIN:
        return None                                                          # місця лише основного парку
    x = json.load(open(ROOT / "web" / "science" / "extra.json"))
    k, hx = x["d_camera"], x["c_cells"]["h3"]
    vis = set(k["visible"])
    cells = [dict(poly=[[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(hx[i])], az=k["az"][i], dist=k["dist"][i], vis=i in vis) for i in k["sector"]]
    return dict(cam=k["cam"], frame_w=k["frame_w"], fps=k["fps"], frames=k["frames"], cells=cells, firms=k["firms_seen"])


def effis():
    """FD у режимі signal: згарища EFFIS у парку з буфером і чи бачив їх FIRMS, як записала задача; точки FIRMS цього парку."""
    k = kept("fire_signal")
    ev = [e for e in k["effis"] if e["park"] == KEY]
    d = here(datasets.load("fd_signal").df)
    return dict(n=len(ev), seen=sum(e["seen"] for e in ev), fires=ev, points=len(d), spring=float(d.month.isin([3, 4]).mean()) if len(d) else 0.,
                forest=float(d.forest.mean()) if len(d) else 0., firms=[[round(a, 5), round(b, 5), int(f), str(t)[:10], "VIIRS" if s == "VIIRS" else "MODIS"]
                                                                    for a, b, f, t, s in zip(d.lat, d.lon, d.forest, d.dt, d.sensor)])


def zones():
    """Зони патрулів на наступний рік із прогнозу переможця LG поруч із «де рубали торік»: скільки нових втрат потрапило на тесті."""
    k = kept("logging_lg")["app"] if IS_MAIN else kept("logging_lg")["parks"].get(KEY)
    rows = [x for x in S["t19"]["results"] if x["part"] == "LG"]
    ch = next(x for x in rows if x.get("champion"))
    last = next((x for x in rows if x["approach"] == "де рубали торік"), {})
    from eaglewings.ds.tasks import logging_greece as LG
    d = here(datasets.load("logging_gr_cells").df)
    ly = d[(d.year == d.year.max()) & (d.ha >= .5)]
    return dict(year=k["year"], approach=k["approach"], cells=k["cells"], share=LG.TOP, capture=pk(ch, "test_metric"), last_capture=pk(last, "test_metric") if last else None,
                last=[[[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(c)] for c in ly.h3])


def routes():
    """Маршрути груп лідером RD (симуляція), лише основний парк: бази груп у ньому."""
    if not IS_MAIN or not (P / "routes.json").exists():
        return None
    return json.loads((P / "routes.json").read_text())


def tasks():
    if not IS_MAIN or not (P / "tasks.json").exists():
        return None
    return json.loads((P / "tasks.json").read_text())


def towers_only():
    """Вежі без моделі ризику цього дослідження (FG_risk лише перевіряє готову модель): вага — ранг готової моделі, якщо є, інакше лісистість."""
    d = here(datasets.load("h3_risk_cells").df)
    p = d.treecover.values.astype(float)
    f = R0 / "results" / "risk_transfer.json"
    if f.exists():
        c = json.loads(f.read_text())["cells"]
        tr = [k_ for k_ in c if k_.startswith("transfer_")]
        if tr:
            p = pd.Series(c[tr[0]], index=c["h3"]).reindex(d.h3).fillna(0).values
    out = towers(d, p)
    out["poly"] = [[[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(c)] for c in d.h3]
    return out


def safe(f):
    """Розділ застосунку від гілки, якої в цьому дослідженні немає (вимкнена в ds/branches.ACTIVE), просто пропускається."""
    try:
        return f()
    except Exception as e:
        print(f"застосунок: {f.__name__} пропущено: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


TOWERS = None
out = dict(camera=safe(camera), park=[latlon(g.exterior.coords) for g in getattr(B.PARK.simplify(.001), "geoms", [B.PARK.simplify(.001)])], burn=safe(burn), fires=safe(fires), day=safe(day),
           risk=safe(risk), towers=None, cp=safe(cameras), spread=safe(spread), logging=safe(logging), forecast=safe(forecast))
out["towers"] = TOWERS if TOWERS is not None or not (P / "tp_final.json").exists() else safe(towers_only)
out.update(effis=safe(effis), zones=safe(zones), routes=safe(routes), tasks=safe(tasks))
(P / "app_extra.json").write_text(json.dumps(out, ensure_ascii=False, default=float))
print({k: len(v) if v else 0 for k, v in out.items()})
print(KEY, out["burn"], (out["fires"] or {}).get("points"), (out["fires"] or {}).get("fires"), sum(c["opera"] for c in (out["logging"] or {}).get("cand", [])))
if "GR_PARK" not in os.environ:                        # без GR_PARK — усі парки дослідження: решта окремими процесами
    for k_ in park.PARKS:
        if k_ != park.MAIN:
            r_ = subprocess.run([sys.executable, __file__], env=dict(os.environ, GR_PARK=k_))
            if r_.returncode:
                raise SystemExit(f"застосунок, парк {k_}: впав (код {r_.returncode})")
