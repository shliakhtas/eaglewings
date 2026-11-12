"""AR · Ареалізація появи на реальних пожежах Греції: чи падають нові пожежі в ареали, побудовані за минулими роками.

Задача      ареал має показувати, де подія повторюється. Перевіряємо це на реальних даних, а не на потоці демо.
Дані        NASA FIRMS (VIIRS + MODIS), Греція, 2021–2025: точки зведені в пожежі правилом FM (2 км, 5 діб). Одиниця — гекс H3 res 7.
Протокол    для року Y ареали будуються лише з пожеж років Y−W…Y−1. Вибір радіуса й вікна на 2023, тест на 2024 і 2025. Усі методи отримують рівно ту саму площу: 1,5 % гексів.
Метрики     PAI: у скільки разів щільність нових пожеж у зоні вища за середню. Також частка нових пожеж у зоні. CI — bootstrap по блоках H3 res 4.
Підходи
  arealization  DBSCAN за центрами минулих пожеж + опукла оболонка: зрозумілі контури
  history       гекси, де вже горіло
  kde           згладжена щільність минулих пожеж
  random        випадкові гекси: PAI ≈ 1
Парки       ареали будуються по всій Греції; для кожного парку дослідження (ds/regions.py) окремо рахуємо PAI пожеж усередині парку
            з буфером 3 км: частка зони теж береться в межах гексів цього парку (колонки *_<парк>). Пожеж у парку мало, інтервали широкі.
Обмеження   даних за 2021–2022 мало, тому вікна короткі. Основа площі — гекси поруч із будь-якими детекціями за всі роки, а не вся суша. Пожежі це не єдиний тип подій демо, вирубка й машини тут не перевіряються.
"""
import inspect

import h3
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from .. import pflow
from .. import datasets, metrics, tracking
from ..common import P
from ..tracking import ml
from . import matching_benchmark as MB

EXPERIMENT = "AR · Arealization (пожежі Греції, FIRMS)"
RES, AREA = 7, .015
EPS = [2., 5., 10., 20.]
WIN = [1, 2]
VAL, TEST = 2023, [2024, 2025]
SEED = 42
K = 111.32 * np.cos(np.radians(39.))


def fires():
    s = MB.signals()
    s = s.reset_index(drop=True)
    s["ev"] = MB.cluster(s, dict(kind="st", km=2., days=5.))
    g = s.groupby("ev").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), n=("dt", "size"), t=("dt", "min"))
    g["year"] = g.t.dt.year
    g["h"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(g.lat, g.lon)]
    g["block"] = [h3.cell_to_parent(c, 4) for c in g.h]
    return g.reset_index(drop=True)


def base(g):
    cells = set()
    for h in g.h.unique():
        cells |= set(h3.grid_disk(h, 3))
    return sorted(cells)


def rank_area(past, cells, eps):
    """Гекси в ареалах спершу: більший ареал вище. Ареал = опукла оболонка кластера + буфер eps/2."""
    lab = DBSCAN(eps=eps / 6371.0088, min_samples=2, metric="haversine").fit_predict(np.radians(past[["lat", "lon"]].values))
    p = past.assign(lab=lab)
    score = dict.fromkeys(cells, 0.)
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    x, y = ll[:, 1] * K, ll[:, 0] * 110.57
    for l, q in p[p.lab >= 0].groupby("lab"):
        px, py = q.lon.values * K, q.lat.values * 110.57
        d = np.full(len(cells), np.inf)
        pts = np.c_[px, py]
        if len(q) >= 3:
            try:
                eq = ConvexHull(pts).equations
                inside = (eq[:, :2] @ np.c_[x, y].T + eq[:, 2:]).max(0) <= eps / 2
            except Exception:
                inside = np.zeros(len(cells), bool)
        else:
            inside = np.zeros(len(cells), bool)
        near = ((x[:, None] - px) ** 2 + (y[:, None] - py) ** 2).min(1) <= (eps / 2) ** 2
        for i in np.nonzero(inside | near)[0]:
            score[cells[i]] += len(q) + 1e-3
    return np.array([score[c] for c in cells])


def rank_history(past, cells):
    n = past.h.value_counts()
    return np.array([float(n.get(c, 0)) for c in cells])


def rank_kde(past, cells, bw=8.):
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    x, y = ll[:, 1] * K, ll[:, 0] * 110.57
    px, py = past.lon.values * K, past.lat.values * 110.57
    out = np.zeros(len(cells))
    for i in range(0, len(cells), 2000):
        out[i:i + 2000] = np.exp(-((x[i:i + 2000, None] - px) ** 2 + (y[i:i + 2000, None] - py) ** 2) / (2 * bw ** 2)).sum(1)
    return out


def scores(name, past, cells, cfg, rng):
    if name == "ареалізація":
        return rank_area(past, cells, cfg["eps"])
    if name == "історія":
        return rank_history(past, cells)
    if name == "KDE":
        return rank_kde(past, cells)
    return rng.random(len(cells))


def pick(s, cells, rng):
    k = int(round(len(cells) * AREA))
    o = np.lexsort((rng.random(len(cells)), -s))[:k]
    return {cells[i] for i in o}, k


ZONES = {}                                       # (підхід, рік) → гекси зони останнього evaluate: для зрізу по парках


def evaluate(g, cells, name, cfg, years, rng):
    rows = []
    for Y in years:
        past = g[(g.year < Y) & (g.year >= Y - cfg["win"])]
        if not len(past):
            continue
        z, k = pick(scores(name, past, cells, cfg, rng), cells, rng)
        ZONES[(name, Y)] = z
        t = g[g.year == Y]
        rows.append(pd.DataFrame(dict(year=Y, hit=t.h.isin(z).values, block=t.block.values, share=len(z) / len(cells), h=t.h.values)))
    return pd.concat(rows, ignore_index=True)


def park_cells(cells):
    """Гекси res RES кожного парку дослідження з буфером 3 км, що входять в основу площі."""
    from .. import common, park
    if len(park.PARKS) < 2:
        return {}
    base_ = set(cells)
    out = {}
    for k in park.PARKS:
        B = common.base(k)
        zone = B.park.to_crs(common.UTM).buffer(common.CFG["buffer_m"]).to_crs(4326).iloc[0]
        out[k] = set(h3.geo_to_cells(zone, RES)) & base_
    return out


def by_park(t, name, PC, n_boot):
    """PAI пожеж усередині кожного парку: попадання серед пожеж парку / частка зони в гексах парку того ж року."""
    out = {}
    for k, pc in PC.items():
        q = t[t.h.isin(pc)].reset_index(drop=True)
        if not len(q) or not pc:
            out |= {f"test_pai_{k}": np.nan, f"n_test_{k}": 0}
            continue
        sh = q.year.map({Y: len(ZONES[(name, Y)] & pc) / len(pc) for Y in q.year.unique()}).values
        q = q.assign(share=sh)
        if not q.share.mean():                               # зона цього підходу не заходить у парк: PAI не визначений, а не 0
            out |= {f"test_pai_{k}": np.nan, f"test_capture_{k}": float(q.hit.mean()), f"share_{k}": 0., f"n_test_{k}": len(q)}
            continue
        lo, hi = metrics.block_bootstrap(q.block.values, lambda i: pai(q.iloc[i]), n=n_boot)
        out |= {f"test_pai_{k}": pai(q), f"lo_{k}": lo, f"hi_{k}": hi, f"test_capture_{k}": float(q.hit.mean()), f"share_{k}": float(q.share.mean()), f"n_test_{k}": len(q)}
    return out


def pai(d):
    return d.hit.mean() / d.share.mean() if d.share.mean() else 0.


def run(n_boot=300):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: merged fires')
    g = fires()
    cells = base(g)
    print(len(g), g.year.value_counts().sort_index().to_dict(), len(cells), flush=True)
    dv = datasets.save(g[["year", "lat", "lon", "n", "h", "block"]].round(5), "ar_fires", unit="пожежа після зведення FM", targets=[], features=["lat", "lon", "year", "n"], description="Пожежі Греції 2021–2025 за FIRMS, зведені правилом FM.",
                       params=dict(res=RES, area=AREA, val=VAL, test=TEST, eps=EPS, win=WIN), sources=["NASA FIRMS VIIRS, MODIS"])
    rng = np.random.default_rng(SEED)
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/areal_fires.py")
        grid = []
        pflow.mark('2. tune on validation')
        for w in WIN:
            for e in EPS:
                d = evaluate(g, cells, "ареалізація", dict(eps=e, win=w), [VAL], rng)
                grid.append(dict(part="grid", approach="ареалізація", eps=e, win=w, val_pai=pai(d), val_capture=float(d.hit.mean())))
        best = max(grid, key=lambda r: r["val_pai"])
        print("best", best, flush=True)
        rows += grid
        T_ = {}
        PC = park_cells(cells)
        pflow.mark('3. approaches on test')
        for name, cfg in (("ареалізація", dict(eps=best["eps"], win=best["win"])), ("історія", dict(win=best["win"])), ("KDE", dict(win=best["win"])), ("випадково", dict(win=best["win"]))):
            v = evaluate(g, cells, name, cfg, [VAL], rng)
            t = evaluate(g, cells, name, cfg, TEST, rng)
            lo, hi = metrics.block_bootstrap(t.block.values, lambda i: pai(t.iloc[i]), n=n_boot)
            T_[name] = t.reset_index(drop=True)
            rows.append(dict(part="test", approach=name, eps=cfg.get("eps"), win=cfg["win"], val_pai=pai(v), val_capture=float(v.hit.mean()), test_pai=pai(t), test_capture=float(t.hit.mean()), lo=lo, hi=hi, n_test=len(t), share=float(t.share.mean()), champion=name == "ареалізація",
                             **by_park(t, name, PC, n_boot)))
        pflow.mark('4. difference vs history and random')
        res = pd.DataFrame(rows)
        res["champion"] = res.champion.fillna(False).astype(bool)
        tc = T_["ареалізація"]
        for b in ("історія", "випадково"):                                    # парна різниця PAI на тих самих блоках
            tb = T_[b]
            if len(tb) == len(tc):
                dd, lo_, hi_ = metrics.paired_block_bootstrap(tc.block.values, lambda i: pai(tc.iloc[i]), lambda i: pai(tb.iloc[i]), n=n_boot)
                res.loc[(res.part == "test") & (res.approach == "ареалізація"), [f"diff_{b}", f"diff_{b}_lo", f"diff_{b}_hi"]] = [dd, lo_, hi_]
        # ареали для мапи: усі роки, обрані параметри
        lab = DBSCAN(eps=best["eps"] / 6371.0088, min_samples=2, metric="haversine").fit_predict(np.radians(g[["lat", "lon"]].values))
        q = g.assign(lab=lab)
        out = []
        for l, x in q[q.lab >= 0].groupby("lab"):
            if len(x) >= 3:
                try:
                    pts = x[["lon", "lat"]].values
                    hull = pts[ConvexHull(pts).vertices]
                    out.append(dict(poly=[[round(b, 4), round(a, 4)] for a, b in hull], n=len(x)))
                except Exception:
                    pass
        import json
        (P / "ar_map.json").write_text(json.dumps(sorted(out, key=lambda a: -a["n"])[:60], ensure_ascii=False))
        for r in res[res.part == "test"].to_dict("records"):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("approach", "val_pai", "test_pai", "test_capture", "lo", "hi", "n_test", "share")}, flush=True)
        pflow.mark('5. approaches: results per algorithm')
        tracking.log_approaches(res, lambda r: r.approach, label=lambda r: "тест" if r.part == "test" else f"eps {r.eps:g} км, вікно {r.win:g}")
        pflow.mark('6. save')
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag)
