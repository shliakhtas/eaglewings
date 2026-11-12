"""AR (режим logging) · Де рубки повторюються: чи падають нові втрати лісу в ареали, побудовані за минулими роками.

Задача      ареал має показувати, де подія повторюється. Тут подія — рубка: гекс H3 res 9, де за рік зникло ≥ 0,5 га лісу (Hansen),
            без згарищ (та сама маска пожеж, що в LD/LG). Спосіб той самий, що в AR на пожежах.
Дані        logging_gr_cells (усі парки, роки втрат Hansen); подія переноситься в гекс H3 res 7. Основа площі — усі гекси res 7 парків із буфером.
Протокол    для року Y ареали будуються лише з подій років Y−W…Y−1. Вибір радіуса й вікна та переможця — на роках валідації,
            тест — на роках тесту (config/regions.py fetch.protocols.AR). Усі методи отримують рівно ту саму частку площі (AREA).
            CI — bootstrap по блоках H3 res 4. Окремо для кожного парку (колонки *_<парк>).
Метрики     PAI: у скільки разів щільність нових рубок у зоні вища за середню (основна). Також частка нових рубок у зоні.
Підходи
  ареалізація   DBSCAN за центрами минулих рубок + опукла оболонка
  історія       гекси, де вже рубали
  KDE           згладжена щільність минулих рубок
  прогноз LG    бали переможця LG на рік Y (вчиться лише на роках < Y), максимум у гексі res 7
  випадково     PAI ≈ 1
Вибір       найбільший PAI на валідації серед підходів, крім випадкового; тест у виборі не бере участі.
Обмеження   Hansen — будь-яка втрата крон (рубка, вітровал, шкідники), а не доведена рубка. Подій мало, інтервали широкі.
"""
import inspect
import json

import h3
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from .. import common, datasets, metrics, park, pflow, tracking
from ..common import P
from ..regions import LAT0, protocol
from ..tracking import ml
from . import areal_fires as AF
from . import logging_greece as LG

EXPERIMENT = "AR · Arealization (рубки, втрати лісу Hansen)"
RES, AREA = AF.RES, AF.AREA
EPS, WIN = [1., 2., 5., 10.], [1, 2, 3]
VAL, TEST = protocol("AR", [2021], [2022, 2023, 2024])
SEED = 42
AF.K = 111.32 * np.cos(np.radians(LAT0))           # км на градус довготи на цій широті (у AR пожеж — Греція)


def events():
    """Рубки: гекс res 9 × рік з ≥ 0,5 га нової втрати; центр гекса, гекс res 7, блок res 4, парк."""
    d = datasets.get("logging_gr_cells").df
    d = d[d.y == 1].copy()
    ll = np.array([h3.cell_to_latlng(c) for c in d.h3])
    g = pd.DataFrame(dict(year=d.year.values, lat=ll[:, 0], lon=ll[:, 1], ha=d.ha.values, h9=d.h3.values,
                          park=d.park.values if "park" in d else park.MAIN))
    g["h"] = [h3.cell_to_parent(c, RES) for c in g.h9]
    g["block"] = [h3.cell_to_parent(c, 4) for c in g.h]
    return g.reset_index(drop=True)


def base_cells():
    """Гекси res 7 усіх парків із буфером: уся площа, де шукаємо рубки."""
    out = set()
    for k in park.PARKS:
        B = common.base(k)
        zone = B.park.to_crs(common.UTM).buffer(common.CFG["buffer_m"]).to_crs(4326).iloc[0]
        out |= set(h3.geo_to_cells(zone, RES))
    return sorted(out)


def lg_scores_by_year(champ):
    """Бали переможця LG на кожен рік (навчання лише на роках < Y) → максимум у гексі res 7."""
    D = datasets.get("logging_gr_cells").df
    rng = np.random.default_rng(SEED)
    out = {}
    for Y in sorted(set(VAL) | set(TEST)):
        t, tr = D[D.year == Y], D[D.year < Y]
        s = LG.lg_scores(tr, t, rng)[champ]
        out[Y] = pd.Series(s, index=[h3.cell_to_parent(c, RES) for c in t.h3]).groupby(level=0).max()
    return out


def lg_champion():
    f = P / "experiments_summary.json"
    S = json.loads(f.read_text()) if f.exists() else {}
    ch = [r["approach"] for r in S.get("t19", {}).get("results", []) if r.get("part") == "LG" and r.get("champion")]
    return ch[0] if ch else "LightGBM"


def evaluate(g, cells, name, cfg, years, rng, lgs=None):
    rows, zones = [], {}
    for Y in years:
        past = g[(g.year < Y) & (g.year >= Y - cfg["win"])]
        if name == "прогноз LG":
            s = lgs[Y].reindex(cells).fillna(-1).values
        elif not len(past):
            continue
        else:
            s = AF.scores(name, past, cells, cfg, rng)
        z, _ = AF.pick(s, cells, rng)
        zones[Y] = z
        t = g[g.year == Y]
        rows.append(pd.DataFrame(dict(year=Y, hit=t.h.isin(z).values, block=t.block.values, share=len(z) / len(cells), h=t.h.values, park=t.park.values)))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["year", "hit", "block", "share", "h", "park"])), zones


def by_park(t, zones, PC, n_boot):
    out = {}
    for k, pc in PC.items():
        q = t[t.h.isin(pc)].reset_index(drop=True)
        if not len(q) or not pc:
            out |= {f"test_pai_{k}": np.nan, f"n_test_{k}": 0}
            continue
        q = q.assign(share=q.year.map({Y: len(zones[Y] & pc) / len(pc) for Y in q.year.unique()}).values)
        if not q.share.mean():
            out |= {f"test_pai_{k}": np.nan, f"test_capture_{k}": float(q.hit.mean()), f"n_test_{k}": len(q)}
            continue
        lo, hi = metrics.block_bootstrap(q.block.values, lambda i: AF.pai(q.iloc[i]), n=n_boot)
        out |= {f"test_pai_{k}": AF.pai(q), f"lo_{k}": lo, f"hi_{k}": hi, f"test_capture_{k}": float(q.hit.mean()), f"n_test_{k}": len(q)}
    return out


def run(n_boot=300):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark("1. dataset: logging events (Hansen loss ≥ 0.5 ha per cell)")
    g, cells = events(), base_cells()
    print(len(g), g.year.value_counts().sort_index().to_dict(), len(cells), flush=True)
    dv = datasets.save(g[["year", "lat", "lon", "ha", "h9", "h", "block", "park"]].round(5), "ar_logging", unit="рубка: гекс res 9 × рік з ≥ 0,5 га втрати",
                       targets=[], features=["lat", "lon", "year", "ha"], description="Втрати лісу Hansen без згарищ у трьох парках, як події для ареалів рубок.",
                       params=dict(res=RES, area=AREA, val=VAL, test=TEST, eps=EPS, win=WIN, min_ha=.5), sources=["Hansen GFC v1.12", "EFFIS", "NASA FIRMS"])
    rng = np.random.default_rng(SEED)
    lgs = lg_scores_by_year(lg_champion())
    PC = AF.park_cells(cells)
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_params(dict(events=len(g), cells=len(cells), val=str(VAL), test=str(TEST), lg_champion=lg_champion(), n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/areal_logging.py")
        pflow.mark("2. tune radius and window on validation")
        grid = []
        for w in WIN:
            for e in EPS:
                d, _ = evaluate(g, cells, "ареалізація", dict(eps=e, win=w), VAL, rng)
                grid.append(dict(part="grid", approach="ареалізація", eps=e, win=w, val_pai=AF.pai(d) if len(d) else np.nan, val_capture=float(d.hit.mean()) if len(d) else np.nan))
        best = max(grid, key=lambda r: -1 if r["val_pai"] != r["val_pai"] else r["val_pai"])
        rows += grid
        pflow.mark("3. approaches: validation and test")
        T_ = {}
        for name in ("ареалізація", "історія", "KDE", "прогноз LG", "випадково"):
            cfg = dict(eps=best["eps"], win=best["win"])
            v, _ = evaluate(g, cells, name, cfg, VAL, rng, lgs)
            t, zones = evaluate(g, cells, name, cfg, TEST, rng, lgs)
            lo, hi = metrics.block_bootstrap(t.block.values, lambda i: AF.pai(t.iloc[i]), n=n_boot) if len(t) else (np.nan, np.nan)
            T_[name] = t
            rows.append(dict(part="test", approach=name, eps=cfg["eps"] if name == "ареалізація" else None, win=cfg["win"],
                             val_pai=AF.pai(v) if len(v) else np.nan, val_capture=float(v.hit.mean()) if len(v) else np.nan,
                             test_pai=AF.pai(t) if len(t) else np.nan, test_capture=float(t.hit.mean()) if len(t) else np.nan, lo=lo, hi=hi,
                             n_test=len(t), n_val=len(v), share=float(t.share.mean()) if len(t) else np.nan, few=len(t) < 30, **by_park(t, zones, PC, n_boot)))
        res = pd.DataFrame(rows)
        res["champion"] = False
        q = res[(res.part == "test") & (res.approach != "випадково")]
        champ = q.loc[q.val_pai.fillna(-1).idxmax(), "approach"]
        res.loc[(res.part == "test") & (res.approach == champ), "champion"] = True
        tc = T_[champ]
        for b in ("історія", "випадково"):
            tb = T_[b]
            if b != champ and len(tb) == len(tc) and len(tc):
                dd, lo_, hi_ = metrics.paired_block_bootstrap(tc.block.values, lambda i: AF.pai(tc.iloc[i]), lambda i: AF.pai(tb.iloc[i]), n=n_boot)
                res.loc[(res.part == "test") & (res.approach == champ), [f"diff_{b}", f"diff_{b}_lo", f"diff_{b}_hi"]] = [dd, lo_, hi_]
        # ареали рубок для мапи й застосунку: усі роки, обрані параметри
        lab = DBSCAN(eps=best["eps"] / 6371.0088, min_samples=2, metric="haversine").fit_predict(np.radians(g[["lat", "lon"]].values))
        out = []
        for _, x in g.assign(lab=lab)[lab >= 0].groupby("lab"):
            if len(x) >= 3:
                try:
                    pts = x[["lon", "lat"]].values
                    hull = pts[ConvexHull(pts).vertices]
                    out.append(dict(poly=[[round(b, 4), round(a, 4)] for a, b in hull], n=len(x), ha=round(float(x.ha.sum()), 1),
                                    years=sorted(int(y) for y in x.year.unique()), park=x.park.mode().iat[0]))
                except Exception:
                    pass
        (P / "ar_map.json").write_text(json.dumps(sorted(out, key=lambda a: -a["n"])[:60], ensure_ascii=False))
        tracking.keep("areal_logging", "areals", sorted(out, key=lambda a: -a["n"])[:60])
        for r in res[res.part == "test"].to_dict("records"):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("approach", "val_pai", "test_pai", "lo", "hi", "n_test", "champion")}, flush=True)
        tracking.log_approaches(res, lambda r: r.approach, label=lambda r: "тест" if r.part == "test" else f"eps {r.eps:g} км, вікно {r.win:g}")
        ml.set_tags(dict(champion=champ, selection="max PAI на валідації, без випадкового"))
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champ)
