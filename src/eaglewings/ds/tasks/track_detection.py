"""SD (режим tracks) · Чи видно лісові колії на знімках Sentinel-2 (10 м, літо).

Задача      колія в лісі — слід машин і джипів. Чи видно на літньому композиті Sentinel-2 ґрунтову колію на відрізку 100 м у лісі.
Еталон      OpenStreetMap (розмічено людьми, незалежно від Sentinel-2). Позитив — відрізок 100 м лінії highway=track у лісі
            (Hansen treecover2000 ≥ 30 %) у парку з буфером. Негатив — та сама лінія, зсунута на 150 м у випадковому напрямку, якщо вона
            лягає в ліс далі 60 м від будь-якої дороги OSM: форма й ландшафт ті самі, різниця лише в наявності колії. Негативи штучні.
Дані        медіанний літній композит Sentinel-2 L2A 15.06–15.09 року YEAR, 10 м (AWS Earth Search), у вікні сітки Hansen кожного парку.
Протокол    вчимося на двох парках, тест на третьому, по черзі (leave-one-park-out). Усередині двох навчальних парків блоки ~1 км
            розкладено: 2/3 — навчання, 1/3 — відкладені блоки. Вибір підходу й поріг F1 — лише на відкладених блоках навчальних парків;
            тестовий парк у виборі не бере участі. CI — bootstrap по блоках ~1 км тестових парків.
Метрики     ROC AUC — основна (середнє по трьох тестових парках: шкали правил між парками не зводяться); PR AUC; F1 при порозі
            з відкладених блоків; ROC AUC окремо в кожному парку.
Підходи
  brightness      база: колія — оголений ґрунт, відрізок яскравіший у видимих каналах
  side_contrast   лінія темніша за NDVI, ніж паралельні смуги лісу за ±30 м
  line_filter     видовжений провал NDVI уздовж лінії (лінійний фільтр, 4 напрями)
  logreg          логістична регресія на ознаках відрізка
  lightgbm        LightGBM на тих самих ознаках
Вибір       найбільший середній ROC AUC на відкладених блоках навчальних парків.
Обмеження   колії під кронами з орбіти не видно; зсув ліній OSM 10–20 м; негативи штучні, тож це перевірка видимості, а не пошук нових колій.
"""
import hashlib
import inspect
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from scipy import ndimage as ndi
from shapely import affinity
from shapely.ops import substring

from .. import common, datasets, metrics, park, pflow, tracking
from ..regions import LAT0
from ..tracking import ml
from . import logging_greece as LG

EXPERIMENT = "SD · Лісові колії на Sentinel-2 10 м"
BANDS10 = {"blue": "blue", "green": "green", "red": "red", "nir": "nir"}
YEAR = 2024
UP = 3                                   # 30 м Hansen → 10 м
SEG_M, SHIFT_M, SIDE_M = 100, 150, 30
KX, KY = 111320 * math.cos(math.radians(LAT0)), 111320.0


def window10(k):
    """Сітка 10 м у вікні Hansen парку k, маска зони (1 — парк, 2 — буфер) і лісу."""
    B = common.base(k)
    tr = rasterio.Affine(B.TR.a / UP, 0, B.TR.c, 0, B.TR.e / UP, B.TR.f)
    shp = (B.H * UP, B.W * UP)
    zone = np.zeros(shp, np.uint8)
    buf = B.park_u.buffer(common.CFG["buffer_m"]).to_crs(4326).iloc[0]
    zone[features.geometry_mask([buf], shp, tr, invert=True)] = 2
    zone[features.geometry_mask([B.PARK], shp, tr, invert=True)] = 1
    forest = np.kron(B.tc >= 30, np.ones((UP, UP), bool))
    bb = (B.TR.c, B.TR.f + B.TR.e * B.H, B.TR.c + B.TR.a * B.W, B.TR.f)
    return tr, shp, zone, forest, bb


def _scene10(it, tr, shp):
    A = it["assets"]
    good = np.isin(LG._read(A["scl"]["href"], tr, shp, Resampling.nearest), [4, 5, 6, 7])
    if good.mean() < .02:
        return None
    out = {}
    for k, a in BANDS10.items():
        raw = LG._read(A[a]["href"], tr, shp, Resampling.bilinear).astype(np.float32)
        x = raw * 1e-4
        x[~good | (raw <= 0)] = np.nan
        out[k] = x.astype(np.float16)
    return out


def composite10(year, k):
    """Медіанний літній композит 10 м у вікні парку k. Кеш у теці парку."""
    d = park.cache(k) / "s2_10m"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"c10_{year}.npz"
    if f.exists():
        z = np.load(f)
        return {b: z[b].astype(np.float32) for b in BANDS10}
    tr, shp, *_, bb = window10(k)

    def job(it):
        try:
            return _scene10(it, tr, shp)
        except Exception as e:
            print("skip", it["id"], str(e)[:120], flush=True)
            return None
    with ThreadPoolExecutor(6) as ex:
        res = [r for r in ex.map(job, LG.scenes(year, bb)) if r is not None]
    comp = {}
    for b in BANDS10:
        comp[b] = np.nanmedian(np.stack([r[b] for r in res]).astype(np.float32), 0) if res else np.full(shp, np.nan, np.float32)
    print(k, year, len(res), "сцен 10 м", f"{np.isfinite(comp['nir']).mean():.0%} покрито", flush=True)
    np.savez_compressed(f, **{b: v.astype(np.float16) for b, v in comp.items()})
    return {b: v.astype(np.float16).astype(np.float32) for b, v in comp.items()}


def _lines(n=7):
    ks = []
    for ang in (0, 45, 90, 135):
        kk = np.zeros((n, n), np.float32)
        c = n // 2
        for t in range(-c, c + 1):
            kk[int(round(c - t * math.sin(math.radians(ang)))), int(round(c + t * math.cos(math.radians(ang))))] = 1
        ks.append(kk / kk.sum())
    return ks


LINES = _lines()
PF = ["vis", "nir", "ndvi", "ndvi_dip_9", "ndvi_dip_17", "vis_peak_9", "line_max", "line_aniso"]


def pix_feats(c):
    nd = (c["nir"] - c["red"]) / np.maximum(c["nir"] + c["red"], 1e-4)
    vis = c["blue"] + c["green"] + c["red"]
    nd0 = np.nan_to_num(nd, nan=np.nanmedian(nd))
    v0 = np.nan_to_num(vis, nan=np.nanmedian(vis))
    F = dict(vis=vis, nir=c["nir"], ndvi=nd)
    for s in (9, 17):
        F[f"ndvi_dip_{s}"] = ndi.uniform_filter(nd0, s) - nd
    F["vis_peak_9"] = vis - ndi.uniform_filter(v0, 9)
    m7 = ndi.uniform_filter(nd0, 7)
    lr = np.stack([m7 - ndi.convolve(nd0, kk) for kk in LINES])
    F["line_max"] = lr.max(0)
    F["line_aniso"] = lr.max(0) - lr.min(0)
    return {k: v.astype(np.float32) for k, v in F.items()}


def _rc(tr, shp, g, n=12):
    inv = ~tr
    pts = [g.interpolate(i / (n - 1), normalized=True) for i in range(n)]
    cc, rr = zip(*[inv * (p.x, p.y) for p in pts])
    rr, cc = np.floor(rr).astype(int), np.floor(cc).astype(int)
    if rr.min() < 0 or cc.min() < 0 or rr.max() >= shp[0] or cc.max() >= shp[1]:
        return None
    return rr, cc


def _side(seg, d):
    (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
    dx, dy = (x1 - x0) * KX, (y1 - y0) * KY
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy / L, dx / L
    return [affinity.translate(seg, s * d * nx / KX, s * d * ny / KY) for s in (1, -1)]


def segments(k):
    """Позитиви (колії OSM у лісі зони парку) і негативи (ті самі лінії, зсунуті на 150 м у ліс без доріг)."""
    tr, shp, zone, forest, _ = window10(k)
    R = common.base(k).roads
    any_road = features.rasterize(((g, 1) for g in R.geometry), shp, transform=tr, all_touched=True, dtype=np.uint8) > 0
    d_road = ndi.distance_transform_edt(~any_road) * 10
    rng = np.random.default_rng(list(park.ALL).index(k) + 1)
    out = []
    for tid, g in zip(R.index, R.geometry):
        if R.highway[tid] != "track" or g.geom_type != "LineString":
            continue
        L = g.length * KX
        for s0 in np.arange(0, L - SEG_M, SEG_M):
            seg = substring(g, s0 / KX, (s0 + SEG_M) / KX)
            p = _rc(tr, shp, seg)
            if p is None:
                continue
            rr, cc = p
            if not ((zone[rr, cc] > 0).all() and forest[rr, cc].mean() > .5):
                continue
            part = "inside" if (zone[rr, cc] == 1).mean() > .5 else "buffer"
            out.append(dict(y=1, geom=seg, track=f"{k}_{tid}", part=part))
            for _ in range(3):
                a = rng.uniform(0, 2 * np.pi)
                sh = affinity.translate(seg, SHIFT_M * np.cos(a) / KX, SHIFT_M * np.sin(a) / KY)
                q = _rc(tr, shp, sh)
                if q is None:
                    continue
                r2, c2 = q
                if (zone[r2, c2] > 0).all() and forest[r2, c2].mean() > .5 and (d_road[r2, c2] > 60).all():
                    out.append(dict(y=0, geom=sh, track=f"{k}_{tid}", part="inside" if (zone[r2, c2] == 1).mean() > .5 else "buffer"))
                    break
    return out


def seg_feats(segs, F, tr, shp):
    rows = []
    for s in segs:
        rr, cc = _rc(tr, shp, s["geom"])
        r = {}
        for kk in PF:
            v = F[kk][rr, cc]
            r[f"{kk}_mean"], r[f"{kk}_max"] = np.nanmean(v), np.nanmax(v)
        sides = [x for x in (_rc(tr, shp, g) for g in _side(s["geom"], SIDE_M)) if x is not None]
        if sides:
            r["side_dndvi"] = np.nanmean([np.nanmean(F["ndvi"][a, b]) for a, b in sides]) - r["ndvi_mean"]
            r["side_dvis"] = r["vis_mean"] - np.nanmean([np.nanmean(F["vis"][a, b]) for a, b in sides])
        else:
            r["side_dndvi"] = r["side_dvis"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


FEATS = [f"{k}_{s}" for k in PF for s in ("mean", "max")] + ["side_dndvi", "side_dvis"]


def table(year):
    out = []
    for k in park.PARKS:
        tr, shp, *_ = window10(k)
        segs = segments(k)
        if not segs:
            print(k, "колій OSM у лісі немає", flush=True)
            continue
        F = pix_feats(composite10(year, k))
        d = seg_feats(segs, F, tr, shp)
        c = [s["geom"].interpolate(.5, normalized=True) for s in segs]
        d = d.assign(y=[s["y"] for s in segs], park=k, part=[s["part"] for s in segs], track=[s["track"] for s in segs],
                     lat=[p.y for p in c], lon=[p.x for p in c])
        d["block"] = [f"{k}_{int(la / .009)}_{int(lo / .0135)}" for la, lo in zip(d.lat, d.lon)]   # ~1 км
        out.append(d)
        print(k, year, "колій", int(d.y.sum()), "контролів", int((d.y == 0).sum()), flush=True)
    return pd.concat(out, ignore_index=True)


def build_dataset():
    df = table(YEAR).round(5)
    return datasets.save(
        df, "track_segments", unit=f"відрізок 100 м у лісі парку з буфером (композит Sentinel-2 літо {YEAR}, 10 м)",
        description="Позитив — відрізок колії OSM (highway=track) у лісі; негатив — та сама лінія, зсунута на 150 м у ліс далі 60 м від доріг OSM.",
        targets=["y"], features=FEATS, params=dict(year=YEAR, seg_m=SEG_M, shift_m=SHIFT_M, side_m=SIDE_M, res_m=10),
        sources=["Sentinel-2 L2A (AWS Earth Search)", "OpenStreetMap highway=*", "Hansen GFC v1.12 treecover2000"])


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=15, min_child_samples=30, subsample=.8, subsample_freq=1,
                          colsample_bytree=.8, verbose=-1, random_state=0)


def _lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=.5, class_weight="balanced"))


APPROACHES = {
    "brightness": dict(kind="baseline", hyp="Колія — оголений ґрунт: відрізок яскравіший у видимих каналах", score=lambda d: d.vis_mean),
    "side_contrast": dict(kind="rule", hyp="Лінія темніша за NDVI, ніж паралельні смуги лісу за ±30 м", score=lambda d: d.side_dndvi),
    "line_filter": dict(kind="rule", hyp="Видовжений провал NDVI уздовж лінії (лінійний фільтр, 4 напрями)", score=lambda d: d.line_aniso_mean),
    "logreg": dict(kind="model", hyp="Логістична регресія на ознаках відрізка (середнє, максимум, бічний контраст)", make=_lr),
    "lightgbm": dict(kind="model", hyp="LightGBM на тих самих ознаках відрізка", make=_lgbm),
}


def _X(d):
    return d[FEATS].fillna(0)


def _score(a, fit, d):
    A = APPROACHES[a]
    if A["kind"] == "model":
        return A["make"]().fit(_X(fit), fit.y).predict_proba(_X(d))[:, 1]
    return np.nan_to_num(A["score"](d).values.astype(float), nan=-1e9)


def _auc(y, s):
    return metrics.ranking(y, s)["roc_auc"] if 0 < y.sum() < len(y) else np.nan


def _best_thr(y, s):
    qs = np.nanquantile(s, np.linspace(.05, .95, 37))
    f = lambda t: metrics.area(((s >= t) & (y == 1)).sum(), ((s >= t) & (y == 0)).sum(), ((s < t) & (y == 1)).sum())["f1"]
    return max(qs, key=f)


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark("1. dataset: OSM tracks and shifted controls on Sentinel-2 10 m")
    dv = build_dataset()
    df = dv.df
    PK = list(dict.fromkeys(df.park))
    y = df.y.values
    held = np.array([int(hashlib.md5(b.encode()).hexdigest(), 16) % 3 == 0 for b in df.block])   # відкладені блоки ~1 км, детерміновано
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "training", "y", files=True)
        ml.log_params(dict(protocol="leave-one-park-out; вибір і поріг на відкладених блоках навчальних парків", year=YEAR, parks=",".join(PK), n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/track_detection.py")
        pflow.mark("2. approaches: pick on held-out blocks of training parks, test on the unseen park")
        for a, A in APPROACHES.items():
            sc, pred, vals = np.full(len(df), np.nan), np.zeros(len(df), bool), []
            for k in PK:
                te, trp = df.park.values == k, df.park.values != k
                fit, hv = trp & ~held, trp & held
                s_h = _score(a, df[fit], df[hv])
                vals.append(_auc(y[hv], s_h))
                thr = _best_thr(y[hv], s_h)
                s_te = _score(a, df[fit], df[te])
                sc[te], pred[te] = s_te, s_te >= thr
            per = {k: _auc(y[df.park.values == k], sc[df.park.values == k]) for k in PK}
            auc = lambda idx: np.nanmean([_auc(y[idx][df.park.values[idx] == k], sc[idx][df.park.values[idx] == k]) for k in PK])
            lo, hi = metrics.block_bootstrap(df.block.values, auc, n=n_boot)
            f = metrics.area((pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum())
            r = dict(approach=a, kind=A["kind"], hypothesis=A["hyp"], val_roc_auc=float(np.nanmean(vals)), test_roc_auc=float(np.nanmean(list(per.values()))),
                     ci_lo=lo, ci_hi=hi, test_pr_auc=float(np.nanmean([metrics.ranking(y[df.park.values == k], sc[df.park.values == k])["pr_auc"] for k in PK])),
                     test_f1=f["f1"], test_precision=f["precision"], test_recall=f["recall"], prevalence=float(y.mean()), n=len(df), tracks=int(y.sum()))
            r |= {f"test_roc_auc_{k}": v for k, v in per.items()} | {f"tracks_{k}": int(y[df.park.values == k].sum()) for k in PK}
            with ml.start_run(run_name=a, nested=True):
                ml.set_tags(dict(approach=a, kind=A["kind"], hypothesis=A["hyp"], role="approach"))
                ml.log_metrics(tracking.clean({kk: v for kk, v in r.items() if isinstance(v, (int, float))}))
            rows.append(r)
            print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items() if kk != "hypothesis"}, flush=True)
        R = pd.DataFrame(rows)
        champ = R.sort_values("val_roc_auc", ascending=False).approach.iloc[0]
        R["champion"] = R.approach.eq(champ)
        ml.set_tags(dict(champion=champ, selection="max ROC AUC на відкладених блоках навчальних парків"))
        tracking.log_fig(_fig(R), "approaches")
        ml.log_table(R, "results.json")
        return R, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champ)


def _fig(R):
    import matplotlib.pyplot as plt
    KC = {"model": "#3987e5", "rule": "#199e70", "baseline": "#9aa59b"}
    t = R.sort_values("test_roc_auc")
    fig, ax = plt.subplots(figsize=(7, .5 * len(t) + 1.2))
    for i, r in enumerate(t.itertuples()):
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=KC[r.kind], lw=2, alpha=.5)
        ax.scatter(r.test_roc_auc, i, color=KC[r.kind], s=60, zorder=3, edgecolors="k" if r.champion else "none")
    ax.axvline(.5, ls="--", color="#999")
    ax.set_yticks(range(len(t)), t.approach)
    ax.set_xlabel("ROC AUC на парку, якого модель не бачила (95 % CI)")
    ax.set_title("SD · лісові колії на Sentinel-2 10 м", loc="left")
    fig.tight_layout()
    return fig
