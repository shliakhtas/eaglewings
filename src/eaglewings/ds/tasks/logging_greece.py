"""LD · LM · LG для Греції: детекція, матчинг і розвиток вирубки в НП Дадья.

Задача      перевірити вирубку на грецькому парку й чесно показати, що можна виміряти там, де рубок мало.
Дані        Sentinel-2 L2A (AWS Earth Search): літні композити 15.06–15.09; NASA OPERA DIST-ALERT через GIBS; Hansen GFC v1.12 як лейбл; рельєф SRTM, ліс, дороги й села.
Лейбл       Hansen: втрата крон саме в році Y. Це не «рубка», а будь-яка втрата лісу.
Пожежі      це не вирубка, тому пікселі ≤ 1 км від точок FIRMS (є лише з 2021) і в контурах EFFIS + 500 м за роки Y і Y−1 відкинуті. Лишкові згарища все одно можливі.
Протокол    кілька парків: квадрати ~6 км усіх парків розкладені на фолди (вибір і тест на різних квадратах); окремо LD_transfer і LM_transfer —
            вчимося на інших парках, тест на парку, якого модель не бачила (не для вибору). З одним парком — половини парку за довготою.
  LD        вибір 2018–2021, тест 2022–2024. CI — bootstrap по блоках ~5 км.
  LM        Sentinel-2 навчається на 2018–2021, правило OPERA обирається на 2023, тест 2024.
  LG        гекс H3 res 9 × рік; прогноз на рік Y вчиться на роках < Y; вибір підходу 2018–2020, тест 2021–2024. CI — bootstrap по блоках H3 res 6.
Метрики     LD, LM — F1 площі; LG — частка нових втрат у 11 % найризикованіших гексів.
Обмеження   нефорестних змін мало: 11–190 га на рік поза згарищами. Позитивів мало, тому інтервали широкі. 2023–2024 найзабрудненіші наслідками пожежі.
"""
import glob
import inspect
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
import requests
from lightgbm import LGBMClassifier
from rasterio import features
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage as ndi
from shapely.geometry import box, shape
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from .. import common, datasets, metrics, opera, tracking
from ..common import CACHE, RAW
from ..tracking import ml

EXPERIMENT = "LD · LM · LG Греція (вирубка в НП Дадья)"
YEARS = LOG_YEARS
BANDS = {"red": "red", "nir": "nir08", "sw1": "swir16", "sw2": "swir22"}
NEG, SEED = .10, 42
S2 = CACHE / "s2"
S2.mkdir(exist_ok=True)
FEATS = [f"{b}_{w}" for w in ("pre", "post") for b in ("red", "nir", "sw1", "sw2", "ndvi", "nbr", "ndmi")] + ["d_ndvi", "d_nbr", "d_ndmi", "r_nir", "elev", "slope", "treecover"]
RULES = dict(post_ndvi=("ndvi_post", -1), dndvi=("d_ndvi", 1), dnbr=("d_nbr", 1), dndmi=("d_ndmi", 1))
SETS = {"conf50": (6, 8), "conf": (3, 6, 7, 8), "prov50": (5, 6, 8), "all50": (4, 5, 6, 8)}
EFFIS = "https://maps.effis.emergency.copernicus.eu/effis"
CELL_F = ["elev", "slope", "treecover", "d_road", "d_village", "track_km_km2", "loss1", "loss2", "loss3_5", "loss_cum", "nb_loss1"]
TOP = .11


def scenes(year, bb):
    r = requests.post("https://earth-search.aws.element84.com/v1/search", timeout=120, json={"collections": ["sentinel-2-l2a"], "bbox": list(bb), "datetime": f"{year}-06-15T00:00:00Z/{year}-09-15T00:00:00Z",
                                                                                             "query": {"eo:cloud_cover": {"lt": 25}}, "limit": 200}).json()
    by = defaultdict(list)
    for f in r["features"]:
        if shape(f["geometry"]).intersects(box(*bb)):
            by[f["properties"].get("grid:code") or f["id"].split("_")[1]].append(f)
    return [x for fs in by.values() for x in sorted(fs, key=lambda f: f["properties"]["eo:cloud_cover"])[:2]]


def _read(href, tr, shp, rs):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE"):
        with rasterio.open(href) as src, WarpedVRT(src, crs="EPSG:4326", transform=tr, width=shp[1], height=shp[0], resampling=rs, nodata=0) as v:
            return v.read(1)


def _scene(it, tr, shp):
    A = it["assets"]
    good = np.isin(_read(A["scl"]["href"], tr, shp, Resampling.nearest), [4, 5, 6, 7])
    if good.mean() < .02:
        return None
    out = {}
    for k, a in BANDS.items():
        raw = _read(A[a]["href"], tr, shp, Resampling.average).astype(np.float32)
        x = raw * 1e-4
        x[~good | (raw <= 0)] = np.nan
        out[k] = x.astype(np.float16)
    return out


def composite(year, key=None):
    """Літній медіанний композит Sentinel-2 року year на сітці Hansen парку key (кеш у теці парку)."""
    from .. import park
    d = park.cache(key or park.KEY) / "s2"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"comp_{year}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k].astype(np.float32) for k in BANDS}, list(z["ids"])
    B = common.base(key)
    tr, shp = B.TR, (B.H, B.W)
    bb = (B.TR.c, B.TR.f + B.TR.e * B.H, B.TR.c + B.TR.a * B.W, B.TR.f)

    def job(it):
        try:
            return it["id"], _scene(it, tr, shp)
        except Exception as e:
            print("skip", it["id"], str(e)[:100], flush=True)
            return it["id"], None
    with ThreadPoolExecutor(6) as ex:
        res = [r for r in ex.map(job, scenes(year, bb)) if r[1] is not None]
    comp = {k: np.nanmedian(np.stack([r[1][k] for r in res]).astype(np.float32), 0) if res else np.full(shp, np.nan, np.float32) for k in BANDS}
    ids = [r[0] for r in res]
    print(year, len(ids), "сцен", f"{np.isfinite(comp['nir']).mean():.0%}", flush=True)
    np.savez_compressed(f, ids=np.array(ids), **{k: v.astype(np.float16) for k, v in comp.items()})
    return {k: v.astype(np.float16).astype(np.float32) for k, v in comp.items()}, ids     # як із кешу: перший і повторний запуск дають однакові числа


def _idx(c):
    nd = lambda a, b: (a - b) / np.maximum(a + b, 1e-4)
    return dict(ndvi=nd(c["nir"], c["red"]), nbr=nd(c["nir"], c["sw2"]), ndmi=nd(c["nir"], c["sw1"]))


def effis_year(y):
    d = RAW / "benchmarks" / "effis"
    f = d / f"effis_{y}_{COUNTRY.lower()}.geojson"
    if not f.exists():
        r = requests.get(EFFIS, timeout=300, params=dict(service="WFS", version="1.1.0", request="GetFeature", typename=f"ms:modis.ba.poly.{y}", bbox=f"{EFFIS_BBOX},EPSG:4326", outputFormat="application/json; subtype=geojson"))
        r.raise_for_status()
        f.write_bytes(r.content)
    return gpd.read_file(f)


def fire_masks(key=None):
    """Рік → пікселі, де втрату лісу не можна вважати вирубкою: згарища й теплові точки цього й попереднього року."""
    B = common.base(key)
    col, row = ~B.TR * (B.firms.longitude.values, B.firms.latitude.values)
    per = {}
    for y in range(2016, 2025):
        m = np.zeros((B.H, B.W), bool)
        q = B.firms.year.values == y
        r, c = row[q].astype(int), col[q].astype(int)
        ok = (r >= 0) & (r < B.H) & (c >= 0) & (c < B.W)
        m[r[ok], c[ok]] = True
        m = ndi.binary_dilation(m, iterations=40)
        g = effis_year(y)
        if len(g):
            g = g.to_crs(common.UTM).buffer(500).to_crs(4326)
            g = g[g.intersects(box(B.TR.c, B.TR.f + B.TR.e * B.H, B.TR.c + B.TR.a * B.W, B.TR.f))]
            if len(g):
                m |= features.geometry_mask(list(g), (B.H, B.W), B.TR, invert=True)
        per[y] = m
    return {y: per[y] | per[y - 1] for y in range(2017, 2025)}


def feature_rasters(year, key=None):
    """Ознаки FEATS для кожного пікселя сітки парку key за пару знімків year−1 → year і маска, де знімки є.
    Одна функція і для датасету (pixels), і для кандидатів на мапі (candidates)."""
    B = common.base(key)
    Z = common.dem(key)
    gy, gx = np.gradient(Z)
    slope = np.degrees(np.arctan(np.hypot(gx / (B.TR.a * 111320 * np.cos(np.radians(LAT0))), gy / (abs(B.TR.e) * 111320))))
    pre, post = composite(year - 1, key)[0], composite(year, key)[0]
    ip, iq = _idx(pre), _idx(post)
    F = dict(elev=Z, slope=slope, treecover=B.tc)
    for k in BANDS:
        F[f"{k}_pre"], F[f"{k}_post"] = pre[k], post[k]
    for k in ("ndvi", "nbr", "ndmi"):
        F[f"{k}_pre"], F[f"{k}_post"], F[f"d_{k}"] = ip[k], iq[k], ip[k] - iq[k]
    F["r_nir"] = post["nir"] / np.maximum(pre["nir"], 1e-4)
    ok = np.isfinite(pre["nir"]) & np.isfinite(post["nir"]) & np.isfinite(pre["sw2"]) & np.isfinite(post["sw2"])
    return F, ok


def pixels(fire):
    B = common.base()
    rng = np.random.default_rng(SEED)
    rows = []
    for Y in YEARS:
        F, ok = feature_rasters(Y)
        base = (B.tc >= 40) & ~fire[Y] & ok
        for lab, m, w in ((1, base & (B.loss == Y - 2000), 1.), (0, base & (B.loss == 0) & (rng.random(B.loss.shape) < NEG), 1 / NEG)):
            r, c = np.nonzero(m)
            d = dict(year=Y, row=r, col=c, y=lab, w=w, pix_ha=B.PIX_HA[r, c], elev=F["elev"][r, c], slope=F["slope"][r, c], treecover=F["treecover"][r, c],
                     side=(c >= B.W // 2).astype(int))
            for k in BANDS:
                d[f"{k}_pre"], d[f"{k}_post"] = F[f"{k}_pre"][r, c], F[f"{k}_post"][r, c]
            for k in ("ndvi", "nbr", "ndmi"):
                d[f"{k}_pre"], d[f"{k}_post"], d[f"d_{k}"] = F[f"{k}_pre"][r, c], F[f"{k}_post"][r, c], F[f"d_{k}"][r, c]
            d["r_nir"] = F["r_nir"][r, c]
            rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True).round(4)
    return datasets.save(df, "logging_gr_pixels", unit="піксель лісу 30 м × рік", targets=["y"], features=FEATS, description="Пікселі лісу НП Дадья зі смугами Sentinel-2 до й після, без згарищ; лейбл Hansen.",
                         params=dict(years=list(YEARS), neg=NEG, split="половина за довготою × час"), sources=["Sentinel-2 L2A (AWS Earth Search)", "Hansen GFC v1.12", "SRTM", "EFFIS", "NASA FIRMS"])


def _f1(y, p, w):
    return metrics.area((w * (y & p)).sum(), (w * (~y & p)).sum(), (w * (y & ~p)).sum())


def _gbm():
    return LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=SEED)


def rule_threshold(name, cal):
    """Поріг правила: максимум F1 площі на cal. Одна функція і для метрик, і для кандидатів на мапі застосунку."""
    yc, wc = cal.y.values.astype(bool), (cal.w * cal.pix_ha).values
    col, sg = RULES[name]
    sc = sg * cal[col].values
    return max(np.nanquantile(sc, np.linspace(.5, .999, 120)), key=lambda q: _f1(yc, sc > q, wc)["f1"])


def polys(mask, tol=.0004, min_ha=0., key=None):
    """Маска на сітці Hansen → контури [{poly, ha, lat, lon}], найбільші спершу."""
    from rasterio import features as rfeatures
    B = common.base(key)
    out = []
    for g, v in rfeatures.shapes(mask.astype(np.uint8), mask=mask, transform=B.TR):
        s = shape(g)
        ha = s.area * (111320 * np.cos(np.radians(LAT0))) * 111320 / 1e4
        if ha < min_ha:
            continue
        s = s.simplify(tol)
        if s.is_empty or s.geom_type != "Polygon":
            continue
        out.append(dict(poly=[[round(y, 5), round(x, 5)] for x, y in s.exterior.coords], ha=round(ha, 1), lat=round(s.centroid.y, 5), lon=round(s.centroid.x, 5)))
    return sorted(out, key=lambda x: -x["ha"])


_MODEL = {}                                                             # (рік, рядків) → модель і поріг: одна на всі парки


def candidates(D, champ, year=2024, key=None):
    """Кандидати на рубки року year для застосунку: переможець LD на всій мапі парку, ліс ≥ 40 %, без пожеж;
    поруч позначка, чи бачить ту саму ділянку OPERA (її правило обране на 2023). Це кандидати, а не доведені рубки.
    Правило: поріг на 2018–2021, як у виборі переможця. Модель: навчена на всіх роках до year−1 з усіх парків, поріг —
    максимум F1 на перехресній перевірці по цих роках (_model); лейбли року year не беруться."""
    B = common.base(key)
    F, ok = feature_rasters(year, key)
    if champ in RULES:
        thr = rule_threshold(champ, D[D.year <= 2021])                # поріг з усіх парків, як і у виборі переможця
        col, sg = RULES[champ]
        hit = sg * F[col] > thr
    else:
        k_ = (year, len(D))
        if k_ not in _MODEL:
            _MODEL[k_] = _model(D[D.year <= year - 1])[:2]
        clf, thr = _MODEL[k_]
        hit = np.zeros((B.H, B.W), bool)
        idx = np.flatnonzero((ok & (B.tc >= 40)).ravel())              # лише ліс зі знімками: решту модель не бачила
        X = pd.DataFrame({f: F[f].ravel()[idx] for f in FEATS})
        hit.ravel()[idx] = clf.predict_proba(X)[:, 1] > thr
    m = hit & ok & (B.tc >= 40) & ~fire_masks(key)[year]
    o23 = D[(D.year == 2023) & ((D.park == (key or common._park.KEY)) if "park" in D else True)]
    ya, wa = o23.y.values.astype(bool), (o23.w * o23.pix_ha).values
    rules = [(v, px) for v in SETS.values() for px in (1, 4, 11)]
    best = max(rules, key=lambda r: _f1(ya, opera_mask(2023, *r, key=key)[o23.row.values, o23.col.values], wa)["f1"])
    wide = ndi.binary_dilation(opera_mask(year, *best, key=key), iterations=4)
    cand = polys(m, tol=.0002, min_ha=.3, key=key)[:25]
    col_, row_ = ~B.TR * (np.array([c["lon"] for c in cand]), np.array([c["lat"] for c in cand]))
    for c, r, k in zip(cand, row_.astype(int), col_.astype(int)):
        c["opera"] = bool(wide[np.clip(r, 0, B.H - 1), np.clip(k, 0, B.W - 1)])
    return dict(year=year, approach=champ, cand=cand, kind="rule" if champ in RULES else "model")


def _model(cal):
    """LightGBM гілки: поріг — максимум F1 площі на перехресній перевірці по роках cal, потім модель на всьому cal.
    Одна функція і для метрик (fit), і для кандидатів на мапі (candidates)."""
    yc, wc = cal.y.values.astype(bool), (cal.w * cal.pix_ha).values
    yrs = sorted(cal.year.unique())
    oof = np.zeros(len(cal))
    for k in (yrs[::2], yrs[1::2]):
        m = cal.year.isin(k).values
        oof[m] = _gbm().fit(cal.loc[~m, FEATS], yc[~m], sample_weight=cal.w.values[~m]).predict_proba(cal.loc[m, FEATS])[:, 1]
    thr = max(np.linspace(.05, .95, 37), key=lambda q: _f1(yc, oof > q, wc)["f1"])
    return _gbm().fit(cal[FEATS], yc, sample_weight=cal.w.values), thr, _f1(yc, oof > thr, wc)["f1"]


def fit(name, cal, te):
    yc, wc = cal.y.values.astype(bool), (cal.w * cal.pix_ha).values
    if name in RULES:
        col, sg = RULES[name]
        sc, st = sg * cal[col].values, sg * te[col].values
        thr = rule_threshold(name, cal)
        return st > thr, _f1(yc, sc > thr, wc)["f1"]
    clf, thr, f1 = _model(cal)
    return clf.predict_proba(te[FEATS])[:, 1] > thr, f1


K_FOLDS = 5


def folds(D, transfer=False):
    """Просторові фолди. З одним парком — половина парку (вибір на одній, тест на іншій).
    З кількома: основний протокол — квадрати ~6 км (200×200 пікселів) усіх парків, розкладені на K_FOLDS фолдів, тож у кожному фолді є
    шматки всіх парків; transfer=True — фолд це парк (вибір на інших парках, тест на цьому), окрема перевірка перенесення."""
    if "park" not in D or D.park.nunique() < 2:
        return D.side.values, [0, 1]
    if transfer:
        return D.park.values, list(dict.fromkeys(D.park))
    g = blocks(D)
    u = np.unique(g)
    f = dict(zip(np.random.default_rng(SEED).permutation(u), np.arange(len(u)) % K_FOLDS))
    return np.array([f[x] for x in g]), list(range(K_FOLDS))


def blocks(T):
    """Блоки ~6 км для CI: квадрати 200×200 пікселів сітки свого парку (сітки парків різні, тому з кодом парку)."""
    g = (T.row.values // 200) * 10000 + T.col.values // 200
    if "park" in T:
        g = g + pd.factorize(T.park)[0] * 10 ** 8
    return g


def by_park_f1(T, n_boot):
    """Метрики окремо для кожного парку на тих самих тестових передбаченнях (колонки *_<парк>)."""
    out = {}
    if "park" not in T or T.park.nunique() < 2:
        return out
    for k in dict.fromkeys(T.park):
        q = T[T.park == k]
        y, w, p = q.y.values.astype(bool), (q.w * q.pix_ha).values, q.p.values
        m = _f1(y, p, w)
        if not y.any():                                      # у тесті парку немає рубок: F1 не визначений, а не 0
            m = dict(m, f1=np.nan, precision=np.nan, recall=np.nan)
        lo, hi = metrics.block_bootstrap(blocks(q), lambda i: _f1(y[i], p[i], w[i])["f1"], n=n_boot) if y.any() else (np.nan, np.nan)
        out |= {f"test_metric_{k}": m["f1"], f"ci_{k}": f"{lo:.2f}–{hi:.2f}", f"precision_{k}": m["precision"], f"recall_{k}": m["recall"], f"true_ha_{k}": m["true_ha"]}
    return out


def ld(D, n_boot, transfer=False):
    pred, val = {}, {}
    fold, FOLDS = folds(D, transfer)
    cols = ["row", "col", "y", "w", "pix_ha"] + (["park"] if "park" in D else [])
    for f in FOLDS:
        cal, te = D[(fold != f) & (D.year <= 2021).values], D[(fold == f) & (D.year >= 2022).values]
        for a in (*RULES, "lightgbm"):
            p, v = fit(a, cal, te)
            pred.setdefault(a, []).append(te[cols].assign(p=p))
            val.setdefault(a, []).append(v)
    out = []
    for a, x in pred.items():
        T = pd.concat(x)
        y, w, p = T.y.values.astype(bool), (T.w * T.pix_ha).values, T.p.values
        g = blocks(T)
        m = _f1(y, p, w)
        lo, hi = metrics.block_bootstrap(g, lambda i: _f1(y[i], p[i], w[i])["f1"], n=n_boot)
        out.append(dict(part="LD_transfer" if transfer else "LD", approach=a, test_metric=m["f1"], ci=f"{lo:.2f}–{hi:.2f}", val_metric=float(np.mean(val[a])),
                        precision=m["precision"], recall=m["recall"], true_ha=m["true_ha"], **by_park_f1(T, n_boot)))
    if not transfer and "park" in D and D.park.nunique() > 1:      # окремо: чи переноситься на парк, якого не бачили (не для вибору)
        out += ld(D, n_boot, transfer=True)
    return out


def opera_mask(year, states, min_px, key=None):
    B = common.base(key)
    m = np.zeros((B.H, B.W), bool)
    for d in opera.dates(year):
        m |= np.isin(opera.snapshot(d, key), states)
    if min_px > 1:
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        size = np.bincount(lab.ravel(), minlength=n + 1)
        size[0] = 0
        m = size[lab] >= min_px
    return m


def lm(D, n_boot, transfer=False):
    rules = [(k, v, px) for k, v in SETS.items() for px in (1, 4, 11)]
    dv = datasets.get("opera_gr_snapshots")
    pk = D.park.values if "park" in D else np.array([common._park.KEY] * len(D))
    yr = D.year.values
    vals = {}                                                    # правило OPERA → значення маски в рядках D (2023 і 2024), кожен парк зі своєї маски
    for k in dict.fromkeys(pk):
        for y in (2023, 2024):
            i = np.flatnonzero((pk == k) & (yr == y))
            for r in rules:
                m = opera_mask(y, r[1], r[2], key=k)
                vals.setdefault(r[:1] + (r[2],), np.zeros(len(D), bool))[i] = m[D.row.values[i], D.col.values[i]]
    fold, FOLDS = folds(D, transfer)
    cols = ["row", "col", "y", "w", "pix_ha"] + (["park"] if "park" in D else [])
    parts = []
    for f in FOLDS:
        ci_ = (yr == 2023) & (fold != f)
        ti_ = (yr == 2024) & (fold == f)
        c, t = D[ci_], D[ti_]
        yc, wc = c.y.values.astype(bool), (c.w * c.pix_ha).values
        best = max(vals, key=lambda r: _f1(yc, vals[r][ci_], wc)["f1"])
        trn = D[(yr <= 2021) & (fold != f)]
        s2, _ = fit("lightgbm", trn, t)
        parts.append(t[cols].assign(opera=vals[best][ti_], sentinel2=s2))
    T = pd.concat(parts)
    T["будь-яке"], T["обидва"] = T.opera | T.sentinel2, T.opera & T.sentinel2
    y, w = T.y.values.astype(bool), (T.w * T.pix_ha).values
    g = blocks(T)
    out = []
    for a in ("opera", "sentinel2", "будь-яке", "обидва"):
        p = T[a].values
        m = _f1(y, p, w)
        lo, hi = metrics.block_bootstrap(g, lambda i: _f1(y[i], p[i], w[i])["f1"], n=n_boot)
        out.append(dict(part="LM_transfer" if transfer else "LM", approach=a, test_metric=m["f1"], ci=f"{lo:.2f}–{hi:.2f}", precision=m["precision"], recall=m["recall"],
                        true_ha=m["true_ha"], **by_park_f1(T.assign(p=p), n_boot)))
    if not transfer and "park" in D and D.park.nunique() > 1:
        out += lm(D, n_boot, transfer=True)[0]
    return out, dv


def cells(fire):
    B = common.base()
    S = datasets.load("h3_risk_cells").df[["h3", "parent6", "elev", "slope", "treecover", "d_road", "d_village", "track_km_km2"]]
    known = set(S.h3)
    r, c = np.nonzero((B.loss > 0) & B.forest)
    lon, lat = B.TR * (c + .5, r + .5)
    hx = np.array([h3.latlng_to_cell(a, b, common.H3_RES) for a, b in zip(lat, lon)])
    yr = 2000 + B.loss[r, c].astype(int)
    keep = np.array([not fire[y][i, j] if y >= 2017 else True for y, i, j in zip(yr, r, c)])
    L = pd.DataFrame(dict(h3=hx[keep], year=yr[keep], ha=B.PIX_HA[r, c][keep]))
    L = L[L.h3.isin(known)].groupby(["h3", "year"]).ha.sum().unstack(fill_value=0).reindex(index=S.h3, columns=range(2001, 2025), fill_value=0)
    nb = {h: [x for x in h3.grid_disk(h, 1) if x != h and x in known] for h in S.h3}
    rows = []
    for Y in range(2017, 2025):
        d = S.copy()
        d["year"], d["ha"] = Y, L[Y].values
        d["y"] = (d.ha >= .5).astype(int)
        l1 = L[Y - 1]
        d["loss1"], d["loss2"], d["loss3_5"] = l1.values, L[Y - 2].values, L[[Y - 3, Y - 4, Y - 5]].sum(axis=1).values
        d["loss_cum"] = L[[y for y in range(2001, Y)]].sum(axis=1).values
        d["nb_loss1"] = [l1.reindex(nb[h]).sum() for h in S.h3]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True).round(4)
    return datasets.save(df, "logging_gr_cells", unit="гекс H3 res 9 × рік", targets=["y"], features=CELL_F, description="Гекси НП Дадья: історія втрат без згарищ і ландшафт; лейбл: ≥ 0,5 га нової втрати в році.",
                         params=dict(val="2018–2020", test="2021–2024", min_ha=.5), sources=["Hansen GFC v1.12", "SRTM", "OpenStreetMap", "EFFIS", "NASA FIRMS"])


def lg_scores(tr, t, rng):
    """Бали всіх підходів LG для рядків t, навчання на tr. Одна функція і для метрик, і для прогнозу застосунку."""
    one = len(np.unique(tr.y)) > 1
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return {"випадково": rng.random(len(t)), "де рубали торік": t.loss1.values + .25 * t.nb_loss1.values, "близькість": t.nb_loss1.values,
            "логрег": lr().fit(tr[CELL_F], tr.y).predict_proba(t[CELL_F])[:, 1] if one else np.zeros(len(t)),
            "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=SEED).fit(tr[CELL_F], tr.y).predict_proba(t[CELL_F])[:, 1] if one else np.zeros(len(t))}


def forecast(d, champ, Y=2025, key=None):
    """Прогноз переможця LG на рік Y для застосунку: ознаки року Y з історії до Y−1, навчання на всіх роках до Y (усіх парків);
    гекси прогнозу — лише парку key, верхні TOP у межах цього парку."""
    dk = d[d.park == (key or common._park.KEY)] if "park" in d else d
    ly = dk[dk.year == Y - 1].copy()
    hist = dk.pivot(index="h3", columns="year", values="ha")
    ly["loss2"], ly["loss3_5"] = ly.loss1.values, ly.h3.map(hist[[Y - 3, Y - 4, Y - 5]].sum(axis=1)).values
    ly["loss_cum"] = ly.loss_cum + ly.ha
    ly["loss1"] = ly.ha
    known = set(ly.h3)
    l1 = ly.set_index("h3").loss1
    ly["nb_loss1"] = [l1.reindex([x for x in h3.grid_disk(h, 1) if x != h and x in known]).sum() for h in ly.h3]
    ly["p"] = lg_scores(d, ly, np.random.default_rng(SEED))[champ]
    top = ly.nlargest(int(round(len(ly) * TOP)), "p")
    return dict(year=Y, approach=champ, cells=[[[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(h)] for h in top.h3])


def lg(D, n_boot):
    scores = {}
    rng = np.random.default_rng(SEED)
    for Y in range(2018, 2025):
        t = D[D.year == Y]
        tr = D[D.year < Y]
        S = lg_scores(tr, t, rng)
        for k, v in S.items():
            scores.setdefault(k, []).append(pd.DataFrame(dict(year=Y, h=t.h3.values, g=t.parent6.values, ha=t.ha.values, s=v + rng.random(len(t)) * 1e-9,
                                                              park=t.park.values if "park" in t else "_")))

    def capture(x):
        tot, hit = 0., 0.
        for _, q in x.groupby("year"):
            top = q.nlargest(int(round(len(q) * TOP)), "s")
            hit += top.ha.sum()
            tot += q.ha.sum()
        return hit / max(tot, 1e-9)
    out = []
    TE = {k: pd.concat(v).query("year >= 2021").reset_index(drop=True) for k, v in scores.items()}
    for k, v in scores.items():
        A = pd.concat(v)
        va, te = A[A.year <= 2020], TE[k]
        g = te.g.values
        lo, hi = metrics.block_bootstrap(g, lambda i: capture(te.iloc[i]), n=n_boot)
        row = dict(part="LG", approach=k, val_metric=capture(va), test_metric=capture(te), ci=f"{lo:.2f}–{hi:.2f}", true_ha=float(te.ha.sum()), n_events=int((te.ha >= .5).sum()))
        PKs = list(dict.fromkeys(te.park))
        for pk_ in (PKs if len(PKs) > 1 else []):                    # той самий підхід, верхні TOP гексів у межах кожного парку
            q = te[te.park == pk_].reset_index(drop=True)
            lo_, hi_ = metrics.block_bootstrap(q.g.values, lambda i: capture(q.iloc[i]), n=n_boot) if q.ha.sum() > 0 else (np.nan, np.nan)
            row |= {f"test_metric_{pk_}": capture(q) if q.ha.sum() > 0 else np.nan, f"ci_{pk_}": f"{lo_:.2f}–{hi_:.2f}",
                    f"true_ha_{pk_}": float(q.ha.sum()), f"n_events_{pk_}": int((q.ha >= .5).sum())}
        for b, tag in (("випадково", "rand"), ("де рубали торік", "hist")):       # парна різниця на тих самих блоках
            dd, dlo, dhi = metrics.paired_block_bootstrap(g, lambda i: capture(te.iloc[i]), lambda i: capture(TE[b].iloc[i]), n=n_boot)
            row.update({f"diff_{tag}": dd, f"diff_{tag}_lo": dlo, f"diff_{tag}_hi": dhi})
        out.append(row)
        for Y, q in A.groupby("year"):                                       # прогресивно: кожен рік окремо
            out.append(dict(part="LG_year", approach=k, year=int(Y), test_metric=capture(q), true_ha=float(q.ha.sum()), n_events=int((q.ha >= .5).sum())))
        for share in (.02, .05, .11, .2, .3, .5):                            # уся крива замість одного порога 11 %
            def cap(x, s_=share):
                tot, hit = 0., 0.
                for _, q in x.groupby("year"):
                    hit += q.nlargest(int(round(len(q) * s_)), "s").ha.sum()
                    tot += q.ha.sum()
                return hit / max(tot, 1e-9)
            out.append(dict(part="LG_curve", approach=k, share=share, test_metric=cap(te)))
    return out


# Окремі частини: спершу дані (run_data), потім кожна гілка своїм запуском (run_ld, run_lm, run_lg). Усі гілки беруть ті самі
# датасети з диска, тому дорогий збір композитів і масок іде один раз. Результати гілок зводяться в один запис t19.
NAMES = dict(data="LD · LM · LG: дані рубок", ld="LD · рубка на знімках Sentinel-2", lm="LM · збіг Sentinel-2 і NASA OPERA",
             lg="LG · де рубатимуть наступного року")


def build_opera():
    """Функція збору opera_gr_snapshots (datasets.get): скільки пікселів кожного статусу OPERA на кожну дату."""
    snaps = pd.DataFrame([dict(date=d, **{f"s{v}": int(c) for v, c in zip(*np.unique(opera.snapshot(d), return_counts=True))}) for y in (2023, 2024) for d in opera.dates(y)]).fillna(0)
    return datasets.save(snaps, "opera_gr_snapshots", unit="знімок стану OPERA на дату", targets=[], features=[c for c in snaps if c != "date"], description="Кількість пікселів кожного статусу OPERA DIST-ALERT над НП Дадья.",
                         params=dict(layer=opera.LAYER), sources=["NASA OPERA L3 DIST-ALERT-HLS через NASA GIBS"])


def build_pixels():
    """Функція збору датасету logging_gr_pixels (викликає datasets.get)."""
    fire = fire_masks()
    for y, m in fire.items():
        print(y, "виключено", round(float(m.mean()), 3), flush=True)
    dv = pixels(fire)
    print(dv.df.groupby(["year", "side"]).y.sum().to_dict(), flush=True)
    return dv


def build_cells():
    """Функція збору датасету logging_gr_cells (викликає datasets.get, після h3_risk_cells)."""
    return cells(fire_masks())


def run_data():
    """Обидва датасети рубок (для ручного запуску; у потоці їх збирає крок datasets)."""
    dv, dc = datasets.get("logging_gr_pixels"), datasets.get("logging_gr_cells")
    return pd.DataFrame(), dict(dataset=f"{dv.tag} + {dc.tag}")


def _part(part, n_boot):
    exp = tracking.experiment(NAMES[part], __doc__)
    tracking.keep_reset(f"logging_{part}")
    pflow.mark("0. dataset")
    dv = datasets.get("logging_gr_cells" if part == "lg" else "logging_gr_pixels")
    dc = dv
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "training", "y", files=True)
        ml.log_text(inspect.getsource(inspect.getmodule(_part)), "code/logging_greece.py")
        pflow.mark({"ld": "1. LD: logging on imagery", "lm": "1. LM: two-source agreement", "lg": "1. LG: next-year forecast"}[part])
        rows = ld(dv.df, n_boot) if part == "ld" else lm(dv.df, n_boot)[0] if part == "lm" else lg(dv.df, n_boot)
        pflow.mark("2. pick on validation")
        res = pd.DataFrame(rows)
        res["champion"] = False
        if part in ("ld", "lg"):                     # LM порівнює правила без вибору переможця, як і раніше
            q = res[(res.part == part.upper()) & (res.approach != "випадково")]    # базова лінія не може бути переможцем, як і в інших гілках
            res.loc[q.val_metric.idxmax(), "champion"] = True
        for r in rows:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if part in ("ld", "lg"):                     # переможець одразу застосовується до останніх даних: це бере застосунок
            ch = res[res.champion].approach.iloc[0]
            mk = lambda k: candidates(dv.df, ch, key=k) if part == "ld" else forecast(dv.df, ch, key=k)
            PK = list(dict.fromkeys(dv.df.park)) if "park" in dv.df else [common._park.KEY]
            tracking.keep(f"logging_{part}", "app", mk(common._park.MAIN if common._park.MAIN in PK else PK[0]))   # застосунок: основний парк
            if len(PK) > 1:
                tracking.keep(f"logging_{part}", "parks", {k: mk(k) for k in PK})          # мапи всіх парків для сторінки дослідження
        pflow.mark("3. approaches: results per algorithm")
        cut = lambda r: r.part if r.part in ("LD", "LM", "LG") or r.part.endswith("_transfer") else f"{r.part}_{int(r.year) if r.year == r.year else r.share}"
        tracking.log_approaches(res, lambda r: r.approach, prefix=cut)
        pflow.mark("4. save")
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag)


def run_ld(n_boot=300):
    return _part("ld", n_boot)


def run_lm(n_boot=300):
    return _part("lm", n_boot)


def run_lg(n_boot=300):
    return _part("lg", n_boot)


def run(n_boot=300):
    """Усе одним викликом (ноутбук, ручний запуск): дані, потім LD, LM, LG; результати разом."""
    run_data()
    parts = [_part(p, n_boot) for p in ("ld", "lm", "lg")]
    return pd.concat([r for r, _ in parts], ignore_index=True), parts[-1][1]
