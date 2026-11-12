"""T5 · Поширення пожежі: які гекси згорять за наступні 24 і 48 годин.

Задача      пожежа триває; відомо, що згоріло до моменту t (детекції VIIRS, ~2 прольоти на добу).
            Для кожного ще не згорілого гекса H3 res 8 (~0,74 км²) у радіусі 12 км від згарища — чи згорить він до t+24 год / t+48 год.
Події       28 багатоденних пожеж Греції 2021–2025 (FIRMS VIIRS S-NPP, ST-DBSCAN 1,5 км / 2 доби, ≥ 80 детекцій, ≥ 3 доби з детекціями).
Ознаки      геометрія фронту (відстань до згарища й до активних детекцій за 24 год, частка згорілих сусідів), динаміка
            (площа, приріст і просування фронту за 24 год, вектор руху), вітер NASA POWER (погодинно: швидкість, напрямок → косинус
            кута між вітром і напрямком на гекс), рельєф SRTM (ухил від фронту до гекса), паливо Hansen (лісистість 2000, недавні втрати).
            Погода на наступну добу береться фактична — як заміна прогнозу погоди (в експлуатації — прогноз).
Протокол    Еврос-2023 — недоторканий тест (не бере участі ні у виборі підходу, ні в підборі параметрів).
            Вибір підходу — leave-one-fire-out на решті 27 пожежах; параметри правил і поріг моделі підбираються всередині кожного фолду.
Метрики     F1 нової площі (основна; macro — кожна пожежа з однаковою вагою), precision, recall, IoU, похибка фронту
            (медіана відстані від реально згорілого гекса до найближчого спрогнозованого, м). 95 % CI — bootstrap по пожежах.
Гіпотези
  iso        базова: фронт росте рівномірно на всі боки зі швидкістю за останню добу
  motion     правило: еліпс уздовж напрямку руху пожежі за останню добу
  wind       правило: еліпс за вітром (подовження за Андерсоном, 1983) + поправка на схил
  logreg     ML: логістична регресія на всіх ознаках
  lightgbm   ML: градієнтний бустинг на всіх ознаках
"""
import glob
import inspect
import io
import json
import math

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from lightgbm import LGBMClassifier
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from .. import common, datasets, metrics, tracking
from ..tracking import ml

EXPERIMENT = "T5 · Fire spread +24/48h (куди піде пожежа)"
RES, RADIUS, STEP_H, ANCHOR_H = 8, 12000, 24, 13          # гекси res 8; кандидати ≤ 12 км; крок 24 год, «зріз» о 13:00 UTC (після денного прольоту)
_SP = common.CFG.get("spread", {})                           # config/regions.py fetch.spread: де шукати пожежі й яка тестова
SPREAD_BBOX = _SP.get("bbox")                                 # lat_min,lon_min,lat_max,lon_max; без нього — уся країна архіву FIRMS
TEST_KEY = _SP.get("test", "2023-08 41.0 26.0")               # Еврос-2023 у кейсі Фракії; "auto" — найбільша знайдена пожежа
MIN_N, MIN_DAYS = 80, 3                                       # критерій багатоденної пожежі
SP = common.CACHE / "spread"
SP.mkdir(exist_ok=True)
FEATS = ["d_front", "d_active", "nb1", "nb2", "log_area", "adv_km", "log_growth", "wind_ms", "cos_wind", "wind_along",
         "cos_motion", "motion_km", "slope_up", "elev", "treecover", "loss_recent", "t2m_max", "rh_min", "day_k"]
WX = ["wind_ms", "cos_wind", "wind_along", "t2m_max", "rh_min"]
FEATS_LAG = [f"{c}_lag" if c in WX else c for c in FEATS]          # чесний варіант: погода лише за минулу добу, без майбутнього
NAMES = {"2023-08 41.0 26.0": "Еврос 2023", "2021-08 38.9 23.3": "Північна Евбея 2021", "2023-07 36.1 27.9": "Родос 2023",
         "2021-08 38.1 23.4": "Віля (Аттика) 2021", "2023-07 38.1 23.5": "Дервенохорія (Аттика) 2023", "2023-08 38.1 23.7": "Парніта (Аттика) 2023",
         "2024-08 38.2 23.9": "Варнавас (Аттика) 2024", "2023-07 39.8 19.9": "Корфу 2023", "2025-06 38.3 26.1": "Хіос 2025",
         "2022-07 41.1 26.2": "Дадья 2022", "2022-07 39.0 26.2": "Лесбос 2022", "2021-08 37.7 21.7": "Ілія 2021", "2021-08 37.3 22.0": "Пелопоннес 2021"}


# ---------------------------------------------------------------- події
def fires():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(common.RAW / "firms" / f"viirs-snpp_*_{COUNTRY}.csv")))], ignore_index=True)
    df = df[df.confidence.isin(["n", "h"])].copy()
    df["dt"] = pd.to_datetime(df.acq_date) + pd.to_timedelta(df.acq_time // 100, "h") + pd.to_timedelta(df.acq_time % 100, "m")
    df["sensor"] = "VIIRS"
    if SPREAD_BBOX:                                               # лише пожежі в рамці території з конфігу
        la0, lo0, la1, lo1 = map(float, SPREAD_BBOX.split(","))
        df = df[df.latitude.between(la0, la1) & df.longitude.between(lo0, lo1)].reset_index(drop=True)
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    g["ev"] = common.st_dbscan(g, 1.5, 2.0) if len(g) else []
    g["day"] = g.dt.dt.floor("D")
    s = g.groupby("ev").agg(n=("dt", "size"), start=("dt", "min"), end=("dt", "max"), lat=("latitude", "mean"), lon=("longitude", "mean"), ndays=("day", "nunique"))
    s = s[(s.n >= MIN_N) & (s.ndays >= MIN_DAYS)].copy()
    s["key"] = [f"{a:%Y-%m} {b:.1f} {c:.1f}" for a, b, c in zip(s.start, s.lat, s.lon)]
    s["name"] = [NAMES.get(k, f"{a:%Y-%m} · {b:.2f}°N {c:.2f}°E") for k, a, b, c in zip(s.key, s.start, s.lat, s.lon)]
    pts = {ev: g[g.ev == ev][["latitude", "longitude", "dt", "frp"]].reset_index(drop=True) for ev in s.index}
    return s.sort_values("start"), pts


# ---------------------------------------------------------------- допоміжні шари на будь-яку ділянку Греції
def dem_at(lats, lons, z=11):
    n = 2 ** z
    out = np.zeros(len(lats))
    tx = ((np.asarray(lons) + 180) / 360 * n)
    ty = ((1 - np.arcsinh(np.tan(np.radians(lats))) / math.pi) / 2 * n)
    cache = {}
    for i, (x, y) in enumerate(zip(tx, ty)):
        key = (int(x), int(y))
        if key not in cache:
            f = SP / f"terr_{z}_{key[0]}_{key[1]}.npy"
            if not f.exists():
                rgb = np.asarray(Image.open(io.BytesIO(requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{key[0]}/{key[1]}.png", timeout=60).content)).convert("RGB"), np.float32)
                np.save(f, (rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768).astype(np.float32))
            cache[key] = np.load(f)
        a = cache[key]
        out[i] = a[min(255, int((y - key[1]) * 256)), min(255, int((x - key[0]) * 256))]
    return out


def hansen_at(lats, lons, year, key):
    f = SP / f"hansen_{key.replace(' ', '_')}.npz"
    lats, lons = np.asarray(lats), np.asarray(lons)
    if not f.exists():
        top, left = int(math.ceil(lats.max() / 10) * 10), int(math.floor(lons.min() / 10) * 10)
        tile = f"{top}N_{left:03d}E"
        b = (lons.min() - .01, lats.min() - .01, lons.max() + .01, lats.max() + .01)
        arr = {}
        for lay in ("treecover2000", "lossyear"):
            with rasterio.open(f"/vsicurl/https://storage.googleapis.com/earthenginepartners-hansen/{common.CFG['hansen']['version']}/Hansen_{common.CFG['hansen']['version']}_{lay}_{tile}.tif") as src:
                w = rasterio.windows.from_bounds(*b, src.transform)
                arr[lay] = src.read(1, window=w)
                tr = src.window_transform(w)
        np.savez(f, tc=arr["treecover2000"], ly=arr["lossyear"], tr=np.array(list(tr)[:6]))
    z = np.load(f)
    tr = rasterio.Affine(*z["tr"])
    r = np.clip(((lats - tr.f) / tr.e).astype(int), 0, z["tc"].shape[0] - 1)
    c = np.clip(((lons - tr.c) / tr.a).astype(int), 0, z["tc"].shape[1] - 1)
    # частка недавніх втрат у вікні ~300 м довкола центру гекса
    ly = z["ly"]
    rec = np.array([((ly[max(0, rr - 5):rr + 6, max(0, cc - 5):cc + 6] >= year - 2003) & (ly[max(0, rr - 5):rr + 6, max(0, cc - 5):cc + 6] < year - 2000)).mean() for rr, cc in zip(r, c)])
    tc = np.array([z["tc"][max(0, rr - 5):rr + 6, max(0, cc - 5):cc + 6].mean() for rr, cc in zip(r, c)])
    return tc, rec


def weather(lat, lon, d0, d1):
    f = SP / f"power_{lat:.2f}_{lon:.2f}_{d0:%Y%m%d}_{d1:%Y%m%d}.json"
    if not f.exists():
        r = requests.get("https://power.larc.nasa.gov/api/temporal/hourly/point", timeout=180, params=dict(
            parameters="WS10M,WD10M,T2M,RH2M", community="RE", longitude=round(lon, 2), latitude=round(lat, 2),
            start=f"{d0:%Y%m%d}", end=f"{d1:%Y%m%d}", format="JSON", **{"time-standard": "UTC"}))
        r.raise_for_status()
        f.write_text(r.text)
    p = json.loads(f.read_text())["properties"]["parameter"]
    w = pd.DataFrame(p)
    w.index = pd.to_datetime(w.index, format="%Y%m%d%H")
    return w.replace(-999, np.nan).interpolate(limit=6)


def _area(row, pts):
    """Гекси-кандидати навколо пожежі: ті самі, що бере build_fire."""
    lat0 = row.lat
    kx, ky = 111320 * math.cos(math.radians(lat0)), 111320
    pad = (RADIUS + 3000) / ky
    box = [(pts.longitude.min() - pad * ky / kx, pts.latitude.min() - pad), (pts.longitude.max() + pad * ky / kx, pts.latitude.max() + pad)]
    poly = h3.LatLngPoly([(box[0][1], box[0][0]), (box[0][1], box[1][0]), (box[1][1], box[1][0]), (box[1][1], box[0][0])])
    cells = np.array(sorted(h3.polygon_to_cells(poly, RES)))
    return cells, np.array([h3.cell_to_latlng(c) for c in cells])


def largest():
    """Найбільша подія в рамці (без критерію): для чесного «нема на чому рахувати»."""
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(common.RAW / "firms" / f"viirs-snpp_*_{COUNTRY}.csv")))], ignore_index=True)
    df = df[df.confidence.isin(["n", "h"])].copy()
    if SPREAD_BBOX:
        la0, lo0, la1, lo1 = map(float, SPREAD_BBOX.split(","))
        df = df[df.latitude.between(la0, la1) & df.longitude.between(lo0, lo1)].reset_index(drop=True)
    if not len(df):
        return dict(n=0, ndays=0)
    df["dt"] = pd.to_datetime(df.acq_date) + pd.to_timedelta(df.acq_time // 100, "h") + pd.to_timedelta(df.acq_time % 100, "m")
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    g["ev"] = common.st_dbscan(g, 1.5, 2.0)
    s = g.groupby("ev").agg(n=("dt", "size"), ndays=("dt", lambda v: v.dt.normalize().nunique()), start=("dt", "min"), lat=("latitude", "mean"), lon=("longitude", "mean"))
    b = s.sort_values("n").iloc[-1]
    return dict(n=int(b.n), ndays=int(b.ndays), start=str(b.start)[:10], lat=round(float(b.lat), 3), lon=round(float(b.lon), 3), events=len(s))


def not_run(found=0):
    """Жодна пожежа в рамці не проходить критерій: гілка не падає, а пише, чому не рахувала, і найбільшу знайдену подію."""
    big = largest()
    why = (f"нема на чому рахувати: пожеж із ≥ {MIN_N} детекцій VIIRS і ≥ {MIN_DAYS} діб знайдено {found} (треба ≥ 3); "
           f"найбільша подія — {big['n']} детекцій за {big['ndays']} діб")
    print("FG_spread:", why, big, flush=True)
    exp = tracking.experiment(EXPERIMENT, __doc__)
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        ml.log_params(dict(protocol="not run", reason=why, **{f"largest_{k}": v for k, v in big.items()}))
        res = pd.DataFrame([dict(approach="not_run", hypothesis=why, champion=False, lofo_f1_24=np.nan, test_f1_24=np.nan, fires=found,
                                 largest_n=big["n"], largest_days=big["ndays"], largest_start=big.get("start"), min_n=MIN_N, min_days=MIN_DAYS)])
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, reason=why)


def prefetch():
    """Скачати в кеш усе, що build_fire бере з інтернету: рельєф, Hansen і погодинну погоду для кожної пожежі."""
    fs, pts = fires()
    for ev, row in fs.iterrows():
        _, ll = _area(row, pts[ev])
        dem_at(ll[:, 0], ll[:, 1])
        hansen_at(ll[:, 0], ll[:, 1], row.start.year, row.key)
        weather(row.lat, row.lon, (row.start - pd.Timedelta(days=2)).normalize(), (row.end + pd.Timedelta(days=3)).normalize())
        print("пожежа", row.key, "у кеші", flush=True)


# ---------------------------------------------------------------- датасет: рядок = (пожежа, момент t, гекс-кандидат)
def build_fire(ev, row, pts):
    lat0, lon0 = row.lat, row.lon
    kx, ky = 111320 * math.cos(math.radians(lat0)), 111320
    pts = pts.copy()
    pts["cell"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(pts.latitude, pts.longitude)]
    pad = (RADIUS + 3000) / ky
    box = [(pts.longitude.min() - pad * ky / kx, pts.latitude.min() - pad), (pts.longitude.max() + pad * ky / kx, pts.latitude.max() + pad)]
    poly = h3.LatLngPoly([(box[0][1], box[0][0]), (box[0][1], box[1][0]), (box[1][1], box[1][0]), (box[1][1], box[0][0])])
    cells = np.array(sorted(h3.polygon_to_cells(poly, RES)))
    ci = {c: i for i, c in enumerate(cells)}
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    X, Y = (ll[:, 1] - lon0) * kx, (ll[:, 0] - lat0) * ky
    elev = dem_at(ll[:, 0], ll[:, 1])
    tc, rec = hansen_at(ll[:, 0], ll[:, 1], row.start.year, row.key)
    ring1 = [[ci[n] for n in h3.grid_ring(c, 1) if n in ci] for c in cells]
    ring2 = [[ci[n] for n in h3.grid_disk(c, 2) if n in ci and n != c] for c in cells]
    w = weather(lat0, lon0, (row.start - pd.Timedelta(days=2)).normalize(), (row.end + pd.Timedelta(days=3)).normalize())
    first_cell_t = pts.groupby("cell").dt.min()
    burn_t = np.full(len(cells), np.datetime64("NaT"), "datetime64[ns]")
    for c, t in first_cell_t.items():
        if c in ci:
            burn_t[ci[c]] = np.datetime64(t)
    px, py = (pts.longitude.values - lon0) * kx, (pts.latitude.values - lat0) * ky
    t = row.start.normalize() + pd.Timedelta(hours=ANCHOR_H)
    if t < row.start:
        t += pd.Timedelta(hours=STEP_H)
    rows, k = [], 0
    while t <= row.end:
        T0, H24 = np.datetime64(t), np.timedelta64(24, "h")
        burned = burn_t <= T0                                       # NaT порівнюється як False
        if burned.sum() == 0:
            t += pd.Timedelta(hours=STEP_H); continue
        b24 = burn_t <= T0 - H24
        new24 = (burn_t > T0) & (burn_t <= T0 + H24)
        new48 = (burn_t > T0) & (burn_t <= T0 + 2 * H24)
        tree_b = cKDTree(np.c_[X[burned], Y[burned]])
        d_front = tree_b.query(np.c_[X, Y])[0]
        act = (pts.dt > t - pd.Timedelta(hours=24)).values & (pts.dt <= t).values
        prev = (pts.dt > t - pd.Timedelta(hours=48)).values & (pts.dt <= t - pd.Timedelta(hours=24)).values
        src = np.c_[px[act], py[act]] if act.any() else np.c_[X[burned], Y[burned]]
        d_act, ia = cKDTree(src).query(np.c_[X, Y])
        bear = np.degrees(np.arctan2(X - src[ia, 0], Y - src[ia, 1]))
        if act.any() and prev.any():
            mv = np.array([px[act].mean() - px[prev].mean(), py[act].mean() - py[prev].mean()])
        else:
            mv = np.zeros(2)
        mkm = float(np.hypot(*mv) / 1000)
        mdir = math.degrees(math.atan2(mv[0], mv[1])) if mkm > 0 else 0.0
        ww = w.loc[t:t + pd.Timedelta(hours=24)]
        u = -(ww.WS10M * np.sin(np.radians(ww.WD10M))).mean()          # вектор «куди дме»
        v = -(ww.WS10M * np.cos(np.radians(ww.WD10M))).mean()
        wsp, wto = float(np.hypot(u, v)), float(math.degrees(math.atan2(u, v)))
        wp = w.loc[t - pd.Timedelta(hours=24):t]
        up_, vp_ = -(wp.WS10M * np.sin(np.radians(wp.WD10M))).mean(), -(wp.WS10M * np.cos(np.radians(wp.WD10M))).mean()
        wsp_l, wto_l = float(np.hypot(up_, vp_)), float(math.degrees(math.atan2(up_, vp_)))
        area = burned.sum() * 0.737
        adv = max(0.0, math.sqrt(burned.sum() / math.pi) - math.sqrt(max(b24.sum(), 0) / math.pi)) * 0.92   # км (≈ радіус у гексах × 0,92 км)
        cand = (~burned) & (d_front <= RADIUS)
        _, fi = tree_b.query(np.c_[X[cand], Y[cand]])
        bidx = np.flatnonzero(burned)[fi]
        bf = burned.astype(float)
        nb1 = np.array([bf[r].mean() if r else 0 for r in (ring1[i] for i in np.flatnonzero(cand))])
        nb2 = np.array([bf[r].mean() if r else 0 for r in (ring2[i] for i in np.flatnonzero(cand))])
        cw = np.cos(np.radians(bear[cand] - wto))
        cw_l = np.cos(np.radians(bear[cand] - wto_l))
        d = pd.DataFrame(dict(fire=row.key, t=t, k=k, cell=cells[cand], lat=ll[cand, 0].round(5), lon=ll[cand, 1].round(5),
                              d_front=d_front[cand], d_active=d_act[cand], nb1=nb1, nb2=nb2, log_area=np.log1p(area),
                              adv_km=adv, log_growth=np.log1p(burned.sum() - b24.sum()), wind_ms=wsp, cos_wind=cw, wind_along=wsp * cw,
                              cos_motion=np.cos(np.radians(bear[cand] - mdir)) if mkm > 0 else 0.0, motion_km=mkm,
                              slope_up=(elev[cand] - elev[bidx]) / np.maximum(d_front[cand], 300), elev=elev[cand], treecover=tc[cand], loss_recent=rec[cand],
                              t2m_max=float(ww.T2M.max()), rh_min=float(ww.RH2M.min()), day_k=k,
                              wind_ms_lag=wsp_l, cos_wind_lag=cw_l, wind_along_lag=wsp_l * cw_l, t2m_max_lag=float(wp.T2M.max()), rh_min_lag=float(wp.RH2M.min()),
                              y24=new24[cand].astype(int), y48=new48[cand].astype(int)))
        rows.append(d)
        t += pd.Timedelta(hours=STEP_H); k += 1
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    meta = dict(cells=cells, ll=ll, burn_t=burn_t)
    return out, meta


def build_dataset():
    s, pts = fires()
    parts, metas = [], {}
    for ev, row in s.iterrows():
        d, m = build_fire(ev, row, pts[ev])
        if len(d) and d.y24.sum() > 0:
            parts.append(d); metas[row.key] = m
    df = pd.concat(parts, ignore_index=True)
    df["t"] = df.t.astype(str)
    num = df.select_dtypes("float").columns
    df[num] = df[num].round(4)
    keep = s[s.key.isin(df.fire.unique())]
    dv = datasets.save(
        df, "fire_spread_cells", unit="(пожежа, момент t, гекс H3 res 8 у радіусі 12 км від згарища)",
        description="Кандидатні гекси довкола активних пожеж Греції 2021–2025 з ознаками фронту, вітру, рельєфу й палива та лейблами «згорить до t+24 / t+48 год» (FIRMS VIIRS).",
        targets=["y24", "y48"], features=FEATS,
        params=dict(h3_res=RES, radius_m=RADIUS, step_h=STEP_H, anchor_utc_h=ANCHOR_H, fires=len(keep), events="ST-DBSCAN 1.5 km / 2 d, VIIRS n/h, >=80 det, >=3 days",
                    weather="NASA POWER hourly (next 24 h as forecast proxy)", test_fire=TEST_KEY),
        sources=["NASA FIRMS VIIRS S-NPP (Greece 2021–2025)", "NASA POWER hourly", "AWS Terrain Tiles (SRTM)", "Hansen GFC v1.12"])
    return dv, keep, metas, pts


# ---------------------------------------------------------------- підходи
def anderson_ecc(ws):
    u = ws * 2.237                                                   # м/с → mph
    lb = 0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397
    lb = np.clip(lb, 1, 8)
    return np.sqrt(1 - 1 / lb ** 2)


def rule_score(d, kind, p, h):
    """Додатна оцінка = гекс усередині прогнозованої області. Радіус = k · max(просування за добу, a0) · (h/24)."""
    r = p["k"] * np.maximum(d.adv_km.values, p["a0"]) * 1000 * (h / 24)
    if kind == "iso":
        return r - d.d_front.values
    if kind == "motion":
        e, cth = p["e"], d.cos_motion.values
        s = np.where(d.motion_km.values > 0, (1 - e) / (1 - e * cth), 1.0)
        return r * s / ((1 - e) / (1 - e)) - d.d_front.values
    e = anderson_ecc(d.wind_ms.values) * p["ew"]
    s = (1 - e) / (1 - e * d.cos_wind.values) * np.exp(p["c"] * np.clip(d.slope_up.values, -.3, .3))
    return r * s / np.maximum(1 - e, 1e-3) * (1 - e) - d.d_front.values


GRID = {"iso": [dict(k=k, a0=a) for k in (.5, .75, 1, 1.5, 2, 3) for a in (.3, .6, 1.2)],
        "motion": [dict(k=k, a0=a, e=e) for k in (.5, .75, 1, 1.5, 2, 3) for a in (.3, .6, 1.2) for e in (.3, .6, .8)],
        "wind": [dict(k=k, a0=a, ew=ew, c=c) for k in (.5, .75, 1, 1.5, 2, 3) for a in (.3, .6, 1.2) for ew in (.5, 1.0) for c in (0, 3)]}
APPROACHES = {"iso": dict(kind="baseline", hyp="Фронт росте рівномірно на всі боки зі швидкістю за останню добу"),
              "motion": dict(kind="rule", hyp="Еліпс уздовж напрямку, куди пожежа рухалась останню добу"),
              "wind": dict(kind="rule", hyp="Еліпс за вітром NASA POWER (подовження за Андерсоном) + швидше вгору по схилу"),
              "logreg": dict(kind="model", hyp="Логістична регресія на ознаках фронту, вітру, рельєфу й палива"),
              "lightgbm": dict(kind="model", hyp="LightGBM на тих самих ознаках (нелінійні взаємодії вітер × схил × паливо)"),
              "lightgbm_lag": dict(kind="model", hyp="LightGBM, але погода лише за минулу добу: так модель працювала б без прогнозу погоди")}


def _ml(name):
    name = name.replace("_lag", "")
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=.5))
    return LGBMClassifier(n_estimators=250, learning_rate=.05, num_leaves=15, min_child_samples=50, subsample=.8, subsample_freq=1,
                          colsample_bytree=.8, verbose=-1, random_state=0)


def _f1(y, p):
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def fit_predict(name, tr, te, h):
    """Навчання / підбір на tr, бінарний прогноз на te. → (bool-масив, параметри)."""
    y = tr[f"y{h}"].values.astype(bool)
    if name in GRID:
        best = max(GRID[name], key=lambda p: _f1(y, rule_score(tr, name, p, h) > 0))
        return rule_score(te, name, best, h) > 0, best, None
    FS = FEATS_LAG if name.endswith("_lag") else FEATS
    # поріг — за OOF-прогнозами всередині навчальних пожеж (3 групові фолди)
    fires_ = tr.fire.unique()
    rng = np.random.default_rng(0)
    fold = dict(zip(fires_, rng.permutation(len(fires_)) % 3))
    oof = np.zeros(len(tr))
    for f in range(3):
        m_ = tr.fire.map(fold).values == f
        if m_.all() or not m_.any():
            continue
        oof[m_] = _ml(name).fit(tr.loc[~m_, FS], y[~m_]).predict_proba(tr.loc[m_, FS])[:, 1]
    thr = max(np.linspace(.05, .95, 37), key=lambda q: _f1(y, oof > q))
    mdl = _ml(name).fit(tr[FS], y)
    return mdl.predict_proba(te[FS])[:, 1] > thr, dict(threshold=round(float(thr), 3)), mdl


def scores_by_fire(df, pred, h):
    """Метрики по кожній пожежі (сумарно за всі її моменти t) + похибка фронту."""
    out = []
    for f, g in df.assign(_p=pred).groupby("fire"):
        y, p = g[f"y{h}"].values.astype(bool), g._p.values
        m = metrics.area((y & p).sum(), (~y & p).sum(), (y & ~p).sum())
        errs = []
        for _, gt in g.groupby("t"):
            yy, pp = gt[f"y{h}"].values.astype(bool), gt._p.values
            if yy.any():
                xy = np.c_[gt.lon.values * 84000, gt.lat.values * 111320]
                errs += list(cKDTree(xy[pp]).query(xy[yy])[0]) if pp.any() else list(gt.d_front.values[yy])
        out.append(dict(fire=f, **{k: m[k] for k in ("f1", "iou", "precision", "recall")}, front_err_m=float(np.median(errs)) if errs else np.nan,
                        new_cells=int(y.sum())))
    return pd.DataFrame(out)


def run(n_boot=1000):
    global TEST_KEY
    s_, _ = fires()
    if len(s_) < 3:                                               # LOFO і окремий тест потребують хоча б трьох пожеж
        return not_run(len(s_))
    if TEST_KEY == "auto":                                        # тестом — найбільша знайдена пожежа
        TEST_KEY = s_.sort_values("n").key.iloc[-1]
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: fire cells')
    dv, fs, metas, pts = build_dataset()
    df = dv.df
    test_mask = df.fire.values == TEST_KEY
    dev, test = df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
    dev_fires = dev.fire.unique()
    rows, evros = [], {}
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "training", "y24", files=True)
        ml.log_params(dict(protocol="LOFO on dev fires; Evros-2023 untouched test", dev_fires=len(dev_fires), test_fire=TEST_KEY, h3_res=RES, radius_m=RADIUS))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/fire_spread.py")
        ml.log_table(fs.reset_index().assign(start=lambda d: d.start.astype(str), end=lambda d: d.end.astype(str)), "fires.json")
        pflow.mark('2. approaches: leave-one-fire-out')
        for name, A in APPROACHES.items():
            with ml.start_run(run_name=name, nested=True):
                ml.set_tags(dict(approach=name, kind=A["kind"], hypothesis=A["hyp"], role="approach"))
                tracking.log_dataset(dv, "training", "y24")
                res = {"approach": name, "kind": A["kind"], "hypothesis": A["hyp"]}
                for h in (24, 48):
                    pred = np.zeros(len(dev), bool)
                    for f in dev_fires:                                     # leave-one-fire-out
                        m_ = dev.fire.values == f
                        pred[m_] = fit_predict(name, dev[~m_], dev[m_], h)[0]
                    sb = scores_by_fire(dev, pred, h)
                    macro = sb.f1.mean()
                    ci = metrics.block_bootstrap(sb.fire.values, lambda idx: sb.f1.values[idx].mean(), n=n_boot)
                    yv = dev[f"y{h}"].values.astype(bool)
                    micro = metrics.area((yv & pred).sum(), (~yv & pred).sum(), (yv & ~pred).sum())
                    # тест: навчання на всіх 27 пожежах → Еврос-2023
                    pt, params, mdl = fit_predict(name, dev, test, h)
                    st = scores_by_fire(test, pt, h).iloc[0]
                    evros[(name, h)] = pt
                    ml.log_params({f"h{h}_{k}": v for k, v in params.items()})
                    ml.log_metrics(tracking.clean({f"lofo_macro_f1_{h}": macro, f"lofo_macro_f1_{h}_ci_lo": ci[0], f"lofo_macro_f1_{h}_ci_hi": ci[1],
                                                       f"lofo_micro_f1_{h}": micro["f1"], f"lofo_precision_{h}": micro["precision"], f"lofo_recall_{h}": micro["recall"],
                                                       f"lofo_front_err_m_{h}": sb.front_err_m.median(),
                                                       f"test_f1_{h}": st.f1, f"test_iou_{h}": st.iou, f"test_precision_{h}": st.precision, f"test_recall_{h}": st.recall,
                                                       f"test_front_err_m_{h}": st.front_err_m}))
                    ml.log_table(sb, f"lofo_by_fire_{h}h.json")
                    if mdl is not None and h == 24 and name == "lightgbm":
                        tracking.log_sklearn(mdl, test[FEATS].head(3))
                    res |= {f"lofo_f1_{h}": macro, f"lofo_f1_{h}_ci": f"{ci[0]:.3f}–{ci[1]:.3f}", f"lofo_precision_{h}": micro["precision"],
                            f"lofo_recall_{h}": micro["recall"], f"lofo_front_err_m_{h}": float(sb.front_err_m.median()),
                            f"test_f1_{h}": float(st.f1), f"test_precision_{h}": float(st.precision), f"test_recall_{h}": float(st.recall),
                            f"test_front_err_m_{h}": float(st.front_err_m), f"params_{h}": json.dumps(params)}
                    res[f"by_fire_{h}"] = sb.to_dict("records")
                rows.append(res)
        pflow.mark('3. pick, test on Evros 2023, difference')
        R = pd.DataFrame(rows)
        champ = R.sort_values("lofo_f1_24", ascending=False).approach.iloc[0]
        R["champion"] = R.approach.eq(champ)
        if champ != "iso":                                                    # парна різниця з «рівномірним ростом» на тих самих 27 пожежах
            fc = pd.DataFrame(R.loc[R.champion, "by_fire_24"].iloc[0]).set_index("fire").f1
            fb = pd.DataFrame(R.loc[R.approach == "iso", "by_fire_24"].iloc[0]).set_index("fire").f1.reindex(fc.index)
            dv_ = (fc - fb).values
            bs = [dv_[np.random.default_rng(i).integers(0, len(dv_), len(dv_))].mean() for i in range(n_boot)]
            R.loc[R.champion, ["diff_iso_lofo", "diff_iso_lofo_lo", "diff_iso_lofo_hi"]] = [float(dv_.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        # фінальна модель чемпіона на всіх 28 пожежах → Model Registry
        if champ in ("logreg", "lightgbm"):
            with ml.start_run(run_name=f"FINAL {champ} (усі пожежі)", nested=True):
                ml.set_tags(dict(role="final_model", approach=champ))
                for h in (24, 48):
                    _, params, mdl = fit_predict(champ, df, df.head(5), h)
                    tracking.log_sklearn(mdl, df[FEATS].head(3), registered_model_name=f"dadia_fire_spread_{h}h")
                    ml.log_params({f"h{h}_{k}": v for k, v in params.items()})
        ml.set_tags(dict(champion=champ, selection="max LOFO macro F1 @24h на 27 пожежах; Еврос-2023 — тест"))
        pflow.mark('4. save')
        tracking.keep_reset("fire_spread")
        spread, daily = spread_view(test, evros, champ, fs, pts)
        tracking.keep("fire_spread", "spread", spread)
        tracking.keep("fire_spread", "daily", daily)
        ml.log_table(R.drop(columns=["by_fire_24", "by_fire_48"]), "results.json")
        ml.log_figure(_fig(R), "plots/approaches.png"); plt.close("all")
        return R, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champ,
                       fires=fs, test=test, evros_pred=evros, metas=metas, pts=pts)


def spread_view(test, evros, champ, fs, pts):
    """Мапа фронту по днях на тестовій пожежі і щоденний F1: з тих самих передбачень evros, що дали метрики тесту."""
    import h3
    r = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)
    show = list(dict.fromkeys([champ, "iso"] + [a for a in ("wind", "motion") if a != champ]))
    ev = fs.index[fs.key == TEST_KEY][0]
    P_ = pts[ev].copy()
    P_["cell"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(P_.latitude, P_.longitude)]
    first = P_.groupby("cell").dt.min()
    times = sorted(test.t.unique())
    cells = sorted(set(test.cell) | set(first.index))
    ci = {c: i for i, c in enumerate(cells)}
    days, daily = [], []
    for t in times:
        g = test[test.t == t]
        T0 = pd.Timestamp(t)
        d = dict(t=t[:16], burned=[ci[c] for c, tt in first.items() if tt <= T0], y24=[ci[c] for c in g.cell[g.y24 == 1]], y48=[ci[c] for c in g.cell[g.y48 == 1]],
                 wind_ms=r(g.wind_ms.iloc[0], 1), wind_to=None, adv_km=r(g.adv_km.iloc[0], 2))
        idx = g.index.values
        for a in show:
            for h in (24, 48):
                d[f"p{h}_{a}"] = [ci[c] for c in g.cell.values[evros[(a, h)][idx]]]
        days.append(d)
        for a in show:
            for h in (24, 48):
                yy, pp = g[f"y{h}"].values.astype(bool), evros[(a, h)][idx]
                if yy.sum():
                    daily.append(dict(t=t[:10], approach=a, h=h, f1=r(metrics.area((yy & pp).sum(), (~yy & pp).sum(), (yy & ~pp).sum())["f1"], 3), new=int(yy.sum())))
    w = weather(fs.loc[ev].lat, fs.loc[ev].lon, (fs.loc[ev].start - pd.Timedelta(days=2)).normalize(), (fs.loc[ev].end + pd.Timedelta(days=3)).normalize())
    for d in days:                                                            # «куди дме» вітер за наступну добу
        ww = w.loc[pd.Timestamp(d["t"]):pd.Timestamp(d["t"]) + pd.Timedelta(hours=24)]
        u = -(ww.WS10M * np.sin(np.radians(ww.WD10M))).mean()
        v = -(ww.WS10M * np.cos(np.radians(ww.WD10M))).mean()
        d["wind_to"] = r(np.degrees(np.arctan2(u, v)) % 360, 0)
    poly = {c: [[round(la, 4), round(lo, 4)] for la, lo in h3.cell_to_boundary(c)] for c in cells}
    spread = dict(fire=TEST_KEY, name=NAMES.get(TEST_KEY), cells=cells, poly=[poly[c] for c in cells], days=days, show=show, champion=champ,
                  dets=dict(lat=P_.latitude.round(4).tolist(), lon=P_.longitude.round(4).tolist(), t=P_.dt.dt.strftime("%Y-%m-%d %H:%M").tolist()))
    return spread, daily


def _fig(R):
    fig, ax = plt.subplots(figsize=(8, 4))
    r = R.sort_values("lofo_f1_24")
    ax.barh(r.approach, r.lofo_f1_24, color=["#3987e5" if c else "#9aa59b" for c in r.champion])
    ax.scatter(r.test_f1_24, r.approach, color="#d95926", zorder=3, label="тест Еврос-2023")
    ax.set_xlabel("F1 нової площі за 24 год (LOFO macro, 27 пожеж)"); ax.legend()
    return fig
