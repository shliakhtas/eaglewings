"""Спільні дані для експериментів (НП Дадья–Лефкімі–Суфлі).

Усе, що качається з мережі, кешується в data/raw/nasa/ — повторні запуски офлайн і детерміновані.
  park / roads           — OSM (data/processed)
  firms                  — NASA FIRMS VIIRS S-NPP + MODIS 2021–2025, парк + 3 км, фільтр впевненості
  hansen                 — Hansen GFC v1.12: treecover2000, lossyear (сітка 0.00025°)
  dem                    — AWS Terrain Tiles (SRTM), перепроєктовано на сітку Hansen
  power                  — NASA POWER daily, центр парку
  modis_ndvi             — MOD13Q1 NDVI (ORNL DAAC) до/після пожежі 2023, на сітці Hansen
  h3grid                 — сітка H3 res 9 + статичні ознаки комірок
"""
import functools
import io
import json
import time
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from PIL import Image
from pyproj import Transformer
from rasterio import features
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from . import park as _park  # noqa: E402
P, RAW = _park.P, ROOT / "data" / "raw"           # P і CACHE свої для кожного парку (GR_PARK), сирі архіви на всю країну спільні
CACHE = _park.CACHE
from .regions import REGION  # noqa: E402
CFG = REGION["fetch"]
UTM = CFG["utm"]
H3_RES = CFG["h3_res"]
YEARS = CFG["firms"]["years"]
SEED = 42

# композити MOD13Q1 навколо пожежі 2023 (21.08–02.09): останній до / перший після
MODIS_PRE, MODIS_POST = "A2023209", "A2023257"


# ---------------------------------------------------------------- базові шари
def _k(key):
    return key or _park.KEY


def per_park(fn):
    """Кеш на парк: base() і base("<основний>") це той самий запис кешу, а не два обчислення."""
    cached = lru_cache(fn)

    @functools.wraps(fn)
    def w(key=None):
        return cached(_k(key))
    w.cache_clear = cached.cache_clear
    return w


@per_park
def base(key=None):
    """Базові шари парку key (за замовчуванням GR_PARK): межа, дороги, FIRMS, сітка Hansen. Кожен парк має свою сітку."""
    key = _k(key)
    PP, CC = _park.processed(key), _park.cache(key)
    park = gpd.read_file(PP / "park.geojson")
    PARK = park.geometry.iloc[0]
    roads = gpd.read_file(PP / "roads.geojson")

    f = gpd.read_file(PP / "firms.geojson")
    if "date" not in f:                                   # у парку за роки FIRMS жодної точки: порожній шар із тими самими колонками
        f = gpd.GeoDataFrame({c: pd.Series(dtype=t) for c, t in dict(latitude=float, longitude=float, date=str, acq_time=int, source=str,
                                                                       confidence=str, frp=float, daynight=str).items()}, geometry=[], crs=4326)
    f["date"] = pd.to_datetime(f.date)
    f["dt"] = f.date + pd.to_timedelta(f.acq_time // 100, "h") + pd.to_timedelta(f.acq_time % 100, "m")
    f["sensor"] = f.source.map({"viirs-snpp": "VIIRS", "modis": "MODIS"})
    ok = (f.sensor.eq("VIIRS") & f.confidence.isin(["n", "h"])) | \
         (f.sensor.eq("MODIS") & (pd.to_numeric(f.confidence, errors="coerce") >= 30))
    firms = f[ok].sort_values("dt").reset_index(drop=True)
    firms["year"] = firms.dt.dt.year

    meta = json.load(open(PP / "hansen_meta.json"))
    TR = Affine(*meta["transform"])
    tc = np.load(PP / "treecover2000.npy")
    H, W = tc.shape
    fl = CC / "hansen_lossyear.npy"
    if not fl.exists():
        url = "/vsicurl/https://storage.googleapis.com/earthenginepartners-hansen/{v}/Hansen_{v}_lossyear_{t}.tif".format(v=CFG["hansen"]["version"], t=CFG["hansen"]["tile"])
        with rasterio.open(url) as src:
            np.save(fl, src.read(1, window=rasterio.windows.from_bounds(*meta["bounds"], src.transform)))
    loss = np.load(fl)[:H, :W]
    row_lat = TR.f + TR.e * (np.arange(H) + 0.5)
    pix_ha = np.repeat(((abs(TR.e) * 111320) * (TR.a * 111320 * np.cos(np.radians(row_lat))) / 1e4)[:, None], W, 1)
    park_mask = features.geometry_mask([PARK], (H, W), TR, invert=True)
    return SimpleNamespace(key=key, park=park, PARK=PARK, park_u=park.to_crs(UTM), roads=roads, firms=firms, meta=meta, TR=TR, tc=tc, loss=loss,
                           H=H, W=W, row_lat=row_lat, PIX_HA=pix_ha, park_mask=park_mask, forest=tc >= 30,
                           EXT=[TR.c, TR.c + TR.a * W, TR.f + TR.e * H, TR.f])


# ---------------------------------------------------------------- рельєф
@per_park
def dem(key=None):
    key = _k(key)
    B = base(key)
    f = _park.cache(key) / "dem_hansen_grid.npy"
    if not f.exists():
        z, n = 12, 2 ** 12
        def t(lat, lon): return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
        minx, miny, maxx, maxy = B.meta["bounds"]
        (x0, y0), (x1, y1) = t(maxy, minx), t(miny, maxx)
        xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
        mos = np.zeros((len(ys) * 256, len(xs) * 256), np.float32)
        for i, tx in enumerate(xs):
            for j, ty in enumerate(ys):
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png"
                rgb = np.asarray(Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("RGB"), np.float32)
                mos[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        size = 40075016.68557849 / n
        src_tr = from_origin(-20037508.342789244 + xs[0] * size, 20037508.342789244 - ys[0] * size, size / 256, size / 256)
        out = np.zeros((B.H, B.W), np.float32)
        reproject(mos, out, src_transform=src_tr, src_crs="EPSG:3857", dst_transform=B.TR, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
        np.save(f, out)
    return np.load(f)


# ---------------------------------------------------------------- FIRMS → події (ST-DBSCAN)
def st_dbscan(firms, eps_km=1.5, eps_days=2.0):
    """Мітки подій: DBSCAN на (x км, y км, t діб · eps_km/eps_days) — циліндрична околиця eps_km × eps_days."""
    fu = firms.to_crs(UTM)
    t = (firms.dt - firms.dt.min()).dt.total_seconds() / 86400
    X = np.c_[fu.geometry.x / 1000, fu.geometry.y / 1000, t * eps_km / eps_days]
    return DBSCAN(eps=eps_km, min_samples=1).fit_predict(X)


def pixel_squares(firms):
    """Реальний слід пікселя сенсора: VIIRS 375 м, MODIS 1 км (UTM)."""
    fu = firms.to_crs(UTM)
    half = np.where(firms.sensor.eq("VIIRS"), 375 / 2, 500)
    return gpd.GeoSeries(shapely.buffer(fu.geometry.values, half, cap_style="square"), crs=UTM, index=firms.index)


def events(firms, labels):
    sq = pixel_squares(firms)
    fp = gpd.GeoDataFrame({"event": labels}, geometry=sq.values, crs=UTM).dissolve("event")
    ev = firms.assign(event=labels).groupby("event").agg(start=("dt", "min"), end=("dt", "max"), n=("dt", "size"), frp_max=("frp", "max"))
    out = gpd.GeoDataFrame(ev, geometry=fp.geometry.reindex(ev.index), crs=UTM)
    out["year"] = out.start.dt.year
    out["area_ha"] = out.area / 1e4
    return out


def to_mask(geoms, key=None):
    """GeoSeries (будь-яка CRS) → bool-маска на сітці Hansen парку key."""
    B = base(key)
    g = gpd.GeoSeries(geoms).to_crs(4326) if len(geoms) else []
    return features.geometry_mask(g, (B.H, B.W), B.TR, invert=True) if len(g) else np.zeros((B.H, B.W), bool)


@per_park
def default_events(key=None):
    """Події з параметрами за замовчуванням (eps 1.5 км / 2 доби) — для лейблів T2/T3."""
    B = base(key)
    ev = events(B.firms, st_dbscan(B.firms))
    fp_year = {y: to_mask(ev[ev.year == y].geometry, key) for y in YEARS}
    return ev, fp_year


# ---------------------------------------------------------------- NASA POWER
@per_park
def power(key=None):
    """Погода NASA POWER по днях у точці парку key: своя точка для кожного парку, а не центр їхнього об'єднання."""
    key = _k(key)
    B = base(key)
    c = B.PARK.representative_point() if B.PARK.geom_type == "MultiPolygon" else B.PARK.centroid
    f = _park.cache(key) / "power_daily.json"
    if not f.exists():
        r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", timeout=180, params=dict(
            parameters="T2M_MAX,T2M,RH2M,WS10M,PRECTOTCORR,GWETTOP", community="AG", longitude=round(c.x, 3), latitude=round(c.y, 3),
            start=CFG["power"]["start"], end=CFG["power"]["end"], format="JSON"))
        r.raise_for_status()
        f.write_text(r.text)
    wx = pd.DataFrame(json.loads(f.read_text())["properties"]["parameter"])
    wx.index = pd.to_datetime(wx.index, format="%Y%m%d")
    return wx.replace(-999, np.nan).interpolate(limit=3)


# ---------------------------------------------------------------- MODIS NDVI (ORNL DAAC)
ORNL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"
SINU = "+proj=sinu +R=6371007.181 +units=m +no_defs"
# з 29.10.2026 сертифікат ORNL підписаний коренем emSign, якого немає в certifi (Python), але є в системному сховищі
ORNL_CA = next((p for p in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt") if Path(p).exists()), True)


def ornl(lat, lon, d0, d1, ab, lr, band="250m_16_days_NDVI"):
    f = CACHE / f"modis_{band}_{lat:.4f}_{lon:.4f}_{d0}_{d1}_{ab}_{lr}.json"
    if not f.exists():
        for attempt, wait in enumerate((30, 90, 180, 300, None), 1):      # сервер ORNL буває недоступний: кілька спроб
            try:
                r = requests.get(f"{ORNL}/subset", headers={"Accept": "application/json"}, timeout=300, verify=ORNL_CA, params=dict(
                    latitude=lat, longitude=lon, band=band, startDate=d0, endDate=d1, kmAboveBelow=ab, kmLeftRight=lr))
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if wait is None:
                    raise
                print(f"MODIS ORNL: спроба {attempt} не вдалась ({type(e).__name__}), повтор через {wait} с", flush=True)
                time.sleep(wait)
        f.write_text(r.text)
    return json.loads(f.read_text())


@per_park
def modis_ndvi(key=None):
    """NDVI до/після пожежі 2023, перенесені на сітку Hansen (nearest: кожен 30-м піксель бере свій 250-м піксель MODIS)."""
    key = _k(key)
    B = base(key)
    pc = B.PARK.centroid
    out = {}
    for k, d in (("pre", MODIS_PRE), ("post", MODIS_POST)):
        js = ornl(round(pc.y, 4), round(pc.x, 4), d, d, 16, 13)
        nr, nc, cs = js["nrows"], js["ncols"], js["cellsize"]
        a = np.array(js["subset"][0]["data"], float).reshape(nr, nc)
        a[a < -2000] = np.nan
        a *= float(js["scale"])
        lon = B.TR.c + B.TR.a * (np.arange(B.W) + .5)
        LON, LAT = np.meshgrid(lon, B.row_lat)
        x, y = Transformer.from_crs("EPSG:4326", SINU, always_xy=True).transform(LON, LAT)
        col = np.floor((x - float(js["xllcorner"])) / cs).astype(int)
        row = nr - 1 - np.floor((y - float(js["yllcorner"])) / cs).astype(int)
        ok = (row >= 0) & (row < nr) & (col >= 0) & (col < nc)
        g = np.full((B.H, B.W), np.nan)
        g[ok] = a[row[ok], col[ok]]
        out[k] = g
    return out


# ---------------------------------------------------------------- H3
@per_park
def h3grid(key=None):
    """Сітка H3 res 9 у межах парку, растр належності пікселів Hansen до комірок, статичні ознаки S."""
    key = _k(key)
    B = base(key)
    cells = sorted(h3.geo_to_cells(B.PARK, H3_RES))
    grid = gpd.GeoDataFrame({"h3": cells, "parent6": [h3.cell_to_parent(c, 6) for c in cells],
                             "parent7": [h3.cell_to_parent(c, 7) for c in cells]},
                            geometry=[Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells], crs=4326)
    cent = grid.to_crs(UTM).centroid
    CXY = np.c_[cent.x, cent.y]
    cell_r = features.rasterize(((g, i + 1) for i, g in enumerate(grid.geometry)), (B.H, B.W), transform=B.TR, fill=0, dtype="int32")
    flat = cell_r.ravel()
    npix = np.bincount(flat, minlength=len(grid) + 1)

    def zmean(a):
        return (np.bincount(flat, weights=np.asarray(a, float).ravel(), minlength=len(grid) + 1) / np.maximum(npix, 1))[1:]

    z = dem(key)
    gy, gx = np.gradient(z)
    dzdx, dzdn = gx / (B.TR.a * 111320 * np.cos(np.radians(B.row_lat)))[:, None], -gy / (abs(B.TR.e) * 111320)
    S = pd.DataFrame({"elev": zmean(z), "slope": zmean(np.degrees(np.arctan(np.hypot(dzdx, dzdn)))),
                      "southness": zmean(dzdn / np.maximum(np.hypot(dzdx, dzdn), 1e-6)), "treecover": zmean(B.tc)})
    ru = B.roads.to_crs(UTM)

    def dist_to(sub):
        return cKDTree(shapely.get_coordinates(shapely.segmentize(sub.geometry.values, 30))).query(CXY)[0]

    S["d_road"] = dist_to(ru[~ru.highway.isin(["track", "path", "footway"])])
    S["d_village"] = dist_to(ru[ru.highway.isin(["residential", "living_street"])])
    tp = shapely.get_coordinates(shapely.segmentize(ru[ru.highway.eq("track")].geometry.values, 25))
    S["track_km_km2"] = np.array([len(v) for v in cKDTree(tp).query_ball_point(CXY, 1000)]) * 0.025 / math.pi
    S["d_boundary"] = shapely.distance(cent.values, B.park_u.boundary.iloc[0])
    return SimpleNamespace(grid=grid, CXY=CXY, cell_r=cell_r, zmean=zmean, S=S)


def dyn_features(Y, key=None):
    """Ознаки, відомі на 1 січня року Y (Hansen є до 2024 включно; FIRMS — з 2021)."""
    B, G = base(key), h3grid(key)
    _, fp_year = default_events(key)
    d = pd.DataFrame(index=G.S.index)
    d["loss_hist"] = G.zmean((B.loss > 0) & (B.loss <= min(Y - 1, 2024) - 2000))
    d["loss_recent3"] = G.zmean((B.loss >= Y - 3 - 2000) & (B.loss <= min(Y - 1, 2024) - 2000))
    fu = B.firms.to_crs(UTM)
    prev = (B.firms.year < Y).values
    if prev.any():
        tree = cKDTree(np.c_[fu.geometry.x, fu.geometry.y][prev])
        d["firms_2km"] = [len(v) for v in tree.query_ball_point(G.CXY, 2000)]
        d["d_firms"] = np.minimum(tree.query(G.CXY)[0], 15000)
    else:
        d["firms_2km"], d["d_firms"] = 0, 15000.0
    prev_fp = [fp_year[y] for y in range(Y - 3, Y) if y in fp_year]
    d["burned_prev3"] = G.zmean(np.any(prev_fp, 0) if prev_fp else np.zeros((B.H, B.W), bool))
    return pd.concat([G.S, d], axis=1)


# ---------------------------------------------------------------- пряма видимість (камери / вежі)
def los(lat0, lon0, lats, lons, mast=15, target_h=20, step=30, key=None):
    """Пряма видимість з вишки (висота mast над рельєфом) на точки: дим висотою target_h видно, якщо його кутова висота
    більша за кутову висоту рельєфу SRTM у всіх проміжних точках (кривизна Землі + рефракція k=0.13). → (видно, відстань м)."""
    B, Z = base(key), dem(key)
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    dist = np.hypot((lats - lat0) * 111320, (lons - lon0) * 111320 * np.cos(np.radians(lat0)))
    f = np.linspace(0, 1, max(int(dist.max() / step), 2) + 1)[1:-1]

    def z_at(la, lo):
        return Z[np.clip(((la - B.TR.f) / B.TR.e).astype(int), 0, B.H - 1), np.clip(((lo - B.TR.c) / B.TR.a).astype(int), 0, B.W - 1)]

    curv = lambda r: 0.87 * r ** 2 / (2 * 6371000)
    z0 = float(z_at(np.array([lat0]), np.array([lon0]))[0]) + mast
    d = dist[:, None] * f[None, :]
    terr = (z_at(lat0 + np.outer(lats - lat0, f), lon0 + np.outer(lons - lon0, f)) - curv(d) - z0) / np.maximum(d, 1)
    tgt = (z_at(lats, lons) + target_h - curv(dist) - z0) / np.maximum(dist, 1)
    return tgt >= terr.max(1), dist


def azimuth(lat, lon, lat0, lon0):
    return (np.degrees(np.arctan2((np.asarray(lon) - lon0) * np.cos(np.radians(lat0)), np.asarray(lat) - lat0)) + 360) % 360
