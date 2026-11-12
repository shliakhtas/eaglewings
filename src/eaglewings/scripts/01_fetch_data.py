"""Збір даних для парку GR_PARK з ds/regions.py (за замовчуванням основний, НП Дадья–Лефкімі–Суфлі; кожен парк окремо):
межа парку (OSM), пожежі NASA FIRMS (VIIRS/MODIS), втрата лісу Hansen GFC (NASA/UMD),
супутникові знімки Esri World Imagery для двох ділянок (GeoTIFF).
"""
import glob
import io
import json
import math
import time
import sys
import urllib.request

import requests
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import Polygon, Point

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
UA = {"User-Agent": "dadia-research/1.0"}
from eaglewings.ds.regions import REGION  # noqa: E402
from eaglewings.ds import park as PK  # noqa: E402
RAW = ROOT / "data" / "raw"                           # архіви на всю країну (FIRMS) спільні для всіх парків
RP, OUT = PK.RAWP, PK.P                               # межа й дороги OSM та оброблені шари: свої для парку GR_PARK
CFG = REGION["fetch"]
AOIS = CFG["aois"]                                       # ділянки для знімків Esri: (назва, lat, lon, half-size у градусах)
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def _overpass(q, path):
    """Запит до OpenStreetMap з повторами й дзеркалом. Готовий файл не перезавантажуємо: OSM змінюється, а числа мають відтворюватись."""
    if path.exists():
        return json.load(open(path))
    for k in range(6):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=180)
                if r.ok:
                    path.write_text(r.text)
                    return r.json()
            except requests.RequestException:
                pass
        time.sleep(20 * (k + 1))
    raise RuntimeError("Overpass недоступний: спробуйте пізніше")


def fetch_raw():
    """Сирі набори, яких немає на диску: межа парку й дороги OSM, архіви FIRMS за країною."""
    kind, oid = PK.INFO["osm"]
    _overpass(f"[out:json];{kind}({oid});out geom;", RP / "park_osm.json")
    minx, miny, maxx, maxy = boundary_geom(json.load(open(RP / "park_osm.json"))["elements"][0]).bounds
    b = (miny - .05, minx - .07, maxy + .05, maxx + .07)
    _overpass(f"[out:json][timeout:180];way[highway]({b[0]},{b[1]},{b[2]},{b[3]});out geom;", RP / "roads_osm.json")
    F = CFG["firms"]
    (RAW / "firms").mkdir(parents=True, exist_ok=True)
    for s in F["sensors"]:
        for y in F["years"]:
            p = RAW / "firms" / f"{s}_{y}_{F['country']}.csv"
            if not p.exists():
                r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/data/country/{s}/{y}/{s}_{y}_{F['country']}.csv", headers=UA, timeout=300)
                r.raise_for_status()
                p.write_bytes(r.content)
                print("FIRMS", p.name, len(r.content) // 1024, "КБ")


ZOOM = 18


# ---------- 1) межа парку ----------
def boundary_geom(el):
    """Контур із відповіді Overpass: way це одне кільце; relation (мультиполігон) складається з кілець зовнішніх членів
    мінус внутрішні (так задані парки з кількох частин, напр. НП «Східна Македонія й Фракія»)."""
    from shapely.geometry import LineString, MultiPolygon
    from shapely.ops import linemerge, polygonize, unary_union
    if el["type"] == "way":
        return Polygon([(p["lon"], p["lat"]) for p in el["geometry"]])
    def rings(role):
        ls = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in el["members"]
              if m["type"] == "way" and m.get("role", "outer") == role and len(m.get("geometry", [])) > 1]
        return list(polygonize(linemerge(ls))) if ls else []
    g = unary_union(rings("outer"))
    inner = rings("inner")
    if inner:
        g = g.difference(unary_union(inner))
    return g if g.geom_type == "MultiPolygon" or g.geom_type == "Polygon" else MultiPolygon([x for x in getattr(g, "geoms", []) if x.geom_type == "Polygon"])


def park_boundary():
    el = json.load(open(RP / "park_osm.json"))["elements"][0]
    poly = boundary_geom(el)
    gdf = gpd.GeoDataFrame({"name": [PK.INFO["name_en"]], "name_el": [el["tags"]["name"]], "park": [PK.KEY]},
                           geometry=[poly], crs=4326)
    gdf.to_file(OUT / "park.geojson", driver="GeoJSON")
    return gdf


def roads(park):
    """Дороги й стежки OSM, що перетинають парк із буфером із конфігурації: з них відстань до доріг, густота й мережа під'їздів."""
    from shapely.geometry import LineString
    rows = [{"highway": e["tags"].get("highway"), "geometry": LineString([(p["lon"], p["lat"]) for p in e["geometry"]])}
            for e in json.load(open(RP / "roads_osm.json"))["elements"] if len(e.get("geometry", [])) > 1]
    g = gpd.GeoDataFrame(rows, crs=4326)
    zone = park.to_crs(CFG["utm"]).buffer(CFG["buffer_m"]).to_crs(4326).iloc[0]
    g = g[g.intersects(zone)].reset_index(drop=True)
    g.to_file(OUT / "roads.geojson", driver="GeoJSON")
    print("OSM roads:", len(g), g.highway.value_counts().head(4).to_dict())
    return g


# ---------- 2) пожежі FIRMS ----------
def firms(park):
    zone = park.to_crs(3857).buffer(CFG["buffer_m"]).to_crs(4326).iloc[0]   # парк + буфер із конфігурації
    frames = []
    for f in glob.glob(str(RAW / "firms" / "*.csv")):
        df = pd.read_csv(f)
        df["source"] = Path(f).name.split("_")[0]                  # viirs-snpp / modis
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    minx, miny, maxx, maxy = zone.bounds
    df = df[df.longitude.between(minx, maxx) & df.latitude.between(miny, maxy)]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    gdf = gdf[gdf.within(zone)].copy()
    gdf["date"] = pd.to_datetime(gdf.acq_date)
    gdf["confidence"] = gdf["confidence"].astype(str)
    keep = ["latitude", "longitude", "date", "acq_time", "source", "confidence", "frp", "daynight", "geometry"]
    gdf = gdf[keep]
    gdf.to_file(OUT / "firms.geojson", driver="GeoJSON")
    print("FIRMS points:", len(gdf), gdf.groupby(gdf.date.dt.year).size().to_dict())
    return gdf


# ---------- 3) втрата лісу Hansen GFC ----------
HANSEN = "/vsicurl/https://storage.googleapis.com/earthenginepartners-hansen/{v}/Hansen_{v}_{{}}_{t}.tif".format(v=CFG["hansen"]["version"], t=CFG["hansen"]["tile"])


def hansen(park):
    minx, miny, maxx, maxy = park.total_bounds
    with rasterio.open(HANSEN.format("lossyear")) as src:
        win = window_from_bounds(minx, miny, maxx, maxy, src.transform)
        loss = src.read(1, window=win)
        tr = src.window_transform(win)
    with rasterio.open(HANSEN.format("treecover2000")) as src:
        cover = src.read(1, window=window_from_bounds(minx, miny, maxx, maxy, src.transform))
    np.save(OUT / "treecover2000.npy", cover)
    json.dump({"bounds": [minx, miny, maxx, maxy], "transform": list(tr)[:6]}, open(OUT / "hansen_meta.json", "w"))
    rows, cols = np.nonzero(loss)
    xs, ys = rasterio.transform.xy(tr, rows, cols)
    gdf = gpd.GeoDataFrame({"year": 2000 + loss[rows, cols].astype(int),
                            "treecover2000": cover[rows, cols].astype(int)},
                           geometry=gpd.points_from_xy(xs, ys), crs=4326)
    gdf = gdf[gdf.within(park.iloc[0].geometry)].copy()
    gdf.to_file(OUT / "hansen_loss.geojson", driver="GeoJSON")
    print("Hansen loss pixels (30 m):", len(gdf), gdf.groupby("year").size().to_dict())
    return gdf


# ---------- 4) супутникові знімки ----------
def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile2deg(x, y, z):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def satellite(name, lat, lon, dlat, dlon, **_):
    path = OUT / f"sat_{name}.tif"
    if path.exists():
        return path
    x0, y0 = deg2tile(lat + dlat, lon - dlon, ZOOM)
    x1, y1 = deg2tile(lat - dlat, lon + dlon, ZOOM)
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{ZOOM}/{ty}/{tx}"
            data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
            mosaic.paste(Image.open(io.BytesIO(data)).convert("RGB"), ((tx - tx0) * 256, (ty - ty0) * 256))
            time.sleep(0.02)
    north, west = tile2deg(tx0, ty0, ZOOM)
    south, east = tile2deg(tx1 + 1, ty1 + 1, ZOOM)
    arr = np.asarray(mosaic).transpose(2, 0, 1)
    # тайли у Web Mercator: зберігаємо в EPSG:3857, щоб пікселі були точними
    import pyproj
    t = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    (mx0, my1), (mx1, my0) = t.transform(west, north), t.transform(east, south)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=3,
                       dtype="uint8", crs="EPSG:3857", transform=from_bounds(mx0, my0, mx1, my1, arr.shape[2], arr.shape[1]),
                       compress="deflate") as dst:
        dst.write(arr)
    mosaic.save(OUT / f"sat_{name}.jpg", quality=90)
    json.dump({"bounds_latlon": [[south, west], [north, east]]}, open(OUT / f"sat_{name}.json", "w"))
    print("saved", path, mosaic.size)
    return path


PARTIAL = 3          # код виходу: частину шарів не скачано; потік продовжує, але показує, чого бракує


def layers():
    """Усі інші сирі шари, які беруть задачі: качаємо тут, щоб задачі працювали лише з диска.
    Якщо джерело недоступне, решту все одно качаємо, а наприкінці кажемо, чого бракує."""
    from eaglewings.ds import common, opera, pflow, pages
    from eaglewings.ds.tasks import fire_spread as FS
    from eaglewings.ds.tasks import logging_greece as LG
    # третій елемент: який набір після цього кроку скачано повністю (сторінка-звіт у Prefect, з --clearml ще й версія в ClearML)
    S2 = list(range(min(LG.YEARS) - 1, max(LG.YEARS) + 1))
    jobs = [("SRTM relief (AWS Terrain Tiles)", common.dem, "srtm"), ("NASA POWER daily weather", common.power, "power"),
            ("MODIS NDVI (NASA ORNL)", common.modis_ndvi, "modis")]
    jobs += [(f"Sentinel-2 composite {y}", lambda y=y: LG.composite(y), "sentinel2" if y == S2[-1] else None) for y in S2]
    jobs += [("EFFIS burnt areas by year", lambda: [LG.effis_year(y) for y in LG.YEARS], None)]
    jobs += [(f"NASA OPERA DIST-ALERT {y}", lambda y=y: [opera.snapshot(d) for d in opera.dates(y)], "opera" if y == 2024 else None) for y in (2023, 2024)]
    if PK.KEY == PK.MAIN:                                   # поширення рахується лише на пожежі Еврос-2023 основного парку
        jobs += [("fire spread: relief, Hansen, hourly weather per fire", FS.prefetch, None)]
    bad = []
    for name, fn, src in jobs:
        pflow.mark(name)
        try:
            fn()
            if src:
                pages.downloaded(src)
        except Exception as e:
            pflow.failed()
            bad.append(name)
            print(f"НЕ ВДАЛОСЯ: {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    return bad


if __name__ == "__main__":
    from eaglewings.ds import pflow, pages
    pflow.mark("raw: OSM boundary and roads, FIRMS archives")
    fetch_raw()
    pflow.mark("park boundary")
    park = park_boundary()
    pages.downloaded("osm_park")
    pflow.mark("roads: park + 3 km")
    roads(park)
    pages.downloaded("osm_roads")
    pflow.mark("FIRMS points: park + 3 km")
    firms(park)
    pages.downloaded("firms")
    pflow.mark("Hansen forest loss")
    hansen(park)
    pages.downloaded("hansen")
    if PK.KEY == PK.MAIN:                                   # знімки Esri лише для застосунку основного парку
        for k, v in AOIS.items():
            pflow.mark(f"Esri area image: {k}")
            satellite(k, **v)
        pages.downloaded("esri")
        json.dump(AOIS, open(OUT / "aois.json", "w"), ensure_ascii=False, indent=1)
    bad = layers()
    if bad:
        print("Не скачано:", "; ".join(bad), "— задачі, яким потрібні ці шари, впадуть", flush=True)
    import os
    import subprocess
    if "GR_PARK" not in os.environ:                   # без GR_PARK — усі парки дослідження: решта парків окремими процесами
        for k in PK.PARKS:
            if k == PK.KEY:
                continue
            print(f"=== парк {k}", flush=True)
            r = subprocess.run([sys.executable, __file__, *sys.argv[1:]], env=dict(os.environ, GR_PARK=k))
            if r.returncode:
                bad.append(f"парк {k} (код {r.returncode})")
    if bad:
        sys.exit(PARTIAL)
