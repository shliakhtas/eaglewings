"""Мапи сирих наборів: функції (firms, hansen …) рахує крок eda (scripts/eda.py) у data/processed/eda/; запуск цього скрипта лише збирає їх у web/science/source_maps.json і копіює картинки.

    eaglewings script 12_source_maps

Лише показує те, що лежить на диску. Контури сцен Sentinel-2 один раз беруться з каталогу STAC і кешуються.
"""
import glob
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from matplotlib import colormaps, colors
from PIL import Image

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from eaglewings.ds import common  # noqa: E402

RAW, P = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = P / "eda" / "maps"                  # картинки мап рахує крок eda; сайт копіює їх у web/assets/src
OUT.mkdir(parents=True, exist_ok=True)
B = common.base()
S, W, N, E = B.TR.f + B.TR.e * B.H, B.TR.c, B.TR.f, B.TR.c + B.TR.a * B.W
BOX = [[round(S, 5), round(W, 5)], [round(N, 5), round(E, 5)]]
CMAP = lambda n: colormaps[n]


def png(name, rgba, step=1):
    a = (np.clip(np.nan_to_num(rgba), 0, 1) * 255).astype(np.uint8)[::step, ::step]
    Image.fromarray(a, "RGBA").save(OUT / f"{name}.png", optimize=True)
    return f"../assets/src/{name}.png"


def ramp(a, cm, lo=None, hi=None, alpha=.92):
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    v = CMAP(cm)(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1))
    v[..., 3] = np.where(np.isfinite(a), alpha, 0)
    return v


def heat(lat, lon, bb, cell, name, cm="inferno"):
    (s, w), (n, e) = bb
    nx, ny = int((e - w) / cell), int((n - s) / cell)
    h = np.histogram2d(lat, lon, bins=[ny, nx], range=[[s, n], [w, e]])[0][::-1]
    a = np.log1p(h)
    a[h == 0] = np.nan
    return dict(img=png(name, ramp(a, cm)), bounds=bb, pixel=True), int(h.max())


def lg(*a):
    return [[c, t] for c, t in a]


def scale(cm, lo, hi):
    return dict(colors=[colors.to_hex(CMAP(cm)(x)) for x in np.linspace(0, 1, 6)], lo=lo, hi=hi)


def firms():
    fs = []
    for f in glob.glob(str(RAW / "firms" / f"*_{COUNTRY}.csv")):
        d = pd.read_csv(f, usecols=["latitude", "longitude"]).assign(s="MODIS" if "modis" in f else "VIIRS")
        fs.append(d)
    d = pd.concat(fs)
    g, mx = heat(d.latitude.values, d.longitude.values, [[34.7, 19.3], [41.8, 28.4]], .05, "firms_gr")
    p = B.firms
    pk, mp = heat(p.latitude.values, p.longitude.values, [[S - .04, W - .04], [N + .04, E + .04]], .006, "firms_park")
    v, _ = heat(d[d.s == "VIIRS"].latitude.values, d[d.s == "VIIRS"].longitude.values, [[34.7, 19.3], [41.8, 28.4]], .05, "firms_v", "viridis")
    return [dict(name="Теплова карта Греції", note="Кількість точок у клітинці 5 км, логарифмічна шкала. Світліше це більше точок.", raster=g, scale=scale("inferno", "1", f"{mx} точок")),
            dict(name="Парк і 3 км", note="Той самий набір у зоні парку, клітинка ~600 м. Майже все тепло дала пожежа 2023.", raster=pk, scale=scale("inferno", "1", f"{mp} точок"), fit=[[S, W], [N, E]])]


def effis():
    fs = sorted(glob.glob(str(RAW / "benchmarks" / "effis" / "*.geojson")))
    gs = pd.concat([gpd.read_file(f).assign(y=Path(f).name.split("_")[1]) for f in fs])
    c = gs.geometry.representative_point()
    g, mx = heat(c.y.values, c.x.values, (lambda b: [[b[0] - .2, b[1] - .2], [b[2] + .1, b[3] + .2]])([float(v) for v in EFFIS_BBOX.split(",")]), .1, "effis_gr", "magma")
    near = gs[gs.intersects(gpd.GeoSeries.from_wkt([f"POLYGON(({W} {S},{E} {S},{E} {N},{W} {N},{W} {S}))"], crs=gs.crs).iloc[0])].to_crs(4326)
    polys = []
    for y, geom in zip(near.y, near.geometry.simplify(.0015)):
        for p in getattr(geom, "geoms", [geom]):
            polys.append([y, [[round(b, 4), round(a, 4)] for a, b in p.exterior.coords]])
    return [dict(name="Де були згарища", note="Кількість контурів у клітинці 11 км за 10 років. Рамка запиту ширша за Грецію.", raster=g, scale=scale("magma", "1", f"{mx} контурів")),
            dict(name="Контури біля парку", note="Усі контури EFFIS, що торкаються рамки парку, колір за роком.", polys=polys, fit=[[S, W], [N, E]])]


def hansen():
    ly = B.loss.astype(float)
    ly[ly == 0] = np.nan
    tc = B.tc.astype(float)
    tc[tc < 1] = np.nan
    return [dict(name="Покриття", note="Плитка Hansen 50N 020E це 10° × 10°. З неї ми вирізаємо лише рамку парку.", rects=[dict(b=[[40, 20], [50, 30]], label="плитка Hansen 50N 020E", color="#6ea8fe"), dict(b=BOX, label="наша вирізка", color="#ff7a3c")], fit=[[39, 19], [51, 31]]),
            dict(name="Рік втрати лісу", note="Колір це рік, коли Hansen побачив втрату крон. Червоне це пожежа 2023, зелена пляма на півдні це втрати 2012 року.", raster=dict(img=png("hansen_loss", ramp(ly, "turbo", 1, 24)), bounds=BOX, pixel=False), scale=scale("turbo", "2001", "2024"), fit=BOX),
            dict(name="Лісистість 2000", note="Частка крон у пікселі, 2000 рік.", raster=dict(img=png("hansen_tc", ramp(tc, "YlGn", 0, 100), 2), bounds=BOX, pixel=False), scale=scale("YlGn", "0 %", "100 %"), fit=BOX)]


def modis():
    nd = common.modis_ndvi()
    pts = sorted({f.split("NDVI_")[1].split("_A")[0] for f in glob.glob(str(RAW / "nasa" / "modis_250m*"))})
    rects = []
    for p in pts:
        a, b = map(float, p.split("_"))
        kx = 111.32 * np.cos(np.radians(a))
        rects.append(dict(b=[[round(a - 16 / 111.32, 4), round(b - 13 / kx, 4)], [round(a + 16 / 111.32, 4), round(b + 13 / kx, 4)]], label=f"вирізка {p.replace('_', ', ')}", color="#6ea8fe"))
    d = nd["pre"] - nd["post"]
    return [dict(name="Де запитували", note="Вікна запитів до ORNL DAAC: ±16 км за широтою і ±13 км за довготою. У датасети йде вікно центру парку.", rects=rects, fit=BOX),
            dict(name="NDVI до", note="28 липня 2023.", raster=dict(img=png("modis_pre", ramp(nd["pre"], "YlGn", 0, .9), 2), bounds=BOX, pixel=True), scale=scale("YlGn", "0", "0,9"), fit=BOX),
            dict(name="NDVI після", note="14 вересня 2023.", raster=dict(img=png("modis_post", ramp(nd["post"], "YlGn", 0, .9), 2), bounds=BOX, pixel=True), scale=scale("YlGn", "0", "0,9"), fit=BOX),
            dict(name="Падіння", note="Де зелень упала. Це і є видиме згарище.", raster=dict(img=png("modis_drop", ramp(np.where(d > .05, d, np.nan), "inferno", .05, .7), 2), bounds=BOX, pixel=True), scale=scale("inferno", "0,05", "0,7"), fit=BOX)]


def sentinel2():
    fp = RAW / "nasa" / "s2" / "footprints.json"
    if not fp.exists():
        ids = [str(i) for f in sorted(glob.glob(str(RAW / "nasa" / "s2" / "comp_*.npz"))) for i in np.load(f)["ids"]]
        r = requests.post("https://earth-search.aws.element84.com/v1/search", timeout=120, json=dict(collections=["sentinel-2-l2a"], ids=ids, limit=50)).json()
        fp.write_text(json.dumps([dict(id=x["id"], date=x["properties"]["datetime"][:10], cloud=x["properties"].get("eo:cloud_cover"), geom=x["geometry"]["coordinates"][0]) for x in r["features"]]))
    it = sorted(json.load(open(fp)), key=lambda x: x["date"])
    polys = [[x["date"], [[round(b, 4), round(a, 4)] for a, b in x["geom"]], f"{x['id']}, хмарність {x['cloud']:.0f} %"] for x in it]
    z = np.load(RAW / "nasa" / "s2" / "comp_2024.npz")
    ndvi = (z["nir"].astype(float) - z["red"]) / np.maximum(z["nir"].astype(float) + z["red"], 1e-4)
    z0 = np.load(RAW / "nasa" / "s2" / "comp_2017.npz")
    idx = lambda q: (q["nir"].astype(float) - q["sw1"]) / np.maximum(q["nir"].astype(float) + q["sw1"], 1e-4)
    d = idx(z0) - idx(z)
    rgb = np.dstack([np.clip(z[k] / .35, 0, 1) for k in ("sw1", "nir", "red")])
    rgb = np.dstack([rgb, np.ones(rgb.shape[:2])])
    return [dict(name="Де були сцени", note=f"Справжні контури {len(it)} сцен із каталогу STAC. Усі лежать на одній плитці 35TMF, парк у рамці. Наведи на контур, щоб побачити дату.", polys=polys, rects=[dict(b=BOX, label="рамка парку", color="#ff7a3c")], fit=[[40.4, 25.4], [41.8, 27.1]]),
            dict(name="Псевдоколір 2024", note="SWIR, NIR, червоний: здорові крони зелені, згарище рудувате.", raster=dict(img=png("s2_rgb", rgb, 2), bounds=BOX, pixel=False), fit=BOX),
            dict(name="NDVI 2024", note="Літній композит 2024.", raster=dict(img=png("s2_ndvi", ramp(ndvi, "YlGn", 0, .9), 2), bounds=BOX, pixel=False), scale=scale("YlGn", "0", "0,9"), fit=BOX),
            dict(name="Падіння вологи 2017 → 2024", note="NDMI 2017 мінус NDMI 2024: де крони висохли або зникли за сім років.", raster=dict(img=png("s2_dndmi", ramp(np.where(d > .05, d, np.nan), "inferno", .05, .6), 2), bounds=BOX, pixel=False), scale=scale("inferno", "0,05", "0,6"), fit=BOX)]


def opera():
    fs = sorted(glob.glob(str(RAW / "nasa" / "opera" / "*.npz")))
    cnt = None
    for f in fs:
        s = np.load(f)["s"]
        m = (s >= 1) & (s <= 8)
        cnt = m.astype(np.int16) if cnt is None else cnt + m
    a = cnt.astype(float)
    a[a == 0] = np.nan
    s = np.load(RAW / "nasa" / "opera" / "2023-09-19.npz")["s"].astype(float)
    s[(s < 1) | (s > 8)] = np.nan
    return [dict(name="Як часто бачив порушення", note=f"Скільки знімків із {len(fs)} показували порушення в цьому пікселі.", raster=dict(img=png("opera_cnt", ramp(a, "inferno", 1, np.nanmax(a)), 2), bounds=BOX, pixel=False), scale=scale("inferno", "1", f"{int(np.nanmax(a))} знімків"), fit=BOX),
            dict(name="Пік, 19 вересня 2023", note="Статус на дату найбільшої хвилі: 1–3 низька втрата, 4–6 висока, 7–8 завершене.", raster=dict(img=png("opera_peak", ramp(s, "turbo", 1, 8), 2), bounds=BOX, pixel=False), scale=scale("turbo", "1", "8"), fit=BOX)]


def esri():
    rects = []
    for f in sorted(glob.glob(str(P / "sat_*.json"))):
        (a, b), (c, e) = json.load(open(f))["bounds_latlon"]
        rects.append(dict(b=[[round(a, 5), round(b, 5)], [round(c, 5), round(e, 5)]], label=Path(f).stem.replace("sat_", ""), color="#ff7a3c"))
    return [dict(name="Де були знімки", note="Дві ділянки для масок і вирізки до подій демо. Вони маленькі, тому парк покривають лише вибірково.", rects=rects, fit=BOX)]


def power():
    c = B.PARK.centroid
    kx = .625 / 2
    return [dict(name="Точка погоди", note="Запит іде в одну точку, центр парку. Прямокутник це приблизний розмір клітинки сітки NASA POWER, 0,5° × 0,625°: весь парк лежить в одній-двох клітинках.", rects=[dict(b=[[c.y - .25, c.x - kx], [c.y + .25, c.x + kx]], label="клітинка POWER", color="#6ea8fe")], pts=[[round(c.y, 3), round(c.x, 3), "центр парку", "#ff7a3c"]], fit=[[40.6, 25.5], [41.6, 26.9]])]


def srtm():
    Z = common.dem()
    gy, gx = np.gradient(Z)
    hs = np.clip(.75 + .012 * (gx - gy), 0, 1)
    v = CMAP("terrain")(np.clip(Z / 650, 0, 1)) * hs[..., None]
    v[..., 3] = .95
    return [dict(name="Рельєф", note="Висота з тінню. Гори на заході й у центрі, низина Евросу на сході.", raster=dict(img=png("srtm_relief", v, 2), bounds=BOX, pixel=False), scale=scale("terrain", "0 м", "650 м"), fit=BOX)]


def osm_park():
    return [dict(name="Межа парку", note="Контур OSM. Він обрізає всі дані, сітка H3 будується в ньому.", rects=[dict(b=BOX, label="рамка всіх растрів", color="#6ea8fe")], fit=BOX)]


def osm_roads():
    r = B.roads
    forest = {"track", "path", "footway", "bridleway"}
    lines = []
    for k, g in zip(r.highway, r.geometry.simplify(.0003)):
        for p in getattr(g, "geoms", [g]):
            lines.append([1 if k in forest else 0, [[round(b, 4), round(a, 4)] for a, b in p.coords]])
    return [dict(name="Мережа доріг", note="Жовте це лісові дороги й стежки, блакитне звичайні дороги.", lines=lines, fit=BOX)]


def wikimedia():
    ph, v = json.load(open(P / "vision" / "photos.json")), json.load(open(P / "vision" / "video.json"))   # результати кроку vision
    pts = [[p["lat"], p["lon"], p["title"], "#ffb27a"] for p in ph if p.get("lat")]
    return [dict(name="Де прив'язано кадри", note="Фото стоять там, де їх зробили. CAM-02 це умовне місце: відео знято в Карсті.", pts=pts + [[v["cam_lat"], v["cam_lon"], "CAM-02, демо", "#6ea8fe"]], fit=BOX)]


def generator():
    d = pd.read_csv(P / "signals.csv")
    col = dict(fire="#ff7a3c", logging="#7dd3a8", vehicle="#6ea8fe")
    return [dict(name="Сигнали потоку", note="Місця реальні, події сценарні. Колір це тип: вогонь, рубка, машини.", pts=[[round(a, 4), round(b, 4), f"{k}, {s}", col[k]] for a, b, k, s in zip(d.lat, d.lon, d.kind, d.source)], fit=BOX)]


MAPS = dict(firms=firms, effis=effis, hansen=hansen, modis=modis, sentinel2=sentinel2, opera=opera, esri=esri, power=power, srtm=srtm, osm_park=osm_park, osm_roads=osm_roads, wikimedia=wikimedia, generator=generator)


if __name__ == "__main__":
    # Лише збирає готові мапи кроку eda у JSON сайту й копіює картинки в web/assets/src: нічого не рахує.
    import shutil
    src = ROOT / "data" / "processed" / "eda" / "sources"
    miss = [k for k in MAPS if not (src / f"{k}.maps.json").exists()]
    if miss:
        sys.exit(f"немає мап наборів {miss}: спершу крок eda (flows/pipeline.py eda)")
    out = {k: json.loads((src / f"{k}.maps.json").read_text()) for k in MAPS}
    dst = ROOT / "web" / "assets" / "src"
    dst.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        shutil.copy(f, dst / f.name)
    (ROOT / "web" / "science" / "source_maps.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=float))
    print({k: len(v or []) for k, v in out.items()})
