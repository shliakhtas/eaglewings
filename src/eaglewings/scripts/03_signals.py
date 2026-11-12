"""Сценарний потік сигналів застосунку для кожного парку → signals.csv, ops.json, areals.geojson, timeline.json у теці парку.
Запуск із теки eaglewings:

    eaglewings script 03_signals                       # усі парки: основний тут, решта окремими процесами
    GR_PARK=evros_delta eaglewings script 03_signals   # один парк

Основний парк (НП Дадья) має камеру-вежу з відео й 12 реальних детекцій диму; в інших парках потік повністю сценарний.
Потік основного парку не залежить від інших парків: на ньому порахована гілка XM.

Це симуляція: події сценарні, розміщені на реальних місцях парку (лісові дороги OSM, сектор камери).
Реальні лише 12 детекцій диму з демо-відео. Еталон для XM: сценарій і день генератора.
Знімки Esri для сценаріїв беруться з кешу data/processed/sat_scene_*; якщо їх немає, завантажуються.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
import h3
from sklearn.cluster import DBSCAN

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import park as PK  # noqa: E402

P = PK.processed()                                    # тека парку GR_PARK (основний: data/processed/)
RAW = PK.raw()                                        # дороги OSM парку
IS_MAIN = PK.KEY == PK.MAIN                           # камера-вежа з відео й реальні детекції диму лише в основному парку
from eaglewings.ds.regions import LAT0 as _LAT0  # noqa: E402
LAT0 = _LAT0 if IS_MAIN else round(float(gpd.read_file(P / "park.geojson").geometry.iloc[0].centroid.y), 2)   # широта для метрів у градусах
NOW = pd.Timestamp("2026-10-19 12:00")
WINDOW_DAYS = 90
H3_RES = 8
rng = np.random.default_rng(7)

TYPE_UA = {"fire": "Пожежа", "logging": "Вирубка", "vehicle": "Незаконний проїзд"}
FRESH_DAYS = 3
PLAN_DAYS = 14
SOURCES = ["satellite", "camera", "trap", "social", "expert"]
SOURCE_UA = {"satellite": "супутник", "camera": "камера", "trap": "фотопастка", "social": "соцмережі", "expert": "гаряча лінія / експерт"}
SOURCE_RELIABILITY = {"satellite": 0.7, "camera": 0.85, "trap": 0.8, "social": 0.35, "expert": 0.85}
# джерела екстреного реагування (вікно 1): встановлені камери, фотопастки, гаряча лінія / експерт
LIVE_SOURCES = {"camera", "trap", "expert"}
CLUSTER_EPS_KM = {"fire": 1.0, "logging": 0.5, "vehicle": 0.8}
AREAL_EPS_KM = {"fire": 2.0, "logging": 1.5, "vehicle": 2.0}

RESPONSE = {
    "fire": dict(title="Виїзд групи реагування", team="Група реагування",
                 checklist=["Підтвердити осередок на місці", "Локалізувати / викликати пожежну службу (199)", "Фото до/після"]),
    "logging": dict(title="Перевірка вирубки", team="Лісова охорона",
                    checklist=["Фотофіксація пнів і техніки", "Звірка з лісорубними квитками", "Протокол у разі порушення"]),
    "vehicle": dict(title="Перехоплення незаконного проїзду", team="Патрульна група",
                    checklist=["Виїзд на трек", "Фіксація номерів / слідів", "Протокол"]),
}
# три види дій в ареалі:
#   one  — одиничні: робляться один раз (інфраструктура / профілактика);
#   post — постекстрені: після кожної підтвердженої події (частина виїзду на подію);
#   check — системні перевірки зробленого одиничним, раз на N днів; patrol — системний патруль активної зони, раз на N днів
ACTIONS = {
    "fire": dict(one=["Мінералізовані смуги (з Пожежною службою)"], post=["Огляд згарища й оцінка площі", "Фотофіксація наслідків"],
                 check=["Перевірка мінералізованих смуг"], patrol="Патруль активної зони в пік спеки"),
    "logging": dict(one=["Шлагбаум на лісовій дорозі"], post=["Фотофіксація пнів", "Звірка з лісорубними білетами"],
                    check=["Перевірка шлагбауму"], patrol="Патруль активної зони"),
    "vehicle": dict(one=["Шлагбаум на в'їзді в трек"], post=["Фіксація слідів і номерів, протокол"],
                    check=["Перевірка шлагбауму"], patrol="Патруль треку"),
}
TEAM = "Лісники (групи А / Б)"
CHECK_FOR = {"Мінералізовані смуги (з Пожежною службою)": "Перевірка мінералізованих смуг", "Шлагбаум на лісовій дорозі": "Перевірка шлагбауму",
             "Шлагбаум на в'їзді в трек": "Перевірка шлагбауму", "Встановити камеру": "Перевірка камери"}


# ================================================================ дані


def load_roads():
    rows = [{"highway": e["tags"].get("highway"), "geometry": LineString([(p["lon"], p["lat"]) for p in e["geometry"]])}
            for e in json.load(open(RAW / "roads_osm.json"))["elements"] if len(e.get("geometry", [])) > 1]
    return gpd.GeoDataFrame(rows, crs=4326)


# ================================================================ сценарний потік сигналів (90 днів)
# кожен сценарій — місце подій зі своєю поведінкою; джерела бачать його з різною ймовірністю
SCENARIOS = [
    dict(kind="logging", every=7, life="active", dyn="expand", sources=["satellite", "camera", "trap", "social", "expert"],
         note="системна щотижнева вирубка, бачать усі джерела, розширюється"),
    dict(kind="logging", every=30, life="active", dyn=None, sources=["satellite"], note="раз на місяць, видно лише на супутнику"),
    dict(kind="logging", every=7, life="faded", dyn=None, sources=["satellite", "expert"], note="була щотижня, згасла"),
    dict(kind="logging", every=10, life="resumed", dyn=None, sources=["satellite", "social", "expert"], note="довго не було, відновилась"),
    dict(kind="logging", every=7, life="active", dyn="shrink", sources=["satellite", "expert", "social"], note="системна, звужується"),
    dict(kind="fire", every=8, life="active", dyn=None, sources=["satellite", "camera", "social"], at_camera=True,
         note="повторні загоряння в секторі камери-вежі"),
    dict(kind="fire", every=None, life="faded", dyn=None, sources=["satellite", "expert", "social"], note="разова пожежа, згасла"),
    dict(kind="fire", every=30, life="active", dyn=None, sources=["social"], note="раз на місяць, лише соцмережі"),
    dict(kind="vehicle", every=7, life="active", dyn="expand", sources=["camera", "trap", "expert"],
         note="щотижневий нічний проїзд, розширюється вздовж треку"),
    dict(kind="vehicle", every=30, life="active", dyn=None, sources=["social"], note="раз на місяць, лише соцмережі"),
    dict(kind="vehicle", every=12, life="resumed", dyn=None, sources=["camera", "trap", "satellite", "social", "expert"],
         note="відновився після паузи, бачать усі"),
    dict(kind="vehicle", every=7, life="new", dyn=None, sources=["satellite"], note="новий, лише супутник"),
]
SEEN_P = {"satellite": 0.75, "camera": 0.9, "trap": 0.7, "social": 0.5, "expert": 0.6}


def event_days(sc):
    """Скільки днів тому відбувалися події сценарію (0 = сьогодні)."""
    every, life = sc["every"], sc["life"]
    if every is None:
        return [60]
    if life == "resumed":                    # активність на початку вікна, пауза, знову зараз
        return list(range(WINDOW_DAYS - 2, 58, -every)) + list(range(13, -1, -every))
    hi, lo = {"active": (WINDOW_DAYS - 2, 0), "faded": (WINDOW_DAYS - 2, 50), "new": (18, 0)}[life]
    return list(range(hi, lo - 1, -every))


def scenario_sites(park, roads, cam):
    """Місця сценаріїв: на реальних лісових треках OSM у парку, ≥ ~4 км одне від одного в межах типу."""
    poly = park.geometry.iloc[0]
    tracks = roads[(roads.highway == "track") & roads.within(poly.buffer(-0.005))].reset_index(drop=True)
    sites, used = [], {k: [] for k in TYPE_UA}
    for sc in SCENARIOS:
        if sc.get("at_camera") and cam is not None:
            c, line = Point(cam), None
        else:
            for _ in range(500):
                line = tracks.geometry.iloc[rng.integers(len(tracks))]
                c = line.interpolate(0.5, normalized=True)
                if line.length > 0.004 and all(c.distance(u) > 0.045 for u in used[sc["kind"]]):
                    break
        used[sc["kind"]].append(c)
        sites.append(dict(center=c, line=line))
    return sites


def footprint(lat, lon, kind):
    """Контур, який знаходить супутник: згарище (120–320 м) або просіка (60–160 м) — витягнута пляма з м'якими краями."""
    r0 = rng.uniform(120, 320) if kind == "fire" else rng.uniform(60, 160)
    stretch, rot = rng.uniform(1.0, 1.6), rng.uniform(0, np.pi)
    ang = np.linspace(0, 2 * np.pi, 9, endpoint=False) + rng.uniform(-0.2, 0.2, 9)
    rad = r0 * rng.uniform(0.82, 1.1, 9)
    pts = np.c_[rad * np.cos(ang) * stretch, rad * np.sin(ang)]
    for _ in range(2):                                   # згладжування Чайкіна
        nxt = np.roll(pts, -1, axis=0)
        pts = np.column_stack([0.75 * pts + 0.25 * nxt, 0.25 * pts + 0.75 * nxt]).reshape(-1, 2)
    x, y = pts[:, 0] * np.cos(rot) - pts[:, 1] * np.sin(rot), pts[:, 0] * np.sin(rot) + pts[:, 1] * np.cos(rot)
    return [[round(lon + dx / (111320 * np.cos(np.radians(lat))), 6), round(lat + dy / 110540, 6)] for dx, dy in zip(x, y)]


DEVICES = []          # встановлені камери й фотопастки: id, тип, координати, сектор огляду


def device_for(k, s, site, cam_video):
    """Камера/фотопастка, що бачить місце сценарію: ставимо за 150–300 м від центру, дивиться на нього."""
    for d in DEVICES:
        if d["scenario"] == k and d["source"] == s:
            return d
    if s == "camera" and SCENARIOS[k].get("at_camera") and cam_video is not None:
        d = dict(id="CAM-02", source="camera", lat=cam_video[1], lon=cam_video[0], bearing=cam_video[2], scenario=k, video=True)
    else:
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(150, 300)
        lat = site["center"].y + dist * np.sin(ang) / 110540
        lon = site["center"].x + dist * np.cos(ang) / (111320 * np.cos(np.radians(LAT0)))
        bearing = (np.degrees(np.arctan2(site["center"].x - lon, site["center"].y - lat)) + 360) % 360
        n = sum(x["source"] == s and not x["video"] for x in DEVICES) + 1
        if s == "camera" and n >= 2:
            n += 1                                     # CAM-02 — камера-вежа з відеострімом
        d = dict(id=f"{'CAM' if s == 'camera' else 'TRAP'}-{n:02d}", source=s, lat=round(lat, 5), lon=round(lon, 5),
                 bearing=round(bearing), scenario=k, video=False)
    DEVICES.append(d)
    return d


# свіжі події «зараз» для екстреного вікна: (сценарій, джерело, годин тому)
NOW_EVENTS = [(7, "expert", 3), (4, "trap", 6), (4, "expert", 5), (3, "expert", 20), (11, "expert", 9)]


SCENES = {}          # сценарій → знімок ділянки (Esri), у межах якого супутник «бачить» події


def satellite_scenes(sites):
    """Для кожного місця, яке спостерігає супутник, знімок ділянки ~1,8 × 1,8 км (Esri, z17). Беремо з кешу, якщо він є."""
    import importlib.util
    import shutil
    web = ROOT / "web" / "assets"
    fd = None
    for k, (sc, site) in enumerate(zip(SCENARIOS, sites)):
        if "satellite" not in sc["sources"]:
            continue
        name = f"scene_s{k:02d}" if IS_MAIN else f"scene_{PK.KEY}_s{k:02d}"
        if not (P / f"sat_{name}.json").exists():
            if fd is None:
                spec = importlib.util.spec_from_file_location("fetch", LIB / "scripts" / "01_fetch_data.py")
                fd = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fd)
                fd.ZOOM = 17
            fd.satellite(name, lat=site["center"].y, lon=site["center"].x, dlat=0.008, dlon=0.0105)
            shutil.copy(P / f"sat_{name}.jpg", web / f"{name}.jpg")
        (s_, w_), (n_, e_) = json.load(open(P / f"sat_{name}.json"))["bounds_latlon"]
        SCENES[k] = dict(id=name, img=f"assets/{name}.jpg", bounds=[[s_, w_], [n_, e_]], kind=sc["kind"])


def inside_scene(k, lat, lon, inset=0.18):
    """Притискаємо точку всередину знімка (з відступом), щоб контур не виходив за його межі."""
    (s_, w_), (n_, e_) = SCENES[k]["bounds"]
    dy, dx = (n_ - s_) * inset, (e_ - w_) * inset
    return float(np.clip(lat, s_ + dy, n_ - dy)), float(np.clip(lon, w_ + dx, e_ - dx))


def simulate_stream(park, roads, cam, cam_video):
    rows = []
    sites = scenario_sites(park, roads, cam)
    satellite_scenes(sites)
    for k, src, h in NOW_EVENTS:
        sc, site = SCENARIOS[k], sites[k]
        dev = device_for(k, src, site, cam_video)["id"] if src in ("camera", "trap") else ""
        if site["line"] is not None and sc["kind"] == "vehicle":
            pt = site["line"].interpolate(float(rng.uniform(0.4, 0.6)), normalized=True)
            lat, lon = pt.y, pt.x
        else:
            lat, lon = site["center"].y + rng.normal(0, 0.002), site["center"].x + rng.normal(0, 0.0026)
        rows.append(dict(kind=sc["kind"], source=src, lat=lat, lon=lon, date=NOW - pd.Timedelta(hours=h), conf=round(float(rng.uniform(0.6, 0.9)), 2),
                         label=f"{SOURCE_UA[src]}: {TYPE_UA[sc['kind']].lower()}", file="", simulated=True, scenario=k, poly=None, device=dev))
    # рідкі додаткові живі події за попередні дні (дзвінки, кадри пасток/камер) — на місцях активних сценаріїв
    for d in range(WINDOW_DAYS - 3, 3, -3):
        live_sc = [k for k, sc in enumerate(SCENARIOS) if d in range(min(event_days(sc)) , max(event_days(sc)) + 1)]
        for k in rng.choice(live_sc, size=min(2, len(live_sc)), replace=False) if live_sc else []:
            sc, site = SCENARIOS[k], sites[k]
            src = str(rng.choice([x for x in ("camera", "trap", "expert") if x in sc["sources"]] or ["expert"]))
            dev = device_for(k, src, site, cam_video)["id"] if src in ("camera", "trap") else ""
            if site["line"] is not None and sc["kind"] == "vehicle":
                pt = site["line"].interpolate(float(rng.uniform(0.35, 0.65)), normalized=True)
                lat, lon = pt.y, pt.x
            else:
                lat, lon = site["center"].y + rng.normal(0, 0.002), site["center"].x + rng.normal(0, 0.0026)
            rows.append(dict(kind=sc["kind"], source=src, lat=lat, lon=lon, date=NOW - pd.Timedelta(days=int(d), hours=int(rng.integers(2, 20))),
                             conf=round(float(rng.uniform(0.55, 0.9)), 2), label=f"{SOURCE_UA[src]}: {TYPE_UA[sc['kind']].lower()}", file="",
                             simulated=True, scenario=k, poly=None, device=dev))
    for k, (sc, site) in enumerate(zip(SCENARIOS, sites)):
        days = sorted(event_days(sc), reverse=True)
        for j, d in enumerate(days):
            prog = j / max(1, len(days) - 1)            # 0 → 1 у часі
            r = 0.003 * {"expand": 0.4 + 2.2 * prog, "shrink": 2.6 - 2.2 * prog}.get(sc["dyn"], 1.0)
            ts = NOW - pd.Timedelta(days=d) - pd.Timedelta(hours=float(rng.integers(0, 10)))
            seen = [s for s in sc["sources"] if rng.random() < SEEN_P[s]] or [sc["sources"][0]]
            for s in seen:
                dev = device_for(k, s, site, cam_video)["id"] if s in ("camera", "trap") else ""
                for _ in range(int(rng.integers(1, 3))):
                    if site["line"] is not None and sc["kind"] == "vehicle":
                        pos = float(np.clip(0.5 + rng.normal(0, 0.15 * r / 0.003), 0, 1))
                        pt = site["line"].interpolate(pos, normalized=True)
                        lat, lon = pt.y + rng.normal(0, 0.0003), pt.x + rng.normal(0, 0.0003)
                    else:
                        lat, lon = site["center"].y + rng.normal(0, r), site["center"].x + rng.normal(0, r * 1.3)
                    poly = None
                    if s == "satellite":                         # супутник бачить лише в межах свого знімка
                        lat, lon = inside_scene(k, lat, lon)
                        if sc["kind"] in ("fire", "logging"):
                            (s_, w_), (n_, e_) = SCENES[k]["bounds"]
                            g = Polygon(footprint(lat, lon, sc["kind"])).intersection(box(w_, s_, e_, n_))
                            poly = [[round(x, 6), round(y, 6)] for x, y in g.exterior.coords]
                            c = g.centroid                       # точка спостереження = центроїд знайденого полігону
                            lat, lon = c.y, c.x
                    rows.append(dict(kind=sc["kind"], source=s, lat=lat, lon=lon, date=ts, conf=round(float(rng.uniform(0.45, 0.9)), 2),
                                     label=f"{SOURCE_UA[s]}: {TYPE_UA[sc['kind']].lower()}", file="", simulated=True, scenario=k,
                                     poly=poly, device=dev, scene=SCENES[k]["id"] if s == "satellite" else ""))
    return pd.DataFrame(rows)


def real_stream():
    """Реальні детекції, що потрапляють у вікно: дим/вогонь з камери (GroundingDINO)."""
    if not IS_MAIN:
        return pd.DataFrame(columns=["kind", "source", "lat", "lon", "date", "conf", "label", "file", "simulated", "scenario", "device"])
    det = gpd.read_file(P / "detections.geojson")
    det = det[(det.source == "yolo_cam") & det.kind.isin(["fire", "smoke"])]
    return pd.DataFrame(dict(kind="fire", source="camera", lat=det.geometry.y, lon=det.geometry.x,
                             date=pd.to_datetime(det.date), conf=det.conf.astype(float),
                             label="камера: " + det.label + " (реальна детекція)", file=det.file, simulated=False, scenario=-1, device="CAM-02"))


def build():
    park = gpd.read_file(P / "park.geojson")
    roads = load_roads()
    if not IS_MAIN:                                   # без камери-вежі з відео: усі камери й пастки ставляться біля сценаріїв
        return finish(simulate_stream(park, roads, None, None))
    video = json.load(open(P / "vision" / "video.json"))
    b = np.radians(video["bearing"])
    cam = (video["cam_lon"] + video["view_km"] / (111 * np.cos(np.radians(LAT0))) * np.sin(b),
           video["cam_lat"] + video["view_km"] / 111 * np.cos(b))
    ev = pd.concat([simulate_stream(park, roads, cam, (video["cam_lon"], video["cam_lat"], video["bearing"])), real_stream()], ignore_index=True)
    return finish(ev)


def finish(ev):
    ev["date"] = pd.to_datetime(ev.date).dt.tz_localize(None)
    return ev


# ================================================================ стрічка подій і ареали для застосунку (без маршрутів і завдань)
# ================================================================ 01 · кластери
def dbscan(lat, lon, eps_km):
    return DBSCAN(eps=eps_km / 6371.0088, min_samples=1, metric="haversine",
                  algorithm="ball_tree").fit(np.radians(np.c_[lat, lon])).labels_


def confidence(g):
    """1 − Π(1 − надійність × сила) по незалежних джерелах; сила = найкращий conf + бонус за повтор."""
    return round(float(1 - np.prod([1 - SOURCE_RELIABILITY[s] * min(1.0, float(c.conf.max()) + 0.05 * (len(c) - 1))
                                    for s, c in g.groupby("source")])), 2)


def conf_level(p):
    return "висока" if p >= 0.7 else "середня" if p >= 0.45 else "низька"


def clusters(ev):
    ev = ev.assign(day=ev.date.dt.strftime("%Y-%m-%d"))
    out = []
    for (kind, day), g in ev.groupby(["kind", "day"]):
        labels = dbscan(g.lat.values, g.lon.values, CLUSTER_EPS_KM[kind])
        for lab in np.unique(labels):
            c = g[labels == lab]
            pts = list(zip(c.lon, c.lat))
            out.append(dict(kind=kind, day=day, date=c.date.max(), lat=float(c.lat.mean()), lon=float(c.lon.mean()),
                            sources=sorted(c.source.unique()), n=int(len(c)), confidence=confidence(c),
                            simulated=bool(c.simulated.all()), files=[f for f in c.file.unique() if f],
                            labels=sorted(c.label.unique())[:5], pts=pts, idx=list(c.index),
                            devices=sorted(d for d in c.device.fillna("").unique() if d),
                            geometry=MultiPoint(pts).convex_hull.buffer(0.0012)))
    cl = gpd.GeoDataFrame(out, crs=4326).sort_values("date").reset_index(drop=True)
    cl["id"] = [f"C-{i + 1:03d}" for i in range(len(cl))]
    return cl




def response_tasks(cl, now=NOW):
    """Події на дату now: кластери за FRESH_DAYS до now, підтверджені хоча б одним живим джерелом."""
    age = (now - cl.date).dt.total_seconds()
    fresh = cl[(age >= 0) & (age <= FRESH_DAYS * 86400) & cl.sources.apply(lambda ss: bool(LIVE_SOURCES & set(ss)))]
    tasks = []
    for c in fresh.itertuples():
        r = RESPONSE[c.kind]
        prio = "критичний" if c.kind == "fire" and c.confidence >= 0.45 else "високий" if c.confidence >= 0.45 else "середній"
        tasks.append(dict(id="", cluster=c.id, kind=c.kind, title=r["title"], subtitle=TYPE_UA[c.kind], team=r["team"],
                          checklist=r["checklist"], priority=prio, confidence=c.confidence, confidence_level=conf_level(c.confidence),
                          lat=round(c.lat, 5), lon=round(c.lon, 5), h3=h3.latlng_to_cell(c.lat, c.lon, H3_RES),
                          detected=c.date.strftime("%Y-%m-%d %H:%M"), signals=c.n,
                          sources=[SOURCE_UA[s] for s in c.sources], devices=c.devices, files=c.files, simulated=c.simulated,
                          areal=c.areal))
    order = {"критичний": 0, "високий": 1, "середній": 2}
    tasks.sort(key=lambda t: (order[t["priority"]], -t["confidence"]))
    for i, t in enumerate(tasks):
        t["id"] = f"R-{i + 1:03d}"
    return tasks



# ================================================================ 02 · суперареали
def regularity(days_ago):
    if len(days_ago) < 2:
        return "разово", None
    med = float(np.median(np.diff(sorted(days_ago))))
    if med <= 8:
        txt = "щотижня, системно" if len(days_ago) >= 4 else "щотижня"
    elif med <= 16:
        txt = "раз на 1–2 тижні"
    elif med <= 35:
        txt = "раз на місяць"
    else:
        txt = "епізодично"
    return txt, med


def coverage(per_source):
    seen = [s for s in SOURCES if per_source[s]]
    if len(seen) == 1:
        return f"лише {SOURCE_UA[seen[0]]}"
    if len(seen) == len(SOURCES):
        return "всі джерела"
    return " + ".join(SOURCE_UA[s] for s in seen)


def lifecycle(days_ago, med):
    """Згас — якщо тиша довша за 2 власні інтервали (мін. 21 дн.); відновився — якщо всередині була така пауза."""
    d = sorted(days_ago)                                    # 0 = сьогодні
    last, first = d[0], d[-1]
    gap = int(np.diff(d).max()) if len(d) > 1 else 0
    quiet = max(21, 2 * med) if med else 21
    if last > quiet:
        return f"згас ({last} дн. тому)"
    if gap >= quiet:
        return f"відновився після паузи {gap} дн."
    if first <= 21:
        return "новий"
    return "активний"


def dynamics(cs):
    if len(cs) < 4:
        return "мало даних", None
    cs = cs.sort_values("date")
    half = len(cs) // 2

    def spread(part):
        pts = np.array([p for ps in part.pts for p in ps])
        return float(np.sqrt(np.var(pts[:, 0]) + np.var(pts[:, 1])) + 1e-6)
    a, b = spread(cs.iloc[:half]), spread(cs.iloc[half:])
    ch = round(100 * (b - a) / a)
    return ("розширюється" if ch > 30 else "звужується" if ch < -30 else "стабільний"), ch


def recommendations(per_source, life, med):
    seen = {s for s, v in per_source.items() if v}
    rec = []
    if "camera" not in seen and med and med <= 16 and not life.startswith("згас"):
        rec.append("поставити камеру — системна подія без відеоконтролю")
    if seen == {"social"}:
        rec.append("перевірити експертом — є лише соцмережі")
    if seen == {"satellite"}:
        rec.append("наземна перевірка — видно лише на супутнику")
    if life.startswith("відновився"):
        rec.append("з'ясувати причину відновлення, посилити контроль")
    if life.startswith("згас"):
        rec.append("контроль раз на місяць, щоб не відновилось")
    return rec


RHYTHM_UA = {1: "щодня", 2: "через день", 3: "2 рази на тиждень", 7: "щотижня", 14: "раз на 2 тижні", 30: "раз на місяць"}


CONTOUR_DAYS = 30


def current_extent(cs, now, recent_days=CONTOUR_DAYS):
    """Контур = поточна зона: УСІ точки за останні recent_days (свіже не буває викидом).
    Якщо свіжих < 3 (рідкісні події) — усі точки вікна. Буфер ~150 м."""
    rec = cs[(now - cs.date).dt.days <= recent_days]
    pts = [p for ps in rec.pts for p in ps]
    if len(pts) < 3:
        pts = [p for ps in cs.pts for p in ps]
    return MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(0.0018)


def area_actions(kind, cs, life, every, recs, now):
    """Одиничні (зроблено / до виконання), системні (раз на N днів) і постекстрені дії ареалу."""
    A = ACTIONS[kind]
    one = list(A["one"]) + (["Встановити камеру"] if any(r.startswith("поставити камеру") for r in recs) else [])
    first = pd.Timestamp(min(cs.day))
    old_area = (now - first).days > 30 and not life.startswith("новий")    # давній ареал — одиничне вже зроблено
    one_time = [dict(action=x, status="done" if (old_area and x != "Встановити камеру") else "pending",
                     date=(first + pd.Timedelta(days=10)).strftime("%Y-%m-%d") if (old_area and x != "Встановити камеру") else "") for x in one]
    faded = life.startswith("згас")
    systemic = [dict(action="Контрольний обхід", every_days=30)] if faded else [dict(action=A["patrol"], every_days=every)]
    # системна перевірка кожного зробленого одиничного
    systemic += [dict(action=CHECK_FOR[o["action"]], every_days=30 if faded else 14) for o in one_time if o["status"] == "done" and o["action"] in CHECK_FOR]
    for x in systemic:
        x["rhythm"] = RHYTHM_UA[x["every_days"]]
    return dict(one_time=one_time, systemic=systemic, post=list(A["post"]),
                actions=[x["action"] for x in systemic])          # actions — короткий список для старих місць UI


def superareals(cl, now=NOW):
    """Стан суперареалів на дату now: кластери за WINDOW_DAYS до now (майбутнього не бачимо)."""
    win = cl[(cl.date <= now) & ((now - cl.date).dt.days <= WINDOW_DAYS)]
    out = []
    for kind, g in win.groupby("kind"):
        labels = dbscan(g.lat.values, g.lon.values, AREAL_EPS_KM[kind])
        for lab in np.unique(labels):
            cs = g[labels == lab]
            geom = current_extent(cs, now)
            c = geom.representative_point()
            days_ago = sorted({int((now.normalize() - pd.Timestamp(d)).days) for d in cs.day})
            reg, med = regularity(days_ago)
            per_source = {s: int(sum(s in ss for ss in cs.sources)) for s in SOURCES}
            life = lifecycle(days_ago, med)
            dyn, ch = dynamics(cs)
            p = float(np.mean(cs.confidence))
            # ритм дій: удвічі частіше за спостережену частоту подій; для згаслих — раз на місяць
            every = 30 if life.startswith("згас") else 14 if med is None else max((x for x in (1, 2, 3, 7, 14, 30) if x <= med / 2), default=1)
            level = "помірна" if life.startswith("згас") else \
                    "критична" if (med and med <= 8) or dyn == "розширюється" or life.startswith("відновився") else "висока"
            out.append(dict(
                kind=kind, type_name=TYPE_UA[kind], clusters=int(len(cs)), signals=int(sum(cs.n)), active_days=len(days_ago),
                first=min(cs.day), last=max(cs.day), regularity=reg, interval_days=None if med is None else round(med, 1),
                coverage=coverage(per_source), per_source=per_source, lifecycle=life, dynamics=dyn, dyn_change=ch,
                confidence=round(p, 2), confidence_level=conf_level(p), recommendations=recommendations(per_source, life, med),
                **area_actions(kind, cs, life, every, recommendations(per_source, life, med), now),
                team=TEAM, every_days=every, rhythm=RHYTHM_UA[every],
                level=level, simulated=bool(cs.simulated.all()), cluster_ids=list(cs.id),
                area_ha=round(geom.area * (111.32e3 ** 2) * np.cos(np.radians(LAT0)) / 1e4, 1),
                lat=round(c.y, 5), lon=round(c.x, 5), geometry=geom))
    lvl = {"критична": 0, "висока": 1, "помірна": 2}
    out.sort(key=lambda a: (lvl[a["level"]], -a["clusters"]))
    for i, a in enumerate(out):
        a["id"] = f"A-{i + 1:02d}"
    return out




def timeline(cl, final, step=2):
    """Знімки стану суперареалів кожні step днів у межах вікна — для повзунка часу.
    Id беремо з фінального ареалу, що містить ті самі кластери (щоб ареал можна було простежити в часі)."""
    owner = {c: a["id"] for a in final for c in a["cluster_ids"]}
    keep = ["kind", "type_name", "regularity", "coverage", "per_source", "lifecycle", "dynamics", "dyn_change", "clusters", "signals",
            "first", "last", "confidence_level", "recommendations", "actions", "one_time", "systemic", "post", "rhythm", "team", "area_ha", "level", "lat", "lon"]
    snaps = []
    for d in range(WINDOW_DAYS, -1, -step):
        now = NOW - pd.Timedelta(days=d)
        ars = superareals(cl, now)
        rows = []
        for a in ars:
            ids = [owner.get(c) for c in a["cluster_ids"]]
            rows.append({"id": max(set(ids), key=ids.count) if ids else a["id"], **{k: a[k] for k in keep},
                         "geometry": a["geometry"].simplify(0.0003).__geo_interface__})
        evs = []
        for t in response_tasks(cl, now):
            c = cl[cl.id == t["cluster"]].iloc[0]
            pts = np.array(c.pts)
            r = float(np.max(np.hypot((pts[:, 0] - t["lon"]) * 111320 * np.cos(np.radians(LAT0)), (pts[:, 1] - t["lat"]) * 110540))) if len(pts) > 1 else 0.0
            evs.append({k: t[k] for k in ("id", "kind", "title", "priority", "confidence_level", "sources", "devices", "detected", "signals", "lat", "lon")}
                       | {"r": round(r), "areal": owner.get(t["cluster"], "")})
        snaps.append(dict(date=now.strftime("%Y-%m-%d"), areals=rows, events=evs))
    return snaps


# текст нотифікації: що і де побачило джерело
WHAT = {
    "fire": {"camera": "Камера задетектила дим", "satellite": "Супутник: теплова аномалія", "trap": "Фотопастка: дим у кадрі",
             "social": "У соцмережах пишуть про дим / вогонь",
             "expert": ["Дзвінок на гарячу лінію: видно дим", "Експерт зафіксував осередок горіння"]},
    "logging": {"camera": "Камера задетектила техніку й людей з пилкою", "satellite": "Супутник: нова просіка / зміна покриву",
                "trap": "Фотопастка: люди з бензопилою", "social": "У соцмережах фото зрубаних дерев",
                "expert": ["Дзвінок на гарячу лінію: чути бензопилу", "Експерт зафіксував свіжі пні"]},
    "vehicle": {"camera": ["Камера задетектила авто на лісовому треку", "Камера задетектила людину на треку"],
                "trap": ["Фотопастка: авто на треку", "Фотопастка: людина на треку"],
                "satellite": "Супутник: авто поза дорогами", "social": "У соцмережах повідомлення про позашляховики",
                "expert": ["Дзвінок на гарячу лінію: позашляховик у лісі", "Експерт зафіксував сліди шин"]},
}


SAT_TEXT = {
    "logging": {"new": "Супутник: виявлено появу вирубки", "spot": "Супутник: нова ділянка вирубки поруч з відомою", "grow": "Супутник: виявлено розширення вирубки", "again": "Супутник: вирубку підтверджено повторно"},
    "fire": {"new": "Супутник: виявлено нову теплову аномалію", "spot": "Супутник: новий осередок поруч зі згарищем", "grow": "Супутник: виявлено розширення згарища", "again": "Супутник: теплова аномалія повторно"},
    "vehicle": {"new": "Супутник: авто поза дорогами в новому місці", "spot": "Супутник: авто поза дорогами в новому місці", "grow": "Супутник: авто поза дорогами в новому місці", "again": "Супутник: авто поза дорогами повторно"},
}


def satellite_events(items):
    """Що саме виявив супутник відносно попередніх знімків того самого ареалу:
    поява (раніше нічого), розширення (> 30% нового контуру поза вже відомою зоною), повтор."""
    from shapely.ops import unary_union
    sat = sorted([x for x in items if x["source"] == "супутник"], key=lambda x: x["ts"])
    known, last = {}, {}                               # ареал → відома зона (обʼєднання попередніх контурів) і дата останнього знімка
    for x in sat:
        g = Polygon(x["poly"]) if x["poly"] else Point(x["lon"], x["lat"]).buffer(0.002)
        gap = (pd.Timestamp(x["ts"]) - last[x["areal"]]).days if x["areal"] in last else None
        if gap is not None and gap > 30:               # довга тиша — нове спостереження вважаємо появою
            known.pop(x["areal"], None)
        prev = known.get(x["areal"])
        if prev is None:
            ev, add = "new", None                      # у цьому ареалі супутник ще нічого не бачив (або після паузи)
        elif not prev.intersects(g.buffer(0.001)):
            ev, add = "spot", None                     # ареал відомий, але окрема нова пляма
        else:
            new_part = g.difference(prev).area / g.area
            ev = "grow" if new_part > 0.3 and x["poly"] else "again"
            add = round(g.difference(prev).area * (111320 * np.cos(np.radians(LAT0))) * 110540 / 1e4, 1) if ev == "grow" else None
        known[x["areal"]] = g if prev is None else unary_union([prev, g])
        last[x["areal"]] = pd.Timestamp(x["ts"])
        x["sat_event"] = ev
        x["text"] = SAT_TEXT[x["kind"]][ev] + (f" (+{add} га)" if add else "") + (f" — після паузи {gap} дн." if ev == "new" and gap else "")
        if x["poly"]:
            x["area_ha"] = round(g.area * (111320 * np.cos(np.radians(LAT0))) * 110540 / 1e4, 1)


def feed(ev, cl):
    """Стрічка нотифікацій: кожен сигнал — текст, джерело, координати, кластер і ареал."""
    c_of = {i: c.id for c in cl.itertuples() for i in c.idx}
    a_of = dict(zip(cl.id, cl.areal))
    out = []
    for i, e in ev.sort_values("date", ascending=False).iterrows():
        w = WHAT[e.kind][e.source]
        txt = w[i % len(w)] if isinstance(w, list) else w
        if not e.simulated:
            txt += " (реальна детекція GroundingDINO)"
        out.append(dict(ts=e.date.strftime("%Y-%m-%d %H:%M"), kind=e.kind, source=SOURCE_UA[e.source], text=txt,
                        lat=round(e.lat, 5), lon=round(e.lon, 5), conf=float(e.conf), cluster=c_of.get(i, ""),
                        areal=a_of.get(c_of.get(i, ""), ""), simulated=bool(e.simulated), file=e.file or "", count=1,
                        poly=e.poly if isinstance(e.get("poly"), list) else None,
                        device=e.device if isinstance(e.get("device"), str) else "", live=e.source in LIVE_SOURCES,
                        scene=e.scene if isinstance(e.get("scene"), str) else ""))
    satellite_events(out)
    # однакові сигнали (той самий час, джерело, кластер — напр. кадри відео) згортаємо в одну нотифікацію
    merged = []
    for x in out:
        if merged and all(merged[-1][k] == x[k] for k in ("ts", "text", "cluster")):
            merged[-1]["count"] += 1
        else:
            merged.append(x)
    return merged




def main():
    ev = build()
    ev.assign(date=ev.date.astype(str)).to_csv(P / "signals.csv", index=False)
    ev.to_pickle(P / "signals.pkl")                   # потік як є (зі списками й типами): з нього генератор маршрутів і тасок (ds/ops_routes.py)
    cl = clusters(ev)
    areals = superareals(cl)
    cl["areal"] = cl.id.map({c: a["id"] for a in areals for c in a["cluster_ids"]}).fillna("")
    snaps = timeline(cl, areals)
    flat = lambda v: ", ".join(map(str, v)) if isinstance(v, list) else (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
    gpd.GeoDataFrame([{k: flat(v) for k, v in a.items()} for a in areals], crs=4326).to_file(P / "areals.geojson", driver="GeoJSON")
    json.dump(dict(snapshots=snaps, clusters=[dict(id=c.id, kind=c.kind, day=c.day, lat=round(c.lat, 5), lon=round(c.lon, 5), n=c.n,
                                                   sources=[SOURCE_UA[x] for x in c.sources], confidence=c.confidence, areal=c.areal)
                                              for c in cl.itertuples()]), open(P / "timeline.json", "w"), ensure_ascii=False, default=str)
    fd_ = feed(ev, cl)
    scenes = []
    for k, sc_ in SCENES.items():
        ar = [n["areal"] for n in fd_ if n.get("scene") == sc_["id"] and n["areal"]]
        a_id = max(set(ar), key=ar.count) if ar else ""
        last = max((n["ts"] for n in fd_ if n.get("scene") == sc_["id"]), default="")
        scenes.append({**sc_, "areal": a_id, "title": f"Ділянка {a_id} · {TYPE_UA[sc_['kind']].lower()}", "last": last})
    scenes.sort(key=lambda x: x["last"], reverse=True)
    json.dump(dict(sat_scenes=scenes, devices=[{k: v for k, v in d.items() if k != "scenario"} for d in DEVICES], feed=fd_,
                   areals=[{k: v for k, v in a.items() if k != "geometry"} for a in areals],
                   now=str(NOW), window_days=WINDOW_DAYS, contour_days=CONTOUR_DAYS, fresh_days=FRESH_DAYS,
                   clusters=int(len(cl)), signals=int(len(ev)), signals_real=int((~ev.simulated).sum())),
              open(P / "ops.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(len(ev), "сигналів,", int((~ev.simulated.astype(bool)).sum()), "реальних,", len(cl), "подій,", len(areals), "ареалів")
    print(f"парк {PK.KEY}: {P.relative_to(ROOT)}")
    if IS_MAIN:
        from eaglewings.ds import pflow, pages
        pflow.mark("report: generated signal stream (simulation)")
        pages.downloaded("generator")


if __name__ == "__main__":
    main()
    if "GR_PARK" not in os.environ:                   # без GR_PARK — усі парки дослідження: решта окремими процесами
        for k in PK.PARKS:
            if k != PK.MAIN:
                r = subprocess.run([sys.executable, __file__], env=dict(os.environ, GR_PARK=k))
                if r.returncode:
                    raise SystemExit(f"потік парку {k} впав (код {r.returncode})")
