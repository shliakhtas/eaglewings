"""Агент даних парків EagleWings: PandasAI 3 + LiteLLM над таблицями всіх парків, відповідь із таблицею й GeoJSON для мапи.

Таблиці парків мають колонку park. Питання задається про вибраний у застосунку парк: агент отримує рядки лише цього парку,
а всі парки — коли питання порівнює парки чи називає інший парк.

Таблиці з описами, відповідь → мапа, джерело числа на мапі; пісочниця з лімітами й без мережі та перевірка коду.
Без акаунтів, тарифів і обліку токенів: ключ OpenAI чи Anthropic приходить із запитом і живе лише в ньому.

    ask(question, key, recent=None, park=None) -> dict(type, text, table, rows, geojson, geo_source, chart, code, error, park)
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import pandasai as pai
import sqlglot
from pandasai.agent import Agent
from shapely import wkt
from shapely.geometry import Point, Polygon, mapping
from sqlglot import exp

from . import llm
from .sandbox import TimeoutSandbox

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import park as PK  # noqa: E402
from eaglewings.ds.regions import COUNTRY, LAT0, REGION  # noqa: E402
import math  # noqa: E402
UTM = REGION["fetch"]["utm"]                                      # метрична проєкція території (кейс Фракії — UTM 35N)

P = ROOT / "data" / "processed"                                   # спільне: результати задач, детекції, research_results
PARKS = {k: PK.ALL[k] for k in PK.PARKS}
MAX_ROWS = 5000                                                   # скільки об'єктів відповіді йде на мапу
EXPORTS = ROOT / "logs" / "assistant"                              # графіки PandasAI; одразу стають data URI й видаляються
os.environ.setdefault("PANDASAI_LOG_LEVEL", "WARNING")


def named(df, name, description, columns=None):
    """pandasai DataFrame з назвою й описом таблиці та колонок: так модель пише кращий код."""
    out = pai.DataFrame(df, _table_name=name)
    out.schema.description = description
    for col in out.schema.columns:
        if columns and col.name in columns:
            col.description = columns[col.name]
    return out


def _ring(poly):
    return Polygon([(b, a) for a, b in poly]).wkt                  # [[lat, lon], …] → WKT (lon lat)


HANSEN = {k: json.load(open(PK.processed(k) / "hansen_meta.json"))["transform"] for k in PARKS}   # сітка Hansen 0,00025° кожного парку
PIX_HA = round(abs(HANSEN[PK.MAIN][0] * HANSEN[PK.MAIN][4]) * 111320 ** 2 * round(math.cos(math.radians(LAT0)), 4) / 1e4, 4)   # площа пікселя на широті території (41,1° → 0,058 га)


def dataset(name):
    """Поточна версія датасету проєкту (data/datasets/<назва>/vN/data.csv.gz); датасету вимкненої гілки немає — порожня таблиця."""
    if not list((ROOT / "data" / "datasets" / name).glob("v*/data.csv.gz")):
        return pd.DataFrame()
    return pd.read_csv(sorted((ROOT / "data" / "datasets" / name).glob("v*/data.csv.gz"), key=lambda f: int(f.parent.name[1:]))[-1])


def with_latlon(d):
    """Пікселі датасетів (row, col сітки Hansen свого парку) → центр пікселя lat, lon для мапи."""
    keys = d["park"] if "park" in d else pd.Series(PK.MAIN, index=d.index)
    T = pd.DataFrame([HANSEN[k] for k in keys], index=d.index)
    return d.assign(lat=(T[5] + (d["row"] + .5) * T[4]).round(6), lon=(T[2] + (d["col"] + .5) * T[0]).round(6))


def first(df, key):
    """Колонка park першою."""
    return df.assign(park=key)[["park"] + [c for c in df.columns if c != "park"]]


def at_main(df):
    """Датасет, прив'язаний до місць основного парку: колонка park зі значенням основного."""
    return df if "park" in df else first(df, PK.MAIN)


def route_frames(P):
    """route_stops — зупинки маршрутів груп по днях (минулі дні виконані, останній — план на завтра); tasks — усі задачі розділу «Таски»."""
    R = json.load(open(P / "routes.json"))
    rows = []
    for d in R["timeline"] + [R["today"]]:
        for tm in d["teams"]:
            for i, x in enumerate(tm["stops"], 1):
                rows.append(dict(date=d["date"], team=tm["id"], team_name=tm["name"], order=i, eta=x["eta"], type=x["type"], kind=x.get("kind", ""),
                                 title=x["title"], detail=x.get("detail", ""), areal=x.get("areal", ""), status=x["status"], leg_km=x["leg_km"],
                                 rerouted=bool(tm.get("rerouted")), plan_day=not d["past"], lat=x["lat"], lon=x["lon"], simulated=True))
    stops = pd.DataFrame(rows, columns=["date", "team", "team_name", "order", "eta", "type", "kind", "title", "detail", "areal", "status", "leg_km",
                                        "rerouted", "plan_day", "lat", "lon", "simulated"])
    T = json.load(open(P / "tasks.json"))["tasks"]
    tasks = pd.DataFrame(T, columns=["id", "date", "time", "type", "type_ua", "title", "detail", "areal", "team", "team_name", "status", "lat", "lon", "kind"]).assign(simulated=True)
    return dict(route_stops=stops, tasks=tasks)


def park_frames(key):
    """Таблиці одного парку з тих самих файлів, що й застосунок (крок mocks), і результатів задач."""
    P = PK.processed(key)
    ops = json.load(open(P / "ops.json"))
    tl = json.load(open(P / "timeline.json"))
    ar = gpd.read_file(P / "areals.geojson")
    areals = pd.DataFrame(dict(
        id=ar.id, type=ar.kind, type_name=ar.type_name, status=ar.lifecycle.str.split().str[0], lifecycle=ar.lifecycle,
        regularity=ar.regularity, coverage=ar.coverage, dynamics=ar.dynamics, dyn_change_pct=ar.dyn_change, clusters=ar.clusters,
        signals=ar.signals, first_seen=ar["first"].astype(str), last_seen=ar["last"].astype(str), area_ha=ar.area_ha, level=ar.level,
        lat=ar.lat, lon=ar.lon, recommendations=ar.recommendations.fillna(""), simulated=True, geometry_wkt=ar.geometry.to_wkt()))
    feed = pd.DataFrame(ops["feed"])
    signals = feed[["ts", "kind", "source", "text", "lat", "lon", "areal", "device", "simulated", "live", "count"]].copy()
    events = pd.DataFrame(tl["clusters"])
    events["sources"] = events.sources.apply(", ".join)
    fi = gpd.read_file(P / "firms.geojson")
    fires = pd.DataFrame(dict(date=fi.date.astype(str).str[:10], time=fi.acq_time.astype(str).str.zfill(4), sensor=fi.source, confidence=fi.confidence.astype(str),
                              frp=fi.frp, daynight=fi.daynight, lat=fi.latitude.round(5), lon=fi.longitude.round(5)))
    cells = dataset("h3_risk_cells")
    cells = cells[cells.park == key] if "park" in cells else cells
    cells = cells[["h3", "elev", "slope", "treecover", "d_road", "d_village", "y_burn_2022", "y_burn_2023", "y_loss_2023_hansen"]]
    if (P / "risk_h3_2026.geojson").exists():
        rk = gpd.read_file(P / "risk_h3_2026.geojson")
        cen = rk.geometry.representative_point()                     # точка всередині гекса (центроїд у градусах дає попередження)
        risk = pd.DataFrame(dict(h3=rk.h3, level=rk.level, risk_2026=rk.risk_2026.round(4), risk_rank=rk.risk_rank.round(3),
                                 susceptibility=rk.susceptibility.round(4), lat=cen.y.round(5), lon=cen.x.round(5), geometry_wkt=rk.geometry.to_wkt())).merge(cells, on="h3", how="left")
    else:                                                            # своєї моделі ризику немає (FG_risk лише перевіряє готову): гекси без прогнозу
        import h3 as _h3
        ll = [(_h3.cell_to_latlng(h)) for h in cells.h3]
        risk = cells.assign(level="", risk_2026=float("nan"), risk_rank=float("nan"), susceptibility=float("nan"),
                            lat=[round(a, 5) for a, _ in ll], lon=[round(b, 5) for _, b in ll],
                            geometry_wkt=[Polygon([(b, a) for a, b in _h3.cell_to_boundary(h)]).wkt for h in cells.h3])
    cand = []
    f = ROOT / "data" / "processed" / "results" / "logging_ld.json"
    if f.exists():
        j = json.load(open(f))
        d = (j.get("app") if key == PK.MAIN else (j.get("parks") or {}).get(key)) or {}
        for i, c in enumerate(d.get("cand", []), 1):
            cand.append(dict(id=f"L-{i}", year=d.get("year"), area_ha=c["ha"], opera_confirms=bool(c.get("opera")), lat=c["lat"], lon=c["lon"], geometry_wkt=_ring(c["poly"])))
    logging = pd.DataFrame(cand, columns=["id", "year", "area_ha", "opera_confirms", "lat", "lon", "geometry_wkt"])

    # дороги OSM і відстань від кожної теплової точки до найближчої (метри, UTM 35N)
    rd = gpd.read_file(P / "roads.geojson")
    rdm = rd.to_crs(UTM)
    roads = pd.DataFrame(dict(id=[f"R-{i}" for i in range(len(rd))], highway=rd.highway, length_km=(rdm.length / 1000).round(3),
                              lat=rd.geometry.representative_point().y.round(5), lon=rd.geometry.representative_point().x.round(5), geometry_wkt=rd.geometry.to_wkt()))
    pts = gpd.GeoSeries(gpd.points_from_xy(fires.lon, fires.lat), crs=4326).to_crs(UTM)
    fires["d_road_m"] = pts.distance(rdm.geometry.union_all()).round(0).values

    park = gpd.read_file(P / "park.geojson")
    park_t = pd.DataFrame(dict(id=["PARK"], name=park.name, area_km2=(park.to_crs(UTM).area / 1e6).round(1), lat=park.geometry.representative_point().y.round(5),
                               lon=park.geometry.representative_point().x.round(5), geometry_wkt=park.geometry.to_wkt()))

    # втрата лісу Hansen: піксель ~30 м з роком втрати
    hl = json.load(open(P / "hansen_loss.geojson"))["features"]
    forest_loss = pd.DataFrame(dict(year=[f["properties"]["year"] for f in hl], treecover2000=[f["properties"]["treecover2000"] for f in hl],
                                    lat=[round(f["geometry"]["coordinates"][1], 6) for f in hl], lon=[round(f["geometry"]["coordinates"][0], 6) for f in hl]))
    forest_loss["area_ha"] = PIX_HA

    frames = dict(areals=areals, signals=signals, events=events, fires=fires, risk_cells=risk, logging_candidates=logging,
                  forest_loss=forest_loss, roads=roads, park=park_t)
    if (P / "routes.json").exists():                                 # маршрути груп і таски (гілка RD, симуляція): є в парку, де бази груп
        frames |= route_frames(P)
    return {n: first(f, key) for n, f in frames.items()}, ops["now"]


def load_frames():
    """Таблиці всіх парків (колонка park) і спільні: результати дослідження, детекції, датасети моделей."""
    per = {k: park_frames(k) for k in PARKS}
    names = list(dict.fromkeys(n for k in PARKS for n in per[k][0]))
    frames = {n: pd.concat([per[k][0][n] for k in PARKS if n in per[k][0]], ignore_index=True) for n in names}
    weather = dataset("fire_day_daily")
    research = pd.read_csv(P / "research_results.csv") if (P / "research_results.csv").exists() else pd.DataFrame(
        columns=["branch", "question", "kind", "group", "algorithm", "metric", "lower_is_better", "how_chosen", "val", "test", "ci_lo", "ci_hi", "winner", "hypothesis"])
    research["group"] = research["group"].fillna("")

    det = gpd.read_file(P / "detections.geojson")
    detections = pd.DataFrame(det.drop(columns="geometry")).assign(date=lambda d: d["date"].astype(str))

    hex_features = dataset("h3_risk_cells")
    hex_logging = dataset("logging_gr_cells")
    burned_pixels = with_latlon(dataset("burned_area_pixels"))
    logging_pixels = with_latlon(dataset("logging_gr_pixels"))
    opera = dataset("opera_gr_snapshots")
    bench = pd.concat([dataset("cv_benchmark_aerial").assign(view="згори (супутник, дрон)"), dataset("cv_benchmark_ground").assign(view="з землі (камера, фотопастка)")], ignore_index=True)
    frames |= dict(weather_days=weather, research_results=research, detections=detections,
                   hex_features=hex_features, hex_logging_years=hex_logging, burned_area_pixels=burned_pixels, logging_pixels=logging_pixels,
                   opera_snapshots=opera, benchmark_images=bench,
                   # датасети гілок, яких раніше не було в асистента
                   fire_spread=dataset("fire_spread_cells"), fires_greece=dataset("ar_fires"), firms_vs_effis=dataset("matching_firms_effis"),
                   tower_sites=dataset("tp_candidates"), tower_fires=dataset("tp_fires"), road_events=at_main(dataset("cp_events")),
                   camera_zone=at_main(dataset("cv_camera_zone")), signals_benchmark=at_main(dataset("xm_signals")),
                   firms_effis_parks=dataset("fd_signal"), logging_events=dataset("ar_logging"), track_segments=dataset("track_segments"),
                   route_days=dataset("route_instances"))
    frames = {n: f for n, f in frames.items() if len(f) or n in CORE_TABLES}       # таблиць вимкнених гілок немає
    return frames, per[PK.MAIN][1]


CORE_TABLES = {"areals", "signals", "events", "fires", "risk_cells", "logging_candidates", "forest_loss", "roads", "park", "research_results", "detections"}


FRAMES, NOW = load_frames()
# контур за (парк, id): ареал A-01 є в кожному парку
GEOM = {(k, i): g for t in ("areals", "logging_candidates", "roads", "park") for k, i, g in zip(FRAMES[t].park, FRAMES[t].id, FRAMES[t].geometry_wkt)}
HEX = dict(zip(FRAMES["risk_cells"].h3, FRAMES["risk_cells"].geometry_wkt))
# Контури лишаються на сервері (GEOM за id, HEX за h3): модель їх не бачить, бо WKT у прикладах рядків роздуває промпт
# у кожному виклику. Відповідь з id чи h3 сервер сам перетворює на контури на мапі (to_geojson).
FRAMES = {n: f.drop(columns=[c for c in ("geometry_wkt",) if c in f.columns]) for n, f in FRAMES.items()}

PARK_COL = {"park": "ключ парку: " + ", ".join(f"{k} — {v['name']}" for k, v in PARKS.items())}
DESCRIPTIONS = {
    "areals": ("Ареали парку за 90 днів демо-потоку: зони повторюваних подій одного типу (type: fire/logging/vehicle). Сценарні дані на реальних місцях. "
               "Контур ареалу сервер показує на мапі за id.",
               {"status": "коротко: активний / новий / відновився / згас — фільтруй за цією колонкою", "lifecycle": "детально, напр. «згас (28 дн. тому)»",
                "dynamics": "розширюється / звужується / стабільний", "dyn_change_pct": "зміна активності, %", "area_ha": "площа, га", "level": "рівень уваги"}),
    "signals": ("Окремі сигнали демо-потоку за 90 днів: камера, фотопастка, супутник, соцмережі, гаряча лінія / експерт. kind: fire/logging/vehicle. "
                "live=True — живі джерела. ts — час 'YYYY-MM-DD HH:MM'. Більшість сигналів сценарні (simulated=True).",
                {"areal": "id ареалу сигналу", "simulated": "True — сценарні дані", "count": "скільки об'єктів у кадрі чи повідомленні, НЕ кількість сигналів"}),
    "events": ("Події: сигнали одного типу за день, зведені в одну подію (DBSCAN).", {"day": "дата YYYY-MM-DD", "n": "кількість сигналів", "confidence": "впевненість 0..1", "areal": "id ареалу"}),
    "fires": ("Справжні теплові точки NASA FIRMS (VIIRS і MODIS) у парку й 3 км довкола, 2021–2025. Одна точка — один гарячий піксель на прольоті супутника.",
              {"date": "дата YYYY-MM-DD", "time": "час HHMM UTC", "sensor": "VIIRS чи MODIS", "frp": "потужність вогню, МВт", "daynight": "D — день, N — ніч", "d_road_m": "до найближчої дороги, м"}),
    "risk_cells": ("Гекси парку (H3 res 9, ~0,1 км²): ризик пожежі на 2026 рік і що горіло раніше. Контур гекса сервер показує за h3.",
                   {"level": "фонова / помірна / висока / критична", "risk_2026": "ризик 0..1", "risk_rank": "частка гексів з меншим ризиком",
                    "y_burn_2023": "1 — згоріло у 2023 за FIRMS", "y_burn_2022": "1 — згоріло у 2022", "y_loss_2023_hansen": "1 — втрата лісу 2023 за Hansen",
                    "elev": "висота, м", "slope": "ухил, °", "treecover": "лісистість, %", "d_road": "до дороги, м", "d_village": "до села, м"}),
    "logging_candidates": ("Ділянки-кандидати на рубку: супутник Sentinel-2 бачить падіння вологості крон поза згарищами. Це кандидати для перевірки, а не доведені рубки.",
                           {"area_ha": "площа, га", "opera_confirms": "True — NASA OPERA теж бачить порушення поруч"}),
}
DESCRIPTIONS |= {
    "forest_loss": ("Втрата лісу Hansen Global Forest Change у парку й 3 км довкола: кожен рядок — піксель ~30 м, де ліс зник у році year (2001–2024). "
                    "Будь-яка втрата крон: пожежа, рубка, буревій. Площа — сума area_ha.", {"year": "рік втрати", "treecover2000": "лісистість пікселя у 2000 р., %", "area_ha": "площа пікселя, га"}),
    "weather_days": ("Кожен день 2021–2025 у кожному парку: погода NASA POWER у точці парку, індекси пожежної небезпеки й чи була пожежа за FIRMS.",
                     {"date": "дата YYYY-MM-DD", "T2M_MAX": "максимальна температура, °C", "T2M": "середня температура, °C", "RH2M": "вологість повітря, %",
                      "WS10M": "вітер, м/с", "PRECTOTCORR": "опади, мм/добу", "GWETTOP": "вологість верхнього шару ґрунту 0..1", "VPD": "дефіцит пружності пари, кПа (сухість повітря)",
                      "dry_days": "днів поспіль без опадів", "prec_30": "опади за 30 днів, мм", "VPD_7": "VPD за 7 днів", "T_7": "температура за 7 днів",
                      "nesterov": "індекс Нестерова", "fwi": "наближений індекс FWI", "fire_day": "1 — у парку була пожежа (FIRMS)", "ignition": "1 — нове займання цього дня",
                      "fire_day_yesterday": "1 — пожежа була вчора"}),
    "research_results": ("Результати дослідження: кожен рядок — алгоритм однієї гілки (питання дослідження), метрика на валідації (val, нею обирали переможця) і на тесті (test) з 95 % CI. "
                         "winner=True — переможець гілки, обраний лише на валідації. Тут лише пораховані гілки.",
                         {"branch": "код гілки (FD — згарище, FG_day — день пожежі, FG_risk — ризик ділянок, LD — рубки на знімках, LM — збіг із OPERA, LG — прогноз рубок…)",
                          "question": "питання гілки", "kind": "виміряно на реальних даних / сценарний аналіз / симуляція", "group": "підгрупа, напр. лейбл fire_day чи ignition",
                          "metric": "назва метрики", "lower_is_better": "True — менше краще", "val": "метрика на валідації", "test": "метрика на тесті", "winner": "переможець гілки"}),
    "roads": ("Дороги OSM у парку й довкола; лінію сервер показує за id. У fires є d_road_m — відстань точки до найближчої дороги.", {"highway": "тип дороги OSM", "length_km": "довжина, км"}),
    "park": ("Межа національного парку (один рядок, id PARK).", {"area_km2": "площа, км²"}),
    "detections": ("Об'єкти, які детектори (YOLO, GroundingDINO) знайшли на фото й відео демо з Дадьї та інших місць. Спільна для всіх парків.",
                   {"kind": "тип об'єкта", "conf": "впевненість детектора 0..1", "source": "детектор і тип кадру", "simulated": "True — сценарні"}),
    "hex_features": ("Повний датасет гексів H3 res 9 парку для прогнозу ризику: рельєф, паливо, люди, історія пожеж і втрат. "
                     "Префікси f22_/f23_/f26_ — ознаки станом на 1 січня 2022/2023/2026; y_burn_РІК — згоріло в році (FIRMS).",
                     {"track_km_km2": "густота доріг і стежок, км/км²", "d_boundary": "до межі парку, м", "loss_hist": "частка втрат лісу в минулому", "firms_2km": "теплових точок у 2 км",
                      "y_final": "лейбл ризику для моделі"}),
    "hex_logging_years": ("Гекси H3 res 9 × рік (2017–2024) для прогнозу рубок: історія втрат лісу поза згарищами. y=1 — у гексі ≥0,5 га нової втрати в цьому році.",
                          {"ha": "нова втрата лісу в році, га", "loss1": "втрата минулого року, га", "loss_cum": "накопичена втрата, га", "nb_loss1": "втрата сусідів минулого року, га"}),
    "burned_area_pixels": ("Пікселі лісу парку ~30 м для задачі згарища 2023: NDVI MODIS до й після пожежі, лейбл втрати Hansen. lat, lon — центр пікселя.",
                           {"split": "calib — калібрування, test — тест", "y": "1 — втрата лісу 2023–24", "pix_ha": "площа, га", "ndvi_pre": "NDVI до", "ndvi_post": "NDVI після"}),
    "logging_pixels": ("Пікселі лісу ~30 м × рік для детекції рубок Sentinel-2 без згарищ: смуги й індекси до/після, лейбл Hansen. lat, lon — центр пікселя.",
                       {"y": "1 — втрата лісу (Hansen) у цьому році", "d_ndmi": "падіння вологості крон NDMI", "d_nbr": "падіння NBR", "d_ndvi": "падіння NDVI", "pix_ha": "площа, га"}),
    "opera_snapshots": ("NASA OPERA DIST-ALERT над парком: на кожну дату скільки пікселів (30 м) у кожному статусі порушення рослинності.",
                        {"s0": "без порушення", "s1": "перше виявлення, зниження <50 %", "s2": "попереднє, <50 %", "s3": "підтверджене, <50 %",
                         "s4": "перше виявлення, ≥50 %", "s5": "попереднє, ≥50 %", "s6": "підтверджене, ≥50 %", "s7": "завершене, <50 %", "s8": "завершене, ≥50 %", "s255": "немає даних"}),
    "benchmark_images": ("Кадри незалежних еталонів, на яких міряли детектори: VEDAI (машини згори), D-Fire (дим і вогонь), ENA24 (фотопастки). Не з парку.",
                         {"dataset": "еталон і роздільність", "n_boxes": "скільки розмічених об'єктів у кадрі", "view": "згори чи з землі"}),
    "fire_spread": ("Поширення пожеж Греції 2021–2025: гекси H3 res 8 довкола активної пожежі в момент t з ознаками фронту, вітру, рельєфу й палива. "
                    "y24 / y48 — гекс згорить до t+24 / t+48 год (FIRMS VIIRS). Уся Греція, без колонки park.",
                    {"fire": "назва пожежі", "t": "момент", "d_front": "до фронту, м", "wind_ms": "вітер, м/с", "wind_along": "вітер уздовж напрямку на гекс",
                     "slope_up": "ухил угору від фронту", "treecover": "лісистість, %", "y24": "1 — згорить до t+24 год", "y48": "1 — до t+48 год"}),
    "fires_greece": ("Пожежі Греції 2021–2025 за FIRMS, зведені з точок правилом переможця FM (2 км, 5 діб). Уся Греція, без колонки park.",
                     {"n": "скільки теплових точок у пожежі", "h": "тривалість, год", "block": "просторовий блок для перевірки"}),
    "firms_vs_effis": ("Теплові точки FIRMS Греції 2021–2025 з позначкою, чи лежать у згарищі EFFIS: на цьому перевіряли, які точки це одна пожежа (FM). Уся Греція.",
                       {"dt": "час", "sensor": "VIIRS чи MODIS", "conf": "впевненість FIRMS", "effis": "id згарища EFFIS", "in_effis": "1 — точка всередині згарища EFFIS"}),
    "tower_sites": ("Місця-кандидати для веж спостереження в кожному парку: найвищі гекси на відстані від 1 км і скільки гексів видно з кожного (рельєф SRTM). Сценарний аналіз.",
                    {"elev": "висота, м", "visible": "скільки гексів видно з місця"}),
    "tower_fires": ("Пожежі кожного парку 2021–2025 з перших точок FIRMS: на них перевіряли, чи бачили б їх вежі.",
                    {"ev": "id пожежі", "n": "скільки точок", "start": "початок"}),
    "road_events": ("Нові втрати лісу поза згарищами й пожежі основного парку (Дадья): місця, до яких перевіряли під'їзд дорогами для камер. Сценарний аналіз.",
                    {"kind": "рубка чи пожежа", "km": "довжина під'їзду, км"}),
    "camera_zone": ("Гекси основного парку (Дадья) з точками FIRMS 2022 і 2023: справжні місця для перевірки зони камери. Азимут на камері симульовано.",
                    {"part": "калібрування чи тест"}),
    "signals_benchmark": ("Потік сигналів основного парку (Дадья) з еталоном для зведення сигналів (XM): сценарій і день генератора. Симуляція, крім 12 детекцій диму.",
                          {"simulated": "True — сценарні", "scenario": "номер сценарію", "truth": "еталонна подія", "part": "калібрування чи тест"}),
}
DESCRIPTIONS |= {
    "route_stops": ("Зупинки маршрутів двох груп лісників по днях (гілка RD, симуляція на сценарному потоці): минулі дні виконані, plan_day=True — план на завтра. "
                    "type: event — реагування на подію, night — нічний сигнал на ранок, plan — дії ареалу, patrol — патруль зони ризику рубок.",
                    {"date": "дата YYYY-MM-DD", "team": "група", "order": "номер зупинки в маршруті", "eta": "час прибуття HH:MM", "status": "done — пройдено, planned — план, new — нова подія",
                     "leg_km": "км від попередньої зупинки", "rerouted": "True — маршрут групи перебудовано через нову подію", "areal": "id ареалу", "simulated": "True — симуляція"}),
    "tasks": ("Усі задачі розділу «Таски» (симуляція): сьогоднішні виїзди з маршруту, дії ареалів за ритмом на 14 днів, одиничні дії, нічні алерти, передачі партнерам, патрулі ризику.",
              {"type": "event, plan, night, patrol, one, system, night_alert, handoff", "type_ua": "тип українською", "team": "група чи хто виконує", "status": "new — нова, done — виконано",
               "areal": "id ареалу", "date": "дата YYYY-MM-DD", "time": "час HH:MM"}),
    "firms_effis_parks": ("Теплові точки FIRMS у парках із буфером і згарище EFFIS, якому вони належать (−1 — жодного): чи бачать супутники пожежі на території (FD).",
                          {"effis": "id згарища EFFIS або −1", "in_effis": "1 — точка в згарищі", "forest": "1 — точка в лісі (Hansen ≥ 30 %)", "month": "місяць"}),
    "logging_events": ("Рубки як події: гекс H3 res 9 × рік із ≥ 0,5 га нової втрати лісу Hansen поза згарищами (AR, ареали рубок). Hansen — будь-яка втрата крон, не доведена рубка.",
                       {"year": "рік", "ha": "втрата, га", "h": "гекс res 7", "park": "ключ парку"}),
    "track_segments": ("Відрізки 100 м лісових колій OSM (y=1) і ті самі лінії, зсунуті на 150 м у ліс (y=0), з ознаками Sentinel-2 літа 2024 (SD, колії).",
                       {"y": "1 — колія OSM, 0 — зсунутий контроль", "park": "ключ парку", "vis_mean": "яскравість уздовж відрізка", "side_dndvi": "наскільки лінія темніша за NDVI від смуг поруч"}),
    "route_days": ("Сирі задачі маршрутизації по днях до побудови маршруту (RD, симуляція): обов'язкові зупинки й нова подія під час зміни.",
                   {"new": "1 — нова подія під час зміни", "at_h": "коли подія стала відома, год", "prio": "пріоритет (0 — найвищий)"}),
}
if COUNTRY != "Greece":                                # тексти таблиць, написані для кейсу Фракії, — словами цієї території
    _main = PK.ALL[PK.MAIN]["short"]
    DESCRIPTIONS = {n: (d.replace("Уся Греція", "Уся рамка території").replace("Греції", "території").replace("(Дадья)", f"({_main})"), c) for n, (d, c) in DESCRIPTIONS.items()}
DESCRIPTIONS = {n: v for n, v in DESCRIPTIONS.items() if n in FRAMES}
for n, (d, c) in list(DESCRIPTIONS.items()):           # колонка park у таблицях парків
    if "park" in FRAMES[n].columns:
        DESCRIPTIONS[n] = (d + " Колонка park — ключ парку.", {**c, **PARK_COL})
assert set(DESCRIPTIONS) == set(FRAMES), set(FRAMES) ^ set(DESCRIPTIONS)
CORE = ["areals", "signals", "events", "fires", "risk_cells", "logging_candidates"]   # якщо вибір таблиць не вдався
CATALOG = "\n".join(f"- {n} ({len(FRAMES[n])} рядків): {d.split('. ')[0]}" for n, (d, _) in DESCRIPTIONS.items())
ROUTE = """Обери таблиці, потрібні, щоб відповісти на питання про національний парк. Таблиці:
{catalog}
{context}
Питання: {question}
Підказки:
- «ареал», «зона», «розширюється», «згас», «активний» → areals (ареали є для пожеж, рубок і транспорту); «сигнал», «фотопастка», «камера»,
  «гаряча лінія», «соцмережі» → signals; «подія» → events. Це оперативна картина демо.
- «теплові точки», «FIRMS», «пожежі» за датами → fires; «ризик», «гекси» → risk_cells; «кандидати на рубку» → logging_candidates.
- «втрата лісу», «Hansen», «гектари лісу» → forest_loss; «погода», «спека», «опади», «день пожежі» → weather_days;
  «алгоритм», «метрика», «переможець», «точність моделі» → research_results; «дорога» → roads (у fires вже є d_road_m); «межа парку» → park.
- Датасети моделей (hex_features, hex_logging_years, burned_area_pixels, logging_pixels, opera_snapshots, benchmark_images) бери,
  лише коли питають про ознаки, лейбли, пікселі, OPERA чи еталони детекторів.
- «поширення», «фронт», «куди піде пожежа» → fire_spread; «пожежі Греції», «повторюються» → fires_greece; «EFFIS» → firms_vs_effis;
  «вежі» → tower_sites, tower_fires; «камери на дорогах», «під'їзд» → road_events; «зона камери» → camera_zone; «зведення сигналів» → signals_benchmark.
- «маршрут», «група», «лісники», «зупинки», «перебудова», «куди їхати» → route_stops; «таски», «задачі», «статус», «нічні алерти», «партнерам» → tasks
  (обидві — симуляція); «колії», «джипи», «Sentinel-2» → track_segments; «чи бачать супутники пожежі», «EFFIS у парках» → firms_effis_parks;
  «рубки повторюються», «ареали рубок» → logging_events.
Відповідь — лише JSON-список назв таблиць (1–5), напр. ["fires", "roads"]."""


def route(question, model_name, key, recent=None):
    """Які таблиці дати PandasAI: короткий виклик тієї ж моделі з каталогом (назва + перше речення опису).
    Усі 18 описів зі зразками рядків це ~40 тис. символів у кожному виклику: дорого й упирається в ліміти ключа."""
    import litellm
    try:
        r = litellm.completion(model=model_name, api_key=key, temperature=0, max_tokens=80, timeout=20,
                               messages=[{"role": "user", "content": ROUTE.format(catalog=CATALOG, context=context_block(recent), question=question)}])
        names = json.loads(re.search(r"\[.*?\]", r.choices[0].message.content, re.S).group(0))
        names = [n for n in dict.fromkeys(names) if n in DESCRIPTIONS][:5]
        return names or CORE
    except Exception:  # noqa: BLE001 — помилку ключа покаже основний виклик
        return CORE

PROMPT = """Ти аналітик даних національних парків. Зараз вибраний парк: {park_name} (park = '{park}'). {scope}
Умовна дата демо-потоку: {now}. {lang_rule} Таблиці: {tables}.
Правила:
- Якщо питання про місця, зони, «де», «покажи», «на мапі» — поверни DataFrame з колонками lat і lon;
  для ареалів, кандидатів на рубку, доріг, межі парку й гексів обов'язково додай id (або h3): за ними сервер покаже контури. Додай кілька коротких пояснювальних колонок.
- Якщо питання про кількість — поверни число; про опис ситуації — короткий текст (string).
- Не вигадуй даних: усе бери з таблиць. Ареали, сигнали й події здебільшого сценарні; теплові точки, втрата лісу, погода, гекси й пікселі справжні.
- Про якість алгоритмів і переможців відповідай з research_results: переможця обирали на валідації (val), тест (test) лише показує результат.
- Великі таблиці (пікселі, втрата лісу) агрегуй у SQL, коли питають про кількість чи суму.
- Коли питають «покажи» чи «де», поверни всі рядки, що підходять: не додавай LIMIT і не обрізай, мапу обмежить сервер.
{context}
Питання: {question}"""


def context_block(recent):
    if not recent:
        return ""
    return "Попередні повідомлення розмови:\n" + "\n".join(f"{'Користувач' if m.get('role') == 'user' else 'Агент'}: {str(m.get('text', ''))[:300]}" for m in recent[-6:])


def _plain(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        return v.item()
    except AttributeError:
        return str(v)


def sample_idx(n):
    """Індекси рядків для мапи: усі, якщо їх не більше MAX_ROWS, інакше MAX_ROWS рівномірно по всій таблиці."""
    if n <= MAX_ROWS:
        return range(n)
    return sorted({int(i * n / MAX_ROWS) for i in range(MAX_ROWS)})


def to_geojson(frame, park=PK.MAIN):
    """Відповідь-таблиця → GeoJSON: geometry_wkt; або id ареалу / кандидата (контур свого парку); або h3 гекса; або lat/lon як точки."""
    f = frame.copy()
    if "geometry_wkt" not in f.columns:
        keys = list(zip(f["park"] if "park" in f.columns else [park] * len(f), f["id"].astype(str))) if "id" in f.columns else []
        if keys and any(k in GEOM for k in keys):
            f["geometry_wkt"] = [GEOM.get(k) for k in keys]
        elif "h3" in f.columns and f["h3"].astype(str).isin(HEX).any():
            f["geometry_wkt"] = f["h3"].astype(str).map(HEX)
    geoms = None
    if "geometry_wkt" in f.columns:
        geoms = [wkt.loads(s) if isinstance(s, str) and s else None for s in f["geometry_wkt"]]
        if not any(geoms) and {"lat", "lon"} <= set(f.columns):
            geoms = None
    if geoms is None and {"lat", "lon"} <= set(f.columns):
        geoms = [Point(lo, la) if pd.notna(la) and pd.notna(lo) else None for la, lo in zip(f["lat"], f["lon"])]
    if geoms is None:
        return None
    props = f.drop(columns=[c for c in ("geometry_wkt",) if c in f.columns]).reset_index(drop=True)
    keep = sample_idx(len(props))                                  # більше MAX_ROWS: рівномірна вибірка, а не перші рядки
    feats = [dict(type="Feature", geometry=mapping(geoms[i]), properties={k: _plain(v) for k, v in props.iloc[i].items()})
             for i in keep if geoms[i] is not None]
    return dict(type="FeatureCollection", features=feats) if feats else None


SQL_RE = re.compile(r'("""|\'\'\'|"|\')(\s*(?:SELECT|WITH)\b.*?)\1', re.S | re.I)


def duck(frames=None):
    con = duckdb.connect()
    for n, f in (frames or FRAMES).items():
        con.register(n, f)
    return con


def source_geojson(code, frames=None, park=PK.MAIN):
    """Відповідь-число чи текст без координат: беремо SQL із коду моделі, прибираємо агрегацію й виконуємо.
    Це рядки, з яких пораховано відповідь (напр. «4 сигнали фотопасток» → ці 4 сигнали на мапі)."""
    if not code:
        return None
    frames = frames or FRAMES
    for m in SQL_RE.finditer(code):
        try:
            tree = sqlglot.parse_one(m.group(2), read="duckdb")
        except Exception:  # noqa: BLE001 — f-рядки й шматки SQL, що не парсяться
            continue
        for sel in tree.find_all(exp.Select):
            src = sel.args.get("from") or sel.args.get("from_")
            table = src and src.find(exp.Table)
            if not table or table.name not in frames:
                continue
            q = sel.copy()
            for k in ("group", "having", "order", "limit", "offset", "distinct", "qualify"):
                q.set(k, None)
            q.set("expressions", [exp.Column(this=exp.Star(), table=exp.to_identifier(table.alias_or_name))])
            try:
                frame = duck(frames).execute(q.sql(dialect="duckdb")).df()      # усі рядки; на мапу to_geojson візьме рівномірну вибірку
            except Exception:  # noqa: BLE001
                continue
            gj = to_geojson(frame, park) if len(frame) else None
            if gj:
                return gj
    return None


def _friendly(exc):
    """Текст помилки для людини. Дивимось на тип помилки провайдера й на перший рядок повідомлення, а не на весь
    текст: у трасі PandasAI бувають дані (число 429 серед потужностей пожеж колись давало «ліміт ключа»)."""
    full = f"{type(exc).__name__}: {exc}"
    t = full.lower()
    head = t.strip().splitlines()[-1] if t.strip() else t
    if "authenticationerror" in t or "invalid x-api-key" in t or "incorrect api key" in t or "invalid api key" in t:
        return "Ключ не підійшов: перевірте, що його скопійовано повністю і він активний."
    if "insufficient_quota" in t or "exceeded your current quota" in t or "credit balance" in t or "billing" in t:
        return "На ключі закінчились кошти або квота. Поповніть баланс у провайдера або введіть інший ключ."
    if "ratelimiterror" in t or "rate limit" in t or "too many requests" in t:
        return "Провайдер моделі тимчасово обмежив запити з цього ключа. Спробуйте за хвилину."
    if "notfounderror" in t or "does not exist" in t and "model" in t:
        return "Модель недоступна для цього ключа."
    if "wall-clock limit" in head:
        return "Розрахунок ішов задовго й був зупинений. Спробуйте звузити питання: рік, тип, район."
    if "memory limit" in head:
        return "Розрахунок забрав забагато пам'яті й був зупинений. Спробуйте звузити питання."
    return "Не вдалося відповісти на це питання. Спробуйте сформулювати інакше."


def check(key):
    """Чи робочий ключ: запит в один токен до тієї ж моделі. Ловить і невалідний ключ, і ключ без коштів.
    Відповідь: ok, provider і, якщо не ok, текст для людини."""
    import litellm
    m = llm.model(key)
    if not m:
        return dict(ok=False, text="Ключ не схожий ні на OpenAI (sk-…), ні на Anthropic (sk-ant-…).")
    try:
        litellm.completion(model=m, messages=[{"role": "user", "content": "ok"}], max_tokens=1, api_key=key.strip(), timeout=20)
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, provider=llm.provider(key), text=_friendly(exc))
    return dict(ok=True, provider=llm.provider(key))


_GENERIC = {"схід", "півн", "захі", "півд", "наці", "парк"}                 # слова назв, які трапляються й без назви парку
ALIASES = {k: {w[:4] for w in re.findall(r"[а-яіїєґ']{4,}", (v["short"] + " " + v["name"]).lower()) if w[:4] not in _GENERIC} | {k.replace("_", " ")}
           for k, v in PARKS.items()}
ALL_WORDS = ("усі парки", "всі парки", "трьох парк", "три парки", "кожному парку", "кожного парку", "порівняй парки", "між парками", "у парках", "в парках")


def scope(question, park):
    """Чиї рядки бачить агент: вибраного парку, а всіх — коли питання порівнює парки чи називає інший парк."""
    q = question.lower()
    other = [k for k, al in ALIASES.items() if k != park and any(a and a in q for a in al)]
    return "all" if other or any(w in q for w in ALL_WORDS) else park


def frames_for(sc):
    """Таблиці для запиту: рядки одного парку (колонка park лишається) або всі."""
    if sc == "all":
        return FRAMES
    return {n: f[f["park"] == sc].reset_index(drop=True) if "park" in f.columns else f for n, f in FRAMES.items()}


LANG_RULE = {"uk": "Відповідай українською.", "en": "Answer in English: every text, column name you create and chart label in English. Table values from the data may stay as they are."}


def ask(question, key, recent=None, park=None, lang="uk"):
    """Одна відповідь про вибраний парк. Новий агент і нова пісочниця на кожен запит: стан не змішується між розмовами."""
    question = (question or "").strip()[:1000]
    park = park if park in PARKS else PK.MAIN
    if not question:
        return dict(type="error", text="Порожнє питання.")
    try:
        model = llm.make(key)
    except ValueError as exc:
        return dict(type="error", text=str(exc))
    EXPORTS.mkdir(parents=True, exist_ok=True)
    sandbox = TimeoutSandbox(timeout_seconds=float(os.environ.get("SANDBOX_TIMEOUT_SECONDS", 30)))
    tables = route(question, llm.model(key), key.strip(), recent)
    sc = scope(question, park)
    frames = frames_for(sc)
    note = ("Таблиці містять усі парки: порівнюй їх за колонкою park і показуй її у відповіді." if sc == "all" else
            "У таблицях з колонкою park лише рядки цього парку; таблиці без неї спільні (уся Греція, еталони, результати дослідження).")
    prompt = PROMPT.format(now=NOW[:10], tables=", ".join(tables), question=question, context=context_block(recent),
                           park=park, park_name=PARKS[park].get("name_en") if lang == "en" else PARKS[park]["name"], scope=note, lang_rule=LANG_RULE.get(lang, LANG_RULE["uk"]))
    try:
        result = Agent([named(frames[n], n, *DESCRIPTIONS[n]) for n in tables], config={"llm": model}, sandbox=sandbox).chat(prompt)   # графіки PandasAI пише у відносний exports/charts (сервер стартує в EXPORTS)
    except Exception as exc:  # noqa: BLE001 — PandasAI інколи випускає помилки з chat()
        text = _friendly(exc)
        return dict(type="error", text=text, key_problem=text.startswith(("Ключ не підійшов", "На ключі закінчились")))
    code = getattr(result, "last_code_executed", None)
    if result.type == "error":
        policy = bool(sandbox.policy_hits)
        return dict(type="error", code=code, policy=policy,
                    text="Таке питання виходить за межі аналізу даних парку." if policy else _friendly(Exception(str(result.error or result.value))))
    out = dict(type=result.type, code=code, tables=tables, park=park, scope=sc)
    if result.type == "dataframe":
        frame = pd.DataFrame(result.value)
        gj = to_geojson(frame, park)
        shown = frame.drop(columns=[c for c in ("geometry_wkt",) if c in frame.columns]).head(50)
        out |= dict(table=[{k: _plain(v) for k, v in r.items()} for r in shown.to_dict(orient="records")], rows=len(frame), geojson=gj)
        shown_n = len(gj["features"]) if gj else 0
        out["text"] = f"Знайдено {len(frame)} рядків." + (f" Показано на мапі: {shown_n} об'єктів" + (f" (рівномірна вибірка з {len(frame)})." if len(frame) > MAX_ROWS else ".") if gj else "")
    elif result.type == "chart":
        f = Path(str(result.value))
        f = f if f.is_absolute() else EXPORTS / f
        if f.exists():
            out["chart"] = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
            f.unlink()
        out["text"] = "Ось графік."
        gj = source_geojson(code, frames, park)
        if gj:
            out |= dict(geojson=gj, geo_source=True)
    else:
        out["text"] = str(_plain(result.value))
        gj = source_geojson(code, frames, park)
        if gj:
            out |= dict(geojson=gj, geo_source=True)
    return out
