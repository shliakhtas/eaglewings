"""Дані для сторінки /science/: web/science/data.json, extra.json, spread.json, risk_2026.geojson.

Джерела: локальні журнали запусків logs/runs (останній пакет «benchmark» кожної задачі), версіоновані датасети data/datasets/, базові шари ds.common.
Криві ROC/PR перераховуються тим самим кодом задач на тих самих версіях датасетів (детерміновано, seed зафіксовані).

    .venv/bin/python scripts/06_build_science.py
"""
import datetime as dt
import io
import json
import re
import os
import sys
from pathlib import Path
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


from eaglewings.ds import common, datasets, metrics, tracking  # noqa: E402
from eaglewings.ds import branches as BR  # noqa: E402
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from eaglewings.ds.tasks import burned_area, fire_day, fire_spread, spatial_risk  # noqa: E402

OUT = common.ROOT / "web" / "science"
OUT.mkdir(parents=True, exist_ok=True)
CLEARML_URL = None  # заповнюється з ClearML у main()


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


class _Run:
    """Запуск із logs/runs: той самий інтерфейс, яким сторінка користувалась із ClearML."""
    def __init__(self, d):
        self.dir, self.rec = d, json.loads((d / "run.json").read_text())
        self.id, self.project = self.rec["id"], self.rec["project"]


_ALL = {}


def _runs(exp):
    p = tracking.LOCAL_RUNS / tracking._project(exp).replace("/", "_")
    if p not in _ALL:
        _ALL[p] = [_Run(d) for d in sorted(p.iterdir()) if (d / "run.json").exists()] if p.exists() else []
    return _ALL[p]


def _row(t):
    m = {}
    for k, v in t.rec.get("metrics", {}).items():
        try:
            m[f"metrics.{k}"] = float(v)
        except (TypeError, ValueError):
            pass
    return pd.Series({"run_id": t.id, "experiment_id": t.project, "_run": t, **{f"tags.{k}": v for k, v in t.rec.get("props", {}).items()}, **m,
                      **{f"params.{k.split('/', 1)[-1]}": v for k, v in t.rec.get("params", {}).items()}})


_BY_ID = {}


def latest(exp):
    rs = [t for t in _runs(exp) if t.rec.get("parent") is None and t.rec["name"].startswith("benchmark") and t.rec.get("status") == "completed"]
    if not rs:
        raise LookupError(f"немає завершених пакетів у logs/runs для {exp}: спершу scripts/05_experiments.py")
    for t in _runs(exp):
        _BY_ID[t.id] = t
    return _row(rs[-1])


def art(run_id, name):
    return str(_BY_ID[run_id].dir / name)


def results(run_id):
    j = json.load(open(art(run_id, "results.json")))
    return pd.DataFrame(j["data"], columns=j["columns"])


def children(exp, parent_id, extra=""):
    cond = dict(re.findall(r"tags\.([\w.]+) = '([^']*)'", extra))
    rows = [_row(t) for t in _runs(exp) if t.rec.get("parent") == parent_id]
    df = pd.DataFrame(rows)
    for k, v in cond.items():
        df = df[df.get(f"tags.{k}") == v] if len(df) else df
    return df.reset_index(drop=True)


def history(run_id, metric):
    return [tuple(x) for x in _BY_ID[run_id].rec.get("series", {}).get(metric, [])]


def _main_ranks(c):
    """Ранги гексів основного парку з результатів FG_risk (з кількома парками там є колонка park; ранги з усіх парків)."""
    from eaglewings.ds import park
    if "park" not in c:
        return dict(final_rank=c["final_rank"], champ_rank=c["champ_rank"])
    m = [p_ == park.MAIN for p_ in c["park"]]
    pick = lambda v: [x for x, k in zip(v, m) if k]
    return dict(final_rank=pick(c["final_rank"]), champ_rank=pick(c["champ_rank"]))


def kept(task):
    """Дані графіків, які задача записала разом із метриками (ds/tracking.keep). Сайт їх лише перекладає."""
    f = common.P / "results" / f"{task}.json"
    if not f.exists():
        raise SystemExit(f"немає {f}: спершу задача ({task})")
    return json.loads(f.read_text())


def thin(xs, ys, n=160):
    xs, ys = np.asarray(xs), np.asarray(ys)
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [[r(xs[i], 4), r(ys[i], 4)] for i in idx]


def records(df):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", force_ascii=False))


def main():
    B, G = common.base(), common.h3grid()
    data = dict(generated=dt.datetime.now().isoformat(timespec="minutes"), clearml_url=None)

    # ------------------------------------------------ огляд
    all_runs = sum(1 for _ in tracking.LOCAL_RUNS.glob(f"{tracking.ROOT_PROJECT}_*/*/run.json"))
    reg = json.load(open(common.ROOT / "models" / "registry.json")) if (common.ROOT / "models" / "registry.json").exists() else {}
    dom = B.park_mask & B.forest
    data["overview"] = dict(
        park_ha=r(B.park_u.area.iloc[0] / 1e4, 0), forest_ha=r(B.PIX_HA[dom].sum(), 0), firms=int(len(B.firms)),
        hansen_pixels=int(dom.sum()), days=int(len(common.power())), h3_cells=int(len(G.grid)), runs=all_runs,
        experiments=4, models=[dict(name=k, version=1) for k in (reg if isinstance(reg, dict) else {})])
    data["datasets"] = []
    for d in sorted(p for p in datasets.DATASETS.iterdir() if p.is_dir()):
        dv = datasets.load(d.name)
        m = dv.manifest
        data["datasets"].append(dict(name=m["name"], version=m["version"], digest=m["digest"], rows=m["rows"], unit=m["unit"],
                                     description=m["description"], targets=m["targets"], label_stats=m["label_stats"],
                                     sources=m["sources"], params=m["params"], n_features=len(m["features"]), created=m["created"]))

    # ------------------------------------------------ EDA
    f = B.firms
    mon = f.groupby([f.dt.dt.to_period("M"), "sensor"]).size().unstack(fill_value=0)
    mon = mon.reindex(pd.period_range(f"{min(common.YEARS)}-01", f"{max(common.YEARS)}-12", freq="M"), fill_value=0)
    data["eda"] = dict(
        firms_monthly=[dict(m=str(p), viirs=int(row.get("VIIRS", 0)), modis=int(row.get("MODIS", 0))) for p, row in mon.iterrows()],
        firms_by_daynight=f.groupby(["year", "daynight"]).size().unstack(fill_value=0).reset_index().rename(columns={"D": "day", "N": "night"}).to_dict("records"),
        hansen_loss=[dict(year=2000 + y, ha=r(B.PIX_HA[(B.loss == y) & B.park_mask].sum(), 1)) for y in range(1, 25)],
    )
    wx = datasets.main_park(datasets.load("fire_day_daily").df)
    keep = ["date", "T2M_MAX", "RH2M", "VPD_7", "dry_days", "nesterov", "GWET_7", "PRECTOTCORR", "fire_day", "ignition"]
    data["eda"]["weather"] = {c: (wx[c].round(3).tolist() if c != "date" else wx[c].tolist()) for c in keep}
    q = pd.qcut(wx.dry_days.rank(method="first"), 10, labels=False)
    data["eda"]["fire_rate_by_dry_decile"] = [dict(decile=int(k), rate=r(v), lo=int(wx.dry_days[q == k].min()), hi=int(wx.dry_days[q == k].max()))
                                              for k, v in wx.groupby(q).fire_day.mean().items()]
    h3d = datasets.main_park(datasets.load("h3_risk_cells").df)
    feats = ["elev", "slope", "southness", "treecover", "d_road", "d_village", "track_km_km2", "d_boundary"]
    data["eda"]["h3"] = {c: h3d[c].round(3).tolist() for c in feats} | {c: h3d[c].astype(int).tolist() for c in ["y_burn_2022", "y_burn_2023", "y_hist"]}

    # ------------------------------------------------ T1
    if BR.is_on("FD") and not BR.mode("FD"):          # режим signal (FD у config/branches.py MODES) — розділ збирається автоматично
        t1_section(data)
    if BR.is_on("FG_day"):
        t2_section(data)
    if BR.is_on("FG_risk") and not BR.mode("FG_risk"):
        t3_section(data, h3d)
    t4_section(data)
    data["park"] = json.loads(B.park.geometry.set_precision(1e-5).to_json())
    if BR.is_on("FG_spread") and spread_ran():
        t5(data)

    (OUT / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=lambda o: None))
    print("web/science/data.json", round((OUT / "data.json").stat().st_size / 1e6, 2), "MB")
    extra(data)


def spread_ran():
    """Поширення порахувало хоч одну пожежу (на території без великих пожеж гілка пише «нема на чому рахувати»)."""
    s = json.load(open(common.P / "experiments_summary.json")).get("t5", {}) if (common.P / "experiments_summary.json").exists() else {}
    return any(r.get("approach") != "not_run" for r in s.get("results", []))


def t1_section(data):
    p1 = latest(burned_area.EXPERIMENT)
    t1 = results(p1.run_id)
    fam = children(burned_area.EXPERIMENT, p1.run_id, " and tags.role = 'family'")
    cfg_rows = []
    for _, fr in fam.iterrows():
        ch = children(burned_area.EXPERIMENT, fr.run_id)
        for _, c in ch.iterrows():
            params = {k[7:]: v for k, v in c.items() if k.startswith("params.") and v is not None and k != "params.family"}
            cfg_rows.append(dict(family=fr["tags.family"], params=params, calib_f1=r(c["metrics.calib_f1"]),
                                 calib_precision=r(c["metrics.calib_precision"]), calib_recall=r(c["metrics.calib_recall"])))
    data["t1"] = dict(experiment=burned_area.EXPERIMENT, run_id=p1.run_id, experiment_id=p1.experiment_id, dataset=p1["tags.dataset.evaluation"],
                      champion=p1["tags.champion"], families=records(t1), configs=cfg_rows)



def t2_section(data):
    p2 = latest(fire_day.EXPERIMENT)
    t2 = results(p2.run_id)
    K2 = kept("fire_day")                                                   # криві задача намалювала з тих самих передбачень, що й метрики
    by_year, curves, champ_series = K2["by_year"], K2["pr_curves"], K2["champion_series"]
    data["t2"] = dict(experiment=fire_day.EXPERIMENT, run_id=p2.run_id, experiment_id=p2.experiment_id, dataset=p2["tags.dataset.training"],
                      targets=fire_day.TARGETS, test_years=fire_day.TEST_YEARS, results=records(t2), by_year=by_year, pr_curves=curves,
                      champion_series=champ_series, hypotheses={a["name"]: a["hyp"] for a in fire_day.APPROACHES})



def t3_section(data, h3d):
    p3 = latest(spatial_risk.EXPERIMENT)
    t3 = results(p3.run_id)
    K3 = kept("spatial_risk")
    roc = K3["roc"]
    data["t3"] = dict(experiment=spatial_risk.EXPERIMENT, run_id=p3.run_id, experiment_id=p3.experiment_id, dataset=p3["tags.dataset.training"],
                      champion=p3["tags.champion"], final_model=p3["tags.final_model"], upper_bound=r(p3["metrics.upper_bound_cv_roc_auc"]),
                      fuel_factor=r(p3["metrics.fuel_factor"]), results=records(t3), roc=roc,
                      hypotheses={a["name"]: a["hyp"] for a in spatial_risk.APPROACHES},
                      prevalence=K3["prevalence"])

    # ------------------------------------------------ мапа ризику 2026 (+ факт 2023)
    risk = gpd.read_file(common.P / "risk_h3_2026.geojson").merge(h3d[["h3", "y_burn_2023", "y_burn_2022"]], on="h3", how="left")
    risk["geometry"] = risk.geometry.set_precision(1e-5)
    for c in ("susceptibility", "fuel_mult", "risk_2026", "risk_rank"):
        risk[c] = risk[c].round(3)
    risk[["h3", "level", "risk_rank", "susceptibility", "fuel_mult", "y_burn_2023", "y_burn_2022", "geometry"]].to_file(OUT / "risk_2026.geojson", driver="GeoJSON")


def t4_section(data):
    # T4: демо детекторів, рахується на GPU окремо, тут лише його підсумок
    s4 = json.load(open(common.P / "experiments_summary.json")).get("t4")
    if not s4:
        return
    data["t4"] = dict(experiment=s4["experiment"], run_id=s4["info"].get("parent_run_id"), experiment_id=s4["info"].get("experiment_id"), results=s4["results"])


# =============================================================== T5 · поширення пожежі (spread.json)
def t5(data):
    import h3
    T = fire_spread
    p5 = latest(T.EXPERIMENT)
    R = results(p5.run_id)
    kids = children(T.EXPERIMENT, p5.run_id, " and tags.role = 'approach'")
    fires_tbl = json.load(open(art(p5.run_id, "fires.json")))
    fires_df = pd.DataFrame(fires_tbl["data"], columns=fires_tbl["columns"])
    by_fire = {}
    for _, k in kids.iterrows():
        for h in (24, 48):
            j = json.load(open(art(k.run_id, f"lofo_by_fire_{h}h.json")))
            by_fire[f"{k['tags.approach']}_{h}"] = pd.DataFrame(j["data"], columns=j["columns"]).to_dict("records")
    champ = p5["tags.champion"]
    K5 = kept("fire_spread")
    spread, daily = K5["spread"], K5["daily"]
    (OUT / "spread.json").write_text(json.dumps(spread, ensure_ascii=False, separators=(",", ":"), default=lambda o: None))
    data["t5"] = dict(experiment=T.EXPERIMENT, run_id=p5.run_id, experiment_id=p5.experiment_id, dataset=p5["tags.dataset.training"], champion=champ,
                      test_key=T.TEST_KEY, results=records(R), by_fire=by_fire, daily=daily,
                      fires=records(fires_df[["key", "name", "n", "start", "end", "lat", "lon", "ndays"]]),
                      hypotheses={k: v["hyp"] for k, v in T.APPROACHES.items()}, kinds={k: v["kind"] for k, v in T.APPROACHES.items()})
    print("web/science/spread.json", round((OUT / "spread.json").stat().st_size / 1e6, 2), "MB")


# =============================================================== extra.json — діагностика з ноутбука (растри, мапи, камера, вежі)
GLOSSARY = {
    "a": {"firms_pixels": ("Сирі пікселі FIRMS", "baseline"), "firms_st_dbscan": ("Події FIRMS (ST-DBSCAN)", "algo"),
          "firms_hull": ("Оболонка подій FIRMS", "algo"), "modis_dndvi": ("Падіння NDVI (dNDVI)", "rule"),
          "modis_rdndvi": ("Відносне падіння NDVI", "rule"), "hybrid": ("dNDVI + підтвердження FIRMS", "rule"),
          "s2_dnbr": ("Падіння NBR Sentinel-2 (dNBR)", "rule")},
    "b": {"persistence": ("Вчора горіло", "baseline"), "climatology": ("Кліматологія (місяць)", "baseline"),
          "index_dry_days": ("Індекс: сухі дні", "rule"), "index_vpd7": ("Індекс: сухість повітря (VPD)", "rule"),
          "index_nesterov": ("Індекс Нестерова", "rule"), "index_fwi": ("Індекс FWI (стандарт EFFIS)", "rule"), "logreg_weather": ("LogReg: погода", "model"),
          "logreg_weather_season": ("LogReg: погода + сезон", "model"), "logreg_weather_season_unweighted": ("LogReg: без зважування класів", "model"),
          "random_forest": ("RandomForest", "model"), "lightgbm": ("LightGBM", "model")},
    "c": {"random": ("Випадково", "baseline"), "elevation": ("Лише висота", "baseline"), "treecover": ("Лише лісистість", "baseline"),
          "near_village": ("Близькість до сіл", "baseline"), "firms_history": ("Де горіло раніше (FIRMS)", "baseline"),
          "hist_lr_static": ("Історія Hansen · LogReg", "model"), "hist_rf_static": ("Історія Hansen · RandomForest", "model"),
          "hist_lgbm_static": ("Історія Hansen · LightGBM", "model"), "hist_lgbm_terrain": ("Історія Hansen · лише рельєф", "model"),
          "hist_lgbm_human": ("Історія Hansen · лише люди", "model"), "hist_lgbm_terrain_fuel": ("Історія Hansen · рельєф + паливо", "model"),
          "f22_lgbm_static": ("Пожежа 2022 · LightGBM", "model"), "f22_lgbm_all": ("Пожежа 2022 · LightGBM (усі ознаки)", "model"),
          "f22_lr_all": ("Пожежа 2022 · LogReg", "model")},
}


def extra(D):
    import shapely
    from PIL import Image
    from pyproj import Transformer
    B, G = common.base(), common.h3grid()
    X = {"glossary": {t: {k: dict(name=v[0], type=v[1]) for k, v in g.items()} for t, g in GLOSSARY.items()}}

    if BR.is_on("FD") and not BR.mode("FD"):
        fd_extra(X, B)
    if BR.is_on("FG_day"):
        X["b_coef"] = kept("fire_day")["b_coef"]
    h3d = datasets.main_park(datasets.load("h3_risk_cells").df)
    if BR.is_on("FG_risk") and not BR.mode("FG_risk"):
        K3 = kept("spatial_risk")
        X["c_cells"] = dict(h3=h3d.h3.tolist(), **{c_: h3d[c_].astype(int).tolist() for c_ in ["y_hist", "y_burn_2022", "y_burn_2023", "y_loss_2023_hansen", "y_dndvi_2023"]},
                            **{c_: h3d[c_].round(2).tolist() for c_ in ["elev", "slope", "treecover", "d_village", "d_road", "track_km_km2"]},
                            **_main_ranks(K3["cells"]))
        X["c_importance"] = K3.get("importance", [])
    camera_extra(X, B, G, h3d)
    (OUT / "extra.json").write_text(json.dumps(X, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=lambda o: None))
    print("web/science/extra.json", round((OUT / "extra.json").stat().st_size / 1e6, 2), "MB")


def fd_extra(X, B):
    import shapely
    from pyproj import Transformer
    # ---------- A · растр «чемпіон vs Hansen» на сітці 120 м
    X["a_raster"] = kept("burned_area")["raster"]                          # мапа з тих самих передбачень, що дали F1

    # ---------- A · MODIS NDVI до / після / відновлення
    cen = B.PARK.centroid
    dates = {"pre": "A2023209", "post": "A2023257", "2024": "A2024209", "2025": "A2025209", "2026": "A2026209"}
    nd = {}
    for kk, d in dates.items():
        js = common.ornl(round(cen.y, 4), round(cen.x, 4), d, d, 16, 13)
        a = np.array(js["subset"][0]["data"], float).reshape(js["nrows"], js["ncols"])
        a[a < -2000] = np.nan
        nd[kk] = a * float(js["scale"])
    nr, nc, cs = js["nrows"], js["ncols"], js["cellsize"]
    xx = float(js["xllcorner"]) + (np.arange(nc) + .5) * cs
    yy = float(js["yllcorner"]) + (nr - np.arange(nr) - .5) * cs
    LON, LAT = Transformer.from_crs(common.SINU, "EPSG:4326", always_xy=True).transform(*np.meshgrid(xx, yy))
    inpark = shapely.contains_xy(B.PARK, LON, LAT) & np.isfinite(nd["pre"]) & np.isfinite(nd["post"])
    dN = nd["pre"] - nd["post"]
    burned = inpark & (dN > .1)
    rec = {y: float(np.nanmedian(np.clip((nd[y][burned] - nd["post"][burned]) / dN[burned], -.5, 1.5))) for y in ("2024", "2025", "2026")}
    ev, _ = common.default_events()
    main = ev.area_ha.idxmax()
    areas = dict(firms=float(gpd.GeoSeries([ev.loc[main].geometry], crs=common.UTM).to_crs(4326).intersection(B.PARK).to_crs(common.UTM).area.sum() / 1e4),
                 modis=float(burned.sum() * cs * cs / 1e4), hansen=float(B.PIX_HA[np.isin(B.loss, [23, 24]) & B.park_mask].sum()))
    ii = np.flatnonzero(inpark.ravel())
    X["a_ndvi"] = dict(dates={k: v for k, v in zip(dates, ["2023-07-28", "2023-09-14", "2024-07-27", "2025-07-28", "2026-07-28"])},
                       cell_deg=[cs / 111320, cs / (111320 * np.cos(np.radians(LAT0)))], lon=np.round(LON.ravel()[ii], 4).tolist(), lat=np.round(LAT.ravel()[ii], 4).tolist(),
                       **{k: [r(v, 2) for v in nd[k].ravel()[ii]] for k in dates}, recovery=rec, areas=areas)
    # часові ряди трьох ділянок (кеш ноутбука: точкові запити ORNL)
    sites = {}
    for f in sorted(common.CACHE.glob("modis_250m_16_days_NDVI_*_0_0.json")):
        js = json.loads(f.read_text())
        key = (js["latitude"], js["longitude"])
        for s_ in js["subset"]:
            v = s_["data"][0]
            sites.setdefault(key, {})[s_["calendar_date"]] = None if v < -2000 else round(v * float(js["scale"]), 3)
    out_sites = []
    for (la, lo), ts in sites.items():
        j = int(np.nanargmin((LAT - la) ** 2 + (LON - lo) ** 2))
        out_sites.append(dict(lat=la, lon=lo, dndvi=r(dN.ravel()[j], 3), dates=sorted(ts), ndvi=[ts[d] for d in sorted(ts)]))
    out_sites.sort(key=lambda s_: -(s_["dndvi"] or 0))
    for s_, nm in zip(out_sites, ["сильне ураження", "помірне ураження", "контроль (ліс без пожежі)"]):
        s_["name"] = nm
    X["a_ndvi"]["sites"] = out_sites

    # ---------- дані: пожежа 2023 по годинах (головна подія)
    lab = common.st_dbscan(B.firms)
    evm = pd.Series(lab).value_counts().index[0]
    g = B.firms[lab == evm]
    t0 = g.dt.min()
    X["fire2023"] = dict(start=str(t0), lat=g.latitude.round(4).tolist(), lon=g.longitude.round(4).tolist(),
                         h=((g.dt - t0).dt.total_seconds() / 3600).round(1).tolist(), frp=g.frp.round(1).tolist(), sensor=g.sensor.tolist())



def camera_extra(X, B, G, h3d):
    from PIL import Image
    V = json.load(open(common.P / "vision" / "video.json"))
    CAM = dict(lat=V["cam_lat"], lon=V["cam_lon"], bearing=V["bearing"], fov=60, mast=15, rmax=10000)
    cen_ = G.grid.set_index("h3").loc[h3d.h3].geometry.centroid
    glat, glon = cen_.y.values, cen_.x.values
    cv_, cd_ = common.los(CAM["lat"], CAM["lon"], glat, glon, CAM["mast"])
    caz = common.azimuth(glat, glon, CAM["lat"], CAM["lon"])
    sector = (((caz - (CAM["bearing"] - CAM["fov"] / 2)) % 360) <= CAM["fov"]) & (cd_ <= CAM["rmax"])

    # ---------- D · камера ↔ мапа
    frames = [dict(t=f_["t"], d=[[x["kind"], round(x["conf"], 2), round(x["box"][0], 1), round(x["box"][2], 1)] for x in f_["detections"]]) for f_ in V["frames"]]
    f23_ = B.firms[B.firms.year == 2023]
    fv, fd = common.los(CAM["lat"], CAM["lon"], f23_.latitude.values, f23_.longitude.values, CAM["mast"])
    faz = common.azimuth(f23_.latitude.values, f23_.longitude.values, CAM["lat"], CAM["lon"])
    seen = fv & (fd <= CAM["rmax"]) & ((((faz - (CAM["bearing"] - CAM["fov"] / 2)) % 360) <= CAM["fov"]))
    X["d_camera"] = dict(cam=CAM, frame_w=1280, fps=V["fps_out"], video="../" + V["video"], frames=frames,
                         sector=np.flatnonzero(sector).tolist(), visible=np.flatnonzero(sector & cv_).tolist(), az=[r(a_, 1) for a_ in caz],
                         dist=[r(d_, 0) for d_ in cd_], firms_seen=[[r(a_, 4), r(b_, 4), str(t_)[:16]] for a_, b_, t_ in zip(f23_.latitude[seen], f23_.longitude[seen], f23_.dt[seen])])
    ph = json.load(open(common.P / "vision" / "photos.json"))
    photos = []
    for p_ in ph:
        rw_, rh_ = Image.open(common.RAW / "media" / p_["file"]).size
        ww, wh = Image.open(common.ROOT / "web" / p_["img"]).size
        sx = ww / rw_
        photos.append(dict(img="../" + p_["img"], title=p_["title"], w=ww, h=wh, lat=p_["lat"], lon=p_["lon"], author=p_["author"], license=p_["license"],
                           boxes=[[round(b[0] * sx), round(b[1] * sx), round(b[2] * sx), round(b[3] * sx), d_["kind"], d_["conf"]]
                                  for d_ in p_["detections"] for b in [d_["box"]]]))
    X["d_photos"] = photos


if __name__ == "__main__":
    main()
