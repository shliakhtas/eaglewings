"""EDA кожного датасету: функція build рахує, а цей скрипт лише збирає готові результати → web/science/eda.json. Запуск із теки eaglewings:

    eaglewings script 10_dataset_eda

Це опис даних, а не експерименти. Зв'язок ознак із лейблом рахується лише на train: тест не використовується.
"""
import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import common  # noqa: E402
sys.path.insert(0, str(LIB / "scripts"))
from data_lineage import LINEAGE, SRC  # noqa: E402

DATA = ROOT / "data" / "datasets"
SEED = 42
rnd = lambda v, k=4: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), k)
YEAR = lambda d: pd.to_datetime(d).dt.year

DOC = dict(row="рядок пікселя в сітці Hansen", col="колонка пікселя в сітці Hansen", h3="ідентифікатор гекса H3", block="просторовий блок для спліту", split="частина спліту", pix_ha="площа пікселя, га", treecover="лісистість 2000 року, %", ndvi_pre="NDVI до пожежі", ndvi_post="NDVI після пожежі", y="лейбл", y_2023_only="лейбл: втрата саме у 2023",
           dt="час сигналу", sensor="сенсор", conf="впевненість сигналу", lat="широта", lon="довгота", year="рік", effis="номер згарища EFFIS", in_effis="сигнал усередині згарища EFFIS",
           date="дата", T2M_MAX="максимальна температура, °C", T2M="середня температура, °C", RH2M="вологість повітря, %", WS10M="вітер на 10 м, м/с", PRECTOTCORR="опади, мм", GWETTOP="вологість верхнього шару ґрунту", VPD="дефіцит вологи повітря", dry_days="сухих днів поспіль", prec_30="опади за 30 днів, мм", VPD_7="дефіцит вологи за 7 днів", T_7="температура за 7 днів", GWET_7="вологість ґрунту за 7 днів", WS_3="вітер за 3 дні", nesterov="індекс Нестерова", season_sin="сезон, синус", season_cos="сезон, косинус", fire_day="день із вогнем", ignition="нове займання", fire_day_yesterday="вчора був вогонь",
           parent6="гекс-батько рівня 6", parent7="гекс-батько рівня 7", elev="висота, м", slope="ухил, °", southness="наскільки схил південний", d_road="до дороги, м", d_village="до села, м", track_km_km2="густота лісових доріг, км/км²", d_boundary="до межі парку, м",
           y_hist="горіло в минулому (за втратами лісу)", y_burn_2022="згоріло у 2022 за FIRMS", y_burn_2023="згоріло у 2023 за FIRMS", y_burn_2024="згоріло у 2024 за FIRMS", y_burn_2025="згоріло у 2025 за FIRMS", y_loss_2023_hansen="згоріло у 2023 за Hansen", y_dndvi_2023="згоріло у 2023 за падінням NDVI", y_final="згоріло хоч за одним джерелом",
           fire="пожежа", t="момент часу", k="крок, діб від старту", cell="гекс H3 res 8", d_front="до фронту, м", d_active="до активного вогню, м", nb1="сусіди, що горять, кільце 1", nb2="сусіди, що горять, кільце 2", log_area="логарифм площі пожежі", adv_km="просування фронту, км", log_growth="логарифм приросту", wind_ms="вітер, м/с", cos_wind="косинус кута до вітру", wind_along="вітер у бік гекса", cos_motion="косинус кута до руху пожежі", motion_km="рух пожежі, км", slope_up="ухил угору", loss_recent="недавні втрати лісу", t2m_max="максимальна температура", rh_min="мінімальна вологість", day_k="день пожежі", y24="згорить за 24 год", y48="згорить за 48 год",
           ha="нова втрата лісу, га", loss1="втрати торік, га", loss2="втрати 2 роки тому, га", loss3_5="втрати 3–5 років тому, га", loss_cum="усі минулі втрати, га", nb_loss1="втрати сусідів торік, га",
           w="вага пікселя", side="половина парку", red_pre="червоний до", red_post="червоний після", nir_pre="ближній ІЧ до", nir_post="ближній ІЧ після", sw1_pre="SWIR1 до", sw1_post="SWIR1 після", sw2_pre="SWIR2 до", sw2_post="SWIR2 після", d_ndvi="падіння NDVI", nbr_pre="NBR до", nbr_post="NBR після", d_nbr="падіння NBR", ndmi_pre="NDMI, вологість крон, до", ndmi_post="NDMI після", d_ndmi="падіння NDMI", r_nir="відношення ІЧ після й до",
           s0="немає порушення", s1="перше виявлення, втрата < 50 %", s2="попереднє, < 50 %", s3="підтверджене, < 50 %", s4="перше виявлення, ≥ 50 %", s5="попереднє, ≥ 50 %", s6="підтверджене, ≥ 50 %", s7="завершене, < 50 %", s8="завершене, ≥ 50 %", s255="немає даних",
           ev="номер пожежі", n="точок у пожежі", start="перша точка", visible="гексів видно з місця", dataset="набір", image="кадр", n_boxes="рамок на кадрі", part="частина спліту", kind="тип", km="до дороги, км",
           source="джерело", simulated="симульований", scenario="сценарій генератора", truth="еталонна подія", h="гекс H3 res 7")
for c in ["loss_hist", "loss_recent3", "firms_2km", "d_firms", "burned_prev3"]:
    t = dict(loss_hist="минулі втрати лісу", loss_recent3="втрати за 3 роки", firms_2km="точки FIRMS у радіусі 2 км", d_firms="до точки FIRMS, м", burned_prev3="згоріло за 3 роки")[c]
    for y in (22, 23, 26):
        DOC[f"f{y}_{c}"] = f"{t}, станом на 20{y}"

G = {"Пожежі": ["burned_area_pixels", "matching_firms_effis", "fire_day_daily", "h3_risk_cells", "fire_spread_cells", "tp_fires", "tp_candidates"], "Вирубка": ["logging_gr_pixels", "opera_gr_snapshots", "logging_gr_cells"],
     "Детектори й камери": ["cv_benchmark_aerial", "cv_benchmark_ground", "cv_camera_zone", "cp_events"], "Сигнали": ["xm_signals", "ar_fires", "as_areals"],
     "Нові гілки": ["fd_signal", "ar_logging", "track_segments", "toptw_instances", "route_instances"]}
CFG = dict(
    burned_area_pixels=dict(code="FD", task="Детекція пожежі", target="y", group="split", train=lambda d: d.split == "calib", tnote="calib, 50 % блоків", feats=["ndvi_pre", "ndvi_post", "treecover", "d_ndvi"]),
    matching_firms_effis=dict(code="FM", task="Матчинг пожежі", target="in_effis", group="year", train=lambda d: d.year <= 2023, tnote="2021–2023", feats=["conf"]),
    fire_day_daily=dict(code="FG", task="Розвиток: день пожежі", target="fire_day", group="year", train=lambda d: YEAR(d.date) <= 2022, tnote="2021–2022", feats=["T2M_MAX", "RH2M", "WS10M", "PRECTOTCORR", "GWETTOP", "VPD", "dry_days", "prec_30", "VPD_7", "T_7", "GWET_7", "WS_3", "nesterov", "fire_day_yesterday"]),
    h3_risk_cells=dict(code="FG", task="Розвиток: ризик ділянок", target="y_burn_2022", train=lambda d: d.index >= 0, tnote="лейбл 2022, ознаки станом на 2022", feats=["elev", "slope", "southness", "treecover", "d_road", "d_village", "track_km_km2", "d_boundary", "f22_loss_hist", "f22_loss_recent3", "f22_firms_2km", "f22_d_firms", "f22_burned_prev3"]),
    fire_spread_cells=dict(code="FG", task="Розвиток: поширення", target="y24", group="k", train=lambda d: d.fire != "2023-08 41.0 26.0", tnote="усі пожежі, крім тестової Еврос-2023", feats=["d_front", "d_active", "nb1", "nb2", "log_area", "adv_km", "log_growth", "wind_ms", "cos_wind", "wind_along", "cos_motion", "motion_km", "slope_up", "elev", "treecover", "loss_recent", "t2m_max", "rh_min", "day_k"]),
    tp_fires=dict(code="TP", task="Де ставити вежі", group="year", feats=["n"]),
    tp_candidates=dict(code="TP", task="Де ставити вежі", feats=["elev", "visible"]),
    logging_gr_pixels=dict(code="LD · LM", task="Детекція й матчинг вирубки", target="y", group="year", train=lambda d: d.year <= 2021, tnote="2018–2021", feats=["ndvi_pre", "ndvi_post", "d_ndvi", "nbr_pre", "d_nbr", "ndmi_pre", "ndmi_post", "d_ndmi", "r_nir", "elev", "slope", "treecover"]),
    opera_gr_snapshots=dict(code="LM", task="Матчинг вирубки", feats=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]),
    logging_gr_cells=dict(code="LG", task="Розвиток вирубки", target="y", group="year", train=lambda d: d.year <= 2020, tnote="2017–2020", feats=["elev", "slope", "treecover", "d_road", "d_village", "track_km_km2", "loss1", "loss2", "loss3_5", "loss_cum", "nb_loss1"]),
    cv_benchmark_aerial=dict(code="SD", task="Детекція зі супутника", group="dataset", feats=["n_boxes"]),
    cv_benchmark_ground=dict(code="CD", task="Детекція з фото й відео", group="dataset", feats=["n_boxes"]),
    cv_camera_zone=dict(code="CG", task="Камера на координати", group="year"),
    cp_events=dict(code="CP", task="Де ставити камери", group="year", feats=["km"]),
    xm_signals=dict(code="XM", task="Крос-джерельний матчинг", feats=["conf"]),
    ar_fires=dict(code="AR", task="Ареалізація появи", group="year", feats=["n"]),
    as_areals=dict(code="AS", task="Зони інтересу (симуляція)", group="part", feats=["sources"]),
    fd_signal=dict(code="FD", task="Чи бачать супутники пожежі", target="in_effis", group="year", feats=["frp", "forest", "month"]),
    ar_logging=dict(code="AR", task="Де рубки повторюються", group="year", feats=["ha"]),
    track_segments=dict(code="SD", task="Лісові колії на Sentinel-2", target="y", group="park", feats=["vis_mean", "ndvi_mean", "side_dndvi", "line_aniso_mean"]),
    toptw_instances=dict(code="RS", task="Розв'язувач маршрутів", group="split", feats=["best_known", "teams"]),
    route_instances=dict(code="RD", task="Маршрути дня (симуляція)", target="new", group="type", feats=["prio", "at_h"]),
)


def latest(n):
    v = json.load(open(DATA / n / "versions.json"))
    m = json.load(open(DATA / n / f"v{v[-1]['version']}" / "manifest.json"))
    return pd.read_csv(DATA / n / f"v{v[-1]['version']}" / "data.csv.gz"), m, len(v)


def role(c, m, cfg):
    if c in (m.get("targets") or []) or c == cfg.get("target"):
        return "лейбл"
    if c in ("lat", "lon", "row", "col"):
        return "місце"
    if c in ("date", "dt", "t", "start", "year"):
        return "час"
    if c in ("split", "part", "block", "side", "parent6", "parent7", "h"):
        return "спліт"
    if c in ("h3", "cell", "image", "ev", "fire", "truth", "scenario", "effis"):
        return "ключ"
    return "ознака" if c in (cfg.get("feats") or []) or c in (m.get("features") or []) or str(c).startswith("f2") else "інше"


def column(d, c, m, cfg):
    s = d[c]
    o = dict(n=c, role=role(c, m, cfg), doc=DOC.get(c, ""), miss=rnd(s.isna().mean(), 4), uniq=int(s.nunique()))
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        q = s.dropna()
        o.update(t="число", min=rnd(q.min(), 3), med=rnd(q.median(), 3), max=rnd(q.max(), 3), mean=rnd(q.mean(), 3), std=rnd(q.std(), 3), zero=rnd((q == 0).mean(), 3))
    else:
        o.update(t="текст", top=[[str(a), int(b)] for a, b in s.value_counts().head(4).items()] if s.nunique() < len(s) * .5 else [], span=[str(s.min()), str(s.max())] if s.nunique() >= len(s) * .5 and s.dtype == object else None)
    return o


def hist(d, c, tg=None):
    s = d[c].astype(float)
    ok = s.notna()
    q = s[ok]
    if q.nunique() < 2:
        return None
    lo, hi = np.quantile(q, [.005, .995])
    if hi <= lo:
        lo, hi = q.min(), q.max()
    log = bool(lo >= 0 and q.skew() > 4)
    f = np.log1p if log else (lambda v: v)
    e = np.linspace(f(lo), f(hi), 25)
    cnt = lambda v: np.histogram(np.clip(f(v), e[0], e[-1]), e)[0].tolist()
    o = dict(n=c, doc=DOC.get(c, ""), log=log, lo=rnd(lo, 3), hi=rnd(hi, 3), med=rnd(q.median(), 3), edges=[rnd(float(np.expm1(x)) if log else float(x), 3) for x in e])
    if tg is not None:
        y = d[tg].values.astype(bool)
        o.update(neg=cnt(s[ok & ~y]), pos=cnt(s[ok & y]))
    else:
        o["all"] = cnt(q)
    return o


def assoc(d, feats, tg, train):
    t = d[train(d)] if train is not None else d
    t = t.sample(min(len(t), 200000), random_state=SEED)
    y = t[tg].values
    out = []
    if y.min() == y.max():
        return out
    for c in feats:
        if c not in t or not pd.api.types.is_numeric_dtype(t[c]):
            continue
        x = t[c].astype(float)
        if x.notna().sum() < 10 or x.nunique() < 2:
            continue
        a = roc_auc_score(y, x.fillna(x.median()))
        out.append(dict(n=c, doc=DOC.get(c, ""), auc=rnd(a, 3), pos=rnd(x[y == 1].median(), 3), neg=rnd(x[y == 0].median(), 3)))
    return sorted(out, key=lambda r: -abs(r["auc"] - .5))


def corr(d, feats):
    fs = [c for c in feats if c in d and pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 1][:13]
    if len(fs) < 3:
        return None
    t = d[fs].sample(min(len(d), 20000), random_state=SEED).astype(float)
    return dict(names=fs, docs=[DOC.get(c, "") for c in fs], m=np.round(t.corr("spearman").values, 2).tolist())


def geo(n, d):
    r = np.random.default_rng(SEED)
    samp = lambda q, k: q if len(q) <= k else q.iloc[r.choice(len(q), k, replace=False)]
    cen = lambda s: np.array([h3.cell_to_latlng(c) for c in s]).T
    if n == "burned_area_pixels":
        q = d.groupby("h3").y.mean().reset_index()
        a, b = cen(q.h3)
        return dict(label="частка згорілих пікселів у гексі", kind="value", pts=[[rnd(x, 4), rnd(z, 4), rnd(v, 2)] for x, z, v in zip(a, b, q.y)])
    if n == "h3_risk_cells":
        a, b = cen(d.h3)
        return dict(label="гекси, що згоріли у 2022", kind="value", pts=[[rnd(x, 4), rnd(z, 4), int(v)] for x, z, v in zip(a, b, d.y_burn_2022)])
    if n == "logging_gr_cells":
        q = d.groupby("h3").y.sum().reset_index()
        a, b = cen(q.h3)
        return dict(label="скільки років у гексі була рубка", kind="value", pts=[[rnd(x, 4), rnd(z, 4), int(v)] for x, z, v in zip(a, b, q.y) if v > 0])
    if n == "logging_gr_pixels":
        la, lo = np.zeros(len(d)), np.zeros(len(d))
        for k in (d.park.unique() if "park" in d else [None]):          # рядок і стовпчик у сітці свого парку
            i = (d.park == k).values if k is not None else np.ones(len(d), bool)
            B = common.base(k)
            la[i], lo[i] = B.TR.f + d.row.values[i] * B.TR.e, B.TR.c + d.col.values[i] * B.TR.a
        q = pd.DataFrame(dict(a=np.round(la / .008) * .008, b=np.round(lo / .008) * .008, yw=d.y.values * d.w.values, w=d.w.values)).groupby(["a", "b"]).agg(yw=("yw", "sum"), w=("w", "sum")).reset_index()
        q = q[q.w >= 300]
        return dict(label="частка пікселів із втратою з урахуванням ваги, клітинка 0,8 км", kind="value", pts=[[rnd(x, 4), rnd(z, 4), rnd(v, 3)] for x, z, v in zip(q.a, q.b, q.yw / q.w)])
    if n == "fire_spread_cells":
        q = d.groupby("fire").agg(a=("lat", "mean"), b=("lon", "mean"), n=("lat", "size")).reset_index()
        return dict(label="28 пожеж, розмір за кількістю рядків", kind="size", pts=[[rnd(x, 3), rnd(z, 3), int(v), f] for x, z, v, f in zip(q.a, q.b, q.n, q.fire)])
    spec = dict(matching_firms_effis="in_effis", tp_fires="year", tp_candidates="visible", cv_camera_zone="year", cp_events="kind", xm_signals="kind", ar_fires="year")
    if n in spec and "lat" in d:
        c = spec[n]
        q = samp(d, 1500)
        return dict(label=DOC.get(c, c), kind="value" if c == "visible" else "cat", pts=[[rnd(x, 4), rnd(z, 4), (v if isinstance(v, str) else int(v))] for x, z, v in zip(q.lat, q.lon, q[c])])
    return None


def timeline(d, cfg):
    tg = cfg.get("target")
    if "date" in d or "dt" in d:
        s = pd.to_datetime(d["date" if "date" in d else "dt"])
    elif "start" in d:
        s = pd.to_datetime(d.start)
    elif "year" in d:
        s = None
    else:
        return None
    if s is not None:
        k, u = s.dt.strftime("%Y-%m"), "місяць"
    else:
        k, u = d.year.astype(int).astype(str), "рік"
    q = pd.DataFrame(dict(k=k.values, y=d[tg].values if tg else 0)).groupby("k").y.agg(["size", "sum"]).reset_index()
    return dict(unit=u, rows=[[a, int(b), int(c)] for a, b, c in zip(q.k, q["size"], q["sum"])])


def special(n, d):
    if n == "h3_risk_cells":
        ys = ["y_burn_2023", "y_loss_2023_hansen", "y_dndvi_2023"]
        return dict(type="agree", title="Чи згодні три джерела про згарище 2023", names=[DOC[c] for c in ys], m=[[rnd((d[a] == d[b]).mean(), 3) for b in ys] for a in ys])
    if n == "opera_gr_snapshots":
        cs = [f"s{i}" for i in range(1, 9)]
        q = d[d[cs + ["s0"]].sum(axis=1) > 0]
        return dict(type="stack", title="Пікселі з порушенням на кожну дату", dates=q.date.tolist(), names=[DOC[c] for c in cs], rows=[q[c].round(0).astype(int).tolist() for c in cs])
    if n == "fire_spread_cells":
        q = d.groupby("fire").agg(rows=("y24", "size"), y24=("y24", "mean"), days=("k", "max")).sort_values("rows", ascending=False).reset_index()
        return dict(type="fires", title="Найбільші пожежі: скільки рядків дає кожна", rows=[[a, int(b), rnd(c, 3), int(e)] for a, b, c, e in q.head(10).itertuples(index=False)])
    if n in ("cv_benchmark_aerial", "cv_benchmark_ground"):
        q = d.groupby("dataset").n_boxes.agg(frames="size", empty=lambda s: (s == 0).mean(), med="median", p95=lambda s: s.quantile(.95), mx="max", tot="sum").reset_index()
        return dict(type="cv", title="Кадри й рамки за наборами", rows=[[a, int(b), rnd(c, 3), rnd(e, 1), rnd(f, 1), int(g), int(h)] for a, b, c, e, f, g, h in q.itertuples(index=False)])
    if n == "xm_signals":
        q = d.groupby(["source", "kind", "simulated"]).size().reset_index(name="n")
        return dict(type="xm", title="Сигнали за джерелом, типом і походженням", rows=[[a, b, bool(c), int(e)] for a, b, c, e in q.itertuples(index=False)])
    return None



P_ = lambda v: f"{v * 100:.0f} %".replace(".", ",")
P1 = lambda v: f"{v * 100:.1f} %".replace(".", ",")
NF = lambda v: f"{int(v):,}".replace(",", " ")
A2 = lambda v: f"{max(v, 1 - v):.2f}".replace(".", ",")


def notes(n, d, o):
    a = o.get("assoc") or []
    top = lambda k=0: f"{a[k]['doc'] or a[k]['n']} ({A2(a[k]['auc'])})"
    if n == "burned_area_pixels":
        c, t = d[d.split == "calib"].y.mean(), d[d.split == "test"].y.mean()
        return f"{NF(len(d))} пікселів лісу парку, {P_(d.y.mean())} із них втрачені у 2023–2024. Падіння NDVI добре відділяє згарище.", [
            f"Класи майже рівні: {P_(d.y.mean())} проти {P_(1 - d.y.mean())}, тому F1 тут показовий.", f"Частки в calib і test близькі: {P_(c)} і {P_(t)}. Спліт за блоками не перекосив лейбл.",
            f"Найсильніша ознака на calib це падіння NDVI, сила {A2(next(x['auc'] for x in a if x['n'] == 'd_ndvi'))}. Лісистість майже нічого не дає ({A2(next(x['auc'] for x in a if x['n'] == 'treecover'))}).",
            f"NDVI відсутній у {P1(d.ndvi_pre.isna().mean())} пікселів, це пропуски MODIS.", "Сусідні пікселі майже однакові, тому 438 тисяч рядків це не 438 тисяч незалежних прикладів. Через це спліт і бутстреп ідуть за блоками."]
    if n == "matching_firms_effis":
        y = d.groupby("year").in_effis.mean()
        return f"{NF(len(d))} супутникових сигналів про вогонь у Греції, {P_(d.in_effis.mean())} із них лежать усередині згарища EFFIS.", [
            f"Частка сигналів у згарищах сильно гуляє за роками: від {P_(y.min())} у {y.idxmin()} до {P_(y.max())} у {y.idxmax()}. Великі пожежі 2021 і 2023 тягнуть її вгору.",
            f"Сенсори нерівні: VIIRS дає {P_((d.sensor == 'VIIRS').mean())} сигналів, MODIS решту.", f"Впевненість слабко пов'язана з лейблом: сила {A2(a[0]['auc'])} на 2021–2023.", f"Згарищ EFFIS у даних {NF(d[d.effis >= 0].effis.nunique())}. Дрібні пожежі EFFIS часто не картографує, тому сигнал поза згарищем не обов'язково хибний."]
    if n == "fire_day_daily":
        y = d.groupby(YEAR(d.date)).fire_day.sum()
        hot = pd.to_datetime(d.date).dt.month[d.fire_day == 1]
        return f"П'ять років щоденної погоди, лише {NF(d.fire_day.sum())} дні з вогнем ({P1(d.fire_day.mean())}). Це рідкісна подія.", [
            f"Від {y.min()} до {y.max()} пожежних днів на рік: {y.idxmax()} дає найбільше. На такому числі подій інтервали неминуче широкі.", f"{P_(hot.isin([7, 8, 9]).mean())} пожежних днів припадає на липень–вересень: сезон важить, але друга половина розкидана по інших місяцях.",
            f"Найсильніші ознаки на 2021–2022: {top(0)}, {top(1)}. Вони описують сухість, а не вогонь напряму.", "Багато ознак дублюють одна одну: температура, дефіцит вологи й вологість ґрунту майже рухаються разом, кореляція за модулем від 0,84 до 0,96.", "Погода береться в одній точці, центрі парку, тому різницю всередині парку вона не показує."]
    if n == "h3_risk_cells":
        return f"{NF(len(d))} гекси парку з шістьма різними лейблами. У 2022 згоріло {P_(d.y_burn_2022.mean())}, у 2023 {P_(d.y_burn_2023.mean())}: тренування й тест дуже різні.", [
            f"Головна проблема видна вже тут: лейбл 2022 покриває {P_(d.y_burn_2022.mean())} парку, а 2023 {P_(d.y_burn_2023.mean())}. Обидва рази згоріли лише {int(((d.y_burn_2022 == 1) & (d.y_burn_2023 == 1)).sum())} гекси з {int(d.y_burn_2022.sum())}.", f"Три джерела про 2023 розходяться: згода FIRMS і Hansen {P_((d.y_burn_2023 == d.y_loss_2023_hansen).mean())}, Hansen і NDVI {P_((d.y_dndvi_2023 == d.y_loss_2023_hansen).mean())}.",
            f"Лейбл 2024 порожній, у 2025 лише {int(d.y_burn_2025.sum())} гекс: після 2023 великих пожеж у парку не було, тому майбутніх перевірок ризику майже немає.", f"Найсильніші ознаки для 2022: {top(0)}, {top(1)}. Близькість до точок FIRMS це майже той самий вогонь, тому вона не доводить причину.", "Ознаки мають три копії, 2022, 2023 і 2026: кожна знає лише минуле свого року."]
    if n == "fire_spread_cells":
        q = d.groupby("fire").size()
        return f"{NF(len(d))} рядків із 28 пожеж Греції, згорить за добу лише {P1(d.y24.mean())} кандидатів. Одна тестова пожежа дає {P_(q.max() / len(d))} рядків.", [
            f"Клас дуже рідкісний: {P1(d.y24.mean())} за 24 години, {P1(d.y48.mean())} за 48.", f"Пожежі дуже нерівні: найбільша дає {NF(q.max())} рядків, медіанна {NF(q.median())}. Без групування за пожежею одна пожежа визначала б усе.",
            f"Найсильніші ознаки поза тестом: {top(0)}, {top(1)}. Відстань до вогню дає більшість сигналу.", f"З 5-го дня пожежі лейблів мало: {int(d[d.k >= 5].y24.sum())} проти {int(d[d.k < 5].y24.sum())} у перші дні. Пізні стадії модель майже не бачить.", "Погода в рядку це годинні дані наступної доби: у реальній роботі замість них буде прогноз, він гірший."]
    if n == "tp_fires":
        return f"Лише {len(d)} пожеж у парку за п'ять років. Найбільша має {NF(d.n.max())} точок, медіанна {int(d.n.median())}.", [
            f"Пожежі дуже нерівні: {P_((d.n <= 3).mean())} мають до трьох точок, а Еврос-2023 дає {P_(d.n.max() / d.n.sum())} усіх точок. Тому метрика рахує пожежі, а не точки.", f"На тест припадає 13 пожеж, на валідацію 6: через це інтервали такі широкі.", "Це перша точка FIRMS, а не момент займання: супутник бачить вогонь із запізненням.", "Датасет CP для тих самих років має 36 пожеж: він бере ширшу рамку навколо парку, а TP лише точки біля гексів."]
    if n == "tp_candidates":
        r = d[["elev", "visible"]].corr("spearman").iloc[0, 1]
        return f"300 місць-кандидатів для веж. З одного місця видно від {NF(d.visible.min())} до {NF(d.visible.max())} гексів із {NF(3664)}.", [
            f"Висота й огляд пов'язані лише помірно: кореляція {r:.2f}.".replace(".", ",", 1).replace(",", ".", 0), f"Найвище місце, {d.elev.max():.0f} м, бачить {NF(d.visible[d.elev.idxmax()])} гексів, а найоглядовіше стоїть на {d.elev[d.visible.idxmax()]:.0f} м і бачить {NF(d.visible.max())}. Тому 30 найвищих кандидатів гірші за 300.", "Кандидати обрані за висотою, з відстанню від 1 км. Доступ, дороги й дозволи тут не враховані."]
    if n == "logging_gr_pixels":
        y = d.groupby("year").y.sum()
        rw = (d.y * d.w).sum() / d.w.sum()
        return f"{NF(len(d))} пікселів лісу поза згарищами, втрата лише у {P1(d.y.mean())}, а зважено ще менше. Більшість лейблів дає один 2018 рік.", [
            f"Рядки відібрані: стовпець «вага» показує, що негативні пікселі взяті приблизно кожен десятий. Справжня частка втрат близько {P1(rw)}, а не {P1(d.y.mean())}.", f"Лейбли дуже нерівні за роками: 2018 дає {P_(y[2018] / y.sum())}, тест 2022–2024 лише {NF(y[[2022, 2023, 2024]].sum())} пікселів.", f"Рядків у 2023–2024 майже вдвічі менше: пікселі біля пожежі вирізані, бо це не рубка.",
            f"Ознаки на 2018–2021 дуже сильні: {top(0)}, {top(1)}. Але клас рідкісний, тому висока сила не дає високого F1: на тесті F1 лише 0,23.", "Лейбл Hansen це будь-яка втрата крон, не тільки рубка. Тому в таблиці це кандидати, а не доведені рубки."]
    if n == "opera_gr_snapshots":
        act = [c for c in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8") if c in d]
        t = d[act].sum(axis=1)
        gone = int((d[act + ["s0"]].sum(axis=1) == 0).sum())
        return f"{len(d)} знімків стану NASA OPERA над парком, від {d.date.min()} до {d.date.max()}. Це кількість пікселів, а не окремі пікселі.", [
            f"Пік порушень {NF(t.max())} пікселів припадає на {d.date[t.idxmax()]}. Ця хвиля це пожежа, а не рубка.", f"{gone} знімків із {len(d)} порожні: на цю дату над парком немає даних.", f"OPERA покриває лише {d.date.min()[:4]}–{d.date.max()[:4]}, тому старіші роки перевірити нею не можна.", "Статуси накопичуються: піксель лишається в них і після події, тому числа різних дат не можна просто додавати.", "Де саме спрацювало, цей датасет не показує: тут обсяг, а маски лежать окремо."]
    if n == "logging_gr_cells":
        y = d.groupby("year").y.sum()
        return f"{NF(len(d))} рядків, але рубок лише {int(d.y.sum())}, {P1(d.y.mean())}. У 2020 їх немає взагалі, у 2023 і 2024 по одній.", [
            f"Це {d.h3.nunique()} гексів по {d.year.nunique()} років: рядки не незалежні.", f"Рік 2018 дає {P_(y[2018] / y.sum())} усіх рубок. Тест 2021–2024 має лише {int(y[[2021, 2022, 2023, 2024]].sum())}.", f"Гексів із рубкою {d[d.y == 1].h3.nunique()}: повторів мало, тому «де рубали торік» працює слабко.",
            f"Найсильніші ознаки до 2020: {top(0)}, {top(1)}. Сила низька: рубки слабко пов'язані з ландшафтом.", f"Втрата ≥ 0,5 га це поріг. Дрібніші втрати сюди не входять."]
    if n == "cv_benchmark_aerial":
        v = d[d.dataset == "vedai_12cm"]
        return f"{NF(v.shape[0])} знімків VEDAI, кожен у трьох роздільностях, разом {NF(len(d))} рядків. Це ті самі кадри, а не різні.", [
            f"Незалежних кадрів {NF(len(v))}, а не {NF(len(d))}: 25 і 50 см це розмитий оригінал.", f"Рамок {NF(v.n_boxes.sum())}, медіана {int(v.n_boxes.median())} на кадр, максимум {int(v.n_boxes.max())}.", "Порожніх кадрів немає: тут неможливо перевірити хибні тривоги на порожньому знімку, а в парку порожні майже всі.", "VEDAI це переважно села й дороги, а не ліс."]
    if n == "cv_benchmark_ground":
        f, e = d[d.dataset == "dfire"], d[d.dataset == "ena24"]
        return f"{NF(len(f))} кадрів D-Fire з димом і вогнем і {NF(len(e))} кадрів ENA24 із фотопасток. Машин на ENA24 лише {int((e.n_boxes > 0).sum())}.", [
            f"D-Fire збалансований: порожніх {P_((f.n_boxes == 0).mean())}.", f"ENA24 майже порожній: {P_((e.n_boxes == 0).mean())} кадрів без машин. Тому ROC-AUC тут високий, але спирається лише на {int((e.n_boxes > 0).sum())} кадри.", "Це фото, відео в еталоні немає.", "D-Fire це здебільшого близькі кадри, а вежа бачить дим за кілометри."]
    if n == "cv_camera_zone":
        return f"{NF(len(d))} гексів, де FIRMS бачив вогонь: {int((d.year == 2022).sum())} у 2022 і {int((d.year == 2023).sum())} у 2023.", [
            f"Гекс рахується раз за рік: {NF(d.h3.nunique())} унікальних гексів.", "Це справжні місця, але напрямок із камери в цьому датасеті немає: він симулюється в експерименті.", "Майже всі рядки дає одна пожежа Еврос-2023, тому тест це фактично одна подія."]
    if n == "cp_events":
        lg, fr = d[d.kind == "logging"], d[d.kind == "fire"]
        return f"{len(lg)} рубок і {len(fr)} пожеж. Половина подій лежить ближче {d.km.median() * 1000:.0f} м від дороги.", [
            f"Рубки ближчі до доріг за пожежі: медіана {lg.km.median() * 1000:.0f} м проти {fr.km.median() * 1000:.0f} м.", f"Рубки сильно зміщені в минуле: 2018 рік дає {P_((lg.year == 2018).mean())}, а 2023–2024 лише {int(lg.year.isin([2023, 2024]).sum())}. Тому перший тест не мав сенсу.", "Рубка це гекс із лейблом LG, тому одна велика рубка може дати кілька подій.", "У 2020 рубок немає: це відповідає LG."]
    if n == "xm_signals":
        sim = d.simulated.mean()
        return f"{len(d)} сигналів потоку демо, із них реальних лише {int((~d.simulated).sum())}. Симульовано {P_(sim)}.", [
            f"Це сценарний потік: еталон узятий із генератора, тому на цих даних не можна стверджувати якість на реальних сигналах.", f"Подій в еталоні {d.truth.nunique()}, у середньому {len(d) / d.truth.nunique():.1f} сигнала на подію.".replace(".", ",", 1), f"Спліт: {int((d.part == 'val').sum())} на валідації, {int((d.part == 'test').sum())} на тесті.", f"Реальні сигнали лише про вогонь із камери: {int(((~d.simulated) & (d.kind == 'fire')).sum())}."]
    if n == "ar_fires":
        q = d.n
        return f"{NF(len(d))} пожеж Греції за п'ять років. {P_((q <= 2).mean())} із них мають одну-дві точки.", [
            f"Розподіл дуже нерівний: типова пожежа це одна точка, найбільша має {NF(q.max())}.", f"Роки схожі: від {NF(d.groupby('year').size().min())} до {NF(d.groupby('year').size().max())} пожеж, тому перевірка на 2024–2025 не на порожньому місці.", f"Блоків для бутстрепа {d.block.nunique()}.", "Це вся Греція, а не лише парк: у самому парку пожеж замало для ареалів."]
    return "", []


def parks_eda(d, m, feats, tg):
    """Чим парки відрізняються в датасеті: рядки, лейбл, медіани ознак по парках і найбільші відмінності.
    Відмінність ознаки — найбільша стандартизована різниця середніх між двома парками (|Δμ| / σ усіх рядків)."""
    from eaglewings.ds import park as PKM
    PK = [k for k in PKM.ALL if (d.park == k).any()]
    nm = lambda k: PKM.ALL[k]["short"]
    rows = [dict(park=k, name=nm(k), rows=int((d.park == k).sum()), pos=int(d.loc[d.park == k, tg].sum()) if tg else None,
                 prev=rnd(d.loc[d.park == k, tg].mean(), 4) if tg else None) for k in PK]
    fs = []
    for c in [x for x in feats if pd.api.types.is_numeric_dtype(d[x]) and d[x].nunique() > 2]:
        sd = float(d[c].std()) or 1.0
        mu = {k: float(d.loc[d.park == k, c].mean()) for k in PK}
        a_, b_ = max(mu, key=mu.get), min(mu, key=mu.get)
        fs.append(dict(n=c, doc=DOC.get(c, ""), med={k: rnd(d.loc[d.park == k, c].median(), 4) for k in PK},
                       q={k: [rnd(d.loc[d.park == k, c].quantile(.25), 4), rnd(d.loc[d.park == k, c].quantile(.75), 4)] for k in PK},
                       smd=rnd(abs(mu[a_] - mu[b_]) / sd, 3), hi=a_, lo=b_))
    fs.sort(key=lambda x: -(x["smd"] or 0))
    diffs = []
    if tg:
        pr = [r for r in rows if r["prev"] is not None]
        if pr:
            a_, b_ = max(pr, key=lambda r: r["prev"]), min(pr, key=lambda r: r["prev"])
            diffs.append(f"Частка позитивних «{tg}»: найбільша в парку «{a_['name']}» ({P1(a_['prev'])}), найменша в «{b_['name']}» ({P1(b_['prev'])})."
                         + (" В одному з парків позитивних немає: метрики там не рахуються." if b_["pos"] == 0 else ""))
    for x in fs[:3]:
        if (x["smd"] or 0) >= .3:
            g_ = lambda v: f"{v:.1f}".replace(".", ",") if v is not None and abs(v) >= 1 else (f"{v:.3f}".replace(".", ",") if v is not None else "—")
            diffs.append(f"{(x['doc'] or x['n'])[:1].upper() + (x['doc'] or x['n'])[1:]}: у парку «{nm(x['hi'])}» вище, ніж у «{nm(x['lo'])}» "
                         f"(медіани {g_(x['med'][x['hi']])} і {g_(x['med'][x['lo']])}, різниця {x['smd']:.2f} σ).".replace(f"{x['smd']:.2f}", f"{x['smd']:.2f}".replace(".", ",")))
    small = [r for r in rows if r["rows"] < .1 * len(d)]
    if small:
        diffs.append("Мало рядків: " + ", ".join(f"«{r['name']}» {r['rows']:,}".replace(",", " ") for r in small) + " — оцінки в цьому парку найменш певні.")
    return dict(rows=rows, feats=fs[:12], diffs=diffs, parents=[f"{p['name']}@v{p['version']}" for p in m.get("parents", [])])


def base_name(n):
    """Датасет одного парку <назва>_<парк> бере налаштування EDA й опис походження своєї назви."""
    return n if n in CFG else next((b for b in sorted(CFG, key=len, reverse=True) if n.startswith(b + "_")), n)


def build(n):
    d, m, nv = latest(n)
    n0 = base_name(n)
    cfg = CFG[n0]
    tg = cfg.get("target")
    DOC.update(y=dict(burned_area_pixels="втрата лісу Hansen 2023–2024", logging_gr_pixels="втрата лісу Hansen у цьому році", logging_gr_cells="нова втрата ≥ 0,5 га").get(n0, "лейбл"))
    if n0 == "logging_gr_pixels":
        DOC.update(ndvi_pre="NDVI до", ndvi_post="NDVI після")
    col = [column(d, c, m, cfg) for c in d.columns]
    if n0 == "burned_area_pixels":
        d["d_ndvi"] = d.ndvi_pre - d.ndvi_post                                 # похідна лише для графіків
    feats = [c for c in (cfg.get("feats") or []) if c in d]
    o = dict(name=n, version=m["version"], versions=nv, digest=m["digest"], rows=len(d), cols=len(col), unit=m.get("unit"), description=m.get("description"), sources=m.get("sources"), params=m.get("params"), created=m.get("created"),
             code=cfg["code"], task=cfg["task"], columns=col, dup=int(d[[c["n"] for c in col]].duplicated().sum()), miss=rnd(d[[c["n"] for c in col]].isna().mean().mean(), 4), tnote=cfg.get("tnote"),
             hist=[h for h in (hist(d, c, tg if tg and d[tg].nunique() == 2 else None) for c in [x for x in feats if d[x].nunique() > 2][:16]) if h], cats=[], corr=corr(d, feats), time=None if n0 == "opera_gr_snapshots" else timeline(d, cfg), geo=geo(n0, d), special=special(n0, d))
    for c in d.columns:
        if d[c].nunique() <= 14 and not pd.api.types.is_float_dtype(d[c]) and c not in (m.get("targets") or []) and c != "year" and c != cfg.get("group"):
            o["cats"].append(dict(n=c, doc=DOC.get(c, ""), rows=[[str(a), int(b)] for a, b in d[c].value_counts().items()]))
    if tg:
        o["target"] = dict(n=tg, doc=DOC.get(tg, ""), pos=int(d[tg].sum()), prev=rnd(d[tg].mean(), 4))
        o["targets"] = [dict(n=t, doc=DOC.get(t, ""), pos=int(d[t].sum()), prev=rnd(d[t].mean(), 4)) for t in (m.get("targets") or []) if t in d]
        o["assoc"] = assoc(d, feats, tg, cfg.get("train"))
    g = cfg.get("group")
    if g:
        k = YEAR(d.date) if g == "year" and "date" in d else d[g]
        if tg:
            q = pd.DataFrame(dict(g=k.values, y=d[tg].values)).groupby("g").y.agg(["size", "sum"]).reset_index()
            o["bygroup"] = dict(n=g, rows=[[str(a), int(b), int(c)] for a, b, c in zip(q.g, q["size"], q["sum"])])
        else:
            q = pd.Series(k.values).value_counts().sort_index()
            o["bygroup"] = dict(n=g, rows=[[str(a), int(b), 0] for a, b in q.items()])
    if "park" in d and d.park.nunique() > 1:
        o["parks"] = parks_eda(d, m, feats, tg)
    o["short"], o["seen"] = notes(n0, d, o)
    L = LINEAGE[n0]
    o["src"] = [dict(id=x, name=SRC[x]["name"]) for x in L["src"]]
    o["also"] = [dict(id=x, name=SRC[x]["name"]) for x in L.get("also", [])]
    o["from_ds"] = L["ds"]
    o["steps"] = L["steps"]
    return o


if __name__ == "__main__":
    # Лише збирає готові результати EDA датасетів (їх рахує ds/pages.dataset, коли задача або крок datasets бере датасет).
    E = ROOT / "data" / "processed" / "eda" / "datasets"
    built = lambda n: (ROOT / "data" / "datasets" / n).exists()           # лише датасети гілок цього проєкту
    G = {t: [n for n in v if built(n)] for t, v in G.items()}
    G = {t: v for t, v in G.items() if v}
    names = [n for v in G.values() for n in v]
    miss = [n for n in names if not (E / f"{n}.json").exists()]
    if miss:
        sys.exit(f"немає EDA датасетів {miss}: спершу кроки datasets і задачі")
    out = dict(groups=[dict(t=t, items=v) for t, v in G.items()], data={n: json.loads((E / f"{n}.json").read_text()) for n in names})
    for n, o in out["data"].items():
        print(n, o["rows"], o["cols"], "dup", o["dup"], "miss", o["miss"], o.get("target", {}).get("prev"), [(a["n"], a["auc"]) for a in o.get("assoc", [])[:4]])
    (ROOT / "web" / "science" / "eda.json").write_text(json.dumps(out, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, "item") else str(v), separators=(",", ":")))
