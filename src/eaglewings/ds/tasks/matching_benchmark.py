"""T13 · Крос-джерельний матчинг на реальних даних: VIIRS + MODIS проти офіційних периметрів EFFIS.

Задача      матчинг сигналів працює на моковому потоці, тож його якість звідти не виміряти. Тут той самий алгоритм групує реальні сигнали двох
            незалежних супутникових сенсорів про пожежі, а правильну відповідь дає незалежний третій джерело — картографовані EFFIS згарища.
Дані        NASA FIRMS, Греція, 2021–2025: VIIRS S-NPP (n, h) і MODIS (≥ 30). Еталон: EFFIS Burnt Areas, COUNTRY = EL.
            Сигнал належить згарищу, якщо лежить у його полігоні + 500 м і в датах [початок − 2 дні; кінець + 2 дні]. Оцінюються сигнали, що мають еталон.
Протокол    вибір параметрів — 2021–2023, тест — 2024–2025. 95 % CI — bootstrap по згарищах.
Метрики     BCubed F1 — основна: кластеризація оцінюється по кожному сигналу, великі пожежі не домінують. ARI — допоміжна.
            merge_rate — частка згарищ, які бачили обидва сенсори, що зведені в одну подію (сама суть крос-джерельного матчингу);
            over_merge — частка подій, що зліпили кілька різних згарищ.
Гіпотези
  singleton     без матчингу: кожен сигнал — окрема подія
  day_dbscan    поточна логіка контуру: DBSCAN у межах (тип, день) — багатоденна пожежа розпадається на події
  st_dbscan     простір + час: сигнали однієї події можуть належати різним дням
  transfer      ST-DBSCAN із налаштуваннями іншого проєкту (REGION["transfer"]) як є, без підбору; у виборі лідера не бере участі
Калібрування надійність джерел r_s у noisy-OR оцінюється на 2021–2023 максимальною правдоподібністю (чи збіглася подія зі згарищем EFFIS) і порівнюється на 2024–2025
            із заданими вручну вагами: ROC-AUC і Brier. Обмеження: EFFIS картографує великі пожежі (~від 30 га), тож «не збіглося» ≠ «хибний сигнал», і r_s занижена.
Вихід       data/processed/matching_calibration.json: параметри матчингу й r_s.
            Надійність камери, фотопастки, гарячої лінії й соцмереж цей набір не перевіряє: вона лишається експертною.
"""
import glob
import inspect
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, brier_score_loss, roc_auc_score

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS, EFFIS_COUNTRY, protocol  # noqa: E402  (параметри території з конфігу)
from .. import datasets, metrics, tracking, transfer
from ..common import P, RAW, UTM
from ..tracking import ml

EXPERIMENT = "T13 · Cross-source matching (VIIRS + MODIS проти EFFIS)"
VAL, TEST = protocol("FM", [2021, 2022, 2023], [2024, 2025])   # роки з config/regions.py або кейсу Фракії
BUF_M, MARGIN_D = 500, 2
HAND_R = dict(VIIRS=.7, MODIS=.7)                 # вага «супутник», задана вручну
GRID_KM, GRID_D = (.5, 1., 1.5, 2., 3., 5.), (1., 2., 3., 5.)


def signals():
    rows = []
    for f in sorted(glob.glob(str(RAW / "firms" / f"*_{COUNTRY}.csv"))):
        d = pd.read_csv(f)
        modis = "modis" in f
        ok = pd.to_numeric(d.confidence, errors="coerce") >= 30 if modis else d.confidence.isin(["n", "h"])
        d = d[ok].copy()
        d["sensor"] = "MODIS" if modis else "VIIRS"
        d["conf"] = pd.to_numeric(d.confidence, errors="coerce") / 100 if modis else d.confidence.map(dict(n=.65, h=.95))
        rows.append(d[["latitude", "longitude", "acq_date", "acq_time", "sensor", "conf", "frp"]])
    s = pd.concat(rows, ignore_index=True)
    la0, lo0, la1, lo1 = map(float, EFFIS_BBOX.split(","))            # лише точки в рамці, де є еталон EFFIS
    s = s[s.latitude.between(la0, la1) & s.longitude.between(lo0, lo1)].reset_index(drop=True)
    s["dt"] = pd.to_datetime(s.acq_date) + pd.to_timedelta(s.acq_time // 100, "h") + pd.to_timedelta(s.acq_time % 100, "m")
    s["year"] = s.dt.dt.year
    g = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.longitude, s.latitude), crs=4326).to_crs(UTM)
    return g.sort_values("dt").reset_index(drop=True)


def effis(buf=BUF_M):
    fs = [gpd.read_file(f) for f in sorted(glob.glob(str(RAW / "benchmarks" / "effis" / "*.geojson")))]
    e = gpd.GeoDataFrame(pd.concat(fs, ignore_index=True), crs=4326)
    e = e[e.COUNTRY.eq(EFFIS_COUNTRY)].to_crs(UTM).reset_index(drop=True)
    e["start"], e["end"] = pd.to_datetime(e.FIREDATE), pd.to_datetime(e.FINALDATE)
    e["area_ha"] = pd.to_numeric(e.AREA_HA, errors="coerce")
    e["geometry"] = e.buffer(0).buffer(buf) if buf else e.buffer(0)
    return e[["id", "start", "end", "area_ha", "geometry"]]


def label(sig, e, margin=MARGIN_D):
    """Еталонне згарище кожного сигналу (−1 = немає): просторово + за датами; при кількох — найближчий початок."""
    j = gpd.sjoin(sig[["dt", "geometry"]], e, how="inner", predicate="within")
    d = pd.Timedelta(days=margin)
    j = j[(j.dt >= j.start - d) & (j.dt <= j.end + d + pd.Timedelta(days=1))]
    j = j.assign(gap=(j.dt - j.start).abs()).sort_values("gap")
    t = pd.Series(-1, index=sig.index)
    first = j[~j.index.duplicated()]
    t.loc[first.index] = first.index_right.values
    return t.values


def cluster(sig, a):
    x, y = sig.geometry.x.values / 1000, sig.geometry.y.values / 1000
    if a["kind"] == "singleton":
        return np.arange(len(sig))
    if a["kind"] == "day":
        out = np.zeros(len(sig), int)
        top = 0
        for _, i in sig.groupby(sig.dt.dt.normalize()).indices.items():
            lab = DBSCAN(eps=a["km"], min_samples=1).fit_predict(np.c_[x[i], y[i]])
            out[i] = lab + top
            top += lab.max() + 1
        return out
    t = (sig.dt - sig.dt.min()).dt.total_seconds().values / 86400
    return DBSCAN(eps=a["km"], min_samples=1).fit_predict(np.c_[x, y, t * a["km"] / a["days"]])   # st і transfer


def bcubed(p, t):
    """BCubed precision / recall / F1 для міток p і еталону t (через таблицю збігів)."""
    tab = pd.crosstab(p, t).values.astype(float)
    n = tab.sum()
    prec = (tab ** 2 / np.maximum(tab.sum(1, keepdims=True), 1)).sum() / n
    rec = (tab ** 2 / np.maximum(tab.sum(0, keepdims=True), 1)).sum() / n
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def scores(sig, lab, truth, mask):
    if not mask.any():                                    # на території немає згарищ EFFIS із сигналом у цих роках: міряти нічого
        return dict(bcubed_f1=np.nan, bcubed_precision=np.nan, bcubed_recall=np.nan, ari=np.nan, merge_rate=np.nan, over_merge=np.nan, events=0, fires=0, n=0)
    p, t, sen = lab[mask], truth[mask], sig.sensor.values[mask]
    pr, rc, f1 = bcubed(p, t)
    d = pd.DataFrame(dict(p=p, t=t, s=sen))
    both = d.groupby("t").filter(lambda g: g.s.nunique() == 2)
    m = both.groupby(["t", "s"]).p.agg(lambda v: v.mode().iat[0]).unstack()
    return dict(bcubed_f1=f1, bcubed_precision=pr, bcubed_recall=rc, ari=adjusted_rand_score(t, p), merge_rate=float((m.VIIRS == m.MODIS).mean()) if len(m) else np.nan,
                over_merge=float((d.groupby("p").t.nunique() > 1).mean()), events=d.p.nunique(), fires=d.t.nunique(), n=len(d))


def event_table(sig, lab, truth):
    """Подія → сила кожного сенсора (найкращий conf + 0,05 за кожен повтор, як у контурі), чи збіглася зі згарищем, рік."""
    d = sig[["sensor", "conf", "year"]].assign(p=lab, t=truth)
    g = d.groupby(["p", "sensor"]).conf.agg(["max", "size"])
    st = (g["max"] + .05 * (g["size"] - 1)).clip(upper=1).unstack(fill_value=0.)
    for k in ("VIIRS", "MODIS"):
        if k not in st:
            st[k] = 0.
    e = d.groupby("p").agg(hit=("t", lambda v: int((v >= 0).any())), year=("year", "first"))
    return e.join(st)


def noisy_or(e, r):
    return 1 - (1 - r["VIIRS"] * e.VIIRS.values) * (1 - r["MODIS"] * e.MODIS.values)


def fit_r(e):
    """Максимальна правдоподібність r_s на сітці (2 параметри)."""
    grid, y, best, bl = np.arange(.05, 1.0, .05), e.hit.values, None, np.inf
    for a in grid:
        for b in grid:
            q = np.clip(noisy_or(e, dict(VIIRS=a, MODIS=b)), 1e-4, 1 - 1e-4)
            ll = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
            if ll < bl:
                best, bl = (float(a), float(b)), ll
    return dict(VIIRS=best[0], MODIS=best[1])


def run(n_boot=300):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: FIRMS signals, EFFIS reference')
    sig, e = signals(), effis()
    truth = label(sig, e)
    strict = label(sig, effis(0), 0)                                          # перевірка циклічності: без буфера й без запасу днів
    yr = sig.year.values
    va, te = np.isin(yr, VAL) & (truth >= 0), np.isin(yr, TEST) & (truth >= 0)
    dv = datasets.save(pd.DataFrame(dict(dt=sig.dt.astype(str), sensor=sig.sensor, conf=sig.conf, lat=sig.latitude, lon=sig.longitude, year=sig.year, effis=truth, in_effis=(truth >= 0).astype(int))),
                       "matching_firms_effis", unit="супутниковий сигнал про вогонь (VIIRS або MODIS)", targets=["in_effis"], features=["sensor", "conf", "lat", "lon"],
                       description="Реальні сигнали FIRMS (Греція, 2021–2025) з еталонним згарищем EFFIS для перевірки крос-джерельного матчингу.",
                       params=dict(buffer_m=BUF_M, margin_days=MARGIN_D, val=VAL, test=TEST), sources=["NASA FIRMS VIIRS S-NPP, MODIS", "EFFIS Burnt Areas (Copernicus)"])
    pflow.mark('2. rule grid on validation')
    grid = [dict(name="singleton", kind="singleton", hyp="Без матчингу")]
    grid += [dict(name=f"day_{k}km", kind="day", km=k, hyp="Поточна логіка: DBSCAN у межах дня") for k in GRID_KM]
    grid += [dict(name=f"st_{k}km_{d:g}d", kind="st", km=k, days=d, hyp="Простір + час") for k in GRID_KM for d in GRID_D]
    labs = {a["name"]: cluster(sig, a) for a in grid}
    val = pd.DataFrame([dict(name=a["name"], kind=a["kind"], **scores(sig, labs[a["name"]], truth, va)) for a in grid])
    pick = {k: g.sort_values("bcubed_f1", ascending=False).iloc[0]["name"] for k, g in val.groupby("kind")}
    pick["day_current"] = "day_1.0km"
    tm = transfer.matching()                                                 # налаштування іншого проєкту як є: лише передбачає, нічого не підбирає
    if tm and tm.get("km") and tm.get("days"):
        a = dict(name=transfer.label(f"st_{tm['km']:g}km_{tm['days']:g}d"), kind="transfer", km=float(tm["km"]), days=float(tm["days"]), hyp=transfer.hyp())
        grid.append(a)
        labs[a["name"]] = cluster(sig, a)
        val = pd.concat([val, pd.DataFrame([dict(name=a["name"], kind="transfer", **scores(sig, labs[a["name"]], truth, va))])], ignore_index=True)
        pick["transfer"] = a["name"]
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", "in_effis", files=True)
        ml.log_params(dict(val=VAL, test=TEST, buffer_m=BUF_M, margin_days=MARGIN_D, signals=len(sig), matched=int((truth >= 0).sum()), fires=len(np.unique(truth[truth >= 0])), n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/matching_benchmark.py")
        ml.log_table(val, "validation_grid.json")
        pflow.mark('3. chosen rules on test')
        for fam, name in pick.items():
            a = next(x for x in grid if x["name"] == name)
            with ml.start_run(run_name=f"{fam} · {name}", nested=True):
                ml.set_tags(dict(approach=name, kind=a["kind"], hypothesis=a["hyp"], role="approach"))
                tracking.log_dataset(dv, "evaluation", "in_effis")
                ml.log_params({k: v for k, v in a.items() if k in ("km", "days")})
                lab = labs[name]
                m = scores(sig, lab, truth, te)
                idx = np.flatnonzero(te)
                groups = truth[idx]
                lo, hi = metrics.block_bootstrap(groups, lambda i: bcubed(lab[idx][i], truth[idx][i])[2], n=n_boot)
                ts = np.isin(yr, TEST) & (strict >= 0)
                m |= dict(bcubed_f1_ci_lo=lo, bcubed_f1_ci_hi=hi, val_bcubed_f1=float(val[val.name == name].bcubed_f1.iloc[0]),
                          bcubed_f1_strict=float(bcubed(lab[ts], strict[ts])[2]), n_strict=int(ts.sum()),
                          unmatched_share=float(1 - te.sum() / np.isin(yr, TEST).sum()))
                ml.log_metrics(tracking.clean(m))
                rows.append(dict(family=fam, approach=name, kind=a["kind"], hypothesis=a["hyp"], **m))
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "hypothesis"}, flush=True)
        pflow.mark('4. winner, difference vs baseline')
        res = pd.DataFrame(rows)
        res["few"] = res.fires < 10                                           # «замало»: менше 10 згарищ у тесті
        champ = res[~res.kind.isin(["singleton", "transfer"])].merge(val[["name", "bcubed_f1"]].rename(columns=dict(name="approach", bcubed_f1="v")), on="approach").sort_values("v", ascending=False).iloc[0].approach
        res["champion"] = res.approach == champ
        idx = np.flatnonzero(te)                                              # парна різниця з «без зведення» на тих самих згарищах
        dd, dlo, dhi = metrics.paired_block_bootstrap(truth[idx], lambda i: bcubed(labs[champ][idx][i], truth[idx][i])[2], lambda i: bcubed(labs["singleton"][idx][i], truth[idx][i])[2], n=n_boot)
        res.loc[res.champion, ["diff_base", "diff_lo", "diff_hi"]] = [dd, dlo, dhi]
        lab = labs[champ]
        ev = event_table(sig, lab, truth)
        r_fit = fit_r(ev[ev.year.isin(VAL)])
        eve = ev[ev.year.isin(TEST)]
        y = eve.hit.values
        cal = {}
        for nm, r in dict(hand=HAND_R, fitted=r_fit).items():
            q = noisy_or(eve, r)
            cal[nm] = dict(roc_auc=roc_auc_score(y, q), brier=brier_score_loss(y, q), mean_pred=float(q.mean()))
        cal["base_rate"] = float(y.mean())
        ml.log_metrics(tracking.clean({f"{k}_{a}": b for k, v in cal.items() if isinstance(v, dict) for a, b in v.items()}))
        a = next(x for x in grid if x["name"] == champ)
        out = dict(champion=champ, km=a.get("km"), days=a.get("days"), r_fitted=r_fit, r_hand=HAND_R, reliability_test=cal, test=res[res.champion].iloc[0].to_dict())
        (P / "matching_calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        print(json.dumps(out["reliability_test"], ensure_ascii=False), r_fit, flush=True)
        ml.set_tags(dict(champion=champ, selection="max BCubed F1 на 2021–2023"))
        tracking.log_fig(_fig(val), "validation_grid")
        pflow.mark('5. save')
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champ, r_fitted=str(r_fit))


def _fig(val):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    v = val.sort_values("bcubed_f1", ascending=False).head(14)
    ax.barh(v.name, v.bcubed_f1, color="#2a7de1")
    ax.invert_yaxis()
    ax.set(xlabel="BCubed F1, 2021–2023", title="T13 · вибір параметрів матчингу")
    fig.tight_layout()
    return fig
