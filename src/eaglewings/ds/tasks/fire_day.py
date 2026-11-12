"""T2 · Прогноз пожежного дня за погодою.

Задача      щоденна бінарна класифікація для зони парку (+3 км):
              fire_day — у цей день є хоча б одна детекція FIRMS (активна пожежа);
              ignition — у цей день починається нова подія ST-DBSCAN (нове займання).
Лейбли      NASA FIRMS (VIIRS + MODIS), фільтр впевненості; події — ST-DBSCAN (1.5 км / 2 доби).
Ознаки      NASA POWER у центрі парку: T max, RH, вітер, опади, вологість ґрунту + похідні (VPD, сухі дні поспіль,
            суми/середні 7–30 діб, індекс Нестерова), сезонність (sin/cos дня року). Усе відомо на ранок дня (без витоку).
Протокол    expanding-window forward chaining за роками: тест 2023 (навчання 2021–22), тест 2024 (2021–23), тест 2025 (2021–24).
            Гіперпараметри зафіксовані заздалегідь; оцінка — на зібраних out-of-fold прогнозах 2023–2025 + по кожному фолду.
Метрики     PR-AUC — основна: позитивів ~2–3 %, ROC-AUC на такому дисбалансі оптимістичний;
            recall@10 % — частка пожежних днів, якщо оголошувати «тривогу» в 10 % найризиковіших днів (операційний бюджет);
            Brier — калібрування (лише для ймовірнісних моделей); ROC-AUC — допоміжна.
            95 % CI — moving-block bootstrap (14-денні блоки: пожежні дні йдуть серіями).
Парки      кілька парків (ds/regions.py): рядок це парк × день, погода NASA POWER у точці свого парку. Моделі вчаться на всіх
            парках разом, кліматологія окремо для кожного парку; тест рахується для всіх разом і окремо для кожного парку (*_<парк>).
Вибір       чемпіон = max PR-AUC на 2022 (навчання лише на 2021) серед «прогнозних» підходів (persistence лише орієнтир).
            transfer_FG_day — готова модель іншого проєкту (REGION["transfer"]) як є: стоїть поруч на тих самих днях, у виборі не бере участі.
            Раніше вибір робився за OOF 2023–2025, тобто на тесті; це виправлено, тест у виборі не бере участі.
"""
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import pflow
from .. import common, datasets, metrics, tracking, transfer
from ..tracking import ml

EXPERIMENT = "T2 · Fire-day forecast (погода → пожежний день)"
from eaglewings.ds.regions import protocol  # noqa: E402
VAL_YEARS, TEST_YEARS = protocol("FG_day", [2022], [2023, 2024, 2025])   # вибір підходу на VAL (навчання на роках до нього), тест — далі
WXF = ["T2M_MAX", "RH2M", "WS10M", "PRECTOTCORR", "GWETTOP", "VPD", "dry_days", "prec_30", "VPD_7", "T_7", "GWET_7", "WS_3", "nesterov"]
SEAS = ["season_sin", "season_cos"]
TARGETS = {"fire_day": "день з активною пожежею", "ignition": "день нового займання"}


def fwi(T, RH, W, P, month):
    """Канадський індекс пожежної небезпеки FWI (Van Wagner, 1987), денний крок.

    T °C, RH %, W км/год, P мм. Без полуденних спостережень беремо денні значення NASA POWER: T max і середню RH,
    тому абсолютні значення наближені, а порядок днів зберігається.
    """
    LE = [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6]
    LF = [-1.6, -1.6, -1.6, .9, 3.8, 5.8, 6.4, 5, 2.4, .4, -1.6, -1.6]
    ffmc, dmc, dc, out = 85., 6., 15., []
    for t, h, w, p, m in zip(T, RH, W, P, month):
        h = min(max(h, 0), 100)
        mo = 147.2 * (101 - ffmc) / (59.5 + ffmc)                            # FFMC
        if p > .5:
            rf = p - .5
            mo = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf)) + (.0015 * (mo - 150) ** 2 * rf ** .5 if mo > 150 else 0)
            mo = min(mo, 250)
        ed = .942 * h ** .679 + 11 * np.exp((h - 100) / 10) + .18 * (21.1 - t) * (1 - np.exp(-.115 * h))
        if mo > ed:
            ko = .424 * (1 - (h / 100) ** 1.7) + .0694 * w ** .5 * (1 - (h / 100) ** 8)
            mo = ed + (mo - ed) * 10 ** (-ko * .581 * np.exp(.0365 * t))
        else:
            ew = .618 * h ** .753 + 10 * np.exp((h - 100) / 10) + .18 * (21.1 - t) * (1 - np.exp(-.115 * h))
            if mo < ew:
                kl = .424 * (1 - ((100 - h) / 100) ** 1.7) + .0694 * w ** .5 * (1 - ((100 - h) / 100) ** 8)
                mo = ew - (ew - mo) * 10 ** (-kl * .581 * np.exp(.0365 * t))
        ffmc = min(max(59.5 * (250 - mo) / (147.2 + mo), 0), 101)
        tt = max(t, -1.1)                                                     # DMC
        if p > 1.5:
            re = .92 * p - 1.27
            mo_ = 20 + np.exp(5.6348 - dmc / 43.43)
            b = 100 / (.5 + .3 * dmc) if dmc <= 33 else (14 - 1.3 * np.log(dmc) if dmc <= 65 else 6.2 * np.log(dmc) - 17.2)
            mr = mo_ + 1000 * re / (48.77 + b * re)
            dmc = max(244.72 - 43.43 * np.log(mr - 20), 0)
        dmc = dmc + 1.894 * (tt + 1.1) * (100 - h) * LE[m - 1] * 1e-6 * 100
        tt = max(t, -2.8)                                                     # DC
        if p > 2.8:
            rd = .83 * p - 1.27
            qr = 800 * np.exp(-dc / 400) + 3.937 * rd
            dc = max(400 * np.log(800 / qr), 0)
        dc = dc + max(.5 * (.36 * (tt + 2.8) + LF[m - 1]), 0)
        fm = 147.2 * (101 - ffmc) / (59.5 + ffmc)                            # ISI, BUI, FWI
        isi = .208 * np.exp(.05039 * w) * 91.9 * np.exp(-.1386 * fm) * (1 + fm ** 5.31 / 4.93e7)
        bui = .8 * dmc * dc / (dmc + .4 * dc) if dmc <= .4 * dc else dmc - (1 - .8 * dc / (dmc + .4 * dc)) * (.92 + (.0114 * dmc) ** 1.7)
        bui = max(bui, 0)
        fd = .626 * bui ** .809 + 2 if bui <= 80 else 1000 / (25 + 108.64 * np.exp(-.023 * bui))
        bb = .1 * isi * fd
        out.append(np.exp(2.72 * (.434 * np.log(bb)) ** .647) if bb > 1 else bb)
    return np.array(out)


def weather_features(wx):
    """Похідні погодні ознаки з добового ряду NASA POWER ."""
    wx = wx.copy()
    es = 0.6108 * np.exp(17.27 * wx.T2M_MAX / (wx.T2M_MAX + 237.3))
    wx["VPD"] = es * (1 - wx.RH2M / 100)
    rain = wx.PRECTOTCORR >= 1
    wx["dry_days"] = rain.groupby(rain.cumsum()).cumcount()
    wx["prec_30"] = wx.PRECTOTCORR.rolling(30, 1).sum()
    wx["VPD_7"] = wx.VPD.rolling(7, 1).mean()
    wx["T_7"] = wx.T2M_MAX.rolling(7, 1).mean()
    wx["GWET_7"] = wx.GWETTOP.rolling(7, 1).mean()
    wx["WS_3"] = wx.WS10M.rolling(3, 1).max()
    # індекс Нестерова: Σ T·(T − Td) за дні від останнього дощу > 3 мм (Td — точка роси за Магнусом)
    g = 17.27 * wx.T2M / (237.7 + wx.T2M) + np.log(wx.RH2M.clip(1, 100) / 100)
    td = 237.7 * g / (17.27 - g)
    term = (wx.T2M_MAX * (wx.T2M_MAX - td)).clip(lower=0)
    wet = wx.PRECTOTCORR > 3
    wx["nesterov"] = term.groupby(wet.cumsum()).cumsum().where(~wet, 0)
    doy = wx.index.dayofyear
    wx["fwi"] = fwi(wx.T2M_MAX.values, wx.RH2M.values, wx.WS10M.values * 3.6, wx.PRECTOTCORR.values, wx.index.month.values)
    wx["season_sin"], wx["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return wx


def build_dataset():
    wx = weather_features(common.power())

    B = common.base()
    ev, _ = common.default_events()
    wx["fire_day"] = wx.index.isin(B.firms.dt.dt.normalize()).astype(int)
    wx["ignition"] = wx.index.isin(ev.start.dt.normalize()).astype(int)
    wx["fire_day_yesterday"] = wx.fire_day.shift(1, fill_value=0)
    df = wx.reset_index(names="date").round(4)
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    return datasets.save(
        df, "fire_day_daily", unit="доба (зона парку + 3 км)",
        description="Щоденна погода NASA POWER (центр парку) з похідними індексами пожежної небезпеки та лейблами пожежного дня / нового займання з FIRMS.",
        targets=list(TARGETS), features=WXF + SEAS + ["fire_day_yesterday"],
        params=dict(power_point="park centroid (MultiPolygon: representative point)", firms_filter="VIIRS n/h, MODIS>=30", events="ST-DBSCAN 1.5 km / 2 d", period="2021-01-01..2025-12-31"),
        sources=["NASA POWER daily (AG community)", "NASA FIRMS VIIRS S-NPP + MODIS"])


def _lr(cw="balanced"):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight=cw))


APPROACHES = [
    dict(name="persistence", kind="persistence", hyp="Нульова: сьогодні горить, якщо горіло вчора (потребує вчорашнього спостереження)", forecast=False),
    dict(name="climatology", kind="climatology", hyp="Частота пожежних днів за місяцем у навчальних роках"),
    dict(name="index_dry_days", kind="index", feature="dry_days", hyp="Довга суха серія → пожежа"),
    dict(name="index_vpd7", kind="index", feature="VPD_7", hyp="Сухе повітря за тиждень (VPD) → пожежа"),
    dict(name="index_nesterov", kind="index", feature="nesterov", hyp="Класичний індекс пожежної небезпеки Нестерова"),
    dict(name="index_fwi", kind="index", feature="fwi", hyp="Канадський індекс FWI: стандарт EFFIS для Європи, з денної погоди NASA POWER"),
    dict(name="logreg_weather", kind="model", features=WXF, make=_lr, hyp="Лінійна комбінація погодних ознак"),
    dict(name="logreg_weather_season", kind="model", features=WXF + SEAS, make=_lr, hyp="Погода + сезонність (лінійно)"),
    dict(name="logreg_weather_season_unweighted", kind="model", features=WXF + SEAS, make=lambda: _lr(None),
         hyp="Те саме без зважування класів: краще калібрування (Brier), але на трьох роках разом ранжує гірше"),
    dict(name="random_forest", kind="model", features=WXF + SEAS, hyp="Нелінійні взаємодії (RandomForest)",
         make=lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", random_state=0, n_jobs=-1)),
    dict(name="lightgbm", kind="model", features=WXF + SEAS, hyp="Градієнтний бустинг (LightGBM, малі дерева)",
         make=lambda: LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=15, subsample=.8, subsample_freq=1,
                                     verbose=-1, random_state=0)),
]


if transfer.on() and transfer.card("FG_day_fire_day"):       # готова модель іншого проєкту як є: нічого не вчить, у виборі не бере участі
    APPROACHES.append(dict(name=transfer.label("FG_day"), kind="transfer", hyp=transfer.hyp()))


def _oof(a, df, tgt, years=None):
    """Out-of-fold прогнози на роках years (expanding window: рік бачить лише попередні)."""
    yr = pd.to_datetime(df.date).dt.year.values
    s = np.full(len(df), np.nan)
    for ty in (years or TEST_YEARS):
        tr, te = yr < ty, yr == ty
        if a["kind"] == "persistence":
            s[te] = df.fire_day_yesterday.values[te]
        elif a["kind"] == "climatology":                              # частота пожежних днів за місяцем, своя в кожного парку
            m = pd.to_datetime(df.date).dt.month.values
            pk = df.park.values if "park" in df else np.zeros(len(df), int)
            clim = pd.Series(df[tgt].values[tr]).groupby([pk[tr], m[tr]]).mean()
            s[te] = pd.Series(list(zip(pk[te], m[te]))).map(clim).fillna(0).values
        elif a["kind"] == "index":
            s[te] = df[a["feature"]].values[te]
        elif a["kind"] == "transfer":
            mdl, _ = transfer.model(f"FG_day_{tgt}")
            feats = list(mdl.feature_name_)
            s[te] = mdl.predict_proba(df.loc[te, feats])[:, 1]
        else:
            mdl = a["make"]().fit(df.loc[tr, a["features"]], df.loc[tr, tgt])
            s[te] = mdl.predict_proba(df.loc[te, a["features"]])[:, 1]
    return s


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: days, weather, fires')
    dv = datasets.get("fire_day_daily")
    df = dv.df
    yr = pd.to_datetime(df.date).dt.year.values
    te = np.isin(yr, TEST_YEARS)
    pk = df.park.values if "park" in df else np.array(["_"] * len(df))
    PK = list(dict.fromkeys(pk))
    code = pd.Series(pk).map({k: i for i, k in enumerate(PK)}).values
    within = pd.Series(np.arange(len(df))).groupby(pk).cumcount().values           # номер дня всередині свого парку
    block = (code * 100000 + within // 14)[te]                      # 14-денні блоки не перетинають межу парку
    out, champions = [], {}
    tracking.keep_reset("fire_day")
    pr, by_year, series, coef = {}, [], {}, {}                        # дані графіків з тих самих передбачень, що й метрики
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "training", files=True)
        ml.log_params(dict(test_years=str(TEST_YEARS), protocol="expanding-window forward chaining", n_boot=n_boot, ci_block_days=14))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/fire_day.py")
        pflow.mark('2. approaches: pick on 2022, test, difference vs baselines')
        for tgt, tname in TARGETS.items():
            y = df[tgt].values[te]
            curves = {}
            pr[tgt] = {}
            for a in APPROACHES:
                with ml.start_run(run_name=f"{tgt} · {a['name']}", nested=True):
                    ml.set_tags(dict(target=tgt, approach=a["name"], kind=a["kind"], hypothesis=a["hyp"], role="approach"))
                    tracking.log_dataset(dv, "training", tgt)
                    ml.log_params(dict(target=tgt, approach=a["name"], kind=a["kind"], features=",".join(a.get("features", [a.get("feature", "")]))))
                    s = _oof(a, df, tgt)[te]
                    m = metrics.ranking(y, s)
                    ci = metrics.block_bootstrap(block, lambda idx: metrics.ranking(y[idx], s[idx])["pr_auc"], n=n_boot)
                    m |= dict(pr_auc_ci_lo=ci[0], pr_auc_ci_hi=ci[1])
                    if a["kind"] in ("model", "climatology"):
                        m["brier"] = metrics.brier(y, s)
                    ml.log_metrics(tracking.clean(m))
                    by_park = {}
                    for k in (PK if len(PK) > 1 else []):          # той самий підхід, тест окремо в кожному парку
                        q = pk[te] == k
                        if y[q].sum() > 0:
                            mk = metrics.ranking(y[q], s[q])
                            ck = metrics.block_bootstrap(block[q], lambda idx: metrics.ranking(y[q][idx], s[q][idx])["pr_auc"], n=n_boot)
                            by_park |= {f"pr_auc_{k}": mk["pr_auc"], f"pr_auc_ci_{k}": f"{ck[0]:.3f}–{ck[1]:.3f}", f"roc_auc_{k}": mk["roc_auc"],
                                        f"positives_{k}": mk["positives"], f"prevalence_{k}": mk["prevalence"]}
                        else:
                            by_park |= {f"pr_auc_{k}": np.nan, f"positives_{k}": 0}
                    for ty in TEST_YEARS:
                        sel = yr[te] == ty
                        if y[sel].sum() > 0:
                            v_ = metrics.ranking(y[sel], s[sel])["pr_auc"]
                            ml.log_metric("pr_auc_by_year", v_, step=ty)
                            by_year.append(dict(target=tgt, approach=a["name"], year=int(ty), pr_auc=round(float(v_), 4)))
                    p_, rc_, _ = precision_recall_curve(y, s)
                    pr[tgt][a["name"]] = metrics.thin(rc_[::-1], p_[::-1])
                    curves[a["name"]] = s
                    vsel = np.isin(yr, VAL_YEARS)
                    sv = _oof(a, df, tgt, VAL_YEARS)[vsel]
                    val_pr = metrics.ranking(df[tgt].values[vsel], np.nan_to_num(sv))["pr_auc"]
                    out.append(dict(target=tgt, approach=a["name"], kind=a["kind"], hypothesis=a["hyp"], forecast=a.get("forecast", True),
                                    val_pr_auc=val_pr, pr_auc=m["pr_auc"], pr_auc_ci=f"{ci[0]:.3f}–{ci[1]:.3f}", roc_auc=m["roc_auc"], recall_at_10=m["recall_at_10"],
                                    brier=m.get("brier", np.nan), positives=m["positives"], prevalence=m["prevalence"], **by_park))
            r = pd.DataFrame([o for o in out if o["target"] == tgt and o["forecast"] and o["kind"] != "transfer"])
            best = r.sort_values("val_pr_auc", ascending=False, kind="mergesort").iloc[0]     # вибір на 2022, тест 2023–2025 не бере участі
            champions[tgt] = best.approach
            for b in ("climatology", "index_fwi"):                           # парна різниця з базами на тих самих 14-денних блоках
                sc, sb = curves[best.approach], curves[b]
                dd, lo_, hi_ = metrics.paired_block_bootstrap(block, lambda i: metrics.ranking(y[i], sc[i])["pr_auc"], lambda i: metrics.ranking(y[i], sb[i])["pr_auc"], n=n_boot)
                for o in out:
                    if o["target"] == tgt and o["approach"] == best.approach:
                        o.update({f"diff_{b}": dd, f"diff_{b}_lo": lo_, f"diff_{b}_hi": hi_})
            ml.log_figure(_pr_fig(y, curves, best.approach, tname), f"plots/pr_{tgt}.png"); plt.close("all")
            q0 = pk[te] == PK[0]                                       # ряд переможця: основний парк, як і раніше
            sc_ = curves[best.approach]
            series[tgt] = dict(approach=best.approach, date=df.date[te][q0].tolist(), score=[None if not np.isfinite(v) else round(float(v), 4) for v in sc_[q0]],
                               y=df[tgt].values[te][q0].astype(int).tolist())
            if len(PK) > 1:
                series[tgt]["parks"] = {k: dict(date=df.date[te][pk[te] == k].tolist(), y=df[tgt].values[te][pk[te] == k].astype(int).tolist(),
                                                score=[None if not np.isfinite(v) else round(float(v), 4) for v in sc_[pk[te] == k]]) for k in PK}
            a = next(x for x in APPROACHES if x["name"] == best.approach)
            if a["kind"] == "model":                        # фінальна модель: навчання на всіх роках → Model Registry
                with ml.start_run(run_name=f"{tgt} · FINAL {a['name']}", nested=True):
                    ml.set_tags(dict(target=tgt, approach=a["name"], role="final_model", hypothesis=a["hyp"]))
                    tracking.log_dataset(dv, "training", tgt)
                    mdl = a["make"]().fit(df[a["features"]], df[tgt])
                    ml.log_params(dict(target=tgt, approach=a["name"], train_period="2021–2025", features=",".join(a["features"])))
                    ml.log_metrics(tracking.clean({"oof_pr_auc": best.pr_auc, "oof_roc_auc": best.roc_auc}))
                    tracking.log_sklearn(mdl, df[a["features"]].head(3), registered_model_name=f"dadia_{tgt}")
                    if "logreg" in a["name"]:
                        coef[tgt] = dict(approach=a["name"], features=a["features"], coef=[round(float(c_), 3) for c_ in mdl[-1].coef_[0]])
        pflow.mark('3. save')
        for k_, v_ in dict(pr_curves=pr, by_year=by_year, champion_series=series, b_coef=coef).items():
            tracking.keep("fire_day", k_, v_)
        res = pd.DataFrame(out)
        res["champion"] = [champions[t] == a for t, a in zip(res.target, res.approach)]
        ml.set_tags({f"champion_{k}": v for k, v in champions.items()} | {"selection": "max PR-AUC на 2022 (навчання 2021) серед прогнозних підходів; тест 2023–2025 у виборі не бере участі"})
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champions=champions)


def _pr_fig(y, curves, champ, tname):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, s in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, lw=2.5 if k == champ else 1, alpha=1 if k == champ else .6, label=k)
    ax.axhline(y.mean(), ls=":", color="gray", label="випадково")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"PR-криві OOF 2023–2025 · {tname}"); ax.legend(fontsize=7, frameon=False)
    return fig
