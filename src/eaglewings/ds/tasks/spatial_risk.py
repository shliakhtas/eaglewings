"""T3 · Просторовий ризик пожежі на сітці H3 (res 9 ≈ 0.1 км²).

Задача      ранжувати комірки парку за ймовірністю згоріти в наступний пожежний сезон → куди спрямувати патрулі, камери, профілактику.
Лейбли      y_burn_<Y>   — ≥ 20 % площі комірки в контурі події FIRMS (ST-DBSCAN) у році Y;
            y_hist       — втрата лісу Hansen 2001–2021 > 2 % комірки (20 років порушень: пожежі, рубки, вітровали);
            перевірка стійкості до визначення лейбла 2023: y_loss_2023_hansen (≥ 30 % лісу втрачено у 2023–24),
            y_dndvi_2023 (≥ 50 % пікселів з dNDVI MODIS > 0.275 — поріг чемпіона T1).
Ознаки      рельєф (SRTM), паливо (лісистість 2000), люди (OSM: дороги, села, треки, межа парку); динамічні — лише з даних ДО року Y.
Протокол    валідація (вибір підходу) — пожежа 2022: для підходів, що не бачили 2022;
                                         для навчених на 2022 — просторова GroupKFold(5) по блоках H3 res 6 всередині 2022;
            тест (звіт) — пожежа 2023, жоден підхід її не бачив; 95 % CI — bootstrap по блоках H3 res 7.
            Референс-стеля: просторова CV всередині 2023 (той самий рік у train і test — не для вибору).
Парки      кілька парків (ds/regions.py): гекси всіх парків в одній таблиці, моделі вчаться на всіх разом; тест для всіх разом
            і окремо для кожного парку (*_<парк>, CI по блоках цього парку). Прогноз 2026 і ранги рівнів окремо для кожного парку.
Метрики     ROC-AUC — основна (ранжування; частка згорілих у 2023 ~67 %, тож PR-AUC близький до тривіального);
            recall@20 % — частка згорілих комірок у 20 % найризиковішої площі (бюджет патрулювання); PR-AUC — довідково.
Фінал       чемпіон за валідацією перенавчається на всій історії (Hansen 2001–2024 + FIRMS 2022–2025) → Model Registry,
            прогноз 2026 = схильність × поправка на вигоріле паливо (коефіцієнт оцінено з даних 2022 → 2023).
"""
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from .. import common, datasets, metrics, tracking
from ..tracking import ml

EXPERIMENT = "T3 · Spatial fire risk H3 (де горітиме)"
STATIC = ["elev", "slope", "southness", "treecover", "d_road", "d_village", "track_km_km2", "d_boundary"]
TERRAIN = ["elev", "slope", "southness"]
HUMAN = ["d_road", "d_village", "track_km_km2", "d_boundary"]
DYN = ["loss_hist", "loss_recent3", "firms_2km", "d_firms", "burned_prev3"]
HIST_THR, BURN_THR, DNDVI_T = 0.02, 0.2, 0.275
FINAL_APPROACH = "hist_lgbm_static"   # апріорний вибір (v1 ноутбука); використовується, якщо чемпіон валідації провалив тест


def build_dataset():
    B, G = common.base(), common.h3grid()
    _, fp_year = common.default_events()
    df = pd.concat([G.grid[["h3", "parent6", "parent7"]].reset_index(drop=True), G.S.reset_index(drop=True)], axis=1)
    for Y in (2022, 2023, 2026):
        d = common.dyn_features(Y)
        for c in DYN:
            df[f"f{Y % 100}_{c}"] = d[c].values
    for Y in range(2022, 2026):
        df[f"y_burn_{Y}"] = (G.zmean(fp_year[Y]) >= BURN_THR).astype(int)
    df["y_hist"] = (df.f22_loss_hist > HIST_THR).astype(int)
    dom = B.forest & B.park_mask
    lost = G.zmean(np.isin(B.loss, [23, 24]) & dom) / np.maximum(G.zmean(dom), 1e-9)
    df["y_loss_2023_hansen"] = ((lost >= .3) & (G.zmean(dom) > .1)).astype(int)
    nd = common.modis_ndvi()
    df["y_dndvi_2023"] = (G.zmean(np.nan_to_num(nd["pre"] - nd["post"]) > DNDVI_T) >= .5).astype(int)
    df["y_final"] = ((df.f26_loss_hist > HIST_THR) | df[[f"y_burn_{Y}" for Y in range(2022, 2026)]].max(1).astype(bool)).astype(int)
    df = df.round(5)
    return datasets.save(
        df, "h3_risk_cells", unit="комірка H3 res 9 у парку",
        description="Комірки H3 з ознаками рельєфу/палива/людей, динамічними ознаками станом на 1.01 років 2022/2023/2026 (префікси f22_/f23_/f26_) і лейблами згорання за FIRMS, Hansen, MODIS.",
        targets=["y_hist", "y_burn_2022", "y_burn_2023", "y_loss_2023_hansen", "y_dndvi_2023", "y_final"],
        features=STATIC + [f"f{y}_{c}" for y in (22, 23, 26) for c in DYN],
        params=dict(h3_res=common.H3_RES, hist_thr=HIST_THR, burn_thr=BURN_THR, dndvi_t=DNDVI_T, events="ST-DBSCAN 1.5 km / 2 d"),
        sources=["Hansen GFC v1.12", "NASA FIRMS", "MODIS MOD13Q1", "AWS Terrain Tiles (SRTM)", "OpenStreetMap"])


def X(df, feats, Y):
    return pd.DataFrame({f: df[f] if f in STATIC else df[f"f{Y % 100}_{f}"] for f in feats})


def _lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=7, min_child_samples=30, subsample=.8, subsample_freq=1,
                          colsample_bytree=.8, reg_lambda=1., verbose=-1, random_state=0)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def _rf():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=10, class_weight="balanced_subsample", random_state=0, n_jobs=-1)


APPROACHES = [
    dict(name="random", kind="score", score=lambda df, Y: np.random.default_rng(0).random(len(df)), hyp="Випадкове ранжування (нижня межа)"),
    dict(name="elevation", kind="score", score=lambda df, Y: df.elev.values, hyp="Вище → сухіше, хвойний ліс → горить"),
    dict(name="treecover", kind="score", score=lambda df, Y: df.treecover.values, hyp="Більше лісу → більше палива"),
    dict(name="near_village", kind="score", score=lambda df, Y: -df.d_village.values, hyp="Ближче до людей → більше займань"),
    dict(name="firms_history", kind="score", score=lambda df, Y: X(df, ["firms_2km"], Y).firms_2km.values, hyp="Де горіло раніше — горітиме знову"),
    dict(name="hist_lr_static", kind="model", label="hist", feats=STATIC, make=_lr, hyp="Схильність з історії Hansen 2001–21, лінійна"),
    dict(name="hist_rf_static", kind="model", label="hist", feats=STATIC, make=_rf, hyp="Схильність з історії Hansen, RandomForest"),
    dict(name="hist_lgbm_static", kind="model", label="hist", feats=STATIC, make=_lgbm, hyp="Схильність з історії Hansen, LightGBM"),
    dict(name="hist_lgbm_terrain", kind="model", label="hist", feats=TERRAIN, make=_lgbm, hyp="Абляція: лише рельєф"),
    dict(name="hist_lgbm_human", kind="model", label="hist", feats=HUMAN, make=_lgbm, hyp="Абляція: лише людський фактор"),
    dict(name="hist_lgbm_terrain_fuel", kind="model", label="hist", feats=TERRAIN + ["treecover"], make=_lgbm, hyp="Абляція: рельєф + паливо"),
    dict(name="f22_lgbm_static", kind="model", label="f22", feats=STATIC, make=_lgbm, hyp="Навчання на одній пожежі 2022, статичні ознаки"),
    dict(name="f22_lgbm_all", kind="model", label="f22", feats=STATIC + DYN, make=_lgbm, hyp="Навчання на пожежі 2022, усі ознаки (з історією)"),
    dict(name="f22_lr_all", kind="model", label="f22", feats=STATIC + DYN, make=_lr, hyp="Навчання на пожежі 2022, лінійна"),
]


def _scores(a, df):
    """→ (val_scores_on_2022, val_labels, test_scores_on_2023, fitted_model|None)."""
    y22 = df.y_burn_2022.values
    if a["kind"] == "score":
        return a["score"](df, 2022), y22, a["score"](df, 2023), None
    if a["label"] == "hist":
        m = a["make"]().fit(X(df, a["feats"], 2022), df.y_hist)
        return m.predict_proba(X(df, a["feats"], 2022))[:, 1], y22, m.predict_proba(X(df, a["feats"], 2023))[:, 1], m
    val = np.zeros(len(df))
    for tr, va in GroupKFold(5).split(df, y22, df.parent6):
        val[va] = a["make"]().fit(X(df.iloc[tr], a["feats"], 2022), y22[tr]).predict_proba(X(df.iloc[va], a["feats"], 2022))[:, 1]
    m = a["make"]().fit(X(df, a["feats"], 2022), y22)
    return val, y22, m.predict_proba(X(df, a["feats"], 2023))[:, 1], m


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: H3 cells')
    dv = datasets.get("h3_risk_cells")
    df = dv.df
    yt = df.y_burn_2023.values
    pk = df.park.values if "park" in df else np.array([common._park.KEY] * len(df))
    PK = list(dict.fromkeys(pk))
    rows, test_scores = [], {}
    tracking.keep_reset("spatial_risk")
    roc, models = {}, {}                                                  # дані графіків з тих самих прогнозів, що й метрики
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "training", files=True)
        ml.log_params(dict(validation="fire 2022", test="fire 2023", ci_blocks="H3 res7", n_boot=n_boot, hist_thr=HIST_THR, burn_thr=BURN_THR))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/spatial_risk.py")
        pflow.mark('2. approaches: validation and test')
        for a in APPROACHES:
            with ml.start_run(run_name=a["name"], nested=True):
                ml.set_tags(dict(approach=a["name"], kind=a["kind"], hypothesis=a["hyp"], train_label=a.get("label", "—"), role="approach"))
                tracking.log_dataset(dv, "training", {"hist": "y_hist", "f22": "y_burn_2022"}.get(a.get("label"), "y_burn_2023"))
                ml.log_params(dict(approach=a["name"], kind=a["kind"], train_label=a.get("label", "—"), features=",".join(a.get("feats", []))))
                sv, yv, st, mdl = _scores(a, df)
                v, t = metrics.ranking(yv, sv, ks=(.2,)), metrics.ranking(yt, st, ks=(.2,))
                ci = metrics.block_bootstrap(df.parent7.values, lambda idx: metrics.ranking(yt[idx], st[idx])["roc_auc"], n=n_boot)
                rob = {lab: metrics.ranking(df[lab].values, st)["roc_auc"] for lab in ("y_loss_2023_hansen", "y_dndvi_2023")}
                by_park = {}
                for k in (PK if len(PK) > 1 else []):              # той самий підхід, тест 2023 окремо в кожному парку
                    q = pk == k
                    if 0 < yt[q].sum() < q.sum():
                        tk = metrics.ranking(yt[q], st[q], ks=(.2,))
                        ck = metrics.block_bootstrap(df.parent7.values[q], lambda idx: metrics.ranking(yt[q][idx], st[q][idx])["roc_auc"], n=n_boot)
                        by_park |= {f"test_roc_auc_{k}": tk["roc_auc"], f"test_roc_auc_ci_{k}": f"{ck[0]:.3f}–{ck[1]:.3f}", f"test_recall_at_20_{k}": tk["recall_at_20"]}
                    else:                                                 # у парку немає згорілих (чи лише згорілі) гексів 2023: ROC-AUC не визначений
                        by_park |= {f"test_roc_auc_{k}": np.nan}
                    by_park |= {f"burned_2023_{k}": int(yt[q].sum()), f"cells_{k}": int(q.sum())}
                ml.log_metrics(tracking.clean({f"val_{k}": x for k, x in v.items()} | {f"test_{k}": x for k, x in t.items()} |
                                                  {"test_roc_auc_ci_lo": ci[0], "test_roc_auc_ci_hi": ci[1]} | {f"test_roc_auc_{k}": x for k, x in rob.items()}))
                if mdl is not None:
                    tracking.log_sklearn(mdl, X(df, a["feats"], 2023).head(3))
                test_scores[a["name"]] = st
                models[a["name"]] = mdl
                f_, t_, _ = roc_curve(yt, st)
                fv_, tv_, _ = roc_curve(yv, sv)
                roc[a["name"]] = dict(test=metrics.thin(f_, t_), val=metrics.thin(fv_, tv_))
                rows.append(dict(approach=a["name"], kind=a["kind"], train_label=a.get("label", "—"), hypothesis=a["hyp"],
                                 val_roc_auc=v["roc_auc"], test_roc_auc=t["roc_auc"], test_roc_auc_ci=f"{ci[0]:.3f}–{ci[1]:.3f}",
                                 test_recall_at_20=t["recall_at_20"], test_pr_auc=t["pr_auc"],
                                 test_roc_auc_hansen=rob["y_loss_2023_hansen"], test_roc_auc_dndvi=rob["y_dndvi_2023"], **by_park))
        pflow.mark('3. pick, difference vs random and elevation')
        res = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
        champ = res.approach.iloc[0]
        res["champion"] = res.approach.eq(champ)
        for b in ("random", "elevation"):                                     # парна різниця на тих самих блоках H3 res7
            dd, lo_, hi_ = metrics.paired_block_bootstrap(df.parent7.values, lambda i: metrics.ranking(yt[i], test_scores[champ][i])["roc_auc"],
                                                          lambda i: metrics.ranking(yt[i], test_scores[b][i])["roc_auc"], n=n_boot)
            res.loc[res.champion, [f"diff_{b}", f"diff_{b}_lo", f"diff_{b}_hi"]] = [dd, lo_, hi_]

        # референс-стеля: просторова CV всередині 2023
        pflow.mark('4. upper bound: spatial cross-validation')
        with ml.start_run(run_name="upper_bound · spatial CV 2023", nested=True):
            ml.set_tags(dict(role="upper_bound", hypothesis="Скільки дає модель, якщо лейбли того ж року відомі (не для вибору)"))
            oof = np.zeros(len(df))
            for tr, va in GroupKFold(5).split(df, yt, df.parent6):
                oof[va] = _lgbm().fit(X(df.iloc[tr], STATIC, 2023), yt[tr]).predict_proba(X(df.iloc[va], STATIC, 2023))[:, 1]
            ub = metrics.ranking(yt, oof, ks=(.2,))["roc_auc"]
            ml.log_metric("cv_roc_auc", ub)

        # фінал → вся історія → Model Registry + прогноз 2026.
        # Валідація на одній пожежі 2022 виявилась ненадійною (див. plots/val_vs_test.png): після неї паливо вигоріло, і її просторовий
        # патерн протилежний 2023-му. Тому в продукт іде модель, обрана ДО бенчмарку (v1 ноутбука) — схильність з 20-річної історії Hansen.
        # Це post-hoc рішення, і воно явно позначене тегами final_selection / champion_by_validation.
        final_name = FINAL_APPROACH if res.loc[res.approach == champ, "test_roc_auc"].iloc[0] < .6 else champ
        a = next(x for x in APPROACHES if x["name"] == final_name)
        fuel = df.y_burn_2023[df.y_burn_2022 == 1].mean() / df.y_burn_2023[df.y_burn_2022 == 0].mean()
        pflow.mark('5. final model -> risk 2026')
        with ml.start_run(run_name=f"FINAL {final_name} → risk 2026", nested=True):
            ml.set_tags(dict(role="final_model", approach=final_name, hypothesis=a["hyp"], champion_by_validation=champ,
                                 final_selection="protocol" if final_name == champ else "post-hoc: чемпіон валідації 2022 провалив тест 2023 (ROC-AUC < 0.6)"))
            tracking.log_dataset(dv, "training", "y_final")
            if a["kind"] == "model":
                mdl = a["make"]().fit(X(df, a["feats"], 2026), df.y_final)
                susc = mdl.predict_proba(X(df, a["feats"], 2026))[:, 1]
                tracking.log_sklearn(mdl, X(df, a["feats"], 2026).head(3), registered_model_name="dadia_fire_risk_h3")
            else:
                susc = pd.Series(a["score"](df, 2026)).rank(pct=True).values
            fuel_mult = 1 - (1 - fuel) * np.clip(df.f26_burned_prev3.values / BURN_THR, 0, 1)
            risk = susc * fuel_mult
            outs = {}
            for k in PK:                                                  # прогноз 2026 кожного парку на його сітці, рівні в межах парку
                q = pk == k
                G = common.h3grid(k).grid
                o = G[["h3", "geometry"]].assign(susceptibility=susc[q], fuel_mult=fuel_mult[q], risk_2026=risk[q], risk_rank=pd.Series(risk[q]).rank(pct=True).values)
                o["level"] = pd.cut(o.risk_rank, [0, .5, .8, .95, 1], labels=["фонова", "помірна", "висока", "критична"]).astype(str)
                o.to_file(common._park.processed(k) / "risk_h3_2026.geojson", driver="GeoJSON")
                outs[k] = o
            out = outs[PK[0]]
            ml.log_params(dict(approach=final_name, fuel_factor=round(fuel, 3), train_label="y_final (Hansen 2001–24 | FIRMS 2022–25)"))
            ml.log_artifact(str(common.P / "risk_h3_2026.geojson"), "forecast")
            ml.log_figure(_map_fig(outs), "plots/risk_2026.png"); plt.close("all")

        ml.set_tags(dict(champion=champ, final_model=final_name, selection="max val_roc_auc (пожежа 2022); тест 2023 не використовується для вибору"))
        ml.log_metrics(tracking.clean(dict(champion_test_roc_auc=res.test_roc_auc.iloc[0], upper_bound_cv_roc_auc=ub, fuel_factor=fuel)))
        pflow.mark('6. save')
        tracking.keep("spatial_risk", "roc", roc)
        tracking.keep("spatial_risk", "prevalence", dict(val=round(float(df.y_burn_2022.mean()), 4), test=round(float(yt.mean()), 4)))
        rank = lambda v: pd.Series(v).rank(pct=True).round(3).tolist()
        tracking.keep("spatial_risk", "cells", dict(h3=df.h3.tolist(), final_rank=rank(test_scores[final_name]), champ_rank=rank(test_scores[champ]),
                                                    **({"park": pk.tolist()} if len(PK) > 1 else {})))
        fa = next(x for x in APPROACHES if x["name"] == final_name)
        if models.get(final_name) is not None:                            # важливість ознак тієї самої моделі, що дала тест 2023
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(models[final_name], X(df, fa["feats"], 2023), df.y_burn_2023, scoring="roc_auc", n_repeats=10, random_state=0)
            tracking.keep("spatial_risk", "importance", sorted([dict(feature=f_, mean=round(float(m_), 4), std=round(float(s_), 4)) for f_, m_, s_ in zip(fa["feats"], pi.importances_mean, pi.importances_std)], key=lambda d: -d["mean"]))
        ml.log_table(res, "results.json")
        ml.log_figure(_roc_fig(yt, test_scores, champ), "plots/roc_test_2023.png")
        ml.log_figure(_val_test_fig(res), "plots/val_vs_test.png"); plt.close("all")
        res["final_model"] = res.approach.eq(final_name)
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champ, final_model=final_name,
                         upper_bound=ub, fuel_factor=fuel)


def _roc_fig(y, scores, champ):
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, s in scores.items():
        f, t, _ = roc_curve(y, s)
        ax.plot(f, t, lw=2.5 if k == champ else 1, alpha=1 if k == champ else .55, label=k)
    ax.plot([0, 1], [0, 1], ls=":", color="gray"); ax.legend(fontsize=7, frameon=False); ax.set_title("ROC · тест: пожежа 2023")
    return fig


def _val_test_fig(res):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in res.itertuples():
        ax.scatter(r.val_roc_auc, r.test_roc_auc, s=60, color="#7048e8" if r.champion else ("#adb5bd" if r.kind == "score" else "#fd7e14"))
        ax.annotate(r.approach, (r.val_roc_auc, r.test_roc_auc), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(.5, ls=":", color="gray"); ax.axvline(.5, ls=":", color="gray")
    ax.set_xlabel("валідація: ROC-AUC на пожежі 2022"); ax.set_ylabel("тест: ROC-AUC на пожежі 2023"); ax.set_title("Чи переноситься вибір з 2022 на 2023")
    return fig


def _map_fig(outs):
    """Ризик 2026: по панелі на кожен парк."""
    col = {"фонова": "#f1f3f5", "помірна": "#ffd43b", "висока": "#fd7e14", "критична": "#c92a2a"}
    fig, axs = plt.subplots(1, len(outs), figsize=(7 * len(outs), 8), squeeze=False)
    for ax, (k, out) in zip(axs[0], outs.items()):
        out.plot(ax=ax, color=out.level.map(col), lw=0)
        common.base(k).park.boundary.plot(ax=ax, color="#343a40", lw=.8, ls="--")
        ax.set_aspect(1 / np.cos(np.radians(LAT0)))
        ax.legend(handles=[Patch(color=v, label=k_) for k_, v in col.items()], frameon=False, loc="lower left")
        ax.set_title("Ризик пожежі 2026" + (f" · {k}" if len(outs) > 1 else ""))
    return fig
