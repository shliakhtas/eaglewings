"""T1 · Картування згарища пожежі 2023.

Задача      бінарна сегментація: піксель лісу (30 м) у парку втрачено пожежею 2023 р.?
Лейбли      Hansen GFC lossyear ∈ {2023, 2024} (Landsat; незалежні від FIRMS і MODIS сенсор та алгоритм).
            2024 включено, бо частина дерев гине із запізненням; варіант лише 2023 — як перевірка стійкості до визначення лейбла.
            Обмеження: низова пожежа без загибелі крон = 0 у Hansen → precision будь-якого «вогневого» підходу занижена.
Домен       ліс 2000 р. (treecover ≥ 30 %) у межах парку і сітки H3.
Протокол    просторовий спліт за блоками H3 res 7 (~5 км²) у кожному парку: 50 % блоків — калібрування (вибір порогів / гіперпараметрів),
            50 % — тест. Парків кілька (ds/regions.py): налаштування й переможця обираємо на калібрувальних блоках усіх парків разом,
            тест рахуємо для всіх разом і окремо для кожного парку (колонки *_<парк>), з CI по блоках цього парку.
Метрики     F1 (= Dice) — основна: баланс commission/omission, стандарт валідації burned-area (CEOS LPV);
            IoU, precision/recall, commission/omission error. Accuracy не використовуємо — домінує фон.
            95 % CI — bootstrap по тестових блоках.
Гіпотези
  firms_pixels     сирі пікселі FIRMS 2023 — «де був вогонь, там згарище»
  firms_st_dbscan  події ST-DBSCAN; відкидаємо дрібні події (min_det) — менше хибних поодиноких детекцій
  firms_hull       опукла оболонка подій — заповнює прогалини між пікселями (ризик перебільшення)
  modis_dndvi      падіння NDVI (до − після) > t — спектральний сигнал загибелі рослинності
  modis_rdndvi     відносне падіння dNDVI / NDVI_до > t — нормалізація на початкову зеленість
  hybrid           dNDVI > t І в межах 1 км від події FIRMS — спектр + термічне підтвердження
"""
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from .. import common, datasets, metrics, tracking
from ..tracking import ml

EXPERIMENT = "T1 · Burned area 2023 (згарище)"
SPLIT_SEED = 42


def build_dataset():
    B, G = common.base(), common.h3grid()
    dom = B.park_mask & B.forest & (G.cell_r > 0)
    rr, cc = np.nonzero(dom)
    cell = G.cell_r[rr, cc] - 1
    block = G.grid.parent7.values[cell]
    ub = np.unique(G.grid.parent7)
    test_blocks = set(np.random.default_rng(SPLIT_SEED).choice(ub, len(ub) // 2, replace=False))
    nd = common.modis_ndvi()
    df = pd.DataFrame(dict(row=rr, col=cc, h3=G.grid.h3.values[cell], block=block,
                           split=np.where(np.isin(block, list(test_blocks)), "test", "calib"),
                           pix_ha=B.PIX_HA[rr, cc].round(5), treecover=B.tc[rr, cc],
                           ndvi_pre=nd["pre"][rr, cc].round(4), ndvi_post=nd["post"][rr, cc].round(4),
                           y=np.isin(B.loss[rr, cc], [23, 24]).astype(int), y_2023_only=(B.loss[rr, cc] == 23).astype(int)))
    return datasets.save(
        df, "burned_area_pixels", unit="піксель Hansen 30 м (ліс у парку)",
        description="Пікселі лісу парку з лейблом втрати лісу Hansen 2023–24, NDVI MODIS до/після пожежі та просторовим сплітом за блоками H3 res 7.",
        targets=["y", "y_2023_only"], features=["ndvi_pre", "ndvi_post", "(FIRMS — через row/col)"],
        params=dict(label="Hansen lossyear in {23,24}", domain="park & treecover2000>=30 & in H3 res9", split=f"H3 res7 blocks 50/50 seed {SPLIT_SEED}",
                    modis=[common.MODIS_PRE, common.MODIS_POST]),
        sources=["Hansen GFC v1.12 (UMD/NASA)", "MODIS MOD13Q1 v061 (ORNL DAAC)", "OSM park boundary"])


# ---------------------------------------------------------------- підходи
def _firms23(key=None):
    f = common.base(key).firms
    return f[f.year == 2023]


def _ev_mask(cfg, hull=False, buffer_m=0, key=None):
    f = _firms23(key)
    if not len(f):
        return common.to_mask([], key)
    ev = common.events(f, common.st_dbscan(f, cfg["eps_km"], cfg["eps_days"]))
    ev = ev[ev.n >= cfg["min_det"]]
    g = ev.geometry.convex_hull if hull else ev.geometry
    return common.to_mask(g.buffer(buffer_m) if buffer_m else g, key)


def parks(df):
    """Парки датасету в порядку ds/regions.py; датасет без колонки park — це один основний парк."""
    from .. import park
    return [k for k in park.PARKS if "park" in df and (df.park == k).any()] if "park" in df else [park.KEY]


def per_park(df, fn):
    """Маска на сітці кожного парку (fn(key) → 2D bool) → значення в рядках df (row, col у сітці свого парку)."""
    out = np.zeros(len(df), bool)
    for k in parks(df):
        i = np.flatnonzero(df.park.values == k) if "park" in df else np.arange(len(df))
        out[i] = fn(k)[df.row.values[i], df.col.values[i]]
    return out


FAMILIES = {
    "firms_pixels": dict(hyp="Сирі пікселі FIRMS 2023 (без кластеризації) = згарище",
                         grid=[{}], mask=lambda c, df: per_park(df, lambda k: common.to_mask(common.pixel_squares(_firms23(k)), k) if len(_firms23(k)) else common.to_mask([], k))),
    "firms_st_dbscan": dict(hyp="Події ST-DBSCAN; дрібні події (< min_det детекцій) відкидаємо як шум",
                            grid=[dict(eps_km=e, eps_days=d, min_det=m) for e in (0.75, 1.5, 3.0) for d in (1.0, 2.0, 4.0) for m in (1, 5, 20)],
                            mask=lambda c, df: per_park(df, lambda k: _ev_mask(c, key=k))),
    "firms_hull": dict(hyp="Опукла оболонка подій ST-DBSCAN заповнює прогалини між пікселями",
                       grid=[dict(eps_km=e, eps_days=2.0, min_det=m) for e in (0.75, 1.5, 3.0) for m in (5, 20, 100)],
                       mask=lambda c, df: per_park(df, lambda k: _ev_mask(c, hull=True, key=k))),
    "modis_dndvi": dict(hyp="Падіння NDVI MODIS (до − після) > t", grid=[dict(t=float(round(t, 3))) for t in np.arange(0.05, 0.451, 0.025)],
                        pred=lambda c, df: (df.ndvi_pre - df.ndvi_post).values > c["t"]),
    "modis_rdndvi": dict(hyp="Відносне падіння NDVI (dNDVI / NDVI_до) > t", grid=[dict(t=float(round(t, 3))) for t in np.arange(0.1, 0.81, 0.05)],
                         pred=lambda c, df: ((df.ndvi_pre - df.ndvi_post) / df.ndvi_pre.clip(lower=.05)).values > c["t"]),
    "s2_dnbr": dict(hyp="Стандартний індекс згарищ: падіння NBR Sentinel-2, літо 2023 до пожежі проти літа 2024 > t",
                    grid=[dict(t=float(round(t, 3))) for t in np.arange(0.05, 0.551, 0.05)], pred=None),
    "hybrid": dict(hyp="dNDVI > t і ≤ 1 км від події FIRMS (спектр + термічне підтвердження)",
                   grid=[dict(t=float(round(t, 3))) for t in np.arange(0.05, 0.351, 0.025)], pred=None),
}


def _setup(df):
    """Підходи, яким потрібні зовнішні шари: буфер подій FIRMS і dNBR Sentinel-2 на рядках df."""
    from . import logging_greece as LGR                                       # ті самі літні композити Sentinel-2, що для вирубки
    fp_buf = per_park(df, lambda k: _ev_mask(dict(eps_km=1.5, eps_days=2.0, min_det=5), buffer_m=1000, key=k))
    FAMILIES["hybrid"]["pred"] = lambda c, d: ((d.ndvi_pre - d.ndvi_post).values > c["t"]) & fp_buf
    nbr = lambda c: (c["nir"] - c["sw2"]) / np.maximum(c["nir"] + c["sw2"], 1e-4)
    dnbr = np.full(len(df), np.nan)
    for k in parks(df):
        i = np.flatnonzero(df.park.values == k) if "park" in df else np.arange(len(df))
        dnbr[i] = (nbr(LGR.composite(2023, k)[0]) - nbr(LGR.composite(2024, k)[0]))[df.row.values[i], df.col.values[i]]
    FAMILIES["s2_dnbr"]["pred"] = lambda c, d: np.nan_to_num(dnbr, nan=-1) > c["t"]


def predict(df, family, cfg):
    """Прогноз будь-якого підходу на рядках датасету: так застосунок і сторінка показують саме обраного переможця."""
    _setup(df)
    F = FAMILIES[family]
    cfg = eval(cfg) if isinstance(cfg, str) else cfg
    return F["pred"](cfg, df) if F.get("pred") else F["mask"](cfg, df)


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: burned / not pixels, EFFIS reference')
    dv = datasets.get("burned_area_pixels")
    df = dv.df
    y = df.y.values.astype(bool)
    w = df.pix_ha.values
    blk, blocks = pd.factorize(df.block)
    is_test_blk = np.array([df.split.values[np.argmax(blk == i)] == "test" for i in range(len(blocks))])
    nb = len(blocks)

    def sums(pred, yy=y):
        return (np.bincount(blk, w * (pred & yy), nb), np.bincount(blk, w * (pred & ~yy), nb), np.bincount(blk, w * (~pred & yy), nb))

    def score(s, sel):
        return metrics.area(s[0][sel].sum(), s[1][sel].sum(), s[2][sel].sum())

    _setup(df)
    from . import logging_greece as LGR
    B_ = common.base()
    eff0 = LGR.effis_year(2023)                                               # другий незалежний еталон: офіційний контур EFFIS 2023

    def effis_mask(k):
        eff = eff0[eff0.intersects(common.base(k).PARK)] if len(eff0) else eff0
        return common.to_mask(eff.to_crs(common.UTM).geometry, k) if len(eff) else common.to_mask([], k)
    y_effis = per_park(df, effis_mask)
    T_ = np.flatnonzero(is_test_blk)
    PK = parks(df)
    blk_park = np.array([df.park.values[np.argmax(blk == i)] for i in range(nb)]) if "park" in df else np.array([PK[0]] * nb)
    T_park = {k: np.flatnonzero(is_test_blk & (blk_park == k)) for k in PK}

    rows, champ_preds, sums_ = [], {}, {}
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", "y", files=True)
        ml.log_params(dict(split_seed=SPLIT_SEED, n_blocks=nb, n_test_blocks=int(is_test_blk.sum()), n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/burned_area.py")
        pflow.mark('2. approaches: tune on calibration, test + bootstrap')
        for fam, F in FAMILIES.items():
            with ml.start_run(run_name=fam, nested=True):
                ml.set_tags(dict(family=fam, hypothesis=F["hyp"], role="family"))
                tracking.log_dataset(dv, "evaluation", "y")
                best = None
                for cfg in F["grid"]:
                    pred = F["pred"](cfg, df) if F.get("pred") else F["mask"](cfg, df)
                    s = sums(pred)
                    cal = score(s, ~is_test_blk)
                    with ml.start_run(run_name=f"{fam} {cfg}", nested=True):
                        ml.set_tags(dict(family=fam, role="config"))
                        ml.log_params({"family": fam, **cfg})
                        ml.log_metrics(tracking.clean({f"calib_{k}": v for k, v in cal.items()}))
                    if best is None or cal["f1"] > best[1]["f1"]:
                        best = (cfg, cal, pred, s)
                cfg, cal, pred, s = best
                te = score(s, is_test_blk)
                ci = metrics.block_bootstrap(T_, lambda idx: score(s, T_[idx])["f1"], n=n_boot)
                te_alt = score(sums(pred, df.y_2023_only.values.astype(bool)), is_test_blk)
                te_eff = score(sums(pred, y_effis), is_test_blk)
                by_park = {}
                for k in (PK if len(PK) > 1 else []):          # той самий переможець і поріг, тест окремо в кожному парку
                    tk = score(s, is_test_blk & (blk_park == k))
                    ck = metrics.block_bootstrap(T_park[k], lambda idx: score(s, T_park[k][idx])["f1"], n=n_boot) if len(T_park[k]) else (np.nan, np.nan)
                    if not tk["true_ha"]:                          # у тестових блоках парку немає згарища: F1 не визначений, а не 0
                        tk, ck = dict(tk, f1=np.nan, precision=np.nan, recall=np.nan), (np.nan, np.nan)
                    by_park |= {f"test_f1_{k}": tk["f1"], f"test_f1_ci_{k}": f"{ck[0]:.3f}–{ck[1]:.3f}", f"true_ha_test_{k}": tk["true_ha"],
                                f"pred_ha_test_{k}": tk["pred_ha"], f"test_precision_{k}": tk["precision"], f"test_recall_{k}": tk["recall"]}
                sums_[fam] = s
                ml.log_params({"best_" + k: v for k, v in cfg.items()} | {"n_configs": len(F["grid"])})
                ml.log_metrics(tracking.clean({f"calib_{k}": v for k, v in cal.items()} | {f"test_{k}": v for k, v in te.items()} |
                                                  {"test_f1_ci_lo": ci[0], "test_f1_ci_hi": ci[1], "test_f1_label_2023_only": te_alt["f1"]}))
                ml.log_figure(_confusion_fig(df, pred, f"{fam} {cfg}"), "plots/confusion_map.png"); plt.close("all")
                champ_preds[fam] = pred
                rows.append(dict(family=fam, hypothesis=F["hyp"], best_config=str(cfg), n_configs=len(F["grid"]), calib_f1=cal["f1"],
                                 test_f1=te["f1"], test_f1_ci=f"{ci[0]:.3f}–{ci[1]:.3f}", test_iou=te["iou"], test_precision=te["precision"],
                                 test_recall=te["recall"], test_f1_label_2023_only=te_alt["f1"], test_f1_effis=te_eff["f1"], pred_ha_test=te["pred_ha"], true_ha_test=te["true_ha"], **by_park))
        pflow.mark('3. pick on calibration, difference vs FIRMS')
        res = pd.DataFrame(rows).sort_values("calib_f1", ascending=False).reset_index(drop=True)
        champion = res.family.iloc[0]
        res["champion"] = res.family.eq(champion)
        for b in ("firms_pixels",):                                            # парна різниця з базою на тих самих тестових блоках
            dd, lo_, hi_ = metrics.paired_block_bootstrap(T_, lambda i: score(sums_[champion], T_[i])["f1"], lambda i: score(sums_[b], T_[i])["f1"], n=n_boot)
            res.loc[res.champion, ["diff_base", "diff_lo", "diff_hi"]] = [dd, lo_, hi_]
        ml.set_tags(dict(champion=champion, selection="max calib_f1 (тест не використовується для вибору)"))
        ml.log_metrics(tracking.clean({"champion_test_f1": res.test_f1.iloc[0]}))
        pflow.mark('4. save')
        tracking.keep_reset("burned_area")                               # мапа «переможець проти Hansen» з тих самих передбачень, що дали F1
        champ_t = float(eval(res.best_config.iloc[0]).get("t", .4))
        rasters = {}
        for key in PK:
            i = np.flatnonzero(df.park.values == key) if "park" in df else np.arange(len(df))
            rasters[key] = _raster(df.iloc[i], champ_preds[champion][i], key, champ_t)
        tracking.keep("burned_area", "raster", rasters[PK[0]] if PK[0] == park_main() else rasters.get(park_main(), rasters[PK[0]]))
        if len(PK) > 1:
            tracking.keep("burned_area", "raster_parks", rasters)     # та сама мапа для кожного парку: перемикач на сторінці дослідження
        ml.log_table(res, "results.json")
        ml.log_figure(_bar_fig(res), "plots/families.png"); plt.close("all")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champion)


def park_main():
    from .. import park
    return park.MAIN


def _raster(df, pred, key, champ_t, k=4):
    """Мапа парку key: Hansen, переможець, сирі пікселі FIRMS 2023 і тестові блоки, агреговані 4×4 пікселі."""
    B_ = common.base(key)
    f23 = B_.firms[B_.firms.year == 2023]
    pf = (common.to_mask(common.pixel_squares(f23), key) if len(f23) else common.to_mask([], key))[df.row, df.col]
    agg = pd.DataFrame(dict(r=df.row.values // k, c=df.col.values // k, y=df.y.values, pc=pred, pf=pf, s=(df.split == "test").astype(int).values)).groupby(["r", "c"]).mean().reset_index()
    return dict(lon0=B_.TR.c, lat0=B_.TR.f, dlon=B_.TR.a * k, dlat=B_.TR.e * k, champion_t=champ_t,
                r=agg.r.astype(int).tolist(), c=agg.c.astype(int).tolist(), y=(agg.y * 100).round().astype(int).tolist(),
                pc=(agg.pc * 100).round().astype(int).tolist(), pf=(agg.pf * 100).round().astype(int).tolist(), s=agg.s.round().astype(int).tolist())


def _confusion_fig(df, pred, title):
    """Мапа помилок: по панелі на кожен парк (TP, FP, FN на сітці свого парку)."""
    PK = parks(df)
    fig, axs = plt.subplots(1, len(PK), figsize=(7 * len(PK), 7.5), squeeze=False)
    for ax, key in zip(axs[0], PK):
        i = np.flatnonzero(df.park.values == key) if "park" in df else np.arange(len(df))
        d, pr = df.iloc[i], pred[i]
        B = common.base(key)
        cm = np.full((B.H, B.W), np.nan)
        y = d.y.values.astype(bool)
        cm[d.row, d.col] = np.select([pr & y, pr & ~y, ~pr & y], [3, 1, 2], 0)
        ax.imshow(cm, extent=B.EXT, cmap=ListedColormap(["#e9ecef", "#ffa94d", "#9775fa", "#c92a2a"]), vmin=-.5, vmax=3.5, interpolation="nearest")
        G = common.h3grid(key).grid
        test = d.drop_duplicates("block").query("split=='test'").block
        G[G.parent7.isin(test)].dissolve().boundary.plot(ax=ax, color="k", lw=.6)
        B.park.boundary.plot(ax=ax, color="#343a40", lw=.8, ls="--")
        ax.set_aspect(1 / np.cos(np.radians(LAT0)))
        ax.legend(handles=[Patch(color="#c92a2a", label="TP"), Patch(color="#ffa94d", label="FP (commission)"), Patch(color="#9775fa", label="FN (omission)"),
                           Patch(facecolor="none", edgecolor="k", label="тестові блоки")], frameon=False, fontsize=8, loc="lower left")
        ax.set_title(f"{title} · {key}" if len(PK) > 1 else title, fontsize=9)
    return fig


def _bar_fig(res):
    fig, ax = plt.subplots(figsize=(9, 4))
    r = res.sort_values("test_f1")
    lo = r.test_f1 - r.test_f1_ci.str.split("–").str[0].astype(float)
    hi = r.test_f1_ci.str.split("–").str[1].astype(float) - r.test_f1
    ax.barh(r.family, r.test_f1, xerr=[lo, hi], color=["#7048e8" if c else "#adb5bd" for c in r.champion])
    ax.set_xlabel("F1 на тестових блоках (95 % CI)"); ax.set_title("T1 · сімейства підходів (чемпіон — за калібруванням)")
    return fig
