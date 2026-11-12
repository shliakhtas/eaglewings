"""FG_risk (режим transfer) · Чи вгадує готова модель з іншої території, які ділянки парку горять.

Задача      своєї моделі ризику вчити нема на чому: великих пожеж на території немає. Беремо модель-переможця гілки FG_risk
            з іншого проєкту EagleWings (REGION["transfer"]) без змін і перевіряємо, чи ранжує вона гекси, де були теплові точки.
Дані        h3_risk_cells (усі парки, H3 res 9): рельєф SRTM, лісистість Hansen, дороги, села й межа з OSM — ті самі ознаки, що в моделі-джерелі.
Еталон      гекс, у який за роки FIRMS потрапила хоч одна теплова точка (VIIRS n/h, MODIS ≥ 30). Це сигнал вогню, а не підтверджена лісова пожежа.
Протокол    лише тест: на цій території нічого не вчиться й нічого не обирається. 95 % CI — bootstrap по блоках H3 res 6.
            Окремо для кожного парку (колонки *_<парк>).
Метрики     ROC AUC — основна (ранжування); recall@20 % — частка гексів із сигналом у 20 % найризиковішої площі; PR AUC — довідково.
Підходи
  random            випадкове ранжування: ROC AUC ≈ 0,5
  near_people       ближче до доріг і сіл — ризикованіше (просте правило без навчання)
  transfer_FG_risk  готова модель іншого проєкту як є
Обмеження   теплові точки на новій території можуть бути переважно весняними пали поза лісом; модель-джерело вчилась на інших пожежах і іншому кліматі.
"""
import inspect

import h3
import numpy as np
import pandas as pd

from .. import common, datasets, metrics, park, pflow, tracking, transfer
from ..tracking import ml

EXPERIMENT = "FG_risk · готова модель ризику на гексах території"


def labels(df):
    """1 — у гекс за роки FIRMS потрапила теплова точка (точки свого парку з буфером)."""
    hit = set()
    for k in park.PARKS:
        f = common.base(k).firms
        hit |= {h3.latlng_to_cell(a, b, common.H3_RES) for a, b in zip(f.latitude, f.longitude)}
    return df.h3.isin(hit).astype(int).values


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark("1. dataset: H3 cells of all parks")
    dv = datasets.get("h3_risk_cells")
    df = dv.df.copy()
    if "park" not in df:
        df["park"] = park.MAIN
    y = labels(df)
    mdl, card = transfer.model("FG_risk")
    sc = {"random": np.random.default_rng(0).random(len(df)),
          "near_people": -(df.d_road.rank(pct=True) + df.d_village.rank(pct=True)).values}
    hyp = {"random": "Випадкове ранжування (нижня межа)", "near_people": "Ближче до доріг і сіл — більше займань"}
    if mdl is not None:
        feats = list(getattr(mdl, "feature_name_", None) or getattr(mdl, "feature_names_in_", None) or [])
        sc[transfer.label("FG_risk")] = mdl.predict_proba(df[feats])[:, 1]
        hyp[transfer.label("FG_risk")] = f"{transfer.hyp()}: {card.get('approach')}, ознаки {', '.join(feats)}"
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_params(dict(cells=len(df), positives=int(y.sum()), source=str(transfer.SRC), source_card=str(card)[:400], n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/risk_transfer.py")
        pflow.mark("2. test only: ranking of cells with heat points")
        for name, s in sc.items():
            m = metrics.ranking(y, s)
            lo, hi = metrics.block_bootstrap(df.parent6.values, lambda i: metrics.ranking(y[i], s[i])["roc_auc"], n=n_boot)
            r = dict(approach=name, hypothesis=hyp[name], test_roc_auc=m["roc_auc"], test_roc_auc_ci=f"{lo:.2f}–{hi:.2f}", ci_lo=lo, ci_hi=hi,
                     recall_at_20=m["recall_at_20"], pr_auc=m["pr_auc"], prevalence=m["prevalence"], positives=m["positives"], cells=len(df),
                     transfer=name.startswith("transfer_"), champion=False)
            for k in park.PARKS:
                i = np.flatnonzero(df.park.values == k)
                ok = len(i) and 0 < y[i].sum() < len(i)
                r[f"test_roc_auc_{k}"] = metrics.ranking(y[i], s[i])["roc_auc"] if ok else np.nan
                r[f"positives_{k}"] = int(y[i].sum()) if len(i) else 0
            with ml.start_run(run_name=name, nested=True):
                ml.set_tags(dict(approach=name, kind="transfer" if r["transfer"] else "baseline", hypothesis=hyp[name], role="approach"))
                ml.log_metrics(tracking.clean({k: v for k, v in r.items() if isinstance(v, (int, float))}))
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "hypothesis"}, flush=True)
        res = pd.DataFrame(rows)
        tracking.keep("risk_transfer", "cells", dict(h3=df.h3.tolist(), park=df.park.tolist(), y=y.tolist(),
                                                      **{n: pd.Series(s).rank(pct=True).round(3).tolist() for n, s in sc.items() if n != "random"}))
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, source=str(transfer.SRC))
