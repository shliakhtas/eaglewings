"""XM · Крос-джерельний матчинг на потоці демо: які параметри найкраще зводять сигнали всіх джерел в одну подію.

Задача      на реальних сенсорах (FM) перевірено лише супутники. Тут весь потік: камери, фотопастки, супутник, гаряча лінія, соцмережі.
Дані        потік демо НП Дадья за 90 днів: 335 сигналів, 12 реальних, решта симульовані.
Еталон      задум генератора: сигнали одного сценарію й одного дня це одна подія. Це еталон симуляції, а не спостереження. Реальні сигнали без сценарію не оцінюються.
Протокол    вибір параметрів окремо для кожного типу події на перших 60 днях, тест на останніх 30. CI — bootstrap по подіях.
Метрика     BCubed F1, плюс точність і повнота.
Підходи     без зведення; DBSCAN у межах дня; ST-DBSCAN із сіткою радіус × час.
Обмеження   результат показує, як алгоритм поводиться на нашій симуляції, а не на реальному потоці. Симуляція сама задає, як далеко розкидані сигнали.
"""
import inspect

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from .. import datasets, metrics, tracking
from ..common import P
from ..tracking import ml
from .matching_benchmark import bcubed

EXPERIMENT = "XM · Cross-source matching (потік демо, симуляція)"
KM = [.25, .5, 1., 2., 3., 5.]
DAYS = [.5, 1., 2., 3., 5., 7.]
VAL_DAYS = 60
K_LAT = LAT0


def st(d, km, days):
    x = np.radians(d.lon.values) * 6371.0088 * np.cos(np.radians(K_LAT))
    y = np.radians(d.lat.values) * 6371.0088
    return DBSCAN(eps=km, min_samples=1).fit_predict(np.c_[x, y, d.t.values * km / days])


def day(d, km):
    out = np.zeros(len(d), int)
    n = 0
    for _, i in d.groupby(d.date.dt.date).indices.items():
        q = d.iloc[i]
        lab = DBSCAN(eps=km, min_samples=1).fit_predict(np.c_[np.radians(q.lon.values) * 6371.0088 * np.cos(np.radians(K_LAT)), np.radians(q.lat.values) * 6371.0088])
        out[i] = lab + n
        n += lab.max() + 1
    return out


def load():
    d = pd.read_csv(P / "signals.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d["t"] = (d.date - d.date.min()).dt.total_seconds() / 86400
    d["truth"] = d.scenario.astype(str) + "_" + d.date.dt.strftime("%Y%m%d")
    d["part"] = np.where(d.t < VAL_DAYS, "val", "test")
    return d


def score(p, t):
    prec, rec, f1 = bcubed(np.asarray(p), pd.factorize(np.asarray(t))[0])
    return dict(f1=f1, precision=prec, recall=rec)


def run(n_boot=300):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: demo stream (simulated)')
    D = load()
    dv = datasets.save(D[["date", "kind", "source", "lat", "lon", "conf", "simulated", "scenario", "truth", "part"]].assign(date=lambda x: x.date.astype(str)), "xm_signals", unit="сигнал потоку демо", targets=[], features=["kind", "source", "lat", "lon", "date", "conf"],
                       description="Потік сигналів демо НП Дадья: реальні й симульовані, еталон це сценарій і день генератора.", params=dict(val_days=VAL_DAYS, km=KM, days=DAYS), sources=["симуляція демо", "NASA FIRMS", "OpenStreetMap"])
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", "truth", files=True)
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/cross_source.py")
        pflow.mark('2. approaches per event type')
        for kind, g in D[D.scenario >= 0].groupby("kind"):
            g = g.reset_index(drop=True)
            cand = [dict(approach="без зведення", km=None, days=None, f=lambda q: np.arange(len(q))), *[dict(approach=f"DBSCAN за день, {k:g} км", km=k, days=None, f=lambda q, k=k: day(q, k)) for k in (1., 5.)],
                    *[dict(approach=f"ST-DBSCAN, {k:g} км, {t:g} діб", km=k, days=t, f=lambda q, k=k, t=t: st(q, k, t)) for k in KM for t in DAYS]]
            va, te = g[g.part == "val"].reset_index(drop=True), g[g.part == "test"].reset_index(drop=True)
            res = []
            for c in cand:
                v, t = score(c["f"](va), va.truth), score(c["f"](te), te.truth)
                pred = c["f"](te)
                ev = pd.factorize(te.truth)[0]
                lo, hi = metrics.block_bootstrap(ev, lambda i: score(pred[i], te.truth.values[i])["f1"], n=n_boot) if c["days"] else (np.nan, np.nan)
                res.append(dict(kind=kind, approach=c["approach"], km=c["km"], days=c["days"], val_f1=v["f1"], val_precision=v["precision"], val_recall=v["recall"], test_f1=t["f1"], test_precision=t["precision"], test_recall=t["recall"], ci_lo=lo, ci_hi=hi, n_val=len(va), n_test=len(te), events_val=va.truth.nunique(), events_test=te.truth.nunique()))
            q = pd.DataFrame(res)
            best = q[q.approach.str.startswith("ST-DBSCAN")].val_f1.idxmax()
            q["champion"] = False
            q.loc[best, "champion"] = True
            rows.append(q)
            r = q.loc[best]
            print(kind, r.approach, round(r.val_f1, 3), round(r.test_f1, 3), {a: round(b, 3) for a, b in zip(q.approach[:3], q.test_f1[:3])}, flush=True)
        res = pd.concat(rows, ignore_index=True)
        pflow.mark('3. approaches: results per algorithm')
        tracking.log_approaches(res, lambda r: f"{r.kind} · {r.approach.split(',')[0]}", label=lambda r: r.approach)
        pflow.mark('4. save')
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag)
