"""AS · Зони інтересу на симуляції: чи правильно ареали застосунку читають, що відбувається на місці.

Задача      ареали застосунку (scripts/03_signals.py, superareals) після зведення сигналів (XM) кажуть про кожну зону інтересу: тип
            (пожежа, вирубка, незаконний проїзд), життєвий цикл (активний, згас, відновився, новий), динаміку (розширюється,
            звужується, стабільний), регулярність і джерела. Тут перевіряємо, наскільки ці риси збігаються з тим, що закладено в сценарій.
Дані        генератор потоку застосунку, прогнаний багато разів з різною випадковістю (місця на треках OSM, які джерела побачили подію,
            розкид точок) окремо для кожного парку. Знімки Esri не качаються: супутник «бачить» у рамці ~0,9 × 0,9 км навколо місця.
Еталон      задум генератора: сценарій = зона інтересу з відомими типом, життєвим циклом, динамікою, регулярністю й джерелами.
            Це еталон симуляції, а не спостереження.
Протокол    прогони 0…N_VAL−1 — вибір підходу, решта — тест. Тест не бере участі у виборі. CI — bootstrap по прогонах.
Метрики     на кожен прогін п'ять оцінок: зведення сигналів у зони (BCubed F1: сигнали одного сценарію в одному ареалі);
            точність життєвого циклу, динаміки (лише де ареал має ≥ 4 подій) і регулярності на рівні сценарію; збіг джерел (Jaccard).
            Головна метрика — середнє п'яти.
Підходи     радіус ареалу (множник до AREAL_EPS_KM застосунку) × поріг динаміки (зміна розкиду, %); «застосунок» — ×1 і 30 %.
            Базова лінія «прості правила»: згас, якщо остання подія давніше 21 дня, інакше активний; динаміка завжди «стабільний».
Обмеження   результат показує, як правила ареалів поводяться на нашій симуляції; генератор сам задає розкид і частоту подій,
            тож висока точність тут не доводить роботу на реальному потоці.
"""
import importlib.util
import inspect

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import pflow
from .. import datasets, metrics, park as PK, tracking
from ..tracking import ml
from ...project import LIB
from .matching_benchmark import bcubed

EXPERIMENT = "AS · Зони інтересу (ареали застосунку на симуляції)"
N_RUNS, N_VAL = 100, 50
EPS = (.5, 1., 1.5)
THR = (15, 30, 50)
TRAITS = ["grouping", "lifecycle", "dynamics", "regularity", "sources"]
LIFE = {"згас": "faded", "відновився": "resumed", "новий": "new", "активний": "active"}
DYN = {"expand": "розширюється", "shrink": "звужується", None: "стабільний"}


def signals():
    """Генератор потоку застосунку як модуль (той самий код, що будує ареали в застосунку)."""
    spec = importlib.util.spec_from_file_location("eaglewings_signals", LIB / "scripts" / "03_signals.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


def true_regularity(every):
    if every is None:
        return "разово"
    return "щотижня" if every <= 8 else "раз на 1–2 тижні" if every <= 16 else "раз на місяць" if every <= 35 else "епізодично"


def pred_life(txt, naive_last=None):
    if naive_last is not None:
        return "faded" if naive_last > 21 else "active"
    return next(v for k, v in LIFE.items() if txt.startswith(k))


def pred_dyn(ch, thr):
    if ch is None:
        return "мало даних"
    return "розширюється" if ch > thr else "звужується" if ch < -thr else "стабільний"


def simulate(S, key, seed):
    """Один прогін генератора для парку key: сигнали й події (кластери XM)."""
    park = gpd.read_file(PK.processed(key) / "park.geojson")
    S.RAW = PK.raw(key)
    S.LAT0 = round(float(park.geometry.iloc[0].centroid.y), 2)
    roads = S.load_roads()
    S.rng = np.random.default_rng(seed)
    S.DEVICES.clear()
    S.SCENES.clear()

    def scenes(sites):                                   # без завантаження знімків: рамка 0,008° × 0,0105° навколо місця
        for k, (sc, site) in enumerate(zip(S.SCENARIOS, sites)):
            if "satellite" in sc["sources"]:
                c = site["center"]
                S.SCENES[k] = dict(id=f"sim{k:02d}", img="", bounds=[[c.y - .004, c.x - .00525], [c.y + .004, c.x + .00525]], kind=sc["kind"])
    S.satellite_scenes = scenes
    ev = S.finish(S.simulate_stream(park, roads, None, None))
    return ev, S.clusters(ev)


def evaluate(S, ev, cl, eps, thr, naive=False):
    """Ареали з радіусом ×eps і оцінка рис проти сценаріїв: рядок на сценарій і BCubed на прогін."""
    base = dict(S.AREAL_EPS_KM)
    S.AREAL_EPS_KM = {k: v * eps for k, v in base.items()}
    try:
        ars = S.superareals(cl)
    finally:
        S.AREAL_EPS_KM = base
    ev_areal = pd.Series("", index=ev.index)
    by_cl = cl.set_index("id").idx
    for a in ars:
        for c in a["cluster_ids"]:
            ev_areal.loc[by_cl[c]] = a["id"]
    sim = ev[(ev.scenario >= 0) & (ev_areal != "")]
    _, _, f1 = bcubed(ev_areal[sim.index].values, sim.scenario.values)
    rows = []
    for k, sc in enumerate(S.SCENARIOS):
        mine = ev_areal[ev.index[ev.scenario == k]]
        mine = mine[mine != ""]
        if not len(mine):
            continue
        a = next(x for x in ars if x["id"] == mine.value_counts().idxmax())
        last = (S.NOW.normalize() - pd.Timestamp(a["last"])).days
        reg = a["regularity"].split(",")[0]
        dyn = "стабільний" if naive else pred_dyn(a["dyn_change"], thr)
        seen = {s for s, n in a["per_source"].items() if n > 0}
        rows.append(dict(scenario=k, kind=sc["kind"], note=sc["note"],
                         life_true=sc["life"], life_pred=pred_life(a["lifecycle"], last if naive else None),
                         dyn_true=DYN[sc["dyn"]], dyn_pred=dyn, reg_true=true_regularity(sc["every"]), reg_pred=reg,
                         src_true=",".join(sorted(sc["sources"])), src_pred=",".join(sorted(seen)),
                         sources=len(seen & set(sc["sources"])) / len(seen | set(sc["sources"]))))
    return pd.DataFrame(rows), f1, len(ars)


def run_scores(q, f1):
    d = q[q.dyn_pred != "мало даних"]
    return dict(grouping=f1, lifecycle=float((q.life_true == q.life_pred).mean()),
                dynamics=float((d.dyn_true == d.dyn_pred).mean()) if len(d) else np.nan,
                regularity=float((q.reg_true == q.reg_pred).mean()), sources=float(q.sources.mean()))


def approaches():
    out = [dict(approach=f"радіус ×{e:g}, поріг {t} %", eps=e, thr=t, naive=False) for e in EPS for t in THR]
    for a in out:
        if a["eps"] == 1 and a["thr"] == 30:
            a["approach"] = "застосунок: радіус ×1, поріг 30 %"
    return out + [dict(approach="прості правила", eps=1., thr=30, naive=True)]


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    S = signals()
    A = approaches()
    parks = list(PK.PARKS)
    pflow.mark("1. dataset: generator runs × approaches (simulated)")
    per_run, per_sc = [], []
    for key in parks:
        for seed in range(N_RUNS):
            ev, cl = simulate(S, key, seed)
            part = "val" if seed < N_VAL else "test"
            for a in A:
                q, f1, n_ar = evaluate(S, ev, cl, a["eps"], a["thr"], a["naive"])
                per_run.append(dict(park=key, seed=seed, part=part, approach=a["approach"], areals=n_ar, signals=len(ev), **run_scores(q, f1)))
                per_sc.append(q.assign(park=key, seed=seed, part=part, approach=a["approach"]))
        print(key, "прогонів:", N_RUNS, flush=True)
    R_ = pd.DataFrame(per_run)
    R_["score"] = R_[TRAITS].mean(axis=1)
    Q = pd.concat(per_sc, ignore_index=True)
    Q["life_ok"] = (Q.life_true == Q.life_pred).astype(int)
    Q["dyn_ok"] = np.where(Q.dyn_pred == "мало даних", np.nan, (Q.dyn_true == Q.dyn_pred).astype(float))
    Q["reg_ok"] = (Q.reg_true == Q.reg_pred).astype(int)
    dv = datasets.save(Q, "as_areals", unit="сценарій × прогін × підхід",
                       targets=["life_ok", "dyn_ok", "reg_ok"], features=["life_true", "life_pred", "dyn_true", "dyn_pred", "reg_true", "reg_pred", "src_true", "src_pred", "sources"],
                       description="Зони інтересу генератора застосунку: еталон сценарію проти рис ареалу, для кожного прогону, парку й підходу. Симуляція.",
                       params=dict(n_runs=N_RUNS, n_val=N_VAL, eps=list(EPS), thr=list(THR)), sources=["симуляція застосунку", "OpenStreetMap"])
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", "life_ok", files=True)
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/areal_sim.py")
        pflow.mark("2. approaches: pick on validation runs, report test runs")
        for a in A:
            g = R_[R_.approach == a["approach"]]
            va, te = g[g.part == "val"], g[g.part == "test"].reset_index(drop=True)
            lo, hi = metrics.block_bootstrap(te.park + te.seed.astype(str), lambda i: float(te.score.values[i].mean()), n=n_boot)
            r = dict(approach=a["approach"], eps=a["eps"], thr=a["thr"], val_score=float(va.score.mean()), test_score=float(te.score.mean()),
                     ci_lo=lo, ci_hi=hi, n_val=len(va), n_test=len(te), areals=float(te.areals.mean()))
            r |= {f"test_{t}": float(te[t].mean()) for t in TRAITS}
            for key in parks:
                r[f"test_score_{key}"] = float(te[te.park == key].score.mean())
            rows.append(r)
        res = pd.DataFrame(rows)
        res["champion"] = False
        res.loc[res[res.approach != "прості правила"].val_score.idxmax(), "champion"] = True
        c = res[res.champion].iloc[0]
        print("лідер:", c.approach, round(c.val_score, 3), "тест", round(c.test_score, 3), f"({c.ci_lo:.3f}–{c.ci_hi:.3f})", flush=True)
        pflow.mark("3. approaches: results per algorithm")
        tracking.log_approaches(res, lambda r: r.approach.split(":")[0], label=lambda r: r.approach)
        pflow.mark("4. save")
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag)
