"""RS · Розв'язувач маршрутів на стандартних задачах TOPTW із відомими найкращими результатами.

Задача      маршрути груп (RD) будує розв'язувач. Наскільки його розв'язок далекий від найкращого відомого — міряємо на стандартних задачах,
            де найкращий результат відомий; від території не залежить.
Дані        Solomon (c, r, rc; 100 точок) і Cordeau (pr01–10; 48–288 точок) у постановці TOPTW Righini & Salani (2006, 2008), бібліотека KU Leuven,
            × 1, 2 і 4 команди. Еталон — найкращі результати з таблиці KU Leuven (Souffriau et al. 2013). Це результати інших алгоритмів,
            а не доведені оптимуми, тому від'ємний розрив можливий: розв'язок кращий за опублікований.
Протокол    відстань = евклідова, округлена вниз до 0,1 (стандарт цих задач). Кожен розв'язок перевіряє окремий симулятор (часові вікна,
            кінець маршруту), а не розв'язувач: недопустимий маршрут дає 0. Вибір лідера на Solomon, тест на Cordeau (інакше лідер обирався б
            на тих самих задачах, на яких міряється). CI — bootstrap по задачах тесту.
Метрики     розрив = (еталон − результат) / еталон, % — основна (менше краще); частка задач із розривом ≤ 5 %; час, с.
Підходи
  greedy        жадібно: найбільша цінність на хвилину дороги; база
  vroom         VROOM: пріоритети = цінність
  ortools_1s    OR-Tools, 1 с
  ortools_5s    OR-Tools, 5 с — скільки дає більше часу
  hybrid_1s     VROOM дає старт, OR-Tools дошліфовує, 1 с
Вибір       найменший середній розрив на Solomon; більше 10 с на задачу не допускається. Лідер стає розв'язувачем для RD.
"""
import inspect
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import datasets, metrics, pflow, routing, tracking
from ..common import ROOT
from ..tracking import ml

EXPERIMENT = "RS · Розв'язувач маршрутів (TOPTW, найкращі відомі результати)"
D = ROOT / "data" / "raw" / "benchmarks" / "toptw"
TEAMS = {"m1": 1, "m2": 2, "m4": 4}
SCALE = 10                                       # 0,1 одиниці відстані = 1 ціла одиниця
APPROACHES = [dict(name="greedy", solver="greedy", limit=0, hyp="Цінність на хвилину дороги; база порівняння"),
              dict(name="vroom", solver="vroom", limit=0, hyp="Готовий розв'язувач із пріоритетами"),
              dict(name="ortools_1s", solver="ortools", limit=1, hyp="Продуктове налаштування: 1 с"),
              dict(name="ortools_5s", solver="ortools", limit=5, hyp="Більше часу на локальний пошук"),
              dict(name="hybrid_1s", solver="hybrid", limit=1, hyp="VROOM обирає, OR-Tools шліфує: 1 с")]


def parse(name):
    """→ (T, depot + точки): матриця часу (цілі ×10), сервіс, цінність, вікна."""
    L = [x.split() for x in (D / "inst" / name).read_text().splitlines() if x.strip()]
    rows = [r for r in L[1:] if len(r) >= 8 and r[0].isdigit()]
    if len(L[1]) >= 8:                           # у частині файлів другого рядка немає
        rows = [r for r in L[1:] if len(r) >= 8]
    p = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[-2]), float(r[-1])] for r in rows])
    T = np.floor(np.hypot(*(p[:, None, :2] - p[None, :, :2]).transpose(2, 0, 1)) * SCALE).astype(int)
    return T, p


def verify(T, p, routes):
    """Незалежна перевірка: вікна, кінець зміни, кожна точка один раз. → цінність або None."""
    seen, score = set(), 0.
    end = p[0, 5] * SCALE
    for seq in routes:
        t, cur = 0., 0
        for i in seq:
            if i in seen:
                return None
            seen.add(i)
            t = max(t + T[cur, i], p[i, 4] * SCALE)
            if t > p[i, 5] * SCALE:
                return None
            t += p[i, 2] * SCALE
            score += p[i, 3]
            cur = i
        if t + T[cur, 0] > end:
            return None
    return score


def greedy(T, p, m):
    left, routes = set(range(1, len(p))), []
    end = p[0, 5] * SCALE
    for _ in range(m):
        t, cur, seq = 0., 0, []
        while True:
            best, bv = None, -1.
            for i in left:
                a = max(t + T[cur, i], p[i, 4] * SCALE)
                if a <= p[i, 5] * SCALE and a + p[i, 2] * SCALE + T[i, 0] <= end:
                    v = p[i, 3] / (a - t + p[i, 2] * SCALE + 1)
                    if v > bv:
                        best, bv = i, v
            if best is None:
                break
            t = max(t + T[cur, best], p[best, 4] * SCALE) + p[best, 2] * SCALE
            cur = best
            seq.append(best)
            left.discard(best)
        routes.append(seq)
    return routes


def library(T, p, m, solver, limit):
    """Ті самі функції routing.py, що в продукті; одиниці: хвилини → цілі ×10 (H = T / 60)."""
    base = dict(lat=0, lon=0)
    teams = [dict(id=f"t{k}", lat=0, lon=0, t0=0., end=p[0, 5] * SCALE / 60, base=base) for k in range(m)]
    prof = p[1:, 3]
    stops = [dict(id=i + 1, lat=0, lon=0, service_h=p[i + 1, 2] * SCALE / 60, value=float(prof[i]) * 1e5, must=False, type="bench", prio=0, known_h=0.,
                  tw=(p[i + 1, 4] * SCALE / 60, p[i + 1, 5] * SCALE / 60), priority=max(1, min(100, int(prof[i])))) for i in range(len(prof))]
    n = 2 * m + len(stops)
    idx = np.r_[[0] * (2 * m), np.arange(1, len(p))]
    H = T[np.ix_(idx, idx)] / 60.
    nodes = routing._nodes(teams, stops)
    plan, _ = routing.SOLVERS[solver](teams, stops, H, nodes, time_limit=limit)
    assert H.shape[0] == n
    return [plan[t["id"]] for t in teams]


def job(a):
    inst, mk, m, best, ap = a
    T, p = parse(inst)
    t = time.perf_counter()
    routes = greedy(T, p, m) if ap["solver"] == "greedy" else library(T, p, m, ap["solver"], ap["limit"])
    secs = time.perf_counter() - t
    v = verify(T, p, routes)
    return dict(instance=inst, teams=m, approach=ap["name"], best_known=best, score=v or 0., feasible=v is not None, gap=100 * (best - (v or 0.)) / best, secs=secs, n=len(p) - 1)


def best_known():
    x = pd.ExcelFile(D / "ResultsTOPTW", engine="xlrd")
    rows = []
    for mk, m in TEAMS.items():
        d = x.parse(mk)
        for r in d.itertuples(index=False):
            if isinstance(r[0], str) and r[0].endswith(".txt") and (D / "inst" / r[0]).exists():
                rows.append(dict(instance=r[0], m_key=mk, teams=m, best_known=float(r[1])))
    B = pd.DataFrame(rows)
    B["split"] = np.where(B.instance.str.startswith("pr"), "test", np.where(B.instance.str.match(r"^(c|r|rc)\d"), "val", "other"))
    return B[B.split != "other"].reset_index(drop=True)


def run(n_boot=1000):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark("1. dataset: TOPTW instances and best known results")
    B = best_known()
    dv = datasets.save(B, "toptw_instances", unit="задача TOPTW × кількість команд", targets=[], features=["instance", "teams", "best_known", "split"],
                       description="Стандартні задачі TOPTW (Solomon — вибір, Cordeau — тест; постановка Righini–Salani) з найкращими відомими результатами KU Leuven.",
                       params=dict(scale=SCALE, teams=TEAMS), sources=["KU Leuven Orienteering Problem Library", "Souffriau et al. 2013"])
    rows, per = [], {}
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_params(dict(instances=B.instance.nunique(), val=int((B.split == "val").sum()), test=int((B.split == "test").sum()), teams=str(list(TEAMS.values())), n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/solver_benchmark.py")
        pflow.mark("2. solvers on Solomon (pick) and Cordeau (test)")
        for a in APPROACHES:
            with ml.start_run(run_name=a["name"], nested=True):
                ml.set_tags(dict(approach=a["name"], kind=a["solver"], hypothesis=a["hyp"], role="approach"))
                with ProcessPoolExecutor(6) as ex:
                    R = pd.DataFrame(ex.map(job, [(r.instance, r.m_key, r.teams, r.best_known, a) for r in B.itertuples()]))
                R["split"] = B.split.values
                per[a["name"]] = R
                v, te = R[R.split == "val"], R[R.split == "test"]
                g = te.gap.values
                lo, hi = metrics.block_bootstrap(np.arange(len(g)), lambda i: g[i].mean(), n=n_boot)
                m = dict(val_gap=v.gap.mean(), test_gap=g.mean(), gap_ci_lo=lo, gap_ci_hi=hi, test_gap_median=float(np.median(g)), test_within_5=(g <= 5).mean(),
                         val_within_5=(v.gap.values <= 5).mean(), feasible=R.feasible.mean(), secs=R.secs.mean(), secs_max=R.secs.max(),
                         **{f"test_gap_m{t}": te[te.teams == t].gap.mean() for t in TEAMS.values()})
                ml.log_metrics(tracking.clean(m))
                ml.log_table(R, f"instances_{a['name']}.json")
                rows.append(dict(approach=a["name"], solver=a["solver"], hypothesis=a["hyp"], gap_ci=f"{lo:.1f}–{hi:.1f}", **m))
                print({k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in rows[-1].items() if k != "hypothesis"}, flush=True)
        res = pd.DataFrame(rows)
        ok = res[res.secs <= 10]
        champ = ok.sort_values("val_gap").iloc[0].approach
        res["champion"] = res.approach == champ
        ml.set_tags(dict(champion=champ, selection="min середній розрив на Solomon, час ≤ 10 с; тест на Cordeau"))
        tracking.keep("solver_benchmark", "champion", dict(approach=champ, solver=res[res.champion].solver.iloc[0]))
        tracking.log_fig(_fig(res), "gap")
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champ)


def _fig(res):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(res.approach, res.test_gap, xerr=[(res.test_gap - res.gap_ci_lo).clip(lower=0), (res.gap_ci_hi - res.test_gap).clip(lower=0)], color=["#2a7de1" if c else "#b8c0cc" for c in res.champion])
    ax.invert_yaxis()
    ax.set(xlabel="розрив до найкращого відомого на Cordeau (тест), %", title="RS · TOPTW")
    fig.tight_layout()
    return fig
