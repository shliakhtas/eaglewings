"""RD · Маршрути дня (симуляція): побудова й перебудова маршрутів двох груп протягом зміни (TOPTW + rolling horizon).

Задача      щодня групи лісників (зміна й бази з config/regions.py fetch.teams) мають обійти обов'язкові зупинки — події на ранок,
            нічні сигнали, дії ареалів — і, якщо лишається час, патрулювати зони найвищого ризику рубок (прогноз LG).
            Під час зміни може надійти нова подія → маршрут перебудовується від поточних позицій груп.
            Постановка — Team Orienteering Problem with Time Windows; модель — ds/routing.py.
Дані        дні сирих задач із ds/ops_routes.py (сценарний потік сигналів на реальних місцях основного парку, scripts/03_signals.py):
            зупинки до маршрутизації + нова подія дня. Це симуляція; час у дорозі — граф доріг і треків OSM.
Еталон      немає: підходи порівнюються між собою.
Протокол    кожен підхід проганяє всі дні послідовно (давність патрулів переходить з дня в день). План будується моделлю часу
            підходу, а виконується за графом доріг. Перша половина днів — вибір, друга — тест. 95 % CI — bootstrap по днях тесту.
Метрики     on_time — частка обов'язкових зупинок, виконаних так, що група звідти ще встигає на базу до кінця зміни (основна);
            response_min — хвилини від нової події до прибуття групи; event_wait_min — очікування ранкових подій від початку зміни;
            risk_covered — сума ризику патрульованих зон за день; overtime_min — повернення після кінця зміни; idle_h — простій;
            km — кілометри обох груп; repeat_jaccard — схожість наборів патрульованих зон у сусідні дні (нижче — менш передбачувано).
Вибір       найбільша частка вчасно на днях вибору; при рівності (± 0,5 п.п.) — швидша реакція, далі — більший покритий ризик.
            Рандомізовані підходи в лідери не беруться: це компроміс «покриття ↔ непередбачуваність», показаний окремо.
Підходи
  greedy_line      база: поділ за широтою, пріоритет + найближча точка, час = пряма × 1,35 / 25 км/год, без патрулів
  greedy_road      те саме, але час — з графа доріг
  greedy_patrol    greedy_road + патрулі зон ризику найнижчим пріоритетом
  ortools          OR-Tools: TOPTW з нагородами, терміновість подій, перебудова для обох груп
  vroom            VROOM: пріоритети 0–100, необов'язкові задачі
  hybrid           VROOM обирає зупинки (лідер RS), OR-Tools шліфує порядок із терміновістю
  ortools_rand     OR-Tools + випадковий вибір зон патруля (Green Security Games / PAWS), щоб порушники не вгадували маршрут
"""
import inspect
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import datasets, metrics, ops_routes, pflow, routing, tracking
from ..tracking import ml

EXPERIMENT = "RD · Маршрути дня (симуляція: побудова й перебудова маршрутів)"
APPROACHES = [
    dict(name="greedy_line", solver="greedy", plan="line", patrol=False, rand=False, hyp="База: пряма × 1,35, поділ за широтою, без патрулів"),
    dict(name="greedy_road", solver="greedy", plan="road", patrol=False, rand=False, hyp="Та сама евристика, але реальний час із графа доріг"),
    dict(name="greedy_patrol", solver="greedy", plan="road", patrol=True, rand=False, hyp="Жадібно + патрулі зон ризику найнижчим пріоритетом"),
    dict(name="ortools", solver="ortools", plan="road", patrol=True, rand=False, hyp="Prize-collecting TOPTW (OR-Tools) з терміновістю подій"),
    dict(name="vroom", solver="vroom", plan="road", patrol=True, rand=False, hyp="VROOM: пріоритети й опційні задачі"),
    dict(name="hybrid", solver="hybrid", plan="road", patrol=True, rand=False, hyp="VROOM обирає зупинки, OR-Tools шліфує порядок із терміновістю"),
    dict(name="ortools_rand", solver="ortools", plan="road", patrol=True, rand=True, hyp="OR-Tools + рандомізація патрулів (Green Security Games / PAWS)"),
]
TIME_LIMIT_S = 1.0


def load_instances():
    days = ops_routes.instances()
    rows = []
    for d in days:
        for k, s in enumerate(d["stops"]):
            rows.append(dict(date=d["date"], stop=f"S{k}", type=s["type"], kind=s["kind"], prio=s["prio"], lat=s["lat"], lon=s["lon"], new=0, at_h=np.nan))
        if d["reroute"]:
            s = d["reroute"]["stop"]
            rows.append(dict(date=d["date"], stop="NEW", type=s["type"], kind=s["kind"], prio=s["prio"], lat=s["lat"], lon=s["lon"], new=1, at_h=d["reroute"]["at"]))
    df = pd.DataFrame(rows)
    TM = ops_routes.teams()
    ri = dict(days=days, teams=[t["id"] for t in TM], bases={t["id"]: t["base"] for t in TM}, service_h=ops_routes.SERVICE_H, shift=list(ops_routes.SHIFT))
    dv = datasets.save(
        df, "route_instances", unit="зупинка × день",
        description="Сирі задачі маршрутизації: обов'язкові зупинки кожного дня до побудови маршруту + нова подія під час зміни. Симуляція.",
        targets=["new"], features=["type", "kind", "prio", "lat", "lon", "at_h"],
        params=dict(teams=ri["teams"], shift=ri["shift"], service_h=ri["service_h"], bases=ri["bases"]),
        sources=["scripts/03_signals.py (сценарний потік сигналів на реальних місцях парку)", "OSM roads"])
    return dv, ri


def _day_stops(d, svc, patrol, last, day):
    stops = [s | dict(id=f"S{k}", service_h=svc[s["type"]], must=True, value=0, known_h=ops_routes.SHIFT[0]) for k, s in enumerate(d["stops"])]
    if patrol:
        for k, p in enumerate(routing.patrol_points()):
            days = (day - last[p["h3"]]).days if p["h3"] in last else routing.PATROL_NEVER_D
            stops.append(dict(id=f"P{k}", type="patrol", kind="patrol", prio=5, lat=p["lat"], lon=p["lon"], h3=p["h3"], risk=p["risk"], service_h=svc["patrol"],
                              must=False, value=routing.patrol_value(p["risk"], days), known_h=ops_routes.SHIFT[0]))
    rr = None
    if d["reroute"]:
        rr = dict(at=d["reroute"]["at"], stop=d["reroute"]["stop"] | dict(id="NEW", service_h=svc["event"], must=True, value=0, known_h=d["reroute"]["at"]))
    return stops, rr


def simulate(a, ri, road, line):
    """Усі дні послідовно одним підходом → рядок метрик на день + маршрути (для графіка)."""
    svc = ri["service_h"] | {"patrol": routing.PATROL_SERVICE_H}
    bases = {t: dict(lat=ri["bases"][t]["lat"], lon=ri["bases"][t]["lon"]) for t in ri["teams"]}
    S0, S1 = ri["shift"]
    teams = [dict(id=t, lat=bases[t]["lat"], lon=bases[t]["lon"], t0=S0, end=S1, base=bases[t]) for t in ri["teams"]]
    last, prev_zones, rows, plans = {}, None, [], {}
    for i, d in enumerate(ri["days"]):
        day = pd.Timestamp(d["date"])
        stops, rr = _day_stops(d, svc, a["patrol"], last, day)
        by_id = {s["id"]: s for s in stops + ([rr["stop"]] if rr else [])}
        ex, dropped, secs = routing.run_day(a["solver"], teams, stops, road, reroute=rr, plan_router=line if a["plan"] == "line" else road,
                                            exec_router=road, seed=i, rand=a["rand"], time_limit=TIME_LIMIT_S)
        visits = [(x, by_id[x["id"]], t) for t, e in ex.items() for x in e["stops"]]
        must = [s for s in by_id.values() if s["must"]]
        # вчасно = зупинку виконано, і група звідти ще встигає на базу до 16:00 (інакше це понаднормова робота)
        on_time = sum(1 for x, s, t in visits if s["must"] and (ex[t]["back_h"] <= S1 + 1e-9 or
                                                                 x["finish_h"] + road.table([s, bases[t]])[0][0, 1] <= S1 + 1e-9))
        zones = {s["h3"] for _, s, _ in visits if s["type"] == "patrol"}
        for z in zones:
            last[z] = day
        new = next((x for x, s, _ in visits if s["id"] == "NEW"), None)
        ev = [x["arrive_h"] - S0 for x, s, _ in visits if s["type"] == "event" and s["id"] != "NEW"]
        rows.append(dict(date=d["date"], must=len(must), done=sum(1 for _, s, _ in visits if s["must"]), on_time=on_time,
                         response_min=(new["arrive_h"] - rr["at"]) * 60 if new else np.nan, new_missed=int(bool(rr) and new is None),
                         event_wait_min=np.mean(ev) * 60 if ev else np.nan,
                         risk_covered=sum(s["risk"] for _, s, _ in visits if s["type"] == "patrol"), patrols=len(zones),
                         overtime_min=sum(max(0.0, e["back_h"] - S1) for e in ex.values()) * 60, idle_h=sum(max(0.0, S1 - e["back_h"]) for e in ex.values()), km=sum(e["km"] for e in ex.values()),
                         repeat_jaccard=(len(zones & prev_zones) / len(zones | prev_zones)) if prev_zones is not None and (zones | prev_zones) else np.nan,
                         rerouted=int(rr is not None), solve_s=secs))
        prev_zones = zones
        plans[d["date"]] = (ex, by_id)
    return pd.DataFrame(rows), plans


def _summary(r):
    return dict(on_time_share=r.on_time.sum() / r.must.sum(), done_share=r.done.sum() / r.must.sum(),
                response_min=r.response_min.mean(), new_missed=r.new_missed.sum(), event_wait_min=r.event_wait_min.mean(), risk_covered=r.risk_covered.mean(),
                patrols_per_day=r.patrols.mean(), overtime_min=r.overtime_min.mean(), overtime_days=(r.overtime_min > 0).mean(), idle_h=r.idle_h.mean(),
                km=r.km.mean(), repeat_jaccard=r.repeat_jaccard.mean(), solve_s=r.solve_s.mean())


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark("1. dataset: daily stops from the simulated stream")
    dv, ri = load_instances()
    road, line = routing.road_router(), routing.LineRouter()
    nd = len(ri["days"])
    half = nd // 2
    out, all_plans = [], {}
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_params(dict(days=nd, val_days=half, test_days=nd - half, reroute_days=sum(d["reroute"] is not None for d in ri["days"]), teams=len(ri["teams"]),
                           shift=str(ri["shift"]), graph_nodes=len(road.coords), graph_main_share=round(road.share_main, 3),
                           patrol_zones=len(routing.patrol_points()), patrol_tau_d=routing.PATROL_TAU_D, time_limit_s=TIME_LIMIT_S, n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/patrol_routing.py")
        ml.log_text(inspect.getsource(routing), "code/routing.py")
        pflow.mark("2. approaches: all days in order; first half pick, second half test")
        for a in APPROACHES:
            with ml.start_run(run_name=a["name"], nested=True):
                ml.set_tags(dict(approach=a["name"], solver=a["solver"], hypothesis=a["hyp"], role="approach"))
                ml.log_params({k: a[k] for k in ("solver", "plan", "patrol", "rand")})
                r, plans = simulate(a, ri, road, line)
                all_plans[a["name"]] = plans
                v, te = r.iloc[:half], r.iloc[half:].reset_index(drop=True)
                mv, mt = _summary(v), _summary(te)
                g = np.arange(len(te))
                lo, hi = metrics.block_bootstrap(g, lambda i: te.on_time.values[i].sum() / max(te.must.values[i].sum(), 1), n=n_boot)
                m = {f"val_{k}": x for k, x in mv.items()} | {f"test_{k}": x for k, x in mt.items()} | dict(ci_lo=lo, ci_hi=hi)
                m["val_on_time"], m["test_on_time"] = mv["on_time_share"], mt["on_time_share"]
                ml.log_metrics(tracking.clean(m))
                ml.log_table(r, f"days_{a['name']}.json")
                out.append(dict(approach=a["name"], solver=a["solver"], hypothesis=a["hyp"], randomized=a["rand"], **m))
                print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in out[-1].items() if k in ("approach", "val_on_time", "test_on_time", "test_response_min", "test_risk_covered", "test_km")}, flush=True)
        res = pd.DataFrame(out)
        cand = res[~res.randomized]
        top = cand[cand.val_on_time >= cand.val_on_time.max() - 0.005]
        champ = top.sort_values(["val_response_min", "val_risk_covered"], ascending=[True, False], na_position="last").iloc[0].approach
        res["champion"] = res.approach == champ
        ml.set_tags(dict(champion=champ, selection="max частка вчасно на днях вибору; ±0,5 п.п. → швидша реакція → більший ризик; без рандомізованих"))
        tracking.keep("patrol_routing", "champion", dict(approach=champ, solver=res[res.champion].solver.iloc[0]))
        tracking.log_fig(_bars(res, champ), "approaches")
        day = next((d["date"] for d in ri["days"] if d["reroute"]), ri["days"][-1]["date"])
        tracking.log_fig(_map(all_plans, ["greedy_line", champ], day, road, ri["bases"]), "routes_reroute_day")
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, champion=champ)


def _bars(res, champ):
    cols = [("test_on_time_share", "обов'язкові вчасно, частка (тест)"), ("test_response_min", "реакція на нову подію, хв"), ("test_risk_covered", "ризик покрито / день"),
            ("test_overtime_min", "понаднормово, хв / день"), ("test_km", "км / день"), ("test_repeat_jaccard", "повтор зон (Жаккар)")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (c, t) in zip(axs.flat, cols):
        ax.barh(res.approach, res[c], color=["#2a7de1" if a == champ else "#b8c0cc" for a in res.approach])
        if c == "test_on_time_share":
            ax.errorbar(res[c], res.approach, xerr=[(res[c] - res.ci_lo).clip(lower=0), (res.ci_hi - res[c]).clip(lower=0)], fmt="none", color="#333", lw=1)
        ax.set_title(t, fontsize=10)
        ax.invert_yaxis()
    fig.suptitle(f"RD · маршрути дня (симуляція); лідер — {champ}")
    fig.tight_layout()
    return fig


def _map(all_plans, names, day, road, bases):
    fig, axs = plt.subplots(1, len(names), figsize=(6 * len(names), 6), sharex=True, sharey=True)
    colors = {t["id"]: t["color"] for t in ops_routes.teams()}
    for ax, n in zip(np.atleast_1d(axs), names):
        ex, by_id = all_plans[n][day]
        base = None
        for t, e in ex.items():
            prev = base = bases[t]
            for x in e["stops"]:
                s = by_id[x["id"]]
                _, path = road.leg(prev, s)
                ax.plot([p[1] for p in path], [p[0] for p in path], color=colors.get(t, "k"), lw=1.8)
                ax.scatter(s["lon"], s["lat"], s=40 if s["must"] else 18, marker="o" if s["must"] else "s",
                           color="#e5484d" if s["id"] == "NEW" else colors.get(t, "k"), zorder=3, edgecolor="k", lw=.4)
                prev = s
            _, path = road.leg(prev, base)
            ax.plot([p[1] for p in path], [p[0] for p in path], color=colors.get(t, "k"), lw=1.8, ls="--")
            ax.text(0.02, 0.97 - 0.05 * list(ex).index(t), f"група {t}: повернення {routing_hm(e['back_h'])}, {e['km']:.0f} км",
                    transform=ax.transAxes, fontsize=9, va="top", color=colors.get(t, "k"))
        ax.scatter(base["lon"], base["lat"], marker="*", s=180, color="k", zorder=4)
        ax.set_title(f"{n} · {day}")
        ax.set_aspect(1 / np.cos(np.radians(routing.LAT0)))
    fig.tight_layout()
    return fig


def routing_hm(h):
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"
