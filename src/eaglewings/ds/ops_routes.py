"""Маршрути груп, таски, ніч і передачі партнерам — поверх сценарного потоку сигналів основного парку (scripts/03_signals.py).

Це симуляція: події — сценарний потік на реальних місцях парку. Вмикається, коли проєкт задає групи в config/regions.py:

    fetch.teams = dict(shift=("08:00", "16:00"), night=("18:00", "08:00"),
                       groups=[dict(key="A", name="Група Рахів", base="Рахів", lat=…, lon=…, people=2, color="#2f6fdb"), …])

Дні задач (instances) — вхід гілки RD: обов'язкові зупинки кожного дня до маршрутизації (події на ранок, нічні сигнали «на ранок»,
дії ареалів за їхнім ритмом) і перша жива подія під час зміни, через яку маршрут перебудовується. Пожежі — не лісникам:
вони йдуть у передачі партнерам (199, за потреби дрон). Патрулі зон ризику — опційні зупинки з прогнозу LG (ds/routing.py).

build(solver) будує маршрути лідером RD на кожен день повзунка (минулі дні виконані) і план на завтра,
таски на 14 днів і пише data/processed/routes.json і tasks.json для застосунку й агента.
"""
import importlib.util
import json

import numpy as np
import pandas as pd

from . import common, routing
from .common import P
from eaglewings.project import LIB  # noqa: E402
from .regions import REGION

T = REGION["fetch"].get("teams") or {}
H = lambda s: int(s[:2]) + int(s[3:5]) / 60
SHIFT = (H(T.get("shift", ("08:00", "16:00"))[0]), H(T.get("shift", ("08:00", "16:00"))[1]))
SERVICE_H = {"event": 1.0, "plan": 0.5, "night": 0.75, "patrol": routing.PATROL_SERVICE_H}      # подія: виїзд + постекстрені дії
PRIO = {"event": 0, "night": 1, "критична": 2, "висока": 3, "помірна": 4, "patrol": 5}
PLAN_DAYS = 14


def on():
    return bool(T.get("groups"))


def teams():
    return [dict(id=g["key"], name=g["name"], base_name=g.get("base", ""), people=g.get("people", 2), color=g.get("color", "#2f6fdb"),
                 base=dict(lat=g["lat"], lon=g["lon"])) for g in T["groups"]]


def hm(h):
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


_S = None


def signals_module():
    """scripts/03_signals.py як модуль: ті самі правила подій, ареалів і стрічки, що в застосунку."""
    global _S
    if _S is None:
        spec = importlib.util.spec_from_file_location("signals_gen", LIB / "scripts" / "03_signals.py")
        _S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_S)
    return _S


_CTX = None


def context():
    """Потік основного парку, події, ареали, знімки по днях і стрічка — так само, як їх бачить застосунок."""
    global _CTX
    if _CTX is None:
        S = signals_module()
        ev = pd.read_pickle(P / "signals.pkl")
        cl = S.clusters(ev)
        areals = S.superareals(cl)
        cl["areal"] = cl.id.map({c: a["id"] for a in areals for c in a["cluster_ids"]}).fillna("")
        snaps = json.loads((P / "timeline.json").read_text())["snapshots"]
        _CTX = dict(S=S, ev=ev, cl=cl, areals=areals, snaps=snaps, feed=S.feed(ev, cl), now=S.NOW)
    return _CTX


def night_action(n):
    """Що робити з нічним сигналом: пожежа — дзвінок 199; вирубка/проїзд — виїзд чергового або ранкова перевірка."""
    if n["kind"] == "fire":
        return "call", "Подзвонити пожежникам (199)"
    if n["conf"] >= 0.7 or n["source"] in ("камера", "фотопастка"):
        return "go", "Виїзд чергового лісника"
    return "morning", "Перевірити зранку — у маршруті"


def night_feed(feed_items, day):
    """Живі сигнали ночі перед днем day: 18:00 попереднього дня — 08:00."""
    a, b = day - pd.Timedelta(hours=24 - H(T.get("night", ("18:00", "08:00"))[0])), day + pd.Timedelta(hours=SHIFT[0])
    return [n for n in feed_items if n["live"] and a <= pd.Timestamp(n["ts"]) <= b]


def day_instance(day, resp, plan_day, night_items, reroute=None):
    """Сирі задачі дня до маршрутизації: обов'язкові зупинки, передачі партнерам, ніч, нова подія під час зміни."""
    S = context()["S"]
    stops, handoff, night = [], [], []
    for t in resp:
        if t["kind"] == "fire":                                     # пожежі — не лісникам
            handoff.append(dict(to="Пожежна служба (199)", title=f'Пожежа · {", ".join(t["sources"])}', ref=t["id"], lat=t["lat"], lon=t["lon"], when=t["detected"],
                                drone="запит дрона до держслужби для підтвердження" if t["confidence_level"] != "висока" else ""))
        else:
            stops.append(dict(type="event", kind=t["kind"], title=t["title"], detail=f'{t["id"]} · впевненість {t["confidence_level"]} · {", ".join(t["sources"])}',
                              post=S.ACTIONS[t["kind"]]["post"], lat=t["lat"], lon=t["lon"], prio=PRIO["event"], areal=t.get("areal", ""), cluster=t["cluster"]))
    for n in night_items:
        act, txt = night_action(n)
        night.append(dict(text=n["text"], ts=n["ts"], source=n["source"], kind=n["kind"], lat=n["lat"], lon=n["lon"], action=act, action_text=txt, areal=n.get("areal", "")))
        if act == "morning" and not any(abs(s_["lat"] - n["lat"]) < 0.004 and abs(s_["lon"] - n["lon"]) < 0.005 for s_ in stops):
            stops.append(dict(type="night", kind=n["kind"], title="Перевірка нічного сигналу", detail=f'{n["text"]} · {n["ts"][11:]}',
                              lat=n["lat"], lon=n["lon"], prio=PRIO["night"], areal=n.get("areal", "")))
    by_zone = {}
    for t in plan_day:                                               # дії одного ареалу — одна зупинка
        z = by_zone.setdefault(t["zone"], dict(type="plan", kind=t["type"], title=f'{t["zone"]} · дії ареалу', actions=[], cats=[],
                                                lat=t["lat"], lon=t["lon"], prio=PRIO.get(t["level"], 4), areal=t["zone"]))
        z["actions"].append(t["action"])
        z["cats"].append(t["cat"])
    for z in by_zone.values():
        z["detail"] = ", ".join(f'{a} ({c})' for a, c in zip(z["actions"], z["cats"]))
        z["one"] = all(c == "одинична" for c in z["cats"])
        stops.append(z)
    new = None
    if reroute:
        new = dict(type="event", kind=reroute["kind"], title=reroute["title"], detail=reroute["detail"], new=True, post=S.ACTIONS[reroute["kind"]]["post"],
                   lat=reroute["lat"], lon=reroute["lon"], prio=PRIO["event"] - 1, areal=reroute.get("areal", ""))
    return dict(date=day.strftime("%Y-%m-%d"), stops=stops, handoff=handoff, night=night, reroute=dict(at=reroute["at"], stop=new) if new else None)


def instances():
    """Дні повзунка застосунку (минулі дні вікна) → сирі задачі кожного дня. Вхід гілки RD."""
    C = context()
    S, cl, snaps, fd = C["S"], C["cl"], C["snaps"], C["feed"]
    origin = C["now"].normalize() + pd.Timedelta(days=1)
    out, visited = [], set()                         # на подію виїжджають один раз — наступного дня її вже немає в маршруті
    for sn in snaps:
        day = pd.Timestamp(sn["date"])
        resp = [t for t in S.response_tasks(cl, now=day + pd.Timedelta(hours=SHIFT[0])) if t["cluster"] not in visited]
        visited |= {t["cluster"] for t in resp if t["kind"] != "fire"}
        idx = (day - origin).days
        plan_day = []
        for a in sn["areals"]:
            base = dict(zone=a["id"], type=a["kind"], level=a["level"], lat=a["lat"], lon=a["lon"])
            for k, x in enumerate(a.get("systemic", [])):
                if (idx + k + int(a["id"][2:])) % x["every_days"] == 0:
                    plan_day.append(base | dict(action=x["action"], cat="системна"))
        day_live = sorted([n for n in fd if n["live"] and n["kind"] != "fire" and
                           day + pd.Timedelta(hours=SHIFT[0] + .5) <= pd.Timestamp(n["ts"]) <= day + pd.Timedelta(hours=SHIFT[1] - 1.5)], key=lambda n: n["ts"])
        rr = None
        if day_live:                                  # перший живий сигнал під час зміни (не пожежа) — перебудова маршруту
            n = day_live[0]
            ts = pd.Timestamp(n["ts"])
            rr = dict(at=ts.hour + ts.minute / 60, kind=n["kind"], title=S.RESPONSE[n["kind"]]["title"], lat=n["lat"], lon=n["lon"], areal=n.get("areal", ""),
                      detail=f'НОВА · {n["text"]} · {n["ts"][11:]}')
        out.append(day_instance(day, resp, plan_day, night_feed(fd, day), rr))
    return out


def recurring_tasks(areals, start):
    """План на PLAN_DAYS: системні дії — за своїм ритмом; одиничні «до виконання» — один раз у найближчі дні."""
    tasks = []
    for a in areals:
        base = dict(zone=a["id"], type=a["kind"], level=a["level"], lat=a["lat"], lon=a["lon"])
        for k, x in enumerate(a["systemic"]):
            offset = (k + int(a["id"][2:])) % x["every_days"]
            for day in range(offset, PLAN_DAYS, x["every_days"]):
                tasks.append(base | dict(date=(start + pd.Timedelta(days=day)).strftime("%Y-%m-%d"), action=x["action"], cat="системна"))
        for k, x in enumerate([o for o in a["one_time"] if o["status"] == "pending"]):
            tasks.append(base | dict(date=(start + pd.Timedelta(days=(k + int(a["id"][2:])) % 3)).strftime("%Y-%m-%d"), action=x["action"], cat="одинична"))
    return sorted(tasks, key=lambda t: (t["date"], t["zone"]))


PATROL_LAST = {}           # зона ризику → дата останнього патруля


def patrol_stops(day):
    """Опційні патрулі зон найвищого ризику; цінність росте з давністю останнього патруля."""
    out = []
    for p in routing.patrol_points():
        last = PATROL_LAST.get(p["h3"])
        days = (day - last).days if last is not None else routing.PATROL_NEVER_D
        out.append(dict(type="patrol", kind="patrol", title="Патруль зони ризику рубок", h3=p["h3"], risk=p["risk"], lat=p["lat"], lon=p["lon"],
                        detail=f'{p["note"]} · ризик {p["risk"]:.2f} · ' + (f"востаннє {days} дн. тому" if last is not None else "ще не патрулювали"),
                        prio=PRIO["patrol"], areal="", value=routing.patrol_value(p["risk"], days)))
    return out


def solve_day(inst, solver, past=False):
    """Маршрути груп на день за сирими задачами inst: лідер RD, час — граф доріг OSM, перебудова о моменті нової події."""
    day = pd.Timestamp(inst["date"])
    TM = teams()
    stops = inst["stops"]
    rs = [s_ | dict(id=f"S{k}", service_h=SERVICE_H[s_["type"]], must=True, value=0, known_h=SHIFT[0]) for k, s_ in enumerate(stops)]
    rs += [p | dict(id=f"P{k}", service_h=SERVICE_H["patrol"], must=False, known_h=SHIFT[0]) for k, p in enumerate(patrol_stops(day))]
    rr = inst["reroute"]
    rr = dict(at=rr["at"], stop=rr["stop"] | dict(id="NEW", service_h=SERVICE_H["event"], must=True, value=0, known_h=rr["at"])) if rr else None
    tms = [dict(id=t["id"], lat=t["base"]["lat"], lon=t["base"]["lon"], t0=SHIFT[0], end=SHIFT[1], base=t["base"]) for t in TM]
    router = routing.road_router()
    ex, dropped, _ = routing.run_day(solver, tms, rs, router, reroute=rr)
    by_id = {s_["id"]: s_ for s_ in rs + ([rr["stop"]] if rr else [])}
    public = lambda s_: {k: v for k, v in s_.items() if k not in ("id", "service_h", "must", "value", "known_h")}
    out = []
    for tm in TM:
        base = tm["base"]
        e, prev, seq = ex[tm["id"]], base, []
        for k, x in enumerate(e["stops"]):
            s_ = by_id[x["id"]]
            leg_km, path = router.leg(prev, s_)
            status = "done" if past or (e["rerouted"] and k < e["stops_before"]) else "new" if s_.get("new") else "planned"
            seq.append(public(s_) | dict(team=tm["id"], eta=hm(x["arrive_h"]), leg_km=round(leg_km, 1), status=status, path=path))
            prev = s_
            if past and s_["type"] == "patrol":
                PATROL_LAST[s_["h3"]] = day
        home_km, path_home = router.leg(prev, base)
        info = dict(rerouted=False)
        if e["rerouted"]:
            info = dict(rerouted=True, at=hm(rr["at"]), reason=rr["stop"]["detail"], stops_before=e["stops_before"],
                        old_rest=[dict(lat=by_id[i]["lat"], lon=by_id[i]["lon"], title=by_id[i]["title"]) for i in e["old_rest"]])
        out.append(tm | dict(stops=seq, path_home=path_home, km=round(sum(x["leg_km"] for x in seq) + home_km, 1), end=hm(e["back_h"]),
                             hours=round(e["back_h"] - SHIFT[0], 1), patrols=sum(x["type"] == "patrol" for x in seq)) | info)
    deferred = [public(by_id[i]) | dict(team="—") for i in dropped if by_id[i]["must"]]
    B = common.base(common._park.MAIN)
    must = len(stops) + (1 if rr else 0)
    return dict(date=inst["date"], past=past, shift=f"{hm(SHIFT[0])}–{hm(SHIFT[1])}", teams=out, handoff=inst["handoff"], deferred=deferred, night=inst["night"],
                solver=solver, simulated=True,
                coverage=dict(total=must, done=sum(x["type"] != "patrol" for t in out for x in t["stops"]), patrols=sum(t["patrols"] for t in out),
                              people=sum(t["people"] for t in TM), groups=len(TM), area_km2=round(float(B.park_u.area.iloc[0]) / 1e6)))


def build(solver):
    """Маршрути лідером RD: кожен день повзунка (виконані) і план на завтра з перебудовою; таски на 14 днів → routes.json, tasks.json."""
    C = context()
    PATROL_LAST.clear()
    inst = instances()
    tl = [solve_day(d, solver, past=True) for d in inst]                 # спершу історія — з неї давність патрулів для плану на завтра
    S, cl, fd = C["S"], C["cl"], C["feed"]
    day = C["now"].normalize() + pd.Timedelta(days=1)
    resp = S.response_tasks(cl)
    plan = recurring_tasks(C["areals"], day)
    nt = night_feed(fd, C["now"].normalize())
    base = solve_day(day_instance(day, resp, [t for t in plan if t["date"] == day.strftime("%Y-%m-%d")], nt), solver)
    tb = base["teams"][-1]                                              # нова подія посеред дня — біля 3-ї зупинки останньої групи (сценарій)
    anchor = tb["stops"][min(3, len(tb["stops"]) - 1)] if tb["stops"] else tb["base"]
    rr = dict(at=SHIFT[0] + 2 + 40 / 60, kind="logging", title=S.RESPONSE["logging"]["title"], lat=anchor["lat"] - 0.012, lon=anchor["lon"] + 0.018,
              detail="НОВА · дзвінок на гарячу лінію: чути бензопилу просто зараз")
    today = solve_day(day_instance(day, resp, [t for t in plan if t["date"] == day.strftime("%Y-%m-%d")], nt, rr), solver)
    routes = dict(today=today, timeline=tl, solver=solver, teams=teams(), shift=[hm(SHIFT[0]), hm(SHIFT[1])], simulated=True)
    (P / "routes.json").write_text(json.dumps(routes, ensure_ascii=False, default=str))
    tasks = task_list(today, tl, plan)
    (P / "tasks.json").write_text(json.dumps(dict(tasks=tasks, simulated=True), ensure_ascii=False, default=str))
    return routes, tasks


TYPE_UA = {"event": "реагування", "plan": "дії ареалу", "night": "нічний сигнал", "patrol": "патруль ризику", "handoff": "передача партнерам",
           "night_alert": "нічний алерт", "one": "одинична дія", "system": "системна дія"}


def task_list(today, tl, plan):
    """Усі задачі в одній таблиці: сьогоднішні виїзди з маршруту, дії ареалів на 14 днів, ніч, передачі партнерам, патрулі."""
    out = []
    for t in today["teams"]:
        for x in t["stops"]:
            ty = "one" if x["type"] == "plan" and x.get("one") else x["type"]
            out.append(dict(date=today["date"], time=x["eta"], type=ty, type_ua=TYPE_UA.get(ty, ty), title=x["title"], detail=x.get("detail", ""),
                            areal=x.get("areal", ""), team=t["id"], team_name=t["name"], status="new" if x["status"] in ("planned", "new") else "done",
                            lat=x["lat"], lon=x["lon"], kind=x.get("kind", "")))
    for n in today["night"]:
        out.append(dict(date=today["date"], time=n["ts"][11:], type="night_alert", type_ua=TYPE_UA["night_alert"], title=n["action_text"], detail=n["text"],
                        areal=n.get("areal", ""), team="черговий" if n["action"] == "go" else "—", team_name="", status="new", lat=n["lat"], lon=n["lon"], kind=n["kind"]))
    for h in today["handoff"]:
        out.append(dict(date=today["date"], time=h["when"][11:], type="handoff", type_ua=TYPE_UA["handoff"], title=f'{h["to"]}: {h["title"]}', detail=h.get("drone", ""),
                        areal="", team="партнери", team_name="", status="new", lat=h["lat"], lon=h["lon"], kind="fire"))
    first = today["date"]
    for p in plan:
        if p["date"] == first:
            continue                                                # сьогоднішні дії вже в маршруті
        ty = "one" if p["cat"] == "одинична" else "system"
        out.append(dict(date=p["date"], time="", type=ty, type_ua=TYPE_UA[ty], title=p["action"], detail=f'{p["zone"]} · рівень {p["level"]}',
                        areal=p["zone"], team="—", team_name="", status="new", lat=p["lat"], lon=p["lon"], kind=p["type"]))
    for i, x in enumerate(out):
        x["id"] = f"T-{i + 1:03d}"
    return out
