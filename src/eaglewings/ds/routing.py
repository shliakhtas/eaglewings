"""Маршрути патрулів: граф доріг OSM → матриця часу → розв'язувачі TOPTW → перебудова протягом зміни.

Постановка  Team Orienteering Problem with Time Windows (TOPTW): кілька груп, зміна 08:00–16:00, старт і фініш на базі.
            Кожна зупинка має час на місці й «цінність»; усе за зміну не обійти → треба вибрати, що відвідати і в якому порядку.
            Обов'язкові зупинки (події, нічні сигнали, дії ареалів) — штраф за пропуск на порядки більший за будь-який шлях;
            патрульні точки (зони ризику з прогнозу) — опційні, цінність = ризик × давність останнього патрулю.
            Події також мають «терміновість»: кожна хвилина запізнення від моменту, коли подія стала відома, — штраф.
Час у дорозі  власний граф доріг OSM (scipy csgraph, Дейкстра) зі швидкістю за типом дороги; від дороги до точки — пішки
            4 км/год з коефіцієнтом обходу 1,3. Альтернатива для продакшну — OSRM / Valhalla з тим самим інтерфейсом table().
Розв'язувачі
  greedy     проста базова лінія: групи діляться за медіаною широти; далі найвищий пріоритет (±1 рівень) і найближча точка
  ortools    Google OR-Tools Routing: disjunction зі штрафом = цінність (prize-collecting), вимір часу, м'яка верхня межа
             часу прибуття для подій (терміновість), guided local search
  vroom      VROOM (pyvroom): пріоритети 0–100 й опційні задачі, часові вікна груп
  hybrid     VROOM обирає зупинки, OR-Tools стартує з його розв'язку й шліфує порядок із терміновістю
  *_rand     рандомізація цінності патрульних точок (логнормальний шум, seed = день) — ідея Green Security Games / PAWS:
             детермінований маршрут порушники вивчають, тож ціною невеликої втрати покриття маршрут стає менш передбачуваним
Перебудова  rolling horizon: нова подія під час зміни → пройдене фіксується, поточну зупинку група дороблює, решта задачі
            для всіх груп розв'язується заново від їхніх поточних позицій (greedy — лише для групи, якій дісталась подія).
"""
import math
import time
from functools import lru_cache

import geopandas as gpd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .regions import LAT0

SPEED_KMH = {"trunk": 60, "trunk_link": 45, "primary": 60, "secondary": 50, "secondary_link": 40, "tertiary": 40, "tertiary_link": 35,
             "unclassified": 30, "residential": 25, "living_street": 15, "service": 15, "track": 15, "construction": 10,
             "path": 4, "footway": 4, "steps": 3}
WALK_KMH, WALK_DETOUR = 4.0, 1.3
GAP_KM, GAP_KMH = .15, 10.                             # розриви мережі до 150 м долаються на малій швидкості
LINE_KMH, LINE_DETOUR = 25.0, 1.35                     # стара оцінка: пряма × 1,35 на 25 км/год
SHIFT = (8.0, 16.0)
MUST_PENALTY = 1_000_000                               # пропуск обов'язкової зупинки — дорожчий за будь-який маршрут
URGENCY_PER_MIN = {"new": 10, "event": 4, "night": 1}  # штраф за хвилину очікування від моменту, коли подія відома; new — «гаряча» подія під час зміни
KX, KY = 111.32 * math.cos(math.radians(LAT0)), 111.2    # км на градус довготи / широти на широті території


def _xy(lon, lat):
    return np.c_[np.asarray(lon, float) * KX, np.asarray(lat, float) * KY]


def line_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


# ================================================================ час у дорозі
class LineRouter:
    """Проста модель часу: пряма × 1,35 на 25 км/год. Лише для базової лінії."""
    name = "line"

    def table(self, pts):
        n = len(pts)
        km = np.array([[line_km(pts[i], pts[j]) * LINE_DETOUR for j in range(n)] for i in range(n)])
        return km / LINE_KMH, km

    def leg(self, a, b):
        return line_km(a, b) * LINE_DETOUR, [[a["lat"], a["lon"]], [b["lat"], b["lon"]]]


class RoadRouter:
    """Дороги й треки OSM як граф; найшвидший шлях Дейкстрою; від дороги до точки — пішки."""
    name = "road"

    def __init__(self, roads=None):
        if roads is None:
            from . import common
            roads = common.base(common._park.MAIN).roads          # дороги й треки OSM основного парку з буфером
        roads = roads[roads.geometry.geom_type == "LineString"]
        idx, coords, ei, ej, eh, ekm = {}, [], [], [], [], []
        for hw, geom in zip(roads.highway, roads.geometry):
            v = SPEED_KMH.get(hw, 15)
            c = [(round(x, 6), round(y, 6)) for x, y in geom.coords]
            for a, b in zip(c[:-1], c[1:]):
                if a == b:
                    continue
                for p in (a, b):
                    if p not in idx:
                        idx[p] = len(coords)
                        coords.append(p)
                km = float(np.hypot((a[0] - b[0]) * KX, (a[1] - b[1]) * KY))
                ei.append(idx[a]); ej.append(idx[b]); eh.append(km / v); ekm.append(km)
        n = len(coords)
        best = {}                                       # паралельні ребра (дві дороги між тими самими вузлами) — найшвидше
        for a, b, hh, kk in zip(ei + ej, ej + ei, eh + eh, ekm + ekm):
            if (a, b) not in best or best[a, b][0] > hh:
                best[a, b] = (hh, kk)
        xy = _xy(np.array(coords)[:, 0], np.array(coords)[:, 1])

        def graph():
            keys, vals = np.array(list(best)), np.array(list(best.values()))
            return csr_matrix((vals[:, 0], (keys[:, 0], keys[:, 1])), shape=(n, n))

        # у вибірці OSM немає з'їздів (*_link) і частини дрібних доріг: зшиваємо сусідні шматки мережі короткими перемичками
        _, comp = connected_components(graph(), directed=False)
        for p, q in cKDTree(xy).query_pairs(GAP_KM, output_type="ndarray"):
            if comp[p] != comp[q]:
                km = float(np.hypot(*(xy[p] - xy[q])))
                best[int(p), int(q)] = best[int(q), int(p)] = (km / GAP_KMH, km)
        self.gaps = int(sum(1 for v in best.values() if v[0] * GAP_KMH - v[1] < 1e-9 and v[1] > 0) / 2)
        self.H = graph()
        self.KM = {(int(a), int(b)): float(k) for (a, b), (_, k) in best.items()}
        _, comp = connected_components(self.H, directed=False)
        main = np.bincount(comp).argmax()
        self.ok = np.flatnonzero(comp == main)          # прив'язуємо точки лише до найбільшої зв'язної частини мережі
        self.coords = np.array(coords)
        self.tree = cKDTree(_xy(self.coords[self.ok, 0], self.coords[self.ok, 1]))
        self.share_main = len(self.ok) / n
        self._rows = {}

    def snap(self, p):
        d, i = self.tree.query(_xy([p["lon"]], [p["lat"]])[0])
        return int(self.ok[i]), float(d)

    def _row(self, node):
        if node not in self._rows:
            if len(self._rows) > 200:
                self._rows.pop(next(iter(self._rows)))
            self._rows[node] = dijkstra(self.H, indices=node, return_predecessors=True)
        return self._rows[node]

    def _path(self, a, b):
        _, pred = self._row(a)
        out = [b]
        while out[-1] != a and pred[out[-1]] >= 0:
            out.append(int(pred[out[-1]]))
        return out[::-1]

    def table(self, pts):
        """Матриці часу (год) і відстані (км) між точками: пішки до дороги → дорогою → пішки від дороги; або напряму пішки, якщо так швидше."""
        sn = [self.snap(p) for p in pts]
        n = len(pts)
        Hm, Km = np.zeros((n, n)), np.zeros((n, n))
        for i, (ni, di) in enumerate(sn):
            dist, _ = self._row(ni)
            for j, (nj, dj) in enumerate(sn):
                if i == j:
                    continue
                walk_km = (di + dj) * WALK_DETOUR
                road_h = dist[nj] + walk_km / WALK_KMH
                direct_km = line_km(pts[i], pts[j]) * WALK_DETOUR
                if direct_km / WALK_KMH <= road_h:
                    Hm[i, j], Km[i, j] = direct_km / WALK_KMH, direct_km
                else:
                    Hm[i, j] = road_h
                    Km[i, j] = walk_km + self._path_km(ni, nj)
        return Hm, Km

    def _path_km(self, a, b):
        p = self._path(a, b)
        return sum(self.KM[p[k], p[k + 1]] for k in range(len(p) - 1))

    def leg(self, a, b):
        """Відстань (км) і геометрія [[lat, lon], …] одного переходу — для мапи."""
        (na, da), (nb, db) = self.snap(a), self.snap(b)
        direct_km = line_km(a, b) * WALK_DETOUR
        dist, _ = self._row(na)
        if direct_km / WALK_KMH <= dist[nb] + (da + db) * WALK_DETOUR / WALK_KMH:
            return direct_km, [[a["lat"], a["lon"]], [b["lat"], b["lon"]]]
        p = self._path(na, nb)
        line = LineString([(a["lon"], a["lat"])] + [tuple(self.coords[k]) for k in p] + [(b["lon"], b["lat"])]).simplify(0.00015)
        return (da + db) * WALK_DETOUR + self._path_km(na, nb), [[round(y, 5), round(x, 5)] for x, y in line.coords]


@lru_cache
def road_router():
    return RoadRouter()


# ================================================================ патрульні точки з прогнозу ризику
PATROL_ZONES, PATROL_MIN_KM = 16, 3                      # до 16 зон найвищого ризику рубок (прогноз LG); точки не ближче 3 км
PATROL_SERVICE_H = 1 / 3                                 # 20 хв патруля в зоні
PATROL_TAU_D, PATROL_NEVER_D = 7.0, 14                   # цінність відновлюється за ~тиждень; «ніколи» = 14 днів тому


@lru_cache
def patrol_points():
    """Зони найвищого ризику рубок з прогнозу переможця LG (основний парк) → точки на дорозі, не ближче 3 км одна від одної.
    Ризик зони = ранг гекса в прогнозі (1 — найризиковіший)."""
    import json
    from .common import P
    f = P / "results" / "logging_lg.json"
    cells = json.loads(f.read_text()).get("app", {}).get("cells", []) if f.exists() else []
    rt, out, used = road_router(), [], []
    n = max(len(cells), 1)
    for k, poly in enumerate(cells):
        la, lo = float(np.mean([p[0] for p in poly])), float(np.mean([p[1] for p in poly]))
        node, _ = rt.snap(dict(lat=la, lon=lo))
        x = _xy([rt.coords[node, 0]], [rt.coords[node, 1]])[0]
        if any(np.hypot(*(x - u)) < PATROL_MIN_KM for u in used):
            continue
        used.append(x)
        out.append(dict(h3=f"lg{k}", kind="logging", risk=round(1 - k / n, 3), note="прогноз LG",
                        lon=round(float(rt.coords[node, 0]), 6), lat=round(float(rt.coords[node, 1]), 6), zlat=round(la, 5), zlon=round(lo, 5)))
        if len(out) >= PATROL_ZONES:
            break
    return out


def patrol_value(risk, days_since):
    """Цінність патруля = ризик × (1 − e^(−давність / τ)); у хвилинах маршруту: 60 = година дороги."""
    return 60 * risk * (1 - math.exp(-days_since / PATROL_TAU_D))


# ================================================================ розклад і розв'язувачі
# team:  dict(id, lat, lon, t0, end, base) — поточна позиція й час (год), кінець зміни, база (lat/lon)
# stop:  dict(id, lat, lon, service_h, value, must, type, prio, known_h) — known_h: коли подія стала відома (для терміновості)

def _points(teams, stops):
    """Порядок вузлів у матриці: старти груп, бази груп, зупинки."""
    return [dict(lat=t["lat"], lon=t["lon"]) for t in teams] + [t["base"] for t in teams] + [dict(lat=s["lat"], lon=s["lon"]) for s in stops]


def schedule(team, seq, stops_by_id, H, node_of):
    """Прибуття / завершення кожної зупинки за матрицею H; повертає список і час повернення на базу."""
    cur, t, out = node_of[("start", team["id"])], team["t0"], []
    for sid in seq:
        s = stops_by_id[sid]
        j = node_of[sid]
        arrive = max(t, s.get("known_h", 0)) + H[cur, j]     # виїхати раніше, ніж про подію стало відомо, неможливо
        finish = arrive + s["service_h"]
        out.append(dict(id=sid, arrive_h=arrive, finish_h=finish, leg_from=cur))
        cur, t = j, finish
    return out, t + H[cur, node_of[("base", team["id"])]]


def _nodes(teams, stops):
    node_of = {("start", t["id"]): i for i, t in enumerate(teams)}
    node_of |= {("base", t["id"]): len(teams) + i for i, t in enumerate(teams)}
    node_of |= {s["id"]: 2 * len(teams) + k for k, s in enumerate(stops)}
    return node_of


def greedy(teams, stops, H, node_of, assign="lat", **_):
    """Базова лінія: поділ за широтою (2 групи), далі пріоритет ±1 рівень і найближча точка, поки встигаємо на базу."""
    by_team = {t["id"]: [] for t in teams}
    if assign == "lat" and len(teams) > 1:
        mid = float(np.median([s["lat"] for s in stops])) if stops else 0
        north, south = teams[0]["id"], teams[1]["id"]
        for s in stops:
            by_team[s.get("team") or (north if s["lat"] >= mid else south)].append(s)
    else:
        for s in stops:
            by_team[s.get("team") or teams[0]["id"]].append(s)
    plan, dropped = {}, []
    for tm in teams:
        todo, cur, t, seq = list(by_team[tm["id"]]), node_of[("start", tm["id"])], tm["t0"], []
        while todo:
            best_p = min(x["prio"] for x in todo)
            nxt = min([x for x in todo if x["prio"] <= best_p + 1], key=lambda x: H[cur, node_of[x["id"]]])
            todo.remove(nxt)
            j = node_of[nxt["id"]]
            finish = t + H[cur, j] + nxt["service_h"]
            if finish + H[j, node_of[("base", tm["id"])]] > tm["end"]:
                dropped.append(nxt["id"])
                continue
            seq.append(nxt["id"])
            cur, t = j, finish
        plan[tm["id"]] = seq
    return plan, dropped


def _penalty(s):
    return MUST_PENALTY if s["must"] else int(round(s["value"]))


def ortools(teams, stops, H, node_of, time_limit=1.0, initial=None, **_):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    V, n = len(teams), 2 * len(teams) + len(stops)
    T = np.rint(H * 60).astype(int)
    svc = [0] * (2 * V) + [int(round(s["service_h"] * 60)) for s in stops]
    mgr = pywrapcp.RoutingIndexManager(n, V, list(range(V)), list(range(V, 2 * V)))
    R = pywrapcp.RoutingModel(mgr)
    cb = R.RegisterTransitCallback(lambda i, j: int(T[mgr.IndexToNode(i), mgr.IndexToNode(j)] + svc[mgr.IndexToNode(i)]))
    R.SetArcCostEvaluatorOfAllVehicles(cb)
    cap = int(max(t["end"] for t in teams) * 60) + 1
    R.AddDimension(cb, cap if any(s.get("tw") for s in stops) else 0, cap, False, "time")   # очікування — лише коли є часові вікна
    dim = R.GetDimensionOrDie("time")
    for v, tm in enumerate(teams):
        t0 = int(round(tm["t0"] * 60))
        dim.CumulVar(R.Start(v)).SetRange(t0, t0)
        dim.CumulVar(R.End(v)).SetMax(int(tm["end"] * 60))
    for k, s in enumerate(stops):
        idx = mgr.NodeToIndex(2 * V + k)
        R.AddDisjunction([idx], _penalty(s))
        if s.get("tw"):
            dim.CumulVar(idx).SetRange(int(round(s["tw"][0] * 60)), int(round(s["tw"][1] * 60)))
        if s.get("team"):
            R.SetAllowedVehiclesForIndex([next(v for v, t in enumerate(teams) if t["id"] == s["team"])], idx)
        u = URGENCY_PER_MIN["new"] if s.get("new") else URGENCY_PER_MIN.get(s["type"], 0)
        if u:
            dim.SetCumulVarSoftUpperBound(idx, int(round(s["known_h"] * 60)), u)
    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.FromMilliseconds(int(time_limit * 1000))
    if initial:                                        # стартовий розв'язок (гібрид): {team_id: [stop_id, …]}
        at = {s["id"]: mgr.NodeToIndex(2 * V + k) for k, s in enumerate(stops)}
        R.CloseModelWithParameters(prm)
        seed = R.ReadAssignmentFromRoutes([[at[x] for x in initial[t["id"]]] for t in teams], True)
        sol = R.SolveFromAssignmentWithParameters(seed, prm) if seed else R.SolveWithParameters(prm)
    else:
        sol = R.SolveWithParameters(prm)
    plan, seen = {}, set()
    for v, tm in enumerate(teams):
        seq, i = [], R.Start(v)
        while sol and not R.IsEnd(i):
            node = mgr.IndexToNode(i)
            if node >= 2 * V:
                seq.append(stops[node - 2 * V]["id"])
                seen.add(seq[-1])
            i = sol.Value(R.NextVar(i))
        plan[tm["id"]] = seq
    return plan, [s["id"] for s in stops if s["id"] not in seen]


def vroom(teams, stops, H, node_of, **_):
    import vroom as vr
    V = len(teams)
    pb = vr.Input()
    pb.set_durations_matrix(profile="car", matrix_input=np.rint(H * 3600).astype(int).tolist())
    pb.add_vehicle([vr.Vehicle(v + 1, start=v, end=V + v, skills={v + 1}, time_window=vr.TimeWindow(int(tm["t0"] * 3600), int(tm["end"] * 3600)))
                    for v, tm in enumerate(teams)])
    vmax = max([s["value"] for s in stops if not s["must"]], default=1) or 1
    team_no = {t["id"]: v + 1 for v, t in enumerate(teams)}
    pb.add_job([vr.Job(k + 1, location=2 * V + k, default_service=int(round(s["service_h"] * 3600)),
                       priority=s.get("priority") or (100 if s["must"] else max(1, int(round(60 * s["value"] / vmax)))),
                       skills={team_no[s["team"]]} if s.get("team") else set(),
                       time_windows=[vr.TimeWindow(int(s["tw"][0] * 3600), int(s["tw"][1] * 3600))] if s.get("tw") else [])
                for k, s in enumerate(stops)])
    sol = pb.solve(exploration_level=5, nb_threads=4)
    r = sol.routes
    plan = {tm["id"]: [stops[int(j) - 1]["id"] for j in r[(r.vehicle_id == v + 1) & (r.type == "job")].sort_values("arrival").id]
            for v, tm in enumerate(teams)}
    seen = {x for seq in plan.values() for x in seq}
    return plan, [s["id"] for s in stops if s["id"] not in seen]


def hybrid(teams, stops, H, node_of, time_limit=1.0, **_):
    """VROOM обирає зупинки (сильний у відборі, RS), OR-Tools шліфує порядок із терміновістю (RD)."""
    plan, _ = vroom(teams, stops, H, node_of)
    return ortools(teams, stops, H, node_of, time_limit=time_limit, initial=plan)


SOLVERS = {"greedy": greedy, "ortools": ortools, "vroom": vroom, "hybrid": hybrid}


def solve(solver, teams, stops, router, **kw):
    """План: {team_id: [stop_id, …]}, відкинуті зупинки, матриці, час розв'язання (с)."""
    pts = _points(teams, stops)
    H, KM = router.table(pts)
    node_of = _nodes(teams, stops)
    t = time.perf_counter()
    plan, dropped = SOLVERS[solver](teams, stops, H, node_of, **kw)
    return plan, dropped, (H, KM, node_of), time.perf_counter() - t


def randomize(stops, seed, sigma=0.5):
    """PAWS-подібна рандомізація: цінність опційних патрульних точок × логнормальний шум (seed = день)."""
    rng = np.random.default_rng(seed)
    return [s | dict(value=s["value"] * float(rng.lognormal(0, sigma))) if not s["must"] else s for s in stops]


# ================================================================ день із перебудовою
def run_day(solver, teams, stops, router, reroute=None, plan_router=None, exec_router=None, seed=0, rand=False, time_limit=1.0):
    """Ранковий план → (опційно) нова подія о reroute['at'] і перебудова → виконання за exec_router.

    plan_router — модель часу, якою планують (для старої логіки — пряма лінія), exec_router — «реальний» час у дорозі.
    Повертає {team_id: dict(stops=[…з arrive_h/finish_h…], back_h, km, rerouted, …)}, відкладені зупинки, час розв'язання.
    """
    plan_router = plan_router or router
    exec_router = exec_router or router
    st = randomize(stops, seed) if rand else stops
    by_id = {s["id"]: s for s in st}
    plan, dropped, _, secs = solve(solver, teams, st, plan_router, time_limit=time_limit)
    info = {tm["id"]: dict(rerouted=False) for tm in teams}
    if reroute:
        new = reroute["stop"]
        by_id[new["id"]] = new
        # що встигли до моменту at — за реальним часом; поточну зупинку група дороблює
        ex = _execute(teams, plan, by_id, exec_router)
        state, rest = [], []
        for tm in teams:
            done = [x for x in ex[tm["id"]]["stops"] if x["arrive_h"] <= reroute["at"]]
            cur = by_id[done[-1]["id"]] if done else tm
            t_cur = max(reroute["at"], done[-1]["finish_h"]) if done else max(reroute["at"], tm["t0"])
            state.append(dict(tm, lat=cur["lat"], lon=cur["lon"], t0=t_cur, _done=[x["id"] for x in done]))
            rest += [sid for sid in plan[tm["id"]][len(done):]]
            info[tm["id"]] |= dict(old_rest=plan[tm["id"]][len(done):], stops_before=len(done), at_h=reroute["at"])
        if solver == "greedy":
            # стара логіка: подія дістається групі за широтою; перераховує лише вона, інша група їде за ранковим планом
            mid = float(np.median([by_id[s]["lat"] for s in sum(plan.values(), [])] + [new["lat"]]))
            who = teams[0]["id"] if new["lat"] >= mid else teams[1]["id"]
            tw = next(s for s in state if s["id"] == who)
            sub = [by_id[sid] for sid in plan[who][len(tw["_done"]):]] + [new]
            p2, d2, _, s2 = solve("greedy", [tw], [x | dict(team=who) for x in sub], plan_router)
            plan[who] = tw["_done"] + p2[who]
            dropped += d2
            info[who] |= dict(rerouted=True)
        else:
            todo = [by_id[sid] for sid in rest + dropped] + [new]
            p2, dropped, _, s2 = solve(solver, state, todo, plan_router, time_limit=time_limit)
            for s in state:
                plan[s["id"]] = s["_done"] + p2[s["id"]]
                info[s["id"]] |= dict(rerouted=p2[s["id"]] != info[s["id"]]["old_rest"])
        secs += s2
    ex = _execute(teams, plan, by_id, exec_router)
    for k in ex:
        ex[k] |= info[k]
    return ex, dropped, secs


def _execute(teams, plan, by_id, router):
    """Виконання плану з реальним часом у дорозі: прибуття, повернення, кілометри."""
    stops = [by_id[s] for seq in plan.values() for s in seq]
    pts = _points(teams, stops)
    H, KM = router.table(pts)
    node_of = _nodes(teams, stops)
    out = {}
    for tm in teams:
        seq, back = schedule(tm, plan[tm["id"]], by_id, H, node_of)
        km, prev = 0.0, node_of[("start", tm["id"])]
        for x in seq:
            x["leg_km"] = float(KM[x["leg_from"], node_of[x["id"]]])
            km += x["leg_km"]
            prev = node_of[x["id"]]
        km += float(KM[prev, node_of[("base", tm["id"])]])
        out[tm["id"]] = dict(stops=seq, back_h=back, km=km)
    return out
