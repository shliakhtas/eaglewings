"""CP · Де в парку ставити камери: які місця на дорогах перехоплять під'їзд до майбутніх рубок.

Задача      обрати 10 місць на дорогах і стежках для наземних камер і фотопасток. Камера бачить лише свій
            відрізок дороги, тому рельєф тут мало що дає, важливо, через що проходить під'їзд.
Дані        дороги OSM, гекси H3 res 9, нові втрати лісу Hansen поза згарищами (LG), пожежі FIRMS (TP).
Припущення  до місця їдуть найкоротшим шляхом від в'їзду в парк. Це припущення: реальних проїздів у нас немає.
Протокол    прогресивний, як у TP і LG: для кожного року Y місця ставимо лише за тим, що відомо до Y, і перевіряємо на подіях року Y.
            Валідація 2018–2020, тест 2021–2025 разом і по роках. Вибір лише на валідації, за рубками.
            Різниця з випадковими місцями: парний бутстреп за рубками тесту.
            Перший протокол мав тест 2023–2024: там лише 2 рубки, результат 0 %. Він лишається в звіті як narrow.
Метрики     log: частка рубок, під'їзд до яких проходить через камеру, головна. fire: те саме для пожеж.
            forest: частка лісових гексів. Інтервал: бутстреп за подіями.
Підходи
  random    випадкові місця, 1000 спроб
  degree    найбільші перехрестя
  hotspot   найближча дорога до найбільших минулих втрат
  greedy_*  жадібне покриття дерева під'їздів з вагою: flat ліс, past минулі втрати, lg прогноз LG
Параметр    де можна ставити: усі дороги чи лише лісові.
Обмеження   подій мало, а збіг із найкоротшим шляхом не означає, що порушник там їхав.
"""
import inspect
import json

import h3
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from .. import common, datasets, tracking
from ..common import P
from ..tracking import ml
from . import logging_greece as LG
from . import tower_placement as TP

EXPERIMENT = "CP · Де ставити камери (під'їзди до майбутніх рубок)"
SEED, N = 42, 10
FOREST_ROADS = {"track", "path", "unclassified", "service", "footway", "bridleway"}
BUDGETS = [5, 10, 20, 40]
WEIGHTS = ["flat", "past", "lg"]


def graph():
    B = common.base()
    r = B.roads.to_crs(common.UTM)
    ll = B.roads
    G, xy = nx.Graph(), {}
    for geom, g2, kind in zip(r.geometry, ll.geometry, r.highway):
        pts, lon = list(geom.coords), list(g2.coords)
        ids = [(round(a), round(b)) for a, b in pts]
        for k, (u, p) in enumerate(zip(ids, lon)):
            xy[u] = p
        for u, v, a, b in zip(ids, ids[1:], pts, pts[1:]):
            if u != v:
                G.add_edge(u, v, length=float(np.hypot(a[0] - b[0], a[1] - b[1])), kind=kind)
    inside = {n: B.PARK.contains(Point(xy[n])) for n in G}
    gates = [u for u, v in G.edges if inside[u] != inside[v]]
    gates += [v for u, v in G.edges if inside[u] != inside[v]]
    return G, xy, inside, sorted(set(n for n in gates if not inside[n]))


def tree(G, gates, jitter=None):
    H = G.copy()
    if jitter is not None:
        for u, v, d in H.edges(data=True):
            d["length"] *= float(jitter.lognormal(0, .3))
    H.add_node("S")
    H.add_edges_from(("S", g, dict(length=0.)) for g in gates)
    pred, dist = nx.dijkstra_predecessor_and_distance(H, "S", weight="length")
    return {n: p[0] for n, p in pred.items() if p}, dist


def matrix(ends, par, idx):
    """Рядок: вузол; колонка: ціль. 1, якщо вузол лежить на під'їзді до цілі."""
    r, c = [], []
    for j, n in enumerate(ends):
        while n in par and n != "S":
            if n in idx:
                r.append(idx[n])
                c.append(j)
            n = par[n]
    return sparse.csr_matrix((np.ones(len(r), bool), (r, c)), shape=(len(idx), len(ends)))


def greedy(A, w, allow, dist, k):
    """Жадібно; за рівності беремо місце глибше в парку, далі від звичайного руху."""
    un, out = np.ones(A.shape[1], bool), []
    for _ in range(k):
        gain = A.astype(float).dot(w * un) * allow
        if gain.max() <= 0:
            break
        top = np.flatnonzero(gain >= gain.max() * (1 - 1e-9))
        j = int(top[np.argmax(dist[top])])
        out.append(j)
        un &= ~A[j].toarray().ravel()
    return out


def run(**_):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. road graph and events')
    G, xy, inside, gates = graph()
    par, dist = tree(G, gates)
    nodes = [n for n in par if n != "S" and inside.get(n) and dist[n] >= 300]
    idx = {n: i for i, n in enumerate(nodes)}
    Dn = np.array([dist[n] for n in nodes])
    kinds = {}
    for u, v, d in G.edges(data=True):
        kinds.setdefault(u, d["kind"]), kinds.setdefault(v, d["kind"])
    forest = np.array([kinds.get(n) in FOREST_ROADS for n in nodes])
    deg = np.array([G.degree(n) for n in nodes])
    print(len(G), "вузлів,", len(nodes), "у парку,", forest.sum(), "лісових,", len(gates), "в'їздів", flush=True)
    reach = [n for n in par if n != "S"]
    ll = np.array([xy[n] for n in reach])
    kd = cKDTree(np.c_[ll[:, 1], ll[:, 0] * np.cos(np.radians(LAT0))])
    near = lambda la, lo: [reach[i] for i in kd.query(np.c_[la, np.asarray(lo) * np.cos(np.radians(LAT0))])[1]]
    d = datasets.main_park(datasets.get("logging_gr_cells").df)         # камери на дорогах основного парку
    base = d[d.year == 2024].set_index("h3")
    hexes = base.index.tolist()
    hl, ho = np.array([h3.cell_to_latlng(c) for c in hexes]).T
    Hn = near(hl, ho)
    A = matrix(Hn, par, idx)
    forestw = (base.treecover.values >= 30).astype(float)
    ev = d[d.y == 1].copy()
    ev["lat"], ev["lon"] = np.array([h3.cell_to_latlng(c) for c in ev.h3]).T
    ev["kind"] = "logging"
    fr = TP.fires()
    fr = fr[fr.lat.between(hl.min(), hl.max())]
    E = pd.concat([ev[["year", "lat", "lon", "kind"]], fr[["year", "lat", "lon"]].assign(kind="fire")], ignore_index=True)
    E["km"] = kd.query(np.c_[E.lat, E.lon * np.cos(np.radians(LAT0))])[0] * 111.32
    print(E.groupby(["kind", "year"]).size().to_dict(), flush=True)
    M = matrix(near(E.lat.values, E.lon.values), par, idx)
    PART = dict(val=(2018, E.year.between(2018, 2020)),
                test=(2021, E.year >= 2021),
                narrow=(2023, ((E.kind == "logging") & E.year.between(2023, 2024)) | ((E.kind == "fire") & (E.year >= 2023))))   # перший протокол: лише 2 рубки
    dv = datasets.save(E.round(5), "cp_events", unit="подія: рубка або пожежа", targets=[], features=["year", "lat", "lon", "kind"], description="Нові втрати лісу поза згарищами й пожежі в парку: місця, до яких перевіряємо під'їзд.",
                       params=dict(n=N, budgets=BUDGETS, seed=SEED), sources=["Hansen GFC", "NASA FIRMS", "OpenStreetMap"])

    def wgt(name, Y):
        if name == "flat":
            return forestw
        q = d[d.year == Y].set_index("h3").reindex(hexes)
        if name == "past":
            return (q.loss_cum.fillna(0).values + q.loss1.fillna(0).values + .5 * q.nb_loss1.fillna(0).values) * forestw + 1e-6
        tr = d[d.year < Y]
        p = TP.lr().fit(tr[LG.CELL_F], tr.y).predict_proba(q[LG.CELL_F].fillna(0))[:, 1]
        return p * forestw

    def score(S, part, rng=None, Mx=None):
        Mz = M if Mx is None else Mx
        q = PART[part][1].values
        hit = np.asarray(Mz[S].sum(0)).ravel() > 0 if len(S) else np.zeros(Mz.shape[1], bool)
        lg, fi = hit[q & (E.kind == "logging").values], hit[q & (E.kind == "fire").values]
        cov = np.asarray(A[S].sum(0)).ravel() > 0
        out = dict(log=float(lg.mean()), fire=float(fi.mean()), forest=float(cov[forestw > 0].mean()), n_log=len(lg), n_fire=len(fi))
        if rng is not None:
            b = [lg[rng.integers(0, len(lg), len(lg))].mean() for _ in range(1000)]
            out.update(log_lo=float(np.quantile(b, .025)), log_hi=float(np.quantile(b, .975)))
        return out

    rows, picks, hv, RM = [], {}, {}, {}
    rng = np.random.default_rng(SEED)
    fidx = np.flatnonzero(forest)
    draws = [rng.choice(fidx, N, replace=False) for _ in range(1000)]
    hitof = lambda S: np.asarray(M[S].sum(0)).ravel() > 0 if len(S) else np.zeros(M.shape[1], bool)
    covof = lambda S: float((np.asarray(A[S].sum(0)).ravel() > 0)[forestw > 0].mean())
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/camera_placement.py")
        pflow.mark('2. year by year, past data only')
        for Y in range(2018, 2026):                                           # прогресивно: рік Y бачить лише дані до Y
            q = (E.year == Y).values
            if not q.any():
                continue
            Yc = min(Y, 2024)
            RM[Y] = np.array([hitof(S)[q] for S in draws], float)
            sets = {("degree", "усі"): np.argsort(-deg)[:N].tolist()}
            qd = d[d.year == Yc].set_index("h3").reindex(hexes)
            hot = np.argsort(-(qd.loss_cum.fillna(0).values + qd.loss1.fillna(0).values))[:N]
            sets[("hotspot", "усі")] = [idx[Hn[i]] for i in hot if Hn[i] in idx]
            for w in WEIGHTS:
                W = wgt(w, Yc)
                sets[(f"greedy_{w}", "усі")] = greedy(A, W, np.ones(len(nodes)), Dn, N)
                sets[(f"greedy_{w}", "лісові")] = greedy(A, W, forest.astype(float), Dn, N)
            kind = E.kind.values[q]
            for (a, wh), S in list(sets.items()) + [(("random", "лісові"), None)]:
                h = RM[Y].mean(0) if S is None else hitof(S)[q].astype(float)
                hv[(Y, a, wh)] = h
                picks[(Y, a, wh)] = S
                rows.append(dict(part="year", year=Y, approach=a, where=wh, log=float(h[kind == "logging"].mean()) if (kind == "logging").any() else None,
                                 fire=float(h[kind == "fire"].mean()) if (kind == "fire").any() else None, n_log=int((kind == "logging").sum()), n_fire=int((kind == "fire").sum())))
        rngb = np.random.default_rng(SEED)
        pflow.mark('3. pick on 2018-20, test 2021-25, difference vs random')
        for part, ys in (("val", [2018, 2019, 2020]), ("test", [2021, 2022, 2023, 2024, 2025])):
            ys = [y for y in ys if y in RM]
            kind = np.concatenate([E.kind.values[(E.year == y).values] for y in ys])
            lgm = kind == "logging"
            base = np.concatenate([hv[(y, "random", "лісові")] for y in ys])
            rp = np.concatenate([RM[y] for y in ys], 1)
            for (y0, a, wh) in [k for k in hv if k[0] == ys[0]]:
                h = np.concatenate([hv[(y, a, wh)] for y in ys])
                li = np.flatnonzero(lgm)
                bi = [rngb.choice(li, len(li)) for _ in range(2000)]
                bl = [h[i].mean() for i in bi]
                dd = [h[i].mean() - base[i].mean() for i in bi]
                S = picks[(ys[0], a, wh)]
                rows.append(dict(part=part, approach=a, where=wh, log=float(h[lgm].mean()), fire=float(h[~lgm].mean()) if (~lgm).any() else None,
                                 forest=covof(S) if S is not None else float(np.mean([covof(S_) for S_ in draws[:200]])), n_log=int(lgm.sum()), n_fire=int((~lgm).sum()),
                                 log_lo=float(np.quantile(bl, .025)), log_hi=float(np.quantile(bl, .975)),
                                 diff=float(h[lgm].mean() - base[lgm].mean()), diff_lo=float(np.quantile(dd, .025)), diff_hi=float(np.quantile(dd, .975)),
                                 beats_random=None if a == "random" else float((rp[:, lgm].mean(1) < h[lgm].mean()).mean())))
                print(part, a, wh, round(h[lgm].mean(), 3), "різниця", round(h[lgm].mean() - base[lgm].mean(), 3), flush=True)
                if part == "test":
                    picks[("test", a, wh)] = S
        pflow.mark('4. sensitivity')
        res = pd.DataFrame(rows)
        v = res[(res.part == "val") & (res.approach != "random")].sort_values(["log", "fire", "forest"], ascending=False, kind="mergesort").iloc[0]
        res["champion"] = (res.approach == v.approach) & (res["where"] == v["where"]) & (res.part == "test")
        print("обрано на валідації:", v.approach, v["where"], flush=True)
        w_ch, allow = v.approach.replace("greedy_", ""), (forest.astype(float) if v["where"] == "лісові" else np.ones(len(nodes)))
        extra = []
        if v.approach.startswith("greedy_"):
            for k in BUDGETS:
                S = greedy(A, wgt(w_ch, 2021), allow, Dn, k)
                extra.append(dict(part="budget", approach=v.approach, cameras=k, **score(S, "test", np.random.default_rng(SEED))))
        S = picks[("test", v.approach, v["where"])]
        extra.append(dict(part="narrow", approach=v.approach, **score(greedy(A, wgt(w_ch, 2023), allow, Dn, N), "narrow")))
        lo = []
        for i in range(20):                                                   # чи тримається, якщо під'їзд інший
            p2, _ = tree(G, gates, np.random.default_rng(SEED + i))
            Mz = matrix(near(E.lat.values, E.lon.values), p2, idx)
            lo.append(score(S, "test", Mx=Mz)["log"])
        extra.append(dict(part="sens", approach=v.approach, log=float(np.mean(lo)), log_lo=float(np.min(lo)), log_hi=float(np.max(lo))))
        res = pd.concat([res, pd.DataFrame(extra)], ignore_index=True)
        res["champion"] = res.champion.fillna(False).astype(bool)
        fin = greedy(A, wgt(w_ch, 2024), allow, Dn, N) if v.approach.startswith("greedy_") else S
        cov = np.asarray(A[fin].sum(0)).ravel() > 0
        keep = set()
        for t in [Hn[i] for i in np.flatnonzero(cov)]:
            n = t
            while n in par and n != "S" and n not in keep:
                keep.add(n)
                n = par[n]
        seg = [[[round(xy[n][1], 5), round(xy[n][0], 5)], [round(xy[par[n]][1], 5), round(xy[par[n]][0], 5)]] for n in keep if par.get(n) in keep]
        hit = np.asarray(M[fin].sum(0)).ravel() > 0
        (P / "cp_final.json").write_text(json.dumps(dict(chosen=v.approach, where=v["where"], forest=float(cov[forestw > 0].mean()), gates=len(gates), cameras=[dict(lat=round(xy[nodes[j]][1], 5), lon=round(xy[nodes[j]][0], 5), road=kinds.get(nodes[j]), km=round(float(Dn[j]) / 1000, 1), share=round(float(A[j].sum() / A.shape[1]), 3)) for j in fin],
                                                         events=[dict(lat=round(a, 5), lon=round(b, 5), year=int(y), kind=k, seen=bool(s)) for a, b, y, k, s in zip(E.lat, E.lon, E.year, E.kind, hit)], segs=seg), ensure_ascii=False))
        print(res[res.part.isin(["budget", "sens", "narrow", "val", "test"]) & res.approach.isin(["random", v.approach])].drop(columns=["champion"]).round(3).to_string()[:2600], flush=True)
        print("тест", (E.km[PART["test"][1]]).describe().round(2).to_dict(), len(seg), flush=True)
        pflow.mark('5. approaches: results per algorithm')
        cut = lambda r: r.part if r.part != "year" else f"y{int(r.year)}"
        tracking.log_approaches(res[res.part.isin(["year", "val", "test"])], lambda r: f"{r.approach}, {r.where}", prefix=cut)
        pflow.mark('6. save')
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, chosen=f"{v.approach}/{v['where']}")
