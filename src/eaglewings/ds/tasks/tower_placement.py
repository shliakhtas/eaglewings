"""TP · Де ставити вежі: чи бачили б вони майбутні пожежі.

Задача      обрати 3 місця для веж (огляд 360°, щогла 15 м, 10 км), щоб побачити якнайбільше пожеж.
Дані        гекси парку H3 res 9, рельєф SRTM, точки NASA FIRMS 2021–2025, зведені в пожежі правилом FM (2 км, 5 діб).
Протокол    прогресивний: для кожного року Y вежі ставимо лише за тим, що відомо до Y, і перевіряємо на пожежах року Y.
            Валідація 2022, тест 2023, 2024, 2025 разом і окремо. Вибір лише на валідації. Той самий протокол у CP і LG.
            Різниця з випадковими місцями: парний бутстреп за пожежами тесту.
Метрики     first: частка пожеж, перша точка яких у полі зору, головна. any: хоч одна точка.
            burn: частка згорілих гексів у полі зору. Інтервал: бутстреп за пожежами.
Підходи
  random        випадкові 3 місця, 1000 спроб
  peaks         3 найвищі точки на відстані
  spread        рівномірно по парку: k-means, найвища точка кластера
  greedy_*      жадібне покриття з вагою: flat ліс, elev висота, firms точки минулих років,
                loss втрати лісу, suscept логрег на рельєфі й доступності проти минулих втрат
  greedy_risk   вага з моделі вкладки «Ризик». Її вчили на пожежі 2022, тому на валідації її немає
Параметр      пул кандидатів: 30, 100, 300 найвищих точок.
Парки         вежі ставляться всередині парку, тому кожен парк (ds/regions.py) рахується окремо: своя валідація 2022, свій переможець,
              свій тест (колонка park). Парк без пожеж у роках валідації чи тесту пропускається з поясненням.
Обмеження     пожеж мало, а точка FIRMS це не момент займання. Вежі гіпотетичні: дозволів і доріг ми не перевіряли.
"""
import inspect
import json

import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import pflow
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
from .. import common, datasets, tracking
from ..common import P
from ..tracking import ml

EXPERIMENT = "TP · Де ставити вежі (майбутні пожежі FIRMS)"
SEED, K, MAST, RANGE = 42, 3, 15, 10000
POOLS = [30, 100, 300]
WEIGHTS = ["flat", "elev", "firms", "loss", "suscept"]
STATIC = ["elev", "slope", "southness", "treecover", "d_road", "d_village", "track_km_km2", "d_boundary"]
DYN = ["loss_hist", "loss_recent3", "firms_2km", "d_firms", "burned_prev3"]
RK = lambda v: pd.Series(np.asarray(v, float)).rank(pct=True).values
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def fires(key=None):
    B = common.base(key)
    f = B.firms.assign(ev=common.st_dbscan(B.firms, 2., 5.))
    f = f[f.ev >= 0].sort_values("dt")
    return pd.DataFrame([dict(ev=int(e), year=int(q.year.iat[0]), n=len(q), start=q.dt.iat[0], lat=q.geometry.y.iat[0], lon=q.geometry.x.iat[0], lats=q.geometry.y.values, lons=q.geometry.x.values) for e, q in f.groupby("ev")])


def pool(d, lat, lon, n):
    out = []
    for j in np.argsort(-d.elev.values):
        if all(np.hypot(lat[j] - lat[i], lon[j] - lon[i]) > .01 for i in out):
            out.append(int(j))
        if len(out) >= n:
            break
    return out


KEY = None                                     # парк, який зараз рахується (run → _run_park)


def see(lat, lon, c, mast=MAST, rng=RANGE, h=20):
    v, dist = common.los(lat[c], lon[c], lat, lon, mast, h, key=KEY)
    return v & (dist <= rng)


def weight(name, d, Y):
    forest = (d.treecover.values >= 30).astype(float)
    if name == "flat":
        return forest
    if name == "elev":
        return RK(d.elev) * forest
    if name == "firms":
        return RK(d[f"f{Y}_firms_2km"]) * forest
    if name == "loss":
        return RK(d[f"f{Y}_loss_hist"]) * forest
    if name == "suscept":
        m = lr().fit(d[STATIC].fillna(0).values, d[f"f{Y}_loss_hist"].values > .02)
        return RK(m.predict_proba(d[STATIC].fillna(0).values)[:, 1]) * forest
    m = lr().fit(d[STATIC + [f"f22_{k}" for k in DYN]].fillna(0).values, d.y_burn_2022)
    return RK(m.predict_proba(d[STATIC + [f"f{Y}_{k}" for k in DYN]].fillna(0).values)[:, 1]) * forest


def greedy(w, cand, VIS, k=K):
    cov, out = np.zeros(len(w), bool), []
    for _ in range(k):
        j = max((c for c in cand if c not in out), key=lambda c: w[VIS[c] & ~cov].sum())
        out.append(j)
        cov |= VIS[j]
    return out


def score(S, VIS, ev, yb):
    seen = np.any([VIS[j] for j in S], 0)
    return dict(first=np.array([seen[c[0]] for c in ev.cells]), any=np.array([seen[c].any() for c in ev.cells]), burn=float(seen[yb].mean()), area=float(seen.mean()))


def summ(s, rng=None):
    out = dict(first=float(s["first"].mean()), any=float(s["any"].mean()), burn=s["burn"], area=s["area"])
    if rng is not None:
        b = [s["first"][rng.integers(0, len(s["first"]), len(s["first"]))].mean() for _ in range(1000)]
        out.update(first_lo=float(np.quantile(b, .025)), first_hi=float(np.quantile(b, .975)))
    return out


def run(**_):
    global KEY
    from .. import park
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark('1. dataset: fires and tower candidates')
    D = datasets.get("h3_risk_cells").df
    PK = list(dict.fromkeys(D.park)) if "park" in D else [park.KEY]
    out, info, fires_, cands_ = [], {}, {}, {}
    for k in PK:
        KEY = k
        d = (D[D.park == k] if "park" in D else D).reset_index(drop=True)
        r = _run_park(k, d, len(PK) > 1, fires_, cands_)
        if r is None:
            continue
        res, inf = r
        out.append(res.assign(park=k) if len(PK) > 1 else res)
        info[k] = inf
    KEY = None
    meta_f = dict(unit="пожежа після зведення FM", targets=[], features=["year", "lat", "lon", "n"], description="Пожежі в парку 2021–2025 із перших точок FIRMS: перевірка, чи бачили б їх вежі.",
                  params=dict(km=2., days=5., k=K, mast=MAST, range=RANGE, pools=POOLS), sources=["NASA FIRMS VIIRS, MODIS", "SRTM"])
    meta_c = dict(unit="місце для вежі", targets=[], features=["lat", "lon", "elev", "visible"], description="300 найвищих місць парку на відстані від 1 км і скільки гексів видно з кожного.",
                  params=dict(mast=MAST, range=RANGE, pool=max(POOLS)), sources=["SRTM"])
    dv = datasets.save_parks(fires_, "tp_fires", **meta_f)
    datasets.save_parks(cands_, "tp_candidates", **meta_c)
    res = pd.concat(out, ignore_index=True)
    return res, dict(experiment_id=exp.experiment_id, dataset=dv.tag, chosen={k: v["chosen"] for k, v in info.items()})


def _run_park(key, d, multi, fires_, cands_):
    """Вежі одного парку: d — гекси цього парку. Датасети пожеж і кандидатів складаються в fires_/cands_ по парках."""
    from .. import park
    lat, lon = np.array([h3.cell_to_latlng(c) for c in d.h3]).T
    F = fires(key)
    tree = cKDTree(np.c_[lat, lon * np.cos(np.radians(LAT0))])
    def cells(q):
        dist, i = tree.query(np.c_[q.lats, q.lons * np.cos(np.radians(LAT0))])
        return i[dist < .01]
    F["cells"] = [cells(q) for q in F.itertuples()]
    F = F[[len(c) > 0 for c in F.cells]].reset_index(drop=True)
    print(key, F.groupby("year").size().to_dict() if len(F) else {}, flush=True)
    if not len(F) or not (F.year == 2022).any() or not (F.year >= 2023).any():
        print(f"TP {key}: немає пожеж FIRMS у роки валідації (2022) чи тесту (2023–2025), вежі не обираємо", flush=True)
        return None
    fires_[key] = F[["ev", "year", "n", "start", "lat", "lon"]].round(5)
    dv = datasets.save(fires_[key], f"tp_fires_{key}" if multi else "tp_fires", unit="пожежа після зведення FM", targets=[], features=["year", "lat", "lon", "n"],
                       description="Пожежі в парку 2021–2025 із перших точок FIRMS: перевірка, чи бачили б їх вежі.", params=dict(km=2., days=5., k=K, mast=MAST, range=RANGE, pools=POOLS), sources=["NASA FIRMS VIIRS, MODIS", "SRTM"])
    pflow.mark('2. viewshed from peaks')
    P3 = pool(d, lat, lon, max(POOLS))
    POOL = {n: P3[:n] for n in POOLS}
    VIS = {c: see(lat, lon, c) for c in P3}
    print("видимість пораховано", flush=True)
    YEARS = {2022: "val", 2023: "test", 2024: "test", 2025: "test"}         # прогресивно: рік Y бачить лише дані до Y
    rows, picks, hits, RM = [], {}, {}, {}
    rng = np.random.default_rng(SEED)
    Xy = np.c_[lat, lon * np.cos(np.radians(LAT0))]
    lab = KMeans(K, n_init=10, random_state=SEED).fit_predict(Xy)
    draws = [rng.choice(P3, K, replace=False).tolist() for _ in range(1000)]
    top = []
    for c in P3:
        if all(np.hypot(lat[c] - lat[p], lon[c] - lon[p]) > .05 for p in top):
            top.append(c)
    fixed = {("peaks", 300): top[:K], ("spread", 300): [max(np.flatnonzero(lab == k), key=lambda i: d.elev.values[i]) for k in range(K)]}
    for j in {int(x) for v in fixed.values() for x in v} - set(VIS):
        VIS[j] = see(lat, lon, j)
    with ml.start_run(run_name=tracking.batch_name() + (f" · {key}" if multi else "")) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/tower_placement.py")
        pflow.mark('3. year by year, past data only')
        for Y, part in YEARS.items():
            Yf = 22 if Y <= 2022 else 23                                       # ознаки станом на 1 січня 2022 або 2023: знають лише минуле
            ev = F[F.year == Y]
            yb = d[f"y_burn_{Y}"].values == 1
            if not len(ev):
                continue
            first = lambda S: np.array([np.any([VIS[j] for j in S], 0)[c[0]] for c in ev.cells])
            anyh = lambda S: np.array([np.any([VIS[j] for j in S], 0)[c].any() for c in ev.cells])
            RM[Y] = np.array([first(S) for S in draws], float)
            hits[(Y, "random", 300)] = (RM[Y].mean(0), np.mean([anyh(S) for S in draws], 0))
            sets = dict(fixed)
            for w in WEIGHTS + (["risk"] if Y >= 2023 else []):
                W = weight(w, d, Yf)
                for n in POOLS:
                    sets[(f"greedy_{w}", n)] = greedy(W, POOL[n], VIS)
            for (a, n), S in sets.items():
                hits[(Y, a, n)] = (first(S).astype(float), anyh(S).astype(float))
                picks[(Y, a, n)] = S
            for (y_, a, n), (fh, ah) in hits.items():
                if y_ == Y:
                    rows.append(dict(part="year", year=Y, approach=a, pool=n, first=float(fh.mean()), any=float(ah.mean()), n=len(ev),
                                     burn=float(np.any([VIS[j] for j in picks[(Y, a, n)]], 0)[yb].mean()) if a != "random" and yb.any() else None))
        keys = {(a, n) for (_, a, n) in hits}
        rngb = np.random.default_rng(SEED)
        pflow.mark('4. pick on 2022, test 2023-25, difference vs random')
        for part in ("val", "test"):
            ys = [y for y, p in YEARS.items() if p == part]
            yb = d.y_burn_2022.values == 1 if part == "val" else d.y_burn_2023.values == 1
            base = np.concatenate([hits[(y, "random", 300)][0] for y in ys if (y, "random", 300) in hits])
            SIMPLE = ["random", "peaks", "spread"] + [f"greedy_{w}" for w in WEIGHTS + ["risk"]]          # за рівності береться простіший
            for a, n in sorted(keys, key=lambda k: (SIMPLE.index(k[0]) if k[0] in SIMPLE else 99, k[1])):
                if not all((y, a, n) in hits for y in ys if (y, "random", 300) in hits):
                    continue
                fh = np.concatenate([hits[(y, a, n)][0] for y in ys if (y, a, n) in hits])
                ah = np.concatenate([hits[(y, a, n)][1] for y in ys if (y, a, n) in hits])
                yref = min(ys)
                S = picks.get((yref, a, n))
                seen = np.any([VIS[j] for j in S], 0) if S else None
                bi = [rngb.integers(0, len(fh), len(fh)) for _ in range(2000)]
                b1 = [fh[i].mean() for i in bi]
                dd = [fh[i].mean() - base[i].mean() for i in bi]
                rows.append(dict(part=part, approach=a, pool=n, n=len(fh), first=float(fh.mean()), any=float(ah.mean()),
                                 burn=float(seen[yb].mean()) if seen is not None else float(np.mean([np.any([VIS[j] for j in S_], 0)[yb].mean() for S_ in draws[:200]])),
                                 area=float(seen.mean()) if seen is not None else float(np.mean([np.any([VIS[j] for j in S_], 0).mean() for S_ in draws[:200]])),
                                 first_lo=float(np.quantile(b1, .025)), first_hi=float(np.quantile(b1, .975)),
                                 diff=float(fh.mean() - base.mean()), diff_lo=float(np.quantile(dd, .025)), diff_hi=float(np.quantile(dd, .975)),
                                 beats_random=float((np.concatenate([RM[y] for y in ys if y in RM], 1).mean(1) < fh.mean()).mean())))
                print(part, a, n, round(fh.mean(), 3), "різниця", round(fh.mean() - base.mean(), 3), flush=True)
        res = pd.DataFrame(rows)
        v = res[(res.part == "val") & (res.approach != "random")].sort_values(["first", "any", "burn"], ascending=False, kind="mergesort").iloc[0]
        res["champion"] = (res.approach == v.approach) & (res["pool"] == v["pool"]) & (res.part == "test")
        print("обрано на валідації:", v.approach, v["pool"], flush=True)
        w_ch, n_ch = v.approach.replace("greedy_", ""), int(v["pool"])
        ev, yb = F[F.year >= 2023], d.y_burn_2023.values == 1
        extra = []
        pflow.mark('5. how many towers, sensitivity')
        for k in range(1, 7):                                                  # скільки веж потрібно
            S = greedy(weight(w_ch, d, 23), POOL[n_ch], VIS, k)
            extra.append(dict(part="budget", approach=v.approach, towers=k, **summ(score(S, VIS, ev, yb), np.random.default_rng(SEED))))
        S = picks[(2023, v.approach, n_ch)]
        for mast, r, h in [(10, 10000, 20), (15, 10000, 20), (30, 10000, 20), (15, 5000, 20), (15, 15000, 20), (15, 10000, 50)]:      # стійкість: місця ті самі, умови інші
            V2 = {c: see(lat, lon, c, mast, r, h) for c in S}
            extra.append(dict(part="sens", approach=v.approach, mast=mast, range_km=r // 1000, smoke=h, **summ(score(S, V2, ev, yb))))
        res = pd.concat([res, pd.DataFrame(extra)], ignore_index=True)
        res["champion"] = res.champion.fillna(False).astype(bool)
        fin = greedy(weight(w_ch, d, 26), POOL[n_ch], VIS)
        alt = greedy(weight("risk", d, 26), POOL[n_ch], VIS)
        print("остаточні вежі:", fin, "з вагою ризику:", alt, flush=True)
        seen = np.any([VIS[c] for c in fin], 0)
        fr = [dict(lat=round(float(q.lat), 5), lon=round(float(q.lon), 5), year=int(q.year), n=int(q.n), seen=bool(seen[q.cells[0]])) for q in F.itertuples()]
        (park.processed(key) / "tp_final.json").write_text(json.dumps(dict(chosen=v.approach, pool=n_ch, cells=[int(c) for c in fin], risk_cells=[int(c) for c in alt], fires=fr), ensure_ascii=False))
        pflow.mark('6. approaches: results per algorithm')
        cut = lambda r: r.part if r.part != "year" else f"y{int(r.year)}"
        tracking.log_approaches(res[res.part.isin(["year", "val", "test"])], lambda r: f"{r.approach}, {int(r.pool)} місць", prefix=cut)
        pflow.mark('7. save')
        cands_[key] = pd.DataFrame(dict(h3=d.h3.values[P3], lat=lat[P3].round(5), lon=lon[P3].round(5), elev=d.elev.values[P3].round(0), visible=[int(VIS[c].sum()) for c in P3]))
        if not multi:
            datasets.save(cands_[key], "tp_candidates", unit="місце для вежі", targets=[], features=["lat", "lon", "elev", "visible"],
                          description="300 найвищих місць парку на відстані від 1 км і скільки гексів видно з кожного.", params=dict(mast=MAST, range=RANGE, pool=max(POOLS)), sources=["SRTM"])
        print(res[res.part.isin(["val", "test"]) & res.approach.isin(["random"])][["part", "first", "any", "burn"]].round(3).to_string(), flush=True)
        print(res[res.part.isin(["budget", "sens"])][["part", "towers", "mast", "range_km", "smoke", "first", "any", "burn", "area"]].round(3).to_string(), flush=True)
        ml.log_table(res, "results.json")
        return res, dict(parent_run_id=parent.info.run_id, dataset=dv.tag, chosen=f"{v.approach}/{v['pool']}")
