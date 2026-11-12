"""CG · Камера на координати: наскільки точно можна вказати місце диму на мапі.

Задача      камера дає лише напрямок. Зона на мапі це клин, звужений прямою видимістю по рельєфу.
Дані        реальні гекси з точками NASA FIRMS у парку: пожежі 2022 (валідація) і 2023 (тест), рельєф SRTM.
Протокол    місце пожежі реальне, а напрямок симульований: справжній азимут плюс похибка з відхиленням 1, 3, 5, 10°.
            Ширину клина (1, 2 або 3 відхилення) обираємо на 2022: найменша зона, що накриває місце в ≥ 90 % випадків.
Метрики     влучання: справжній гекс у зоні; площа зони; відстань від місця до центру зони.
Підходи
  wedge      клин без рельєфу, одна камера: базова лінія
  wedge_los  клин і пряма видимість, одна камера
  two_cams   перетин двох клинів із прямою видимістю
Окремо      яку частку гексів із пожежами бачить кожна камера за різної висоти щогли й дальності.
Обмеження   справжніх детекцій диму з координатами немає, тому це перевірка геометрії, а не детектора.
"""
import inspect
import json

import h3
import numpy as np
import pandas as pd

from .. import pflow
from .. import common, datasets, tracking
from ..common import P
from ..tracking import ml

EXPERIMENT = "CG · Камера на координати (симуляція азимута на реальних місцях пожеж)"
SEED = 42
from ..regions import protocol  # noqa: E402
_V, _T = protocol("CG", [2022], [2023])                   # роки з config/regions.py (fetch.protocols.CG) або кейсу Фракії
VAL, TEST = _V[0], _T                                      # рік вибору ширини клина і роки тесту
SIGMAS = [1, 3, 5, 10]
KS = [1, 2, 3]
MASTS = [10, 15, 30]
RANGES = [5000, 10000, 15000]


def cameras():
    """Камера CAM-02 із кроку vision і вежі, які обрала гілка TP в основному парку (tp_final.json: номери гексів того ж датасету).
    Раніше вежі бралися з даних застосунку (app_extra.json), які будуються пізніше за гілки: так CG залежав від порядку кроків."""
    v = json.load(open(P / "vision" / "video.json"))
    d = datasets.main_park(datasets.get("h3_risk_cells").df)
    t = [h3.cell_to_latlng(d.h3.iat[int(j)]) for j in json.load(open(P / "tp_final.json"))["risk_cells"]]
    return [dict(name="CAM-02", lat=v["cam_lat"], lon=v["cam_lon"], bearing=v["bearing"], fov=60)] + [dict(name=f"вежа {i + 1}", lat=round(a, 5), lon=round(b, 5), bearing=None, fov=360) for i, (a, b) in enumerate(t)]


def table():
    d = datasets.main_park(datasets.get("h3_risk_cells").df)            # камера стоїть в основному парку
    res = h3.get_resolution(d.h3.iat[0])
    lat, lon = np.array([h3.cell_to_latlng(c) for c in d.h3]).T
    B = common.base()
    f = B.firms[B.firms.year.isin([VAL, *TEST])]
    idx = dict(zip(d.h3, range(len(d))))
    rows = {}
    for y, a, b in zip(f.year, f.geometry.y, f.geometry.x):
        c = h3.latlng_to_cell(a, b, res)
        if c in idx:
            rows[(y, c)] = idx[c]
    t = pd.DataFrame([dict(year=y, h3=c, cell=i, lat=lat[i], lon=lon[i], part="val" if y == VAL else "test") for (y, c), i in rows.items()],
                     columns=["year", "h3", "cell", "lat", "lon", "part"])
    return d, lat, lon, t, h3.cell_area(d.h3.iat[0], "km^2")


def view(cam, lat, lon, mast=15, rng=10000):
    v, dist = common.los(cam["lat"], cam["lon"], lat, lon, mast)
    az = common.azimuth(lat, lon, cam["lat"], cam["lon"])
    ok = dist <= rng
    if cam["fov"] < 360:
        ok &= ((az - (cam["bearing"] - cam["fov"] / 2)) % 360) <= cam["fov"]
    return dict(vis=v & ok, ok=ok, az=az, dist=dist)


def adiff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def locate(q, V, lat, lon, area, sigma, k, approach, rng):
    """Для кожного гекса з пожежею будує зону й міряє влучання, площу, відстань."""
    out = []
    for c in q.cell:
        seen = sorted([v for v in V if v["vis"][c]], key=lambda v: v["dist"][c])
        need = 2 if approach == "two_cams" else 1
        if len(seen) < need:
            continue
        zone = np.ones(len(lat), bool)
        for v in seen[:need]:
            obs = v["az"][c] + rng.normal(0, sigma)
            zone &= adiff(v["az"], obs) <= max(k * sigma, 1.)
            zone &= v["vis"] if approach != "wedge" else v["ok"]
        n = int(zone.sum())
        far = np.nan if not n else float(np.hypot((lat[zone].mean() - lat[c]) * 111.32, (lon[zone].mean() - lon[c]) * 111.32 * np.cos(np.radians(lat[c]))))
        out.append(dict(hit=bool(zone[c]), km2=n * area, km=far))
    return pd.DataFrame(out)


def run(**_):
    pflow.mark('1. dataset: cells and fires')
    d, lat, lon, t, area = table()
    dv = datasets.save(t.drop(columns="cell"), "cv_camera_zone", description=f"Гекси парку з точками NASA FIRMS, пожежі {VAL} (вибір) і {', '.join(map(str, TEST))} (тест): справжні місця для перевірки зони камери. Азимут симульовано.",
                       unit="гекс із пожежею за рік", targets=[], features=["year", "h3", "lat", "lon"], params=dict(seed=SEED, sigmas=SIGMAS, ks=KS, masts=MASTS, ranges=RANGES), sources=["NASA FIRMS", "SRTM"])
    exp = tracking.experiment(EXPERIMENT, __doc__)
    cams = cameras()
    test = t[t.part == "test"]
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", files=True)
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/camera_geo.py")
        rows = []
        pflow.mark('2. cameras on masts')
        for cam in cams:
            for mast in MASTS:
                for r in RANGES:
                    v = view(cam, lat, lon, mast, r)
                    rows.append(dict(part="coverage", camera=cam["name"], mast=mast, range_km=r // 1000, seen=float(v["vis"][test.cell].mean()) if len(test) else np.nan, n=len(test), cells_visible=int(v["vis"].sum())))
        V = [view(c, lat, lon) for c in cams[1:]]
        pflow.mark('3. approaches by azimuth error')
        for sigma in SIGMAS:
            for app in ("wedge", "wedge_los", "two_cams"):
                best = None
                for k in KS:
                    z = locate(t[t.part == "val"], V, lat, lon, area, sigma, k, app, np.random.default_rng(SEED))
                    if len(z) and z.hit.mean() >= .9 and (best is None or z.km2.median() < best[1]):
                        best = (k, z.km2.median(), z.hit.mean())
                k = best[0] if best else KS[-1]
                z = locate(test, V, lat, lon, area, sigma, k, app, np.random.default_rng(SEED + 1))
                v = locate(t[t.part == "val"], V, lat, lon, area, sigma, k, app, np.random.default_rng(SEED))
                m_ = lambda d_, c_, f_: float(getattr(d_[c_], f_)()) if len(d_) else np.nan     # гексів із пожежею в цьому році може не бути
                rows.append(dict(part="locate", approach=app, sigma=sigma, k=k, val_hit=m_(v, "hit", "mean"), val_km2=m_(v, "km2", "median"), test_hit=m_(z, "hit", "mean"),
                                 test_km2=m_(z, "km2", "median"), test_km=m_(z, "km", "median"), n=len(z), champion=bool(app == "two_cams" and sigma == 3)))
        res = pd.DataFrame(rows)
        res["champion"] = res.champion.fillna(False).astype(bool)
        pflow.mark('4. approaches: results per algorithm')
        tracking.log_approaches(res[res.part == "locate"], lambda r: r.approach, label=lambda r: f"азимут ±{r.sigma:g}°, k={r.k:g}")
        pflow.mark('5. save')
        print(res.to_string()[:6000])
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag)
