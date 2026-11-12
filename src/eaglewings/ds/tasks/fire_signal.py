"""FD (режим signal) · Чи бачать супутники пожежі на території: скільки згарищ EFFIS мають сигнал FIRMS.

Задача      там, де великих пожеж немає, згарище за знімками не помалюєш. Спершу треба знати, чи теплові точки FIRMS
            узагалі бачать згарища, які картографує EFFIS, і що саме вони бачать: ліс чи весняні пали на полях.
Дані        NASA FIRMS (VIIRS S-NPP n/h, MODIS ≥ 30) у межах кожного парку + буфер із конфігу, роки FIRMS; EFFIS Burnt Areas тих самих років.
Еталон      згарище EFFIS, що перетинає парк із буфером. EFFIS картографує великі пожежі, приблизно від 30 га.
Протокол    нічого не вчиться: опис на всіх роках. Сигнал належить згарищу, якщо лежить у полігоні + 500 м і в датах ± 2 дні (правило FM).
Метрики     effis_seen — частка згарищ EFFIS, які мають хоч один сигнал (основна); signals_in_effis — частка сигналів у згарищах;
            in_forest — частка сигналів у лісі (лісистість Hansen 2000 ≥ 30 %); spring_share — частка сигналів у березні–квітні (весняні пали).
            95 % CI — bootstrap по згарищах. Окремо для кожного парку (колонки *_<парк>).
Підходи
  viirs        лише VIIRS S-NPP (375 м)
  modis        лише MODIS (1 км)
  both         обидва сенсори разом
Вибір       немає: це опис, а не модель.
Обмеження   згарищ EFFIS на території може бути кілька або жодного — тоді частка з широким інтервалом чи без нього («замало»).
            Сигнал у згарищі не доводить, що пожежа лісова; причину загоряння не перевірено.
"""
import inspect

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import common, datasets, metrics, park, pflow, tracking
from ..tracking import ml
from . import matching_benchmark as MB

EXPERIMENT = "FD · FIRMS проти згарищ EFFIS (чи бачать супутники пожежі)"
APPROACHES = [dict(name="viirs", sensors=["VIIRS"], hyp="VIIRS 375 м бачить і дрібніші згарища"),
              dict(name="modis", sensors=["MODIS"], hyp="MODIS 1 км бачить лише великі й гарячі пожежі"),
              dict(name="both", sensors=["VIIRS", "MODIS"], hyp="Два сенсори разом пропускають менше згарищ")]
FEW = 10                                                    # менше 10 згарищ — «замало»


def signals():
    """Сигнали FIRMS усіх парків (парк + буфер), з ознакою «в лісі»."""
    out = []
    for k in park.PARKS:
        B = common.base(k)
        f = B.firms.copy()
        col, row = ~B.TR * (f.longitude.values, f.latitude.values)
        r_, c_ = np.clip(row.astype(int), 0, B.H - 1), np.clip(col.astype(int), 0, B.W - 1)
        f["forest"] = (B.tc[r_, c_] >= 30).astype(int)
        f["park"] = k
        out.append(f[["dt", "year", "sensor", "latitude", "longitude", "frp", "forest", "park", "geometry"]])
    g = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=4326).to_crs(common.UTM)
    return g.sort_values("dt").reset_index(drop=True)


def zones():
    """Парк + буфер кожного парку в UTM."""
    return {k: common.base(k).park_u.buffer(common.CFG["buffer_m"]).iloc[0] for k in park.PARKS}


def run(n_boot=500):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    pflow.mark("1. dataset: FIRMS signals, EFFIS burnt areas")
    sig, e = signals(), MB.effis()
    Z = zones()
    e = e[e.start.dt.year.isin(common.YEARS)].reset_index(drop=True)
    e["park"] = ""
    for k, z in Z.items():                                  # згарище в зоні парку (перше, що перетинає)
        e.loc[(e.park == "") & e.intersects(z), "park"] = k
    e = e[e.park != ""].reset_index(drop=True)
    truth = MB.label(sig, e) if len(e) else np.full(len(sig), -1)
    df = pd.DataFrame(dict(dt=sig.dt.astype(str), year=sig.year, sensor=sig.sensor, lat=sig.latitude.round(5), lon=sig.longitude.round(5), frp=sig.frp,
                           forest=sig.forest, park=sig.park, effis=truth, in_effis=(truth >= 0).astype(int), month=sig.dt.dt.month))
    dv = datasets.save(df, "fd_signal", unit="теплова точка FIRMS у парку з буфером", targets=["in_effis"], features=["sensor", "frp", "forest", "month"],
                       description="Сигнали FIRMS у трьох парках із буфером і згарище EFFIS, якому вони належать (−1 — жодного).",
                       params=dict(years=list(common.YEARS), buffer_m=common.CFG["buffer_m"], effis_buffer_m=MB.BUF_M, margin_days=MB.MARGIN_D, effis_fires=len(e)),
                       sources=["NASA FIRMS VIIRS S-NPP, MODIS", "EFFIS Burnt Areas (Copernicus)", "Hansen GFC treecover2000"])
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        tracking.log_dataset(dv, "evaluation", "in_effis", files=True)
        ml.log_params(dict(signals=len(sig), effis_fires=len(e), years=str(list(common.YEARS)), n_boot=n_boot))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/fire_signal.py")
        pflow.mark("2. share of EFFIS burnt areas with a signal")
        for a in APPROACHES:
            m_ = df.sensor.isin(a["sensors"]).values
            seen = set(df.effis[m_ & (df.effis >= 0)].unique())
            hit = np.array([i in seen for i in range(len(e))], float)
            r = dict(approach=a["name"], hypothesis=a["hyp"], signals=int(m_.sum()), effis_fires=len(e), effis_seen_n=int(hit.sum()),
                     effis_seen=float(hit.mean()) if len(e) else np.nan,
                     signals_in_effis=float(df.in_effis[m_].mean()) if m_.any() else np.nan,
                     in_forest=float(df.forest[m_].mean()) if m_.any() else np.nan,
                     spring_share=float(df.month[m_].isin([3, 4]).mean()) if m_.any() else np.nan,
                     effis_ha=float(e.area_ha.sum()), few=len(e) < FEW, champion=False)
            if len(e) > 1:
                lo, hi = metrics.block_bootstrap(np.arange(len(e)), lambda i: hit[i].mean(), n=n_boot)
            else:
                lo = hi = np.nan
            r |= dict(effis_seen_lo=lo, effis_seen_hi=hi)
            for k in park.PARKS:                            # той самий опис окремо для кожного парку
                ek = (e.park == k).values
                mk = m_ & (df.park == k).values
                r[f"effis_fires_{k}"] = int(ek.sum())
                r[f"effis_seen_{k}"] = float(hit[ek].mean()) if ek.any() else np.nan
                r[f"signals_{k}"] = int(mk.sum())
                r[f"in_forest_{k}"] = float(df.forest[mk].mean()) if mk.any() else np.nan
                r[f"spring_share_{k}"] = float(df.month[mk].isin([3, 4]).mean()) if mk.any() else np.nan
            with ml.start_run(run_name=a["name"], nested=True):
                ml.set_tags(dict(approach=a["name"], kind="sensor", hypothesis=a["hyp"], role="approach"))
                ml.log_metrics(tracking.clean({k: v for k, v in r.items() if isinstance(v, (int, float))}))
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "hypothesis"}, flush=True)
        res = pd.DataFrame(rows)
        months = df.groupby(["month", "sensor"]).size().unstack(fill_value=0).reindex(range(1, 13), fill_value=0)
        tracking.keep("fire_signal", "months", {str(m): {s: int(v) for s, v in row.items()} for m, row in months.iterrows()})
        tracking.keep("fire_signal", "effis", [dict(id=str(x.id), park=x.park, start=str(x.start)[:10], area_ha=float(x.area_ha or 0),
                                                   seen=bool(i in set(df.effis[df.effis >= 0])), lat=float(c.y), lon=float(c.x))
                                              for i, (x, c) in enumerate(zip(e.itertuples(), gpd.GeoSeries(e.geometry, crs=common.UTM).to_crs(4326).centroid))])
        tracking.log_fig(_fig(months), "signals_by_month")
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag, effis_fires=len(e))


def _fig(months):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bottom = np.zeros(12)
    for s, c in (("VIIRS", "#e8743b"), ("MODIS", "#8aa4c8")):
        v = months[s].values if s in months else np.zeros(12)
        ax.bar(range(1, 13), v, bottom=bottom, color=c, label=s)
        bottom += v
    ax.set(xticks=range(1, 13), xlabel="місяць", ylabel="теплові точки", title="FD · коли супутники бачать вогонь (усі роки, три парки)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
