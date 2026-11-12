"""Вкладка «Набори даних»: лише збирає готові результати кроку eda (scripts/eda.py) у web/science/sources.json. Функції опису (firms, hansen …) рахують EDA для того кроку.

    eaglewings script 11_source_eda

Це опис того, що лежить на диску, а не експерименти.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
sys.path.insert(0, str(LIB / "scripts"))
from data_lineage import GROUPS, LINEAGE, SRC  # noqa: E402
from eaglewings.ds import common, datasets  # noqa: E402

RAW = ROOT / "data" / "raw"
P = ROOT / "data" / "processed"
r_ = lambda v, k=1: round(float(v), k)
N = lambda v: f"{int(v):,}".replace(",", " ")
PC = lambda v: f"{v * 100:.0f} %"
D2 = lambda v, k=2: f"{v:.{k}f}".replace(".", ",")


def du(path):
    try:
        out = subprocess.run(f"du -chs {ROOT / path} 2>/dev/null | tail -1", shell=True, capture_output=True, text=True).stdout.split()
        return out[0] if out else ""
    except Exception:
        return ""


def nfiles(path):
    p = ROOT / path
    return sum(1 for x in p.rglob("*") if x.is_file() and ".cache" not in x.parts) if p.is_dir() else len(glob.glob(str(p)))


def hist(a, n, doc="", log=False):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    lo, hi = np.quantile(a, [.005, .995])
    f = np.log1p if log else (lambda v: v)
    e = np.linspace(f(lo), f(hi), 25)
    return dict(n=n, doc=doc, log=log, lo=r_(lo, 2), hi=r_(hi, 2), edges=e.tolist(), all=np.histogram(np.clip(f(a), e[0], e[-1]), e)[0].tolist())


def firms():
    fs = []
    for f in sorted(glob.glob(str(RAW / "firms" / f"*_{COUNTRY}.csv"))):
        d = pd.read_csv(f)
        d["sensor"] = "MODIS" if "modis" in f else "VIIRS"
        fs.append(d)
    d = pd.concat(fs, ignore_index=True)
    m = d.sensor == "MODIS"
    ok = (m & (pd.to_numeric(d.confidence, errors="coerce") >= 30)) | (~m & d.confidence.isin(["n", "h"]))
    B = common.base()
    d["ym"] = d.acq_date.str[:7]
    q = d.groupby("ym").size()
    s = d.sample(1500, random_state=42)
    y = d.groupby([d.acq_date.str[:4], "sensor"]).size().unstack()
    return dict(rows=len(d), short=f"{N(len(d))} теплових точок у Греції за п'ять років, у зоні парку лише {N(len(B.firms))}.",
                seen=[f"VIIRS дає {PC((~m).mean())} точок, MODIS {PC(m.mean())}: він грубіший і бачить менше.", f"Фільтр впевненості лишає {PC(ok.mean())} точок.", f"Вночі зроблено {PC((d.daynight == 'N').mean())} точок, тож вогонь видно й без сонця.", f"Роки дуже нерівні: {N(q.groupby(q.index.str[:4]).sum().min())} точок у найтихіший рік і {N(q.groupby(q.index.str[:4]).sum().max())} у найгарячіший.", "FIRMS бачить вогонь лише під час прольоту: коротку малу пожежу між прольотами він може пропустити."],
                table=dict(title="Точок за роками", head=["Рік", "VIIRS", "MODIS"], rows=[[a, N(c), N(b)] for a, b, c in y.reset_index().itertuples(index=False)]),
                hist=[hist(d.frp, "frp", "потужність вогню, МВт", True), hist(d.scan, "scan", "розмір пікселя вздовж сканування, км")],
                cats=[dict(n="confidence", doc="впевненість VIIRS", rows=[[str(a), int(b)] for a, b in d[~m].confidence.value_counts().items()]), dict(n="daynight", doc="день чи ніч", rows=[[str(a), int(b)] for a, b in d.daynight.value_counts().items()])],
                time=dict(unit="місяць", rows=[[a, int(b), 0] for a, b in q.items()]), geo=dict(label="вибірка 1500 точок, колір за сенсором", kind="cat", pts=[[r_(a, 3), r_(b, 3), c] for a, b, c in zip(s.latitude, s.longitude, s.sensor)]))


def effis():
    rows, pts = [], []
    for f in sorted(glob.glob(str(RAW / "benchmarks" / "effis" / "*.geojson"))):
        g = gpd.read_file(f)
        y = Path(f).name.split("_")[1]
        ha = pd.to_numeric(g.get("AREA_HA", pd.Series(dtype=float)), errors="coerce")
        c = g.geometry.representative_point()
        rows.append([y, len(g), r_(ha.sum(), 0) if len(ha) else 0, r_(ha.median(), 0) if len(ha) else 0])
        pts += [[r_(a, 3), r_(b, 3), y] for a, b in zip(c.y, c.x)]
    t = sum(r[1] for r in rows)
    return dict(rows=t, short=f"{N(t)} контурів згарищ за {len(rows)} років у рамці навколо Греції.", seen=["Рамка ширша за Грецію: туди потрапляють і сусідні країни.", f"Типове згарище велике: медіана за роками від {N(min(r[3] for r in rows))} до {N(max(r[3] for r in rows))} га. Малі пожежі EFFIS часто не картографує.", "Тому точка FIRMS поза контуром не обов'язково хибна."],
                table=dict(title="Контури за роками", head=["Рік", "Контурів", "Площа, га", "Медіана, га"], rows=[[a, N(b), N(c), N(e)] for a, b, c, e in rows]), geo=dict(label="центри контурів за роками", kind="cat", pts=pts[:1800]))


def hansen():
    B = common.base()
    forest = B.forest
    hist_ = {2000 + int(y): r_(B.PIX_HA[(B.loss == y) & B.park_mask].sum(), 0) for y in np.unique(B.loss) if y > 0}
    tc = B.tc[B.park_mask]
    top = max(hist_, key=hist_.get)
    tot = sum(hist_.values())
    return dict(rows=int(B.H * B.W), short=f"Сітка {N(B.H)} × {N(B.W)} пікселів. Ліс це {PC(forest[B.park_mask].mean())} парку, рік {top} дає {PC(hist_[top] / tot)} усіх втрат із 2001.",
                seen=[f"У {top} році втрачено {N(hist_[top])} га: це пожежа, а не рубка.", "Втрата це будь-яка загибель крон: Hansen не розрізняє вогонь і рубку. Тому для вирубки ми вирізаємо пожежі.", "Низова пожежа, що не вбила крони, для Hansen не існує.", "Лісистість це стан 2000 року: нові насадження тут не видно.", "Частина дерев гине із запізненням, тому втрату від пожежі 2023 ми рахуємо за 2023 і 2024."],
                time=dict(unit="рік", rows=[[str(a), int(b), 0] for a, b in hist_.items()]), hist=[hist(tc, "treecover2000", "лісистість 2000 року, %")], time_note="площа втрат у парку, га")


def modis():
    nd = common.modis_ndvi()
    B = common.base()
    m = B.park_mask & B.forest
    a, b = nd["pre"][m], nd["post"][m]
    d = a - b
    fs = glob.glob(str(RAW / "nasa" / "modis_250m*"))
    return dict(rows=len(fs), short=f"{len(fs)} вирізок MODIS на диску. У датасети входять дві дати: 28 липня і 14 вересня 2023.", seen=[f"Медіана NDVI лісу до пожежі {D2(np.nanmedian(a))}, після {D2(np.nanmedian(b))}.", f"Падіння більше 0,275 має {PC(np.nanmean(d > .275))} лісу парку.", f"Пропусків {D2(100 * np.isnan(a).mean(), 1)} %, це хмари й краї вирізки.", "250 м це грубо: один піксель MODIS накриває багато пікселів Hansen, тому контур виходить сходинками.", "Інші вирізки завантажені для історії NDVI в окремих точках парку й у таблиці датасетів не йдуть."],
                hist=[hist(a, "ndvi_pre", "NDVI лісу 28 липня"), hist(b, "ndvi_post", "NDVI лісу 14 вересня"), hist(d, "d_ndvi", "падіння NDVI")])


def sentinel2():
    rows = []
    for f in sorted(glob.glob(str(RAW / "nasa" / "s2" / "comp_*.npz"))):
        z = np.load(f)
        rows.append([Path(f).stem[-4:], len(z["ids"]), float(np.isfinite(z["nir"]).mean()), float(np.nanmedian(z["nir"]))])
    return dict(rows=len(rows), short=f"{len(rows)} літніх композитів, по одному на рік, із {N(sum(r[1] for r in rows))} сцен.", seen=[f"На рік ідуть лише {rows[0][1]} найменш хмарні сцени літа, хмарність до 25 %.", "Це один знімок на рік: дата рубки всередині року втрачається, а рубка після знімка потрапить у наступний рік.", "Сітка спільна з Hansen, тому обидва джерела збігаються піксель у піксель.", "Рік 2017 потрібен лише як «до» для 2018."],
                table=dict(title="Композити за роками", head=["Рік", "Сцен", "Покриття рамки", "Медіана NIR"], rows=[[a, b, PC(c), f"{e:.3f}".replace(".", ",")] for a, b, c, e in rows]))


def opera():
    d = datasets.main_park(datasets.get("opera_gr_snapshots").df)
    empty = int((d.drop(columns="date").drop(columns="s255").sum(axis=1) == 0).sum())
    return dict(rows=len(d), short=f"{len(d)} знімків статусу за два сезони, {empty} із них без даних над парком.", seen=["Продукт почали випускати недавно, тому старіших років тут немає.", "GIBS віддає картинку, а не дані: статус відновлюється за кольором. Чужий колір зупиняє обробку, тому помилку не пропустимо.", "Це продукт про будь-яке порушення: пожежа дає найбільшу хвилю."],
                table=dict(title="Знімки", head=["Дата", "Пікселів із порушенням"], rows=[[a, N(b)] for a, b in zip(d.date, d[[f"s{i}" for i in range(1, 9)]].sum(axis=1))]))


def esri():
    fs = sorted(glob.glob(str(P / "sat_*.json")))
    return dict(rows=len(fs), short=f"{len(fs)} вирізок знімків високої роздільності.", seen=["Дата зйомки невідома, тому ці знімки не годяться для зміни в часі.", "Самі знімки ми далі не передаємо."])


def power():
    w = common.power()
    return dict(rows=len(w), short=f"{N(len(w))} діб шести змінних у центрі парку.", seen=[f"Пропусків після заповнення {int(w.isna().sum().sum())}.", f"Температура від {D2(w.T2M_MAX.min(), 1)} до {D2(w.T2M_MAX.max(), 1)} °C, опади до {w.PRECTOTCORR.max():.0f} мм на добу.", "Одна точка на весь парк: різницю між ділянками погода не дає.", "Сітка NASA POWER приблизно 50 км, тому місцевий вітер вона згладжує."],
                hist=[hist(w.T2M_MAX, "T2M_MAX", "максимальна температура, °C"), hist(w.RH2M, "RH2M", "вологість повітря, %"), hist(w.WS10M, "WS10M", "вітер, м/с"), hist(w.PRECTOTCORR, "PRECTOTCORR", "опади, мм", True), hist(w.GWETTOP, "GWETTOP", "вологість ґрунту")],
                time=dict(unit="місяць", value="T2M_MAX", rows=[[k.strftime("%Y-%m"), r_(v, 0), 0] for k, v in w.T2M_MAX.resample("MS").max().items()]), time_note="максимальна температура місяця, °C")


def srtm():
    Z, B = common.dem(), common.base()
    z = Z[B.park_mask]
    return dict(rows=int(B.H * B.W), short=f"Висоти парку від {z.min():.0f} до {z.max():.0f} м, медіана {np.median(z):.0f} м.", seen=["Рельєф низький, тому огляд сильно залежить від місця: вершини бачать не найбільше.", "Перенесено на сітку Hansen, тому висота лежить на тих самих пікселях, що й лейбли.", "Дерева в SRTM частково входять у висоту, тому пряма видимість приблизна."], hist=[hist(z, "elev", "висота, м")])


def osm_park():
    B = common.base()
    a = float(B.park_u.area.sum()) / 1e6
    return dict(rows=1, short=f"Один полігон парку, площа близько {a:.0f} км².", seen=["Це межа з OSM, а не офіційний шар, тому можливі дрібні розбіжності.", "Усе інше обрізається нею: якщо межа зміниться, зміняться й датасети."])


def osm_roads():
    B = common.base()
    r = B.roads.to_crs(common.UTM)
    r["km"] = r.length / 1000
    q = r.groupby("highway").km.agg(["sum", "size"]).sort_values("sum", ascending=False).reset_index()
    return dict(rows=len(r), short=f"{N(len(r))} ліній загальною довжиною {N(r.km.sum())} км.", seen=[f"Лісові дороги track це {PC(q[q.highway == 'track']['sum'].sum() / q['sum'].sum())} довжини.", "OSM неповний у лісі: стежка, якої немає на мапі, для нашої мережі під'їздів не існує.", "Тип дороги заповнюють волонтери, тому він не завжди точний."],
                table=dict(title="Дороги за типом", head=["Тип", "Км", "Ліній"], rows=[[a, N(b), N(c)] for a, b, c in q.head(12).itertuples(index=False)]))


def cv(name, sets):
    d = pd.concat([datasets.load(s).df for s in sets]) if len(sets) > 1 else datasets.load(sets[0]).df
    return d


def vedai():
    d = datasets.get("cv_benchmark_aerial").df
    v = d[d.dataset == "vedai_12cm"]
    return dict(rows=len(v), short=f"{N(len(v))} аерофото з {N(v.n_boxes.sum())} рамками машин.", seen=["Це переважно села й дороги, а не ліс.", "Вихідна роздільність 12,5 см, а на наших знімках близько 50 см: тому ми розмиваємо самі.", "Порожніх кадрів немає."], hist=[hist(v.n_boxes, "n_boxes", "рамок на кадрі")])


def dfire():
    d = datasets.get("cv_benchmark_ground").df
    v = d[d.dataset == "dfire"]
    return dict(rows=len(v), short=f"{N(len(v))} фото, {N(v.n_boxes.sum())} рамок диму й вогню.", seen=[f"Порожніх кадрів {PC((v.n_boxes == 0).mean())}.", "Здебільшого близькі кадри, а вежа бачить дим за кілометри.", "Тому результат на D-Fire не переноситься прямо на вежу."], hist=[hist(v.n_boxes, "n_boxes", "рамок на кадрі")])


def ena24():
    d = datasets.get("cv_benchmark_ground").df
    v = d[d.dataset == "ena24"]
    return dict(rows=len(v), short=f"{N(len(v))} кадрів фотопасток, машина лише на {int((v.n_boxes > 0).sum())}.", seen=["Рамок немає, є лише позначка кадру.", "Це не Дадья й не Європа: фон інший.", "Машин дуже мало, тому висновки про них слабкі."])


def wikimedia():
    fs = glob.glob(str(RAW / "media" / "*"))
    v = json.load(open(P / "vision" / "video.json"))
    return dict(rows=len(fs), short="Один запис пожежі на 9,5 секунди і 8 фото з парку.", seen=["Відео знято в Карсті у 2022 році, а не в Дадьї. На сторінці це підписано.", f"Автор відео {v['author']}, ліцензія {v['license']}.", "Розмітки немає, тому це демонстрація, а не вимірювання."])


def generator():
    d = pd.read_csv(P / "signals.csv")
    return dict(rows=len(d), short=f"{N(len(d))} сигналів, {N(d.simulated.sum())} із них симульовані.", seen=["Це сценарний потік на реальних місцях, а не спостереження.", f"Умовні дати від {d.date.min()[:10]} до {d.date.max()[:10]}.", "Код генератора ще треба перенести в цю теку: зараз тут лише його результат."],
                cats=[dict(n="source", doc="джерело", rows=[[str(a), int(b)] for a, b in d.source.value_counts().items()]), dict(n="kind", doc="тип", rows=[[str(a), int(b)] for a, b in d.kind.value_counts().items()])])


F = dict(firms=firms, effis=effis, hansen=hansen, modis=modis, sentinel2=sentinel2, opera=opera, esri=esri, power=power, srtm=srtm, osm_park=osm_park, osm_roads=osm_roads, vedai=vedai, dfire=dfire, ena24=ena24, wikimedia=wikimedia, generator=generator)
UNIT = dict(firms="точок", effis="контурів", hansen="пікселів", modis="вирізок", sentinel2="композитів", opera="знімків", esri="вирізок", power="діб", srtm="пікселів", osm_park="полігон", osm_roads="ліній", vedai="кадрів", dfire="кадрів", ena24="кадрів", wikimedia="файлів", generator="сигналів")
EDA = ROOT / "data" / "processed" / "eda" / "sources"


def by_park(k):
    """Набір, що залежить від місця, у кожному парку дослідження: головні числа поруч і найбільші відмінності."""
    from eaglewings.ds import park as PKM
    if len(PKM.PARKS) < 2 or k not in ("firms", "hansen", "modis", "sentinel2", "opera", "power", "srtm", "osm_park", "osm_roads"):
        return None
    cols, rows, diffs = None, [], []
    vals = {}
    for pk in PKM.PARKS:
        B = common.base(pk)
        nm = PKM.ALL[pk]["short"]
        if k == "firms":
            f = B.firms
            cols = ["Точок 2021–2025", "Уночі", "Медіана FRP, МВт", "Найбільше за рік"]
            yr = f.groupby("year").size()
            v = [len(f), (f.daynight == "N").mean() if len(f) else None, f.frp.median() if len(f) else None, f"{int(yr.idxmax())}: {N(yr.max())}" if len(yr) else "—"]
            vals[pk] = len(f) / max(B.park_u.area.iloc[0] / 1e8, 1e-9)
        elif k == "hansen":
            dom = B.park_mask
            loss = {2000 + y: float(B.PIX_HA[(B.loss == y) & dom].sum()) for y in range(1, 25)}
            top = max(loss, key=loss.get)
            cols = ["Ліс, га", "Частка лісу", "Втрати 2001–2024, га", "Рік найбільших втрат"]
            v = [B.PIX_HA[dom & B.forest].sum(), (B.forest & dom).sum() / max(dom.sum(), 1), sum(loss.values()), f"{top}: {N(loss[top])} га"]
            vals[pk] = sum(loss.values()) / max(B.PIX_HA[dom & B.forest].sum(), 1e-9)
        elif k == "power":
            w = common.power(pk)
            jja = w[w.index.month.isin([6, 7, 8])]
            cols = ["Тмакс літа, °C", "Опади за рік, мм", "Вологість повітря, %", "Днів понад 35 °C за рік"]
            yrs = len(w) / 365.25
            v = [jja.T2M_MAX.mean(), w.PRECTOTCORR.sum() / yrs, w.RH2M.mean(), (w.T2M_MAX > 35).sum() / yrs]
            vals[pk] = jja.T2M_MAX.mean()
        elif k == "srtm":
            z = common.dem(pk)[B.park_mask]
            cols = ["Висота мін, м", "Медіана, м", "Макс, м"]
            v = [z.min(), np.median(z), z.max()]
            vals[pk] = float(np.median(z))
        elif k == "modis":
            nd = common.modis_ndvi(pk)
            dom = B.park_mask & B.forest
            a_, b_ = nd["pre"][dom], nd["post"][dom]
            cols = ["NDVI лісу до пожежі 2023", "Після", "Падіння > 0,275"]
            v = [np.nanmedian(a_), np.nanmedian(b_), np.nanmean((a_ - b_) > .275)]
            vals[pk] = float(np.nanmean((a_ - b_) > .275))
        elif k == "sentinel2":
            from eaglewings.ds.tasks import logging_greece as LG
            n = [len(LG.composite(y, pk)[1]) for y in range(min(LG.YEARS) - 1, max(LG.YEARS) + 1)]
            cols = ["Композитів", "Сцен усього", "Сцен на рік, мін"]
            v = [len(n), sum(n), min(n)]
        elif k == "opera":
            from eaglewings.ds import opera as OP
            sn = [OP.snapshot(d, pk) for y in (2023, 2024) for d in OP.dates(y)]
            dist = [float(np.isin(x, [1, 2, 3, 4, 5, 6, 7, 8]).mean()) for x in sn]
            cols = ["Знімків", "Найбільша частка з порушенням", "Знімків без даних"]
            v = [len(sn), max(dist), sum(float((x == 255).mean()) > .99 for x in sn)]
            vals[pk] = max(dist)
        elif k == "osm_park":
            cols = ["Площа, км²"]
            v = [B.park_u.area.iloc[0] / 1e6]
        elif k == "osm_roads":
            r = B.roads.to_crs(common.UTM)
            km = r.length / 1000
            cols = ["Доріг, км", "Лісові дороги track", "Доріг на км² парку"]
            v = [km.sum(), km[r.highway == "track"].sum() / max(km.sum(), 1e-9), km.sum() / max(B.park_u.area.iloc[0] / 1e6, 1e-9)]
            vals[pk] = km.sum() / max(B.park_u.area.iloc[0] / 1e6, 1e-9)
        rows.append([nm] + [x if isinstance(x, str) else (None if x is None or x != x else round(float(x), 3)) for x in v])
    what = dict(firms="теплових точок на 100 км²", hansen="втрат від площі лісу", power="Тмакс літа", srtm="медіана висоти", modis="частка лісу з падінням NDVI 2023",
                opera="найбільша частка пікселів з порушенням", osm_roads="густота доріг")
    if vals and k in what:
        hi, lo = max(vals, key=vals.get), min(vals, key=vals.get)
        diffs.append(f"Найбільша {what[k]} у парку «{PKM.ALL[hi]['short']}», найменша в «{PKM.ALL[lo]['short']}».")
    return dict(cols=cols, rows=rows, diffs=diffs)


def describe(k):
    """EDA одного набору: те, що показує вкладка «Набори даних». Рахує крок eda (scripts/eda.py)."""
    m = SRC[k]
    o = dict(id=k, **m, **F[k]())
    bp = by_park(k)
    if bp:
        o["parks"] = bp
    o["disk"], o["files"] = du(m["path"]), nfiles(m["path"])
    o["unit_"] = UNIT[k]
    o["datasets"] = [n for n, v in LINEAGE.items() if k in v["src"] or k in v.get("also", [])]
    return o


if __name__ == "__main__":
    # Лише збирає готові результати кроку eda у JSON сайту: нічого не рахує.
    miss = [k for k in SRC if not (EDA / f"{k}.json").exists()]
    if miss:
        sys.exit(f"немає EDA наборів {miss}: спершу крок eda (flows/pipeline.py eda)")
    out = {k: json.loads((EDA / f"{k}.json").read_text()) for k in SRC}
    for k, o in out.items():
        print(k, o["rows"], o["disk"], o["files"], o["short"][:80], flush=True)
    (ROOT / "web" / "science" / "sources.json").write_text(json.dumps(dict(groups=[dict(t=t, items=v) for t, v in GROUPS.items()], data=out), ensure_ascii=False, default=float, separators=(",", ":")))
