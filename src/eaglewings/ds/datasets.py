"""Версіонування датасетів.

Кожен датасет — data/datasets/<name>/v<N>/{data.csv.gz, manifest.json} + data/datasets/<name>/versions.json.
Версія визначається вмістом: digest = sha256(csv + параметри побудови). Той самий вміст → та сама версія;
будь-яка зміна даних або параметрів → нова версія (стара лишається на диску).
У ClearML кожна версія — Dataset у `Dadia/datasets` (`tracking.dataset_id`), запуск посилається на неї параметром `Datasets/<context>` — видно, на якій версії навчено кожну модель.
"""
import datetime as dt
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import os
import subprocess
import sys

from . import park
from .common import ROOT, LIB

DATASETS = ROOT / "data" / "datasets"
# Датасети, що залежать від місця, збираються окремо для кожного парку (<назва>_<парк>, окремий процес з GR_PARK)
# і зливаються в <назва> з колонкою park. Злиття знає своїх батьків: у маніфесті й, з --clearml, у ClearML (parent_datasets).
PER_PARK = {"burned_area_pixels", "fire_day_daily", "h3_risk_cells", "logging_gr_pixels", "logging_gr_cells", "opera_gr_snapshots"}
PARK_BUILD = os.environ.get("GR_PARK_BUILD") == "1"


@dataclass
class DatasetVersion:
    name: str
    version: int
    digest: str
    path: Path
    df: pd.DataFrame = field(repr=False)
    manifest: dict = field(repr=False)

    @property
    def tag(self):
        return f"{self.name}@v{self.version}"


def save(df, name, *, description, targets, features, params, sources, unit):
    if PARK_BUILD:                                       # збір для одного парку: датасет <назва>_<парк>, потім злиття (merge)
        name = f"{name}_{park.KEY}"
        description = f"{park.INFO['name']}. {description}"
    d = DATASETS / name
    d.mkdir(parents=True, exist_ok=True)
    payload = df.to_csv(index=False).encode()
    digest = hashlib.sha256(payload + json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]
    idx_f = d / "versions.json"
    idx = json.loads(idx_f.read_text()) if idx_f.exists() else []
    for v in idx:
        if v["digest"] == digest:
            if not _IN_GET:                            # get() сам робить EDA після збору
                eda(name)
            return load(name, v["version"])
    ver = max([v["version"] for v in idx], default=0) + 1
    stats = {t: dict(positives=int(df[t].sum()), prevalence=round(float(df[t].mean()), 4)) for t in targets if t in df}   # до запису: помилка в мітках не лишає напівзаписаної версії
    vd = d / f"v{ver}"
    vd.mkdir()
    (vd / "data.csv.gz").write_bytes(gzip.compress(payload, mtime=0))
    manifest = dict(name=name, version=ver, digest=digest, created=dt.datetime.now().isoformat(timespec="seconds"), rows=len(df),
                    unit=unit, description=description, targets=targets, features=features, params=params, sources=sources,
                    label_stats=stats, schema={c: str(t) for c, t in df.dtypes.items()})
    (vd / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    idx.append(dict(version=ver, digest=digest, created=manifest["created"], rows=len(df), label_stats=stats))
    idx_f.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    catalog()
    if not _IN_GET:
        eda(name)
    return DatasetVersion(name, ver, digest, vd / "data.csv.gz", df, manifest)


def _report(name):
    """Сторінка-звіт про датасет: Prefect, файл, ClearML Reports; результат EDA у data/processed/eda/datasets/<name>.json."""
    from . import pages
    pages.dataset(name)


def load(name, version=None):
    if PARK_BUILD and name in PER_PARK:                  # під час збору для парку датасет місця — це датасет саме цього парку
        name = f"{name}_{park.KEY}"
    d = DATASETS / name
    idx = json.loads((d / "versions.json").read_text())
    v = idx[-1] if version is None else next(x for x in idx if x["version"] == version)
    vd = d / f"v{v['version']}"
    manifest = json.loads((vd / "manifest.json").read_text())
    df = pd.read_csv(vd / "data.csv.gz", dtype={c: t for c, t in manifest["schema"].items() if t in ("object", "str", "string")})
    return DatasetVersion(name, v["version"], v["digest"], vd / "data.csv.gz", df, manifest)


def catalog():
    """data/datasets/README.md — каталог усіх датасетів і версій."""
    L = ["# Датасети (версіоновані)", "",
         "Генерується автоматично (`ds/datasets.py`). Версія = хеш вмісту + параметрів побудови; старі версії не видаляються.", "",
         "| датасет | версія | digest | рядків | одиниця | лейбли (позитивних / частка) | створено |", "|---|---|---|---|---|---|---|"]
    for d in sorted(p for p in DATASETS.iterdir() if p.is_dir()):
        for v in json.loads((d / "versions.json").read_text()):
            m = json.loads((d / f"v{v['version']}" / "manifest.json").read_text())
            lab = "<br>".join(f"`{k}`: {s['positives']:,} / {s['prevalence']:.1%}" for k, s in m["label_stats"].items())
            L.append(f"| `{m['name']}` | v{m['version']} | `{m['digest']}` | {m['rows']:,} | {m['unit']} | {lab} | {m['created']} |")
    L += ["", "## Опис", ""]
    for d in sorted(p for p in DATASETS.iterdir() if p.is_dir()):
        m = json.loads((d / f"v{json.loads((d / 'versions.json').read_text())[-1]['version']}" / "manifest.json").read_text())
        L += [f"### `{m['name']}`", "", m["description"], "", f"- **Одиниця:** {m['unit']}", f"- **Лейбли:** " + ", ".join(f"`{t}`" for t in m["targets"]),
              f"- **Ознаки:** " + ", ".join(f"`{t}`" for t in m["features"]), f"- **Параметри побудови:** `{json.dumps(m['params'], ensure_ascii=False, default=str)}`",
              f"- **Джерела:** " + "; ".join(m["sources"]), ""]
    (DATASETS / "README.md").write_text("\n".join(L))


# ---------------------------------------------------------------- «дай датасет»: одна функція збору на датасет
# Датасети, які збираються лише з сирих даних (або з інших таких датасетів), мають тут свою функцію збору. Задачі й крок
# datasets беруть їх через get(): якщо датасет уже зібрано з тих самих сирих файлів тим самим кодом, він читається з диска,
# інакше збирається. Копій коду збору немає, тож розійтися нічому. Датасети, які задача збирає зі своїх проміжних
# результатів (наприклад, кандидати веж із видимістю), лишаються в задачі.
BUILDERS = dict(
    burned_area_pixels=("eaglewings.ds.tasks.burned_area", "build_dataset"),
    fire_day_daily=("eaglewings.ds.tasks.fire_day", "build_dataset"),
    h3_risk_cells=("eaglewings.ds.tasks.spatial_risk", "build_dataset"),
    logging_gr_pixels=("eaglewings.ds.tasks.logging_greece", "build_pixels"),
    logging_gr_cells=("eaglewings.ds.tasks.logging_greece", "build_cells"),
    opera_gr_snapshots=("eaglewings.ds.tasks.logging_greece", "build_opera"),
    cv_benchmark_aerial=("eaglewings.ds.tasks.cv_benchmark", "build_aerial"),
    cv_benchmark_ground=("eaglewings.ds.tasks.cv_benchmark", "build_ground"),
)
NEEDS = dict(logging_gr_cells=["h3_risk_cells"])
_IN_GET = False


def _fingerprint(name):
    """Ключ збору: код модуля збору й ds/common.py, сирі файли з data_lineage (розмір і час зміни), ключі датасетів-джерел."""
    import importlib
    import inspect
    import sys
    sys.path.insert(0, str(LIB / "scripts"))
    from data_lineage import LINEAGE, SRC
    from .tracking import source_files
    mod, _ = BUILDERS[name]
    h = hashlib.sha256(inspect.getsource(importlib.import_module(mod)).encode())
    h.update((LIB / "ds" / "common.py").read_bytes())
    for k in LINEAGE[name]["src"] + LINEAGE[name].get("also", []):
        for f in source_files(SRC[k]["path"]):
            st = f.stat()
            h.update(f"{f.relative_to(ROOT)}:{st.st_size}:{st.st_mtime_ns}".encode())
    for d in NEEDS.get(name, []):
        h.update(_fingerprint(d).encode())
    return h.hexdigest()[:16]


def get(name):
    """Датасет name: з диска, якщо його зібрано з тих самих даних тим самим кодом, інакше зібрати. Потім EDA.
    Датасет місця (PER_PARK) збирається для кожного парку окремо й зливається в один."""
    if name in PER_PARK and not PARK_BUILD and len(park.PARKS) > 1:
        return merged(name)
    return _get(name)


def _get(name):
    """name — назва збирача з BUILDERS. Під час збору для парку (GR_PARK_BUILD) датасет місця лежить під <назва>_<парк>."""
    global _IN_GET
    import importlib
    for d in NEEDS.get(name, []):
        _get(d)
    key = _fingerprint(name)
    dname = f"{name}_{park.KEY}" if PARK_BUILD and name in PER_PARK else name
    idx_f = DATASETS / dname / "versions.json"
    idx = json.loads(idx_f.read_text()) if idx_f.exists() else []
    if idx and idx[-1].get("build_key") == key:
        dv = load(dname)
        print(f"датасет {dv.tag}: з диска", flush=True)
    else:
        mod, fn = BUILDERS[name]
        _IN_GET = True
        try:
            dv = getattr(importlib.import_module(mod), fn)()
        finally:
            _IN_GET = False
        idx = json.loads(idx_f.read_text())
        for v in idx:
            if v["version"] == dv.version:
                v["build_key"] = key
        idx_f.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
        dv = load(dname, dv.version)                   # задача завжди бачить датасет таким, яким він лежить на диску
        print(f"датасет {dv.tag}: зібрано", flush=True)
    eda(dname)
    return dv


def eda(name):
    """EDA датасету: окремий підкрок у Prefect зі сторінкою, а результат EDA лежить у data/processed/eda/ для сайту."""
    from . import pflow
    pflow.mark(f"EDA: {name}")
    _report(name)


def _lineage(dv, parents, keys):
    """Злиття dv: батьки (датасети парків) і що змінилось проти попередньої версії злиття — пише в маніфест і версії."""
    man_f = dv.path.parent / "manifest.json"
    man = json.loads(man_f.read_text())
    if "parents" in man:
        return
    man["parents"] = [dict(name=p.name, version=p.version, digest=p.digest, rows=len(p.df), park=k) for p, k in zip(parents, keys)]
    prev = None
    if dv.version > 1:
        pf = DATASETS / dv.name / f"v{dv.version - 1}" / "manifest.json"
        prev = json.loads(pf.read_text()) if pf.exists() else None
    was = {x["name"]: x for x in (prev or {}).get("parents", [])}
    ch = []
    for x in man["parents"]:
        o = was.get(x["name"])
        if o is None:
            ch.append(f"нова пачка {x['name']}@v{x['version']} ({park.ALL[x['park']]['short']}): {x['rows']:,} рядків".replace(",", " "))
        elif o["version"] != x["version"]:
            d = x["rows"] - o.get("rows", x["rows"])
            ch.append(f"оновлено {x['name']}: v{o['version']} → v{x['version']} ({'+' if d >= 0 else ''}{d:,} рядків)".replace(",", " "))
    for nme in set(was) - {x["name"] for x in man["parents"]}:
        ch.append(f"прибрано пачку {nme}")
    man["changes"] = ch if prev else [f"перше злиття {len(parents)} пачок: " + ", ".join(f"{x['name']}@v{x['version']}" for x in man["parents"])]
    man["previous"] = f"{dv.name}@v{dv.version - 1}" if prev else None
    man_f.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str))
    idx_f = DATASETS / dv.name / "versions.json"
    idx = json.loads(idx_f.read_text())
    for v in idx:
        if v["version"] == dv.version:
            v["changes"] = man["changes"]
    idx_f.write_text(json.dumps(idx, ensure_ascii=False, indent=1))


def merged(name):
    """Датасет місця для всіх парків: для кожного парку окремий процес (GR_PARK=<парк>, GR_PARK_BUILD=1) збирає
    <назва>_<парк> або бере його з диска, потім усі зливаються в <назва> з першою колонкою park.
    Злиття стає новою версією лише тоді, коли змінилась версія котрогось батька; батьки записані в params і маніфесті."""
    parts, parents = [], []
    for k in park.PARKS:
        env = dict(os.environ, GR_PARK=k, GR_PARK_BUILD="1")
        code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from eaglewings.ds import datasets; datasets.get({name!r})"
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env)
        if r.returncode:
            raise RuntimeError(f"датасет {name}_{k}: збір для парку {k} упав (код {r.returncode})")
        dv = load(f"{name}_{k}")
        parts.append(dv.df.assign(park=k)[["park"] + list(dv.df.columns)])
        parents.append(dv)
    m0 = parents[0].manifest
    df = pd.concat(parts, ignore_index=True)
    dv = save(df, name, description=f"{m0['description'].split('. ', 1)[-1]} Злиття {len(parents)} парків: колонка park.",
              targets=m0["targets"], features=m0["features"], unit=m0["unit"], sources=m0["sources"],
              params=dict(m0["params"], parents=[p.tag for p in parents], parks=list(park.PARKS)))
    _lineage(dv, parents, list(park.PARKS))
    print(f"датасет {dv.tag}: злиття " + ", ".join(p.tag for p in parents), flush=True)
    return load(name, dv.version)


def save_parks(parts, name, **meta):
    """Датасет, який задача збирає сама по кожному парку: parts = {парк: DataFrame}. Кожен парк — окремий датасет
    <назва>_<парк>, потім злиття <назва> з колонкою park і батьками в маніфесті (як у merged). З одним парком — просто <назва>."""
    if len(parts) == 1:
        return save(next(iter(parts.values())), name, **meta)
    parents = []
    for k, df in parts.items():
        m = dict(meta, description=f"{park.ALL[k]['name']}. {meta['description']}")
        parents.append(save(df, f"{name}_{k}", **m))
    df = pd.concat([dv.df.assign(park=k)[["park"] + list(dv.df.columns)] for k, dv in zip(parts, parents)], ignore_index=True)
    dv = save(df, name, **dict(meta, description=f"{meta['description']} Злиття {len(parents)} парків: колонка park.",
                               params=dict(meta["params"], parents=[p.tag for p in parents], parks=list(parts))))
    _lineage(dv, parents, list(parts))
    return dv


def main_park(df):
    """Рядки основного парку злитого датасету: для гілок, прив'язаних до місць основного парку (камери, вежа CAM-02)."""
    if "park" not in df:
        return df
    return df[df.park == park.MAIN].drop(columns="park").reset_index(drop=True)
