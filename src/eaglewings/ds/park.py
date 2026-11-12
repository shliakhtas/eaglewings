"""Контекст парку: з яким парком дослідження зараз працює код.

    GR_PARK=evros_delta eaglewings script 01_fetch_data    # усе для парку «Дельта Евросу»

Парки задані в ds/regions.py (REGION["fetch"]["parks"]). Перший парк основний: його файли лежать там, де й були
(data/processed/, data/raw/nasa/), і саме його показує застосунок. Інші парки мають свої теки
data/processed/parks/<ключ>/ і data/raw/parks/<ключ>/ (межа OSM, дороги, кеш NASA), тож кеші й результати парків не змішуються.
Сирі архіви на всю країну (FIRMS) спільні: data/raw/firms/.
"""
import os
from pathlib import Path

from .regions import REGION

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
ALL = REGION["fetch"]["parks"]
MAIN = next(iter(ALL))
# GR_PARKS=dadia обмежує дослідження частиною парків: так перевіряємо, що з одним парком числа ті самі, що були до трьох
PARKS = {k: ALL[k] for k in os.environ.get("GR_PARKS", ",".join(ALL)).split(",") if k in ALL}
KEY = os.environ.get("GR_PARK", MAIN)
if KEY not in ALL:
    raise SystemExit(f"GR_PARK={KEY}: такого парку немає в ds/regions.py, є {', '.join(ALL)}")
INFO = ALL[KEY]


def processed(key=None):
    """Тека оброблених даних парку."""
    key = key or KEY
    return ROOT / "data" / "processed" if key == MAIN else ROOT / "data" / "processed" / "parks" / key


def raw(key=None):
    """Тека сирих даних саме цього парку (межа, дороги OSM); архіви на всю країну лежать у data/raw/."""
    key = key or KEY
    return ROOT / "data" / "raw" if key == MAIN else ROOT / "data" / "raw" / "parks" / key


def cache(key=None):
    """Кеш NASA й інших шарів, вирізаних під рамку парку."""
    return raw(key) / "nasa"


P, RAWP, CACHE = processed(), raw(), cache()
for _d in (P, RAWP, CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def local(path, key=None):
    """Шлях сирого набору з scripts/data_lineage.py → той самий набір цього парку (відносно ROOT).
    Межа, дороги й кеш NASA свої в кожного парку; архіви на всю країну (FIRMS, еталони) спільні."""
    key = key or KEY
    path = str(path)
    if key == MAIN:
        return path
    for pre, new in (("data/raw/nasa/", raw(key) / "nasa"), ("data/raw/park_osm.json", raw(key) / "park_osm.json"),
                     ("data/raw/roads_osm.json", raw(key) / "roads_osm.json"), ("data/processed/", processed(key))):
        if path.startswith(pre):
            rest = path[len(pre):]
            return str((new / rest if rest else new).relative_to(ROOT)) + ("/" if path.endswith("/") and rest else "")
    return path
