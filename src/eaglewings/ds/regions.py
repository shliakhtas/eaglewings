"""Конфігурація проєкту: місце, задачі й датасети. Береться з <проєкт>/config/regions.py (див. eaglewings.project).

    python -m eaglewings.ds.regions cpu        # ключі задач для scripts/05_experiments.py
"""
import sys

from ..project import config

_cfg = config("regions")
if _cfg is None:
    raise SystemExit("EagleWings: у проєкті немає config/regions.py (REGION: територія, парки, джерела, роки)")
REGION = _cfg.REGION

# параметри території, якими користуються модулі (значення за замовчуванням — кейс Фракії, тож старі конфіги працюють як були)
_F = REGION.get("fetch", {})
LAT0 = float(_F.get("lat0", 41.1))                            # широта для косинусної поправки в площах і відстанях (центр території)
COUNTRY = _F.get("firms", {}).get("country", "Greece")        # назва країни в архівах FIRMS: <сенсор>_<рік>_<країна>.csv
EFFIS_BBOX = _F.get("effis_bbox", "34,19,42.5,30")             # рамка запиту EFFIS (lat_min,lon_min,lat_max,lon_max)
EFFIS_YEARS = list(_F.get("effis_years", range(2021, 2026)))  # роки згарищ EFFIS
LOG_YEARS = list(_F.get("logging_years", range(2018, 2025)))  # роки втрат лісу Hansen для рубок (LD, LM, LG)
CLEARML_PROJECT = REGION.get("clearml_project", "Dadia")      # корінь проєкту в ClearML
EFFIS_COUNTRY = _F.get("effis_country", "EL")                 # код країни в полі COUNTRY згарищ EFFIS
PROTOCOLS = _F.get("protocols", {})                           # роки валідації й тесту гілок, якщо інші, ніж у кейсі Фракії


def protocol(branch, val, test):
    """Роки валідації й тесту гілки: з config/regions.py (fetch.protocols.<гілка>), інакше значення кейсу Фракії."""
    p = PROTOCOLS.get(branch, {})
    return list(p.get("val", val)), list(p.get("test", test))

if __name__ == "__main__":
    print(" ".join(REGION.get(sys.argv[1], [])) if len(sys.argv) > 1 else REGION["key"])
