"""Конфігурація проєкту: місце, задачі й датасети. Щоб перенести на інший парк, міняється цей файл.

    python -m ds.regions cpu        # ключі задач для scripts/05_experiments.py
"""
import sys

REGION = dict(
    key='greece',
    title='НП Дадья–Лефкімі–Суфлі, Греція',
    about='Пожежі: згарище, день пожежі, просторовий ризик, поширення, зведення сенсорів. Плюс перевірка детекторів на відкритих еталонах.',
    area='НП Дадья–Лефкімі–Суфлі: межа парку з OSM, ділянки знімків і сітка H3 res 9 задані в scripts/01_fetch_data.py і ds/common.py; пожежа-еталон серпень–вересень 2023.',
    sources=['scripts/01_fetch_data.py', 'scripts/00_fetch_benchmarks.py', 'ds/common.py'],
    cpu=['t1', 't2', 't3', 't5', 't13', 't19', 't20', 't21', 't23', 't24'],
    gpu=['t4', 't11', 't18'],
    datasets=['cv_camera_zone', 'tp_fires', 'tp_candidates', 'cp_events', 'burned_area_pixels', 'fire_day_daily', 'h3_risk_cells', 'fire_spread_cells', 'matching_firms_effis', 'cv_benchmark_aerial', 'cv_benchmark_ground', 'logging_gr_pixels', 'logging_gr_cells', 'opera_gr_snapshots', 'xm_signals', 'ar_fires'],
    requires=[],
    # усе, що стосується місця: з цього завантажуються сирі набори (scripts/01_fetch_data.py) і будується сітка (ds/common.py)
    fetch=dict(
        park_osm=("way", 137321368),                 # межа НП Дадья–Лефкімі–Суфлі в OpenStreetMap (основний парк, його бачить застосунок)
        # парки дослідження: кожен рахується окремо (GR_PARK=<ключ>, ds/park.py), датасети потім зливаються в один з колонкою park.
        # Застосунок показує лише основний парк (перший у списку).
        parks={
            "dadia": dict(osm=("way", 137321368), name="НП Дадья–Лефкімі–Суфлі", short="Дадья",
                          name_en="Dadia–Lefkimi–Soufli National Park"),
            "evros_delta": dict(osm=("way", 137321320), name="НП «Дельта Евросу»", short="Дельта Евросу",
                                name_en="Evros Delta National Park"),
            "emt": dict(osm=("relation", 1850516), name="НП «Східна Македонія й Фракія»", short="Східна Македонія й Фракія",
                        name_en="East Macedonia and Thrace National Park"),
        },
        buffer_m=3000,                              # зона довкола парку для FIRMS
        firms=dict(country="Greece", sensors=["viirs-snpp", "modis"], years=[2021, 2022, 2023, 2024, 2025]),
        hansen=dict(version="GFC-2024-v1.12", tile="50N_020E"),
        power=dict(start="20210101", end="20251231"),
        utm=32635, h3_res=9,
        aois={"dadia_village": dict(lat=41.1265, lon=26.2215, dlat=0.0055, dlon=0.0075, title="Село Дадья та візит-центр"),
              "burn_scar_2023": dict(lat=41.0740, lon=26.2120, dlat=0.0055, dlon=0.0075, title="Згарище пожежі 2023 (південь парку)")},
    ),
)
