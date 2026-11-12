"""Реєстр гілок: усе про кожну гілку в одному місці.

Гілка це одне питання дослідження (FD, FM, FG_day …). Тут записано:
  q, why, kind, page   питання, навіщо, тип (виміряно / сценарій / симуляція), якір на сторінці дослідження
  group, title         група в меню сторінки дослідження; коротка назва розділу (для гілки без свого фрагмента сторінки)
  job                  ключ задачі в потоці й у scripts/05_experiments.py
  run                  модуль задачі в ds/tasks/ і, якщо задача рахує кілька гілок, її частина: "logging_greece:ld"
  summary, part        ключ у data/processed/experiments_summary.json і частина (група) рядків гілки, якщо задача спільна
  step, after, name, en   крок потоку (core / cv), після яких задач, назви для плану й для графа Prefect
  datasets             датасети гілки: від них залежить, що переганяти, коли датасет змінився
  lead                 як шукати лідера й що показувати: колонки підходу, метрики вибору й тесту, CI

Решта коду бере ці дані звідси: TASKS (05_experiments), JOBS (flows/pipeline.py), LEAD (ds/pages.py),
BRANCHES (ds/report.py). Нова гілка чи новий алгоритм додаються тут і в модулі задачі, а не в дев'яти місцях.
"""

KIND = {"m": "виміряно", "s": "сценарний аналіз", "x": "симуляція"}

R = dict(
    FD=dict(group="Пожежі · виміряно", q="Що саме згоріло у 2023 році і скільки?", kind="m", page="a",
            why="Площа й межі згарища потрібні для оцінки збитків і відновлення, а польове обстеження великої пожежі триває місяцями.",
            job="t1", run="burned_area", summary="t1", step="core", after=["datasets"], name="FD · що згоріло у 2023", en="FD - burned area 2023",
            datasets=["burned_area_pixels"],
            lead=dict(code="FD", name="family", val="calib_f1", test="test_f1", ci="test_f1_ci", metric="F1", how="найкращий F1 на калібрувальній половині блоків")),
    FM=dict(group="Пожежі · виміряно", q="Які теплові точки з двох супутників це одна пожежа?", kind="m", page="fm",
            why="Одна пожежа дає сотні теплових точок від двох сенсорів. Щоб рахувати пожежі, а не точки, їх треба звести в події.",
            job="t13", run="matching_benchmark", summary="t13", step="core", after=["fetch", "bench"], name="FM · які точки це одна пожежа", en="FM - which points are one fire",
            datasets=["matching_firms_effis"],
            lead=dict(code="FM", name="approach", val="val_bcubed_f1", test="bcubed_f1", ci=("bcubed_f1_ci_lo", "bcubed_f1_ci_hi"), metric="BCubed F1", how="найкращий BCubed F1 на 2021–2023, тест 2024–2025")),
    FG_day=dict(group="Пожежі · виміряно", q="Чи можна за погодою заздалегідь виділити дні з пожежами?", kind="m", page="b",
                why="Чи дозволяють відкриті погодні дані заздалегідь виділити дні з підвищеною ймовірністю вогню.",
                job="t2", run="fire_day", summary="t2", step="core", after=["datasets"], name="FG · день пожежі", en="FG - fire day",
                datasets=["fire_day_daily"],
                lead=dict(code="FG", group="target", name="approach", val="val_pr_auc", test="pr_auc", ci="pr_auc_ci", metric="PR AUC", how="найкращий PR AUC на валідаційних роках, окремо для кожного лейбла")),
    FG_risk=dict(group="Пожежі · виміряно", q="Які ділянки парку згорять наступними?", kind="m", page="c",
                 why="Чи можна з рельєфу, лісистості й історії передбачити, які ділянки згорять, і чи переноситься прогноз з однієї пожежі на іншу.",
                 job="t3", run="spatial_risk", summary="t3", step="core", after=["datasets"], name="FG · ризик ділянок", en="FG - cell risk",
                 datasets=["h3_risk_cells"],
                 lead=dict(code="FG", name="approach", val="val_roc_auc", test="test_roc_auc", ci="test_roc_auc_ci", metric="ROC AUC", how="найкращий ROC AUC на валідації 2022 → 2023")),
    FG_spread=dict(group="Пожежі · виміряно", q="Куди піде фронт пожежі за добу?", kind="m", page="e",
                   why="Чи пояснюють вітер, рельєф і паливо просування фронту краще за рівномірне зростання.",
                   job="t5", run="fire_spread", summary="t5", step="core", after=["fetch"], name="FG · поширення фронту", en="FG - fire spread",
                   datasets=["fire_spread_cells"],
                   lead=dict(code="FG", name="approach", val="lofo_f1_24", test="test_f1_24", ci=None, metric="F1 за 24 год", how="найкращий F1 у перевірці «без однієї пожежі» (LOFO), тест на Еврос-2023")),
    AR=dict(group="Пожежі · виміряно", q="Де пожежі повторюються на тих самих місцях?", kind="m", page="ar",
            why="Чи повторюються пожежі на тих самих місцях настільки, щоб контури минулих подій передбачали нові.",
            job="t21", run="areal_fires", summary="t21", step="core", after=["fetch"], name="AR · де пожежі повторюються", en="AR - recurring fires",
            datasets=["ar_fires"],
            lead=dict(code="AR", rows=("part", ["test"]), name="approach", val="val_pai", test="test_pai", ci=("lo", "hi"), metric="PAI", how="найкращий PAI на валідації 2021–2023")),
    LD=dict(group="Вирубка · виміряно", q="Чи видно рубку на знімках Sentinel-2, коли рубок мало?", kind="m", page="ld",
            why="Перевіряємо, чи працює детекція рубок там, де їх мало.",
            job="t19_ld", run="logging_greece:ld", summary="t19", part="LD", step="core", after=["datasets"], name="LD · рубка на знімках", en="LD - logging on imagery",
            datasets=["logging_gr_pixels"],
            lead=dict(code="LD · LM · LG", rows=("part", ["LD", "LM", "LG"]), group="part", name="approach", val="val_metric", test="test_metric", ci="ci", metric="метрика частини", how="найкраща метрика на валідаційних роках, окремо для LD, LM, LG")),
    LM=dict(group="Вирубка · виміряно", q="Чи точніший збіг Sentinel-2 і NASA OPERA, ніж одне джерело?", kind="m", page="lm",
            why="Перевіряємо, чи допомагає збіг двох джерел у парку з малою кількістю рубок.",
            job="t19_lm", run="logging_greece:lm", summary="t19", part="LM", step="core", after=["datasets"], name="LM · збіг Sentinel-2 і OPERA", en="LM - Sentinel-2 and OPERA agree",
            datasets=["logging_gr_pixels", "opera_gr_snapshots"]),
    LG=dict(group="Вирубка · виміряно", q="Де рубатимуть наступного року?", kind="m", page="lg",
            why="Перевіряємо, чи можна передбачити рубки в парку, де вони рідкісні.",
            job="t19_lg", run="logging_greece:lg", summary="t19", part="LG", step="core", after=["datasets"], name="LG · де рубатимуть", en="LG - next-year logging",
            datasets=["logging_gr_cells"]),
    SD=dict(group="Детектори · виміряно", q="Чи видно машини згори і за якої роздільності?", kind="m", page="sd",
            why="Чи видно транспорт у лісі на відкритих знімках і як точність залежить від роздільності.",
            job="t11", run="cv_benchmark", summary="t11", step="cv", after=["datasets"], name="SD · машини згори на VEDAI", en="SD - vehicles from above, VEDAI",
            datasets=["cv_benchmark_aerial"],
            lead=dict(code="SD", group=("dataset", "target"), name="approach", val=None, test="ap50", ci="ap50_ci", metric="AP50", how="готові детектори без навчання: порівняння на еталоні, вибору на тесті немає")),
    CD=dict(group="Детектори · виміряно", q="Наскільки точно готові моделі бачать дим і вогонь на фото?", kind="m", page="cd",
            why="Наскільки точно готові моделі без донавчання знаходять дим, вогонь і транспорт на фото.",
            job="t18", run="cv_benchmark_ground", summary="t18", step="cv", after=["datasets"], name="CD · дим і вогонь на D-Fire, машини на ENA24", en="CD - smoke and fire, D-Fire; ENA24",
            datasets=["cv_benchmark_ground"],
            lead=dict(code="CD", group=("dataset", "target"), name="approach", val=None, test="ap50", ci="ap50_ci", metric="AP50", how="готові детектори без навчання: порівняння на еталоні, вибору на тесті немає")),
    TP=dict(group="Сценарний аналіз", q="Де поставити вежі, щоб побачити найбільше майбутніх пожеж?", kind="s", page="tp",
            why="Де розмістити кілька точок огляду, щоб у поле зору з урахуванням рельєфу потрапляла найбільша частка майбутніх пожеж.",
            job="t23", run="tower_placement", summary="t23", step="core", after=["datasets"], name="TP · де ставити вежі", en="TP - tower placement",
            datasets=["h3_risk_cells", "tp_fires", "tp_candidates"],
            lead=dict(code="TP", rows=("part", ["test"]), name="approach", val=None, test="first", ci=("first_lo", "first_hi"), metric="частка пожеж, які вежі бачать першими", how="вибір на валідаційних роках, тут тестові роки")),
    CG=dict(group="Сценарний аналіз", q="Де на мапі дим, якщо камера бачить лише напрямок?", kind="s", page="cg",
            why="Наскільки точно можна знайти місце диму за напрямком і рельєфом і що дає друга камера.",
            job="t22", run="camera_geo", summary="t22", step="core", after=["datasets", "vision"], name="CG · камера на координати (сценарний аналіз)", en="CG - camera to map (scenario)",
            datasets=["h3_risk_cells", "cv_camera_zone"],
            lead=dict(code="CG", rows=("part", ["locate"]), name="approach", val="val_km2", test="test_km2", ci=None, metric="площа пошуку, км² (менше краще)", low=True, how="найменша площа пошуку на валідації; азимут симульований")),
    CP=dict(group="Сценарний аналіз", q="Через які місця на дорогах проходить під'їзд до рубок?", kind="s", page="cp",
            why="Чи можна кількома камерами на дорогах побачити більшість під'їздів до місць, де пізніше зрубали ліс.",
            job="t24", run="camera_placement", summary="t24", step="core", after=["datasets"], name="CP · де ставити камери", en="CP - camera placement",
            datasets=["logging_gr_cells", "cp_events"],
            lead=dict(code="CP", rows=("part", ["test"]), name="approach", val=None, test="log", ci=("log_lo", "log_hi"), metric="частка рубок під камерами", how="вибір на валідаційних роках, тут тестові роки")),
    XM=dict(group="Симуляція", q="Як звести сигнали з різних джерел в одну подію?", kind="x", page="xm",
            why="Як звести повідомлення з різних джерел про ту саму подію в одну і які параметри для цього потрібні.",
            job="t20", run="cross_source", summary="t20", step="core", after=["mocks"], name="XM · зведення сигналів (симуляція)", en="XM - signal fusion (simulated)",
            datasets=["xm_signals"],
            lead=dict(code="XM", group="kind", name="approach", val="val_f1", test="test_f1", ci=("ci_lo", "ci_hi"), metric="F1", how="найкращий F1 на валідації; потік симульований")),
    AS=dict(group="Симуляція", q="Які зони інтересу є в парку, якого типу, з якими рисами в часі й просторі і з яких джерел?", kind="x", page="as",
            why="Ареали застосунку кажуть, що відбувається в кожній зоні: тип, чи вона активна, згасла чи відновилась, чи розширюється, як часто там щось стається й хто це бачить. Перевіряємо, наскільки ці риси збігаються із задумом сценарію.",
            job="t25", run="areal_sim", summary="t25", step="core", after=["fetch"], name="AS · зони інтересу (симуляція)", en="AS - zones of interest (simulated)",
            datasets=["as_areals"],
            lead=dict(code="AS", name="approach", val="val_score", test="test_score", ci=("ci_lo", "ci_hi"), metric="середня точність рис зони", how="найкраща середня точність на прогонах вибору; потік симульований")),
    RS=dict(group="Маршрути", q="Наскільки добре розв'язувач підбирає маршрути порівняно з найкращими відомими?", kind="m", page="rs", title="розв'язувач маршрутів",
            why="Маршрути груп будує розв'язувач. Перш ніж йому довіряти, міряємо його на стандартних задачах TOPTW, де відомі найкращі результати.",
            job="t26", run="solver_benchmark", summary="t26", step="core", after=["bench"], name="RS · розв'язувач маршрутів", en="RS - route solver on TOPTW",
            datasets=["toptw_instances"],
            lead=dict(code="RS", name="approach", val="val_gap", test="test_gap", ci=("gap_ci_lo", "gap_ci_hi"), metric="розрив до найкращого відомого, % (менше краще)", low=True,
                      how="найменший середній розрив на Solomon (вибір), тест на Cordeau; понад 10 с на задачу не допускається")),
    RD=dict(group="Маршрути", q="Як щодня розкласти задачі між двома групами лісників і перебудувати маршрут, коли з'являється нова подія?", kind="x", page="rd",
            title="маршрути дня (симуляція)",
            why="Лісників мало: за зміну 08:00–16:00 дві групи мають встигнути на події, нічні сигнали й дії зон, а у вільний час патрулювати зони ризику.",
            job="t27", run="patrol_routing", summary="t27", step="core", after=["mocks", "t19_lg", "t26"], name="RD · маршрути дня (симуляція)", en="RD - daily routes (simulated)",
            datasets=["route_instances"],
            lead=dict(code="RD", name="approach", val="val_on_time", test="test_on_time", ci=("ci_lo", "ci_hi"), metric="частка обов'язкових зупинок вчасно",
                      how="найбільша частка вчасно на днях вибору; ±0,5 п.п. → швидша реакція → більший ризик; рандомізовані не беруться; симуляція")),
)

# Увімкнені гілки дослідження: потік, сторінка дослідження, застосунок і зведення беруть лише їх.
# Нове дослідження вибирає тут свої задачі (промпт eaglewings:new-study); вимкнена гілка лишається в реєстрі.
from ..project import config as _config
_proj = _config("branches")
ACTIVE = list(getattr(_proj, "ACTIVE", None) or [k for k in R if not k.startswith("t")])   # гілки проєкту: <проєкт>/config/branches.py


MODES = dict(getattr(_proj, "MODES", None) or {})   # режим гілки, якщо він інший, ніж у базовому кейсі: <проєкт>/config/branches.py MODES


def mode(code, default=None):
    """Режим гілки з config/branches.py проєкту (наприклад, SD="tracks"); без запису — default (поведінка базового кейсу)."""
    return MODES.get(code, default)


# Режими гілок: що змінюється в реєстрі, коли проєкт вмикає гілку в іншому режимі (config/branches.py MODES).
# Свій якір сторінки без фрагмента в templates/science/parts/ → розділ збирається автоматично з результатів гілки.
VARIANTS = {
    ("FD", "signal"): dict(q="Чи бачать супутники пожежі на території: скільки згарищ EFFIS мають сигнал FIRMS?", page="fd_signal", title="чи бачать супутники пожежі",
                           why="Якщо великих пожеж немає, згарище не помалюєш; спершу треба знати, чи теплові точки FIRMS узагалі бачать згарища, які картографує EFFIS.",
                           name="FD · чи бачить FIRMS згарища EFFIS", en="FD - FIRMS sees EFFIS burnt areas", datasets=["fd_signal"], run="fire_signal", after=["fetch", "bench"],
                           lead=dict(code="FD", name="approach", val=None, test="effis_seen", ci=("effis_seen_lo", "effis_seen_hi"), metric="частка згарищ EFFIS із сигналом",
                                     how="опис без вибору: нічого не вчиться, усі роки")),
    ("FG_risk", "transfer"): dict(run="risk_transfer", q="Чи вгадує готова модель з іншої території, які ділянки парку горять?", page="c_transfer", title="ризик ділянок, готова модель",
                                  why="Своєї моделі ризику вчити нема на чому: великих пожеж немає. Перевіряємо готову модель без змін на гексах, де були теплові точки.",
                                  lead=dict(code="FG", name="approach", val=None, test="test_roc_auc", ci="test_roc_auc_ci", metric="ROC AUC",
                                            how="порівняння на тесті без вибору: на цій території нічого не вчиться")),
    ("AR", "logging"): dict(q="Де рубки повторюються на тих самих місцях?", page="ar_log", group="Вирубка · виміряно", title="де рубки повторюються",
                            why="Чи повторюються втрати лісу на тих самих місцях настільки, щоб контури минулих рубок передбачали нові.",
                            name="AR · де рубки повторюються", en="AR - recurring logging", datasets=["ar_logging"], after=["datasets", "t19_lg"], run="areal_logging",
                            lead=dict(code="AR", rows=("part", ["test"]), name="approach", val="val_pai", test="test_pai", ci=("lo", "hi"), metric="PAI",
                                      how="найкращий PAI на валідаційних роках")),
    ("SD", "tracks"): dict(q="Чи видно лісові колії на знімках Sentinel-2?", page="sd_tracks", title="лісові колії на Sentinel-2", step="core",
                           why="Колії в лісі — сліди машин і джипів. Перевіряємо, чи видно їх на відкритих знімках 10 м, якщо вчитись на двох парках і перевіряти на третьому.",
                           name="SD · лісові колії на Sentinel-2", en="SD - forest tracks on Sentinel-2", datasets=["track_segments"], run="track_detection", after=["fetch"],
                           lead=dict(code="SD", name="approach", val="val_roc_auc", test="test_roc_auc", ci=("ci_lo", "ci_hi"), metric="ROC AUC",
                                     how="найкращий ROC AUC на відкладених блоках навчальних парків")),
}


def _apply_modes():
    for (code, m), over in VARIANTS.items():
        if MODES.get(code) == m and code in R:
            R[code] = dict(R[code], **over, mode=m)


def on():
    """Увімкнені гілки в порядку реєстру."""
    return {c: b for c, b in R.items() if c in ACTIVE}


# Задачі потоку без гілки: рахують, але метрик і лідера не мають.
EXTRA = dict(
    t4=dict(run="cv_inference", step="cv", after=["vision"], name="демо детекторів, без метрик", en="detector demo, no metrics"),
)

# Порядок задач у потоці (і в плані): як у списку R, плюс задачі без гілки в кінці свого кроку.
_apply_modes()
ORDER = ["FD", "FM", "FG_day", "FG_risk", "FG_spread", "AR", "LD", "LM", "LG", "TP", "CP", "CG", "XM", "AS", "RS", "RD", "SD", "CD", "t4"]
if MODES.get("AR") == "logging":                     # AR на рубках бере прогноз LG: рахується після LG
    ORDER.remove("AR")
    ORDER.insert(ORDER.index("LG") + 1, "AR")


# ---------------------------------------------------------------- що будується з реєстру
def jobs():
    """Задачі потоку: (ключ, крок, назва, аргументи, після чого) у порядку ORDER."""
    out, have = [], {"fetch", "bench", "vision", "mocks", "datasets"}
    for k in ORDER:
        if k in R and k not in ACTIVE:
            continue
        b = R.get(k) or EXTRA[k]
        j = b.get("job", k)
        out.append((j, b["step"], b["name"], ["scripts/05_experiments.py", j], [d for d in b["after"] if d in have]))   # залежність від вимкненої гілки відпадає
        have.add(j)
    return out


def tasks():
    """05_experiments: ключ задачі → «модуль[:частина]»."""
    out = {b["job"]: b["run"] for b in on().values()}
    out.update({k: v["run"] for k, v in EXTRA.items()})
    return out


def en():
    """Англійські назви задач для графа Prefect."""
    out = {b["job"]: b["en"] for b in R.values()}
    out.update({k: v["en"] for k, v in EXTRA.items()})
    return out


def _rows(spec):
    if "rows" not in spec:
        return spec
    col, vals = spec["rows"]
    return dict(spec, rows=lambda d, c=col, v=tuple(vals): d[d[c].isin(v)])


def lead():
    """ds/pages.LEAD: правила лідера за ключем підсумку. Гілки однієї задачі (LD, LM, LG) мають спільний запис."""
    out = {}
    for b in on().values():
        if "lead" in b and b["summary"] not in out:
            out[b["summary"]] = _rows(b["lead"])
    return out


def report_branches():
    """ds/report.BRANCHES для ноутбуків: питання, тип, ключ підсумку, модуль, якір, навіщо."""
    return {code: dict(q=b["q"], kind=b["kind"], key=b["summary"], **({"part": b["part"]} if "part" in b else {}),
                       module=b["run"].split(":")[0], page=b["page"], why=b["why"]) for code, b in R.items()}


# Порядок гілок у меню сторінки дослідження. Гілка без page на сторінку не потрапляє.
PAGE_ORDER = ["FD", "FM", "FG_day", "FG_risk", "FG_spread", "AR", "LD", "LM", "LG", "SD", "CD", "TP", "CG", "CP", "XM", "AS", "RS", "RD"]


def page_branches():
    """Гілки сторінки дослідження в порядку меню: код, якір, група, тип. Гілки, яких немає в PAGE_ORDER, ідуть у кінець."""
    codes = [c for c in PAGE_ORDER if c in on()] + [c for c in on() if c not in PAGE_ORDER]
    return [dict(code=c, id=R[c]["page"], group=R[c].get("group", "Інше"), kind=R[c]["kind"]) for c in codes if R[c].get("page")]


def by_job(job):
    return next((dict(b, code=c) for c, b in R.items() if b["job"] == job), None)


def by_dataset(name):
    """Гілки, які беруть датасет name: їх треба переганяти, коли він змінився."""
    return [c for c, b in R.items() if name in b.get("datasets", [])]


def is_on(code):
    """Чи рахує проєкт гілку code (config/branches.py ACTIVE). Розділи сторінок вимкнених гілок пропускаються."""
    return code in ACTIVE
