"""Увесь конвеєр eaglewings/ як один потік Prefect. Кожен скрипт це окрема задача, і в кожної записано, після яких задач
вона стартує: це ті задачі, чиї файли вона читає. У Prefect ці залежності видно стрілками на графі запуску.

    eaglewings run --plan            # показати порядок і залежності, нічого не запускаючи
    eaglewings run                   # усе: data → vision → mocks → eda → datasets → core → cv → models → site_data → notebooks, наприкінці версія прогону
    eaglewings run core site         # лише ці кроки
    eaglewings run t1 science        # лише ці задачі (ключі з JOBS)
    eaglewings run --dry             # пройти потік, але замість скриптів лише надрукувати команди
    eaglewings run t1 --clearml      # ще й записати в ClearML: запуски, датасети, сирі набори, моделі
    eaglewings run LD FG_risk        # гілки з реєстру ds/branches.py
    eaglewings run --from LD         # змінилась гілка LD: лише вона й усе, що від неї залежить
    eaglewings run --from h3_risk_cells   # змінився датасет: його збір, гілки на ньому й далі

Звіти: у кожному місці потоку одна сторінка у вкладці Artifacts (ds/pages.py): скачаний набір, зібраний датасет,
лідери задачі, наприкінці зведення всіх задач.

Задачі виконуються строго по одній, у порядку списку JOBS (паралельні запуски пишуть у ті самі файли і ділять GPU).
Якщо скрипт падає, потік зупиняється на ньому: задачі, що від нього залежать, не беруть неповних файлів.
Prefect тут лише керує порядком і журналом. Уся робота в scripts/ і ds/tasks/.
Інтерфейс: `prefect server start`, потім http://127.0.0.1:4200.
"""
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.context import get_run_context
from prefect.logging import get_run_logger

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import branches as BR  # noqa: E402
PY = sys.executable
DRY = "--dry" in sys.argv

STEPS = {
    "data": "Завантаження сирих даних → data/raw/, data/processed/",
    "vision": "Комп'ютерний зір на справжніх фото, відео й знімках (GPU) → data/processed/vision/",
    "mocks": "Моки: сценарний потік сигналів демо, бере справжній дим із vision → signals.csv, ops.json",
    "eda": "EDA сирих наборів одразу після завантаження й моків → data/processed/eda/sources/",
    "datasets": "Датасети з сирих даних, кожен своєю функцією збору, і їхня EDA → data/datasets/, data/processed/eda/datasets/",
    "core": "Задачі без GPU → data/datasets/, logs/runs/, experiments_summary.json",
    "cv": "Задачі детекторів на GPU → те саме",
    "models": "Моделі-переможці кожної гілки → models/<код>/, з --clearml ще ClearML Model Registry",
    "site_data": "Дані для сайту з готових результатів → web/science/*.json, web/app.html (нічого не рахує)",
    "notebooks": "Ноутбуки, що повторюють сторінку досліджень → notebooks/",
    "docs": "Документи з коду: схема архітектури й таблиця гілок → artifacts/ (пояснення переписує Claude, коли код змінився)",
}

TASKS = [j[0] for j in BR.jobs()]   # усі задачі 05_experiments, з реєстру гілок

# (ключ, крок, назва, аргументи скрипта, після яких задач). Порядок у списку це порядок запуску.
JOBS = [
    ("fetch", "data", "межа парку, дороги, FIRMS, Hansen, знімки ділянок", ["scripts/01_fetch_data.py"], []),
    ("bench", "data", "еталони D-Fire, VEDAI, ENA24 і згарища EFFIS", ["scripts/00_fetch_benchmarks.py"], []),
    ("vision", "vision", "детектори на фото й відео, маски на знімках", ["scripts/02_vision.py"], ["fetch"]),
    ("mocks", "mocks", "генератор потоку сигналів демо", ["scripts/03_signals.py"], ["fetch", "vision"]),
    ("eda", "eda", "EDA сирих наборів: що в кожному, розподіли, час, мапи", ["scripts/eda.py"], ["fetch", "bench", "vision", "mocks"]),
    ("datasets", "datasets", "датасети з сирих даних і їхня EDA", ["scripts/datasets_build.py"], ["fetch", "bench"]),
    *BR.jobs(),                                      # задачі гілок: з реєстру ds/branches.py
    ("models", "models", "моделі-переможці", ["scripts/07_export_models.py"], TASKS),
    ("science", "site_data", "сторінка досліджень: data, extra, spread", ["scripts/06_build_science.py"], TASKS + ["vision"]),
    *([("routes", "site_data", "маршрути груп, таски, ніч і передачі партнерам — лідером RD", ["scripts/16_routes.py"], ["t27"])] if BR.is_on("RD") else []),
    ("app_extra", "site_data", "дані для застосунку", ["scripts/08_app_extra.py"], ["science"] + (["routes"] if BR.is_on("RD") else [])),
    ("more", "site_data", "сторінка досліджень: more", ["scripts/09_science_more.py"], ["app_extra", "mocks"]),
    ("sources", "site_data", "вкладка «Набори даних»: опис з EDA", ["scripts/11_source_eda.py"], ["eda"]),
    ("maps", "site_data", "вкладка «Набори даних»: мапи з EDA", ["scripts/12_source_maps.py"], ["eda"]),
    ("ds_eda", "site_data", "вкладка «Датасети»: EDA датасетів", ["scripts/10_dataset_eda.py"], TASKS + ["datasets"]),
    ("science_page", "site_data", "сторінка досліджень із шаблону: розділи гілок з реєстру", ["scripts/build_science_page.py"], TASKS + ["science"]),
    ("parks_page", "site_data", "сторінка досліджень: по парках (метрики й мапи кожного парку)", ["scripts/15_parks_page.py"], TASKS + ["science"]),
    ("answers", "site_data", "відповіді чату застосунку", ["scripts/14_app_answers.py"], ["mocks"] + (["routes"] if BR.is_on("RD") else [])),
    ("page", "site_data", "сторінка застосунку web/app.html", ["scripts/04_build_page.py"], ["app_extra", "mocks", "vision"]),
    ("app_recordings", "site_data", "записи застосунку для лендінгу", ["scripts/app_recordings.py"], ["page"]),
    ("landing_parks", "site_data", "мапи парків першого екрана лендінгу й слайди", ["scripts/landing_parks.py"], ["parks_page"]),
    ("landing", "site_data", "лендінг з каркаса й блоків, записи застосунку в блоці views", ["scripts/build_landing.py"], ["app_recordings", "landing_parks"]),
    ("notebooks", "notebooks", "ноутбуки", ["scripts/13_notebooks.py"], TASKS + ["ds_eda"]),
    ("i18n", "site_data", "дві мови: копії сторінок у web/ua/ і web/en/ зі скриптом перекладу", ["scripts/i18n_pages.py"], ["landing", "page", "science_page", "parks_page"]),
    ("docs", "docs", "схема архітектури й таблиця гілок з коду й результатів", ["scripts/build_docs.py"], ["i18n", "notebooks"]),
]
NAME = {k: f"{list(STEPS).index(s) + 1}. {s} · {n}" for k, s, n, _, _ in JOBS}

# Назви задач на графі Prefect: його шрифт не має кирилиці, тому там англійською.
EN = dict(fetch="raw data: park, roads, FIRMS, Hansen, layers", bench="benchmarks: D-Fire, VEDAI, ENA24, EFFIS",
          vision="vision: photos, video, area masks", mocks="mocks: demo signal stream", eda="EDA of raw sources", datasets="datasets from raw data + EDA",
          science="site: science data, extra, spread", app_extra="site: app data", more="site: science more",
          sources="site: sources tab", maps="site: source maps", ds_eda="site: dataset EDA tab", models="site: winner models",
          answers="site: app chat answers", science_page="site: research page from template", parks_page="site: per-park metrics and maps", landing_parks="site: landing park maps", i18n="site: ua and en page copies", app_recordings="site: app recordings for landing", landing="site: landing from blocks", page="site: app page app.html",
          routes="site: team routes and tasks by RD leader", notebooks="notebooks", docs="docs: architecture and branch tables from code")
EN.update(BR.en())                                   # назви задач гілок з реєстру

PNAME = {k: f"{list(STEPS).index(s) + 1}. {s} - {EN[k]}" for k, s, _, _, _ in JOBS}

# Залежність може вказувати лише на задачу, що стоїть вище в списку: інакше порядок запуску суперечив би їй.
_seen = set()
for _k, _s, _n, _a, _d in JOBS:
    assert set(_d) <= _seen, f"{_k}: залежить від {set(_d) - _seen}, а ті стоять нижче"
    _seen.add(_k)


@task(retries=0)
def run_script(name: str, args: list[str]):
    """Один скрипт: запускає, пише його вивід у журнал Prefect, падає, якщо скрипт упав."""
    log = get_run_logger()
    log.info("▶ %s: %s", name, " ".join(args))
    if DRY:
        return
    from prefect.context import get_run_context
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
               PREFECT_PIPELINE_TASK_RUN=str(get_run_context().task_run.id))     # скрипт вкладе свої кроки під цю задачу (ds/pflow.py)
    args = [str(LIB / a) if a.startswith("scripts/") else a for a in args]   # скрипти з бібліотеки, робота в теці проєкту
    p = subprocess.Popen([PY, *args], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        log.info(line.rstrip())
    code = p.wait()
    if code == 3:                                    # дані скачано частково: потік іде далі, задачі без потрібних шарів упадуть самі
        log.warning("%s: частину шарів не скачано, дивись рядки «НЕ ВДАЛОСЯ» вище", name)
        create_markdown_artifact(key="partial-data", markdown=f"**{name}**: частину шарів не скачано. Дивись журнал задачі.",
                                 description="Неповні дані")
        return
    if code:
        raise RuntimeError(f"{name}: {' '.join(args)} завершився з кодом {code}")


@flow(name="eaglewings", description="EagleWings project pipeline from raw data to the site")
def pipeline(steps: list[str]):
    """Задачі йдуть строго по одній. Якщо задача падає, решта йде далі, а ті, що від неї залежать, пропускаються:
    вони не беруть неповних файлів. Наприкінці потік чесно падає, якщо хоч щось упало або пропущено."""
    log = get_run_logger()
    from eaglewings import BANNER
    log.info(BANNER)
    done, broken, journal = {}, set(), []
    started = dt.datetime.now(dt.timezone.utc)
    for k, s, n, args, deps in JOBS:
        if s not in steps and k not in steps:
            continue
        t0 = dt.datetime.now(dt.timezone.utc)
        if broken & set(deps):
            log.warning("пропущено %s: залежить від того, що впало (%s)", PNAME[k], ", ".join(PNAME[d] for d in broken & set(deps)))
            broken.add(k)
            journal.append(dict(key=k, step=s, name=NAME[k], state="пропущено", why=", ".join(NAME[d] for d in broken & set(deps)), t0=t0, t1=t0))
            continue
        fut = run_script.with_options(name=PNAME[k], tags=[s]).submit(n, args, wait_for=[done[d] for d in deps if d in done])
        state = "зроблено"
        try:
            fut.result()
        except Exception as e:
            log.error("впало %s: %s", PNAME[k], e)
            broken.add(k)
            state = "впало"
        done[k] = fut
        journal.append(dict(key=k, step=s, name=NAME[k], state=state, t0=t0, t1=dt.datetime.now(dt.timezone.utc)))
    run = get_run_context().flow_run
    version_url = None
    try:                                             # версія прогону: числа, різниця з попередньою, журнал усіх версій
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_version", LIB / "scripts" / "run_version.py")
        rv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rv)
        vmd, v = rv.record(run.name, str(run.id), [k for k in done])
        art = create_markdown_artifact(key="run-version", markdown=vmd, description=f"Версія прогону {v['n']}: що змінилось і журнал версій")
        from eaglewings.ds import pages as _pg
        _pg.save("run-version", vmd)
        rv.to_clearml(vmd, v)
        from eaglewings.ds.pages import _ui
        version_url = f"{_ui()}/artifacts/artifact/{art}"
    except Exception as e:
        log.warning("версію прогону не записано: %s", e)
    try:                                             # сторінка-зміст прогону: кроки по порядку, під кожним його сторінки, лідери, ClearML
        from eaglewings.ds import pages, tracking
        ran = sorted({k.split("_")[0] for k in TASKS if k in done and k not in broken})
        f = ROOT / "data" / "processed" / "experiments_summary.json"
        leaders = pages.summary(json.loads(f.read_text()), ran) if ran and f.exists() else None
        md = pages.run_summary(run.name, str(run.id), started, journal, leaders, tracking.created_since(started), version_url)
        pages.save("summary", md)
        create_markdown_artifact(key="summary", markdown=md, description="Зміст прогону: кроки, сторінки, лідери, ClearML")
        tracking.summary_page(md)
    except Exception as e:
        log.warning("зміст прогону не побудовано: %s", e)
    if broken:
        raise RuntimeError("впало або пропущено: " + "; ".join(PNAME[k] for k in broken))


def short(k):
    """Коротка назва для плану: код гілки для задач, ім'я скрипта для решти."""
    _, s, n, args, _ = next(j for j in JOBS if j[0] == k)
    return n.replace(" · ", " ") if args[0] == "scripts/05_experiments.py" else Path(args[0]).stem


def plan(steps):
    for s in [x for x in STEPS if x in steps or any(j[1] == x and j[0] in steps for j in JOBS)]:
        print(f"\n{list(STEPS).index(s) + 1}. {s}: {STEPS[s]}")
        for k, s_, n, args, deps in JOBS:
            if s_ != s or (s not in steps and k not in steps):
                continue
            rest = [d for d in deps if d not in TASKS] if set(TASKS) <= set(deps) else deps
            after = (["усі задачі 05_experiments"] if set(TASKS) <= set(deps) else []) + [short(d) for d in rest]
            print(f"     {' '.join(args):30s} {n:46s} після: {', '.join(after) or '—'}")


def resolve(names):
    """Що означає кожне ім'я: крок, задача, гілка з реєстру (FD, LM …) або датасет. → ключі задач потоку."""
    import json as _j
    from eaglewings.ds import datasets as DS
    keys, out, bad = [j[0] for j in JOBS], [], []
    for a in names:
        if a in STEPS:
            out += [j[0] for j in JOBS if j[1] == a]
        elif a in keys:
            out.append(a)
        elif a in BR.R:
            out.append(BR.R[a]["job"])
        elif a in DS.BUILDERS or (DS.DATASETS / a).exists() or BR.by_dataset(a):
            names_ = [a]                                     # і датасети, зібрані з нього (h3_risk_cells → logging_gr_cells)
            while True:
                more = [n for n, need in DS.NEEDS.items() if set(need) & set(names_) and n not in names_]
                if not more:
                    break
                names_ += more
            if a in DS.BUILDERS:
                out.append("datasets")                       # датасет із функцією збору: його збирає крок datasets
            out += [BR.R[c]["job"] for n in names_ for c in BR.by_dataset(n)]
        else:
            bad.append(a)
    if bad:
        sys.exit(f"Невідомо: {bad}. Є кроки: {', '.join(STEPS)}; задачі: {', '.join(keys)}; гілки: {', '.join(BR.R)}; або назва датасету")
    return list(dict.fromkeys(out))


def downstream(seeds, whole=False):
    """Задачі seeds і все, що від них залежить (прямо чи через інші задачі): саме це треба переганяти після зміни.
    Крок datasets спільний для всіх задач, тож через нього не поширюємо: які гілки беруть змінений датасет,
    уже вирішив resolve за реєстром."""
    out = set(seeds)
    for k, s_, n, args, deps in JOBS:                    # JOBS упорядковані: залежності завжди вище
        if (out if whole else out - {"datasets"}) & set(deps) or (k not in TASKS and "datasets" in out and "datasets" in deps):
            out.add(k)
    return [j[0] for j in JOBS if j[0] in out]


if __name__ == "__main__":
    asked = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--from" in sys.argv:                         # змінилось це: перегнати лише його і все, що від нього залежить
        steps = downstream(resolve(asked), whole="datasets" in asked)     # весь крок datasets: перегнати все, що від нього залежить
        print("переганяю:", ", ".join(steps))
    else:
        steps = resolve(asked) if asked else list(STEPS)     # кроки, задачі, гілки або датасети; порядок завжди з JOBS
    if "--clearml" in sys.argv:                      # дозвіл писати в ClearML іде всім скриптам через середовище
        os.environ["DADIA_CLEARML"] = "1"
    if "--plan" in sys.argv:
        plan(steps)
    else:
        pipeline(steps)
