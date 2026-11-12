"""ClearML (app.clear.ml): проєкти задач, запуски підходів, датасети й моделі.

Структура в ClearML
    Dadia/T1 · …          проєкт задачі; опис проєкту — постановка, гіпотези, протокол
      benchmark YYYY-MM-DD   батьківський запуск пакета: датасет, протокол, results.json, підсумкові графіки, тег champion
        <підхід>             дочірній запуск (parent = benchmark): гіпотеза, параметри, метрики, CI, графіки
        FINAL <підхід>       фінальна модель (OutputModel) — нащадок того самого пакета
    Dadia/datasets        ClearML Datasets: <name> версії 1, 2 … (наступна версія — нащадок попередньої)
    Dadia/sources         сирі набори, як їх скачано: firms, hansen, modis … ; нове завантаження з іншим вмістом — наступна версія
    Dadia/models          Model Registry: переможець кожної гілки (scripts/07_export_models.py --clearml)

Код задач звертається до запусків через `ml` (start_run / log_metrics / log_figure …); `ml` пише в ClearML.
За замовчуванням запуски пишуться лише локально в logs/runs/ (той самий API), навіть якщо ключі є.
У ClearML — лише за явною командою: scripts/05_experiments.py --clearml. Ключі — у ~/clearml.conf (не в репозиторії).
"""
import contextlib
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from . import pflow
from .common import ROOT  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)
from eaglewings.ds.regions import CLEARML_PROJECT  # noqa: E402
ROOT_PROJECT = CLEARML_PROJECT
DATASETS_PROJECT = f"{ROOT_PROJECT}/datasets"


LOCAL_RUNS = ROOT / "logs" / "runs"


def _project(name):
    return f"{ROOT_PROJECT}/{name}"


JOURNAL = f"{ROOT_PROJECT}/журнал задач"


def _cm(project):
    """Проєкт ClearML для детального журналу задачі: Dadia/журнал задач/<задача>. Гілки лежать окремо, Dadia/<гілка>."""
    return project.replace(f"{ROOT_PROJECT}/", f"{JOURNAL}/", 1)


if "--clearml" in sys.argv:                          # будь-який скрипт: --clearml вмикає запис; flows/pipeline.py передає його через середовище
    os.environ["DADIA_CLEARML"] = "1"


def online():
    """ClearML лише за явною командою: scripts/05_experiments.py --clearml (DADIA_CLEARML=1) і наявні ключі. Інакше все локально, навіть якщо ключі є."""
    if os.environ.get("DADIA_CLEARML") != "1":
        return False
    if os.environ.get("CLEARML_API_ACCESS_KEY") and os.environ.get("CLEARML_API_SECRET_KEY"):
        return True
    from clearml.config import config_obj
    if not config_obj.get("api.credentials.access_key", None):
        raise SystemExit("--clearml: немає ключів у ~/clearml.conf")
    return True


class _LocalLogger:
    def __init__(self, t):
        self.t = t

    def report_single_value(self, name, value):
        self.t.rec["metrics"][name] = value

    def report_scalar(self, title, series, value, iteration):
        self.t.rec["series"].setdefault(title, []).append([iteration, value])

    def report_table(self, title, series, iteration, table_plot):
        pd.DataFrame(table_plot).to_csv(self.t.dir / f"table_{title}.csv", index=False)

    def report_matplotlib_figure(self, title, series, figure, iteration, report_image=True):
        pass                                    # файл зберігає log_figure


class _LocalTask:
    """Той самий набір викликів, що в clearml.Task, але запис у logs/runs/<проєкт>/<час>_<запуск>/run.json."""

    def __init__(self, project, name):
        self.id = uuid.uuid4().hex
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = LOCAL_RUNS / project.replace("/", "_") / f"{stamp}_{name.replace('/', '_')[:60]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.rec = dict(id=self.id, project=project, name=name, parent=None, status="created", tags=[], params={}, props={}, metrics={}, series={})

    def _save(self):
        (self.dir / "run.json").write_text(json.dumps(self.rec, ensure_ascii=False, indent=1, default=str))

    def get_logger(self):
        return _LocalLogger(self)

    def set_parent(self, p):
        self.rec["parent"] = p.id

    def set_comment(self, c):
        self.rec["comment"] = c

    def get_tags(self):
        return self.rec["tags"]

    def add_tags(self, tags):
        self.rec["tags"] += list(tags)

    def set_user_properties(self, **kw):
        self.rec["props"].update(kw)

    def set_parameters(self, d):
        self.rec["params"].update(d)

    def upload_artifact(self, name, artifact_object, wait_on_upload=True):
        shutil.copy(artifact_object, self.dir / Path(name).name)

    def flush(self, wait_for_uploads=True):
        self._save()

    def mark_started(self, force=True):
        self.rec["status"] = "in_progress"

    def mark_failed(self, status_reason="", force=True):
        self.rec["status"] = "failed"
        self._save()

    def mark_completed(self, force=True):
        self.rec["status"] = "completed"
        self._save()


class _BothLogger:
    """Логер, що пише і в ClearML, і в локальний запис (_LocalLogger); локальних аналогів не всі методи мають."""

    def __init__(self, cm, loc):
        self.cm, self.loc = cm, loc

    def __getattr__(self, k):
        a, b = getattr(self.cm, k), getattr(self.loc, k, None)

        def both(*x, **y):
            if b is not None:
                b(*x, **y)
            return a(*x, **y)
        return both


class _Both:
    """Запуск з --clearml: справжній ClearML Task (cm) і його локальна копія в logs/runs (loc) з тим самим id.
    Сайт, ноутбуки й версії прогонів читають локальні записи, тож вони є й тоді, коли запуски йдуть у ClearML."""

    def __init__(self, cm, loc):
        self.cm, self.loc = cm, loc
        loc.id = loc.rec["id"] = cm.id
        self.id, self.name = cm.id, loc.name

    def get_logger(self):
        return _BothLogger(self.cm.get_logger(), self.loc.get_logger())

    def set_parent(self, p):
        self.cm.set_parent(getattr(p, "cm", p))
        self.loc.set_parent(getattr(p, "loc", p))

    def get_tags(self):
        return self.cm.get_tags()

    def __getattr__(self, k):
        a, b = getattr(self.cm, k), getattr(self.loc, k, None)

        def both(*x, **y):
            if b is not None:
                b(*x, **y)
            return a(*x, **y)
        return both


def cm_task(t):
    """Справжній ClearML Task запуску (для OutputModel та інших API ClearML)."""
    return getattr(t, "cm", t)


class _Run:
    def __init__(self, task):
        self.task = task
        self.logger = task.get_logger()
        self.info = SimpleNamespace(run_id=task.id)
        self.props = {}


class _ML:
    """Мінімальний API запусків поверх ClearML. Вкладений start_run → дочірній Task (set_parent)."""

    def __init__(self):
        self.project = None
        self.description = ""
        self.stack = []
        self.figs = {}                  # назва запуску → [(назва графіка, png)]: їх отримує запуск алгоритму в проєкті гілки
        self.last_top = None            # останній пакет «benchmark»: туди йде сторінка-звіт задачі

    # ---- проєкт / запуск
    def set_project(self, name, description):
        self.project, self.description = _project(name), description

    @contextlib.contextmanager
    def start_run(self, run_name, nested=False):
        if online():
            from clearml import Task
            t = _Both(Task.create(project_name=_cm(self.project), task_name=run_name, task_type=Task.TaskTypes.training, add_task_init_call=False),
                      _LocalTask(self.project, run_name))
        else:
            t = _LocalTask(self.project, run_name)
        if self.stack:
            t.set_parent(self.stack[-1].task)
        else:
            t.set_comment(self.description[:4000])
            from eaglewings import __version__
            t.add_tags([f"built-with:eaglewings-{__version__}"])
            self.last_top = t
        t.mark_started(force=True)
        run = _Run(t)
        self.stack.append(run)
        # вкладений запуск це сабфлов у Prefect (якщо скрипт запущено з flows/pipeline.py): підхід, усередині налаштування
        pnode = pflow.node(run_name, "approach" if len(self.stack) == 2 else "config") if nested else contextlib.nullcontext()
        try:
            with pnode:
                yield run
        except BaseException:
            t.flush(wait_for_uploads=True)
            t.mark_failed(status_reason="exception", force=True)
            raise
        else:
            t.flush(wait_for_uploads=True)
            t.mark_completed(force=True)
        finally:
            self.stack.pop()

    @property
    def run(self):
        return self.stack[-1]

    # ---- теги, параметри, метрики
    def set_tags(self, d):
        r = self.run
        r.props.update({k: str(v) for k, v in d.items()})
        r.task.set_user_properties(**{k: str(v)[:1000] for k, v in r.props.items()})
        tags = [f"{k}:{v}" for k, v in d.items() if k in ("approach", "kind", "role", "family", "target")]
        if "champion" in d:
            tags.append(f"champion:{d['champion']}")
        if tags:
            r.task.add_tags(tags)

    def set_tag(self, k, v):
        self.set_tags({k: v})

    def log_params(self, d):
        self.run.task.set_parameters({f"Protocol/{k}": str(v) for k, v in d.items()})

    def log_param(self, k, v):
        self.log_params({k: v})

    def log_metrics(self, d, step=None):
        if step is None:
            pflow.metrics(d)
        lg = self.run.logger
        for k, v in d.items():
            if step is None:
                lg.report_single_value(k, float(v))
            else:
                lg.report_scalar(title=k, series=k, value=float(v), iteration=int(step))

    def log_metric(self, k, v, step=None):
        self.log_metrics({k: v}, step=step)

    # ---- артефакти
    def _upload(self, name, path):
        self.run.task.upload_artifact(name=name, artifact_object=Path(path), wait_on_upload=True)

    def log_text(self, text, artifact_file):
        tmp = Path(tempfile.mkdtemp()) / Path(artifact_file).name
        tmp.write_text(text)
        self._upload(artifact_file, tmp)

    def log_table(self, df, artifact_file):
        df = pd.DataFrame(df)
        self.log_text(df.to_json(orient="split", force_ascii=False, index=False, default_handler=str), artifact_file)
        if len(df) <= 500:
            self.run.logger.report_table(title=Path(artifact_file).stem, series="table", iteration=0, table_plot=df.reset_index(drop=True))

    def log_figure(self, fig, artifact_file):
        name = Path(artifact_file).stem
        self.run.logger.report_matplotlib_figure(title=name, series=name, figure=fig, iteration=0, report_image=True)
        tmp = Path(tempfile.mkdtemp()) / Path(artifact_file).name
        fig.savefig(tmp, dpi=120, bbox_inches="tight")
        self._upload(artifact_file, tmp)
        self.figs.setdefault(self.run.task.name, []).append((name, tmp))

    def log_artifact(self, path, artifact_path=""):
        p = Path(path)
        self._upload(f"{artifact_path}/{p.name}".strip("/"), p)


ml = _ML()


# ---------------------------------------------------------------- проєкти й запуски
def setup():
    print("Запуски:", "ClearML app.clear.ml" if online() else f"локально в {LOCAL_RUNS}")


def experiment(name, description):
    """Проєкт Dadia/<name> з описом задачі (видно у вкладці Overview проєкту)."""
    if not online():
        ml.set_project(name, description)
        return SimpleNamespace(experiment_id="local", name=ml.project)
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    ml.set_project(name, description)
    description = overview(description)
    pid = Task.get_project_id(_cm(ml.project))
    c = APIClient()
    if pid is None:
        pid = c.projects.create(name=_cm(ml.project), description=description[:4000]).id
    else:
        c.projects.update(project=pid, description=description[:4000])
    return SimpleNamespace(experiment_id=pid, name=ml.project)


def overview(doc):
    """Опис проєкту ClearML (вкладка Overview, markdown): рядки докстрінга лишаються рядками, а внизу сказано, де що лежить."""
    lines = [x.strip() for x in (doc or "").strip().splitlines()]
    body = "  \n".join(f"**{x}**" if x and len(x) < 40 and not x.endswith(".") and i == 0 else x for i, x in enumerate(lines))
    return (body + "\n\n---\n**Де дивитися.** Вкладка TASKS: запуск `benchmark …` це пакет задачі, під ним по запуску на кожен підхід, "
            "а під підходом його налаштування. У пакеті: Debug Samples це сторінка-звіт (постановка, лідер, усі підходи з 95 % CI), "
            "Plots це її графіки окремо, Artifacts це results.json. Датасет задачі в Datasets → Dadia/datasets.")[:4000]


def batch_name():
    return "benchmark " + dt.datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- датасети
def _versions(project, name):
    from clearml import Dataset
    return [d for d in Dataset.list_datasets(dataset_project=project, partial_name=name, only_completed=True) if d["name"] == name]


def _num(d):
    v = str(d.get("version") or "0").split(".")[0]
    return int(v) if v.isdigit() else 0


def clearml_dataset(project, name, digest, files, description, tags=(), fill=None, report=None, parents=()):
    """Один ClearML Dataset на назву, версії 1, 2, 3 … Той самий вміст (digest) знаходить наявну версію;
    новий вміст стає наступною версією, нащадком попередньої. Номер версії в ClearML свій, він не залежить від
    локального vN: у репозиторії греції лежить лише поточна версія, а в ClearML лишається вся історія."""
    from clearml import Dataset
    vs = _versions(project, name)
    for d in vs:
        if f"digest:{digest}" in (d.get("tags") or []):
            if report:                               # та сама версія: звіт у Reports і графіки оновлюються (ті самі назви, без дублів)
                from clearml import Task
                report(Task.get_task(task_id=d["id"]).get_logger())
            return d["id"]
    last = max(vs, key=_num, default=None)
    ds = Dataset.create(dataset_project=project, dataset_name=name, dataset_version=str(_num(last) + 1 if last else 1),
                        parent_datasets=([last["id"]] if last else []) + [p for p in parents if p] or None, description=description[:1000],
                        dataset_tags=[f"digest:{digest}", *[t[:120] for t in tags]])
    for p in list(ds.list_files()):                 # нащадок містить лише власний вміст, а не файли батька
        ds.remove_files(p)
    for f in files:
        ds.add_files(str(f))
    if fill:
        fill(ds)
    if report:
        report(ds.get_logger())
    ds.upload()
    ds.finalize()
    print(f"ClearML: {project}/{name} версія {_num(last) + 1 if last else 1}", flush=True)
    return ds.id


def dataset_id(dv):
    """ClearML Dataset для версії датасету в Dadia/datasets (див. clearml_dataset)."""
    if not online():
        return dv.digest
    man = json.loads((dv.path.parent / "manifest.json").read_text())

    def fill(ds):
        lg = ds.get_logger()
        lg.report_table("manifest", "params", iteration=0, table_plot=pd.DataFrame([{"key": k, "value": json.dumps(v, ensure_ascii=False)[:300]} for k, v in man.items()]))
        stats = man.get("label_stats", {})
        if stats:
            lg.report_table("labels", "stats", iteration=0, table_plot=pd.DataFrame([{"label": k, **v} for k, v in stats.items()]))
        ds.set_metadata(dv.df.head(200), metadata_name="preview")

    def report(lg):
        from . import pages
        md = pages.page(f"dataset-{dv.name}", lambda: pages.dataset_md(dv.name))
        if md:
            pages.to_clearml(lg, md, "звіт датасету", f"dataset-{dv.name}", keep=True)
            pages.clearml_report(DATASETS_PROJECT, f"датасет · {dv.name}", md, ["звіт датасету"])
    # злиття парків: батьки — попередня версія злиття (останній датасет тієї ж назви) і датасет кожного парку, з якого його зібрано
    from . import datasets as _ds
    par = [dataset_id(_ds.load(x["name"], x["version"])) for x in man.get("parents", [])]
    desc = man.get("description", "") + ("\nЩо змінилось: " + "; ".join(man["changes"]) if man.get("changes") else "")
    from . import park as _pk
    pk_ = next((k for k in _pk.ALL if man["name"].endswith("_" + k)), None)
    tags = [f"unit:{man.get('unit', '')}", f"local:v{dv.version}"] + ([f"park:{pk_}"] if pk_ else ["merged"] if man.get("parents") else [])
    return clearml_dataset(DATASETS_PROJECT, dv.name, dv.digest, [dv.path, dv.path.parent / "manifest.json"], desc, tags, fill, report, parents=par)


SOURCES_PROJECT = f"{ROOT_PROJECT}/sources"


def source_files(path):
    """Файли сирого набору за шляхом із scripts/data_lineage.py: тека, файл або шаблон."""
    import glob
    from . import park
    p = ROOT / park.local(path)                       # набір парку GR_PARK (межа, дороги, кеш NASA у кожного свої)
    if p.is_dir():
        return sorted(x for x in p.rglob("*") if x.is_file() and ".cache" not in x.parts)
    return sorted(Path(x) for x in glob.glob(str(p)) if Path(x).is_file())


def log_source(key, meta):
    """Сирий набір даних (те, що скачано) → ClearML Dataset Dadia/sources/<key>. Нове завантаження з іншим
    вмістом стає наступною версією тієї самої назви. Лише за --clearml."""
    if not online():
        return None
    import hashlib
    fs = source_files(meta["path"])
    if not fs:
        print(f"ClearML: {key}: файлів за {meta['path']} немає, пропускаю", flush=True)
        return None
    h = hashlib.sha256()
    for f in fs:
        h.update(str(f.relative_to(ROOT)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    desc = f"{meta['name']} ({meta['who']}). {meta['what']} Ліцензія: {meta['license']}. {meta['url']}"
    from . import park
    name, tags = key, [f"license:{meta['license']}"]
    if park.local(meta["path"]) != meta["path"]:       # набір, вирізаний під парк: у кожного парку своя назва й свої версії
        name, tags = (key if park.KEY == park.MAIN else f"{key}_{park.KEY}"), tags + [f"park:{park.KEY}"]
        desc = f"{park.INFO['name']}. {desc}"
    return clearml_dataset(SOURCES_PROJECT, name, h.hexdigest()[:16], fs, desc, tags)   # EDA додає крок eda


def log_dataset(dv, context, target=None, files=False):
    """Зв'язок запуску з версією датасету: параметр Datasets/<context> = id ClearML Dataset + властивість dataset.<context>."""
    did = dataset_id(dv)
    ml.run.task.set_parameters({f"Datasets/{context}": did, f"Datasets/{context}_name": dv.tag, f"Datasets/{context}_digest": dv.digest,
                                **({f"Datasets/{context}_target": target} if target else {})})
    ml.set_tags({f"dataset.{context}": f"{dv.tag} ({dv.digest})"})
    if files:
        ml.log_artifact(str(dv.path.parent / "manifest.json"), f"datasets/{dv.tag}")


def log_approaches(rows, name, label=None, prefix=None):
    """По вкладеному запуску на кожен алгоритм поточного пакета: у Prefect це сабфлов «алгоритм», у ClearML дочірній запуск.
    Для задач, де алгоритми рахуються спільно (рік за роком, зі спільною випадковістю): розрахунок не чіпаємо, бо інакше
    зміняться числа, а результати кожного алгоритму записуємо окремо одразу після нього.
      name(r)    назва алгоритму рядка
      label(r)   якщо рядки алгоритму це налаштування: під алгоритмом по запуску на налаштування
      prefix(r)  якщо рядки це розрізи (val, test, рік): метрики алгоритму з цим префіксом"""
    import math
    d = pd.DataFrame(rows).reset_index(drop=True)
    if d.empty:
        return
    d["_name"] = [name(r) for r in d.itertuples(index=False)]
    num = lambda r: {k: float(v) for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and not math.isinf(v)}
    for nm, g in d.groupby("_name", sort=False):
        with ml.start_run(run_name=str(nm), nested=True):
            ml.set_tags({"approach": nm, **({"champion": True} if "champion" in g and g.champion.fillna(False).astype(bool).any() else {})})
            recs = g.drop(columns="_name").to_dict("records")
            if label:
                for r, rr in zip(g.drop(columns="_name").itertuples(index=False), recs):
                    with ml.start_run(run_name=str(label(r)), nested=True):
                        ml.log_metrics(num(rr))
                        ml.log_params({k: v for k, v in rr.items() if isinstance(v, str)})
            elif prefix:
                ml.log_metrics({f"{prefix(r)}/{k}": v for r, rr in zip(g.drop(columns="_name").itertuples(index=False), recs) for k, v in num(rr).items()})
            else:
                ml.log_metrics({k: v for rr in recs for k, v in num(rr).items()})
            ml.log_table(g.drop(columns="_name"), "rows.json")


RESULTS = ROOT / "data" / "processed" / "results"


def _plain(o):
    """У JSON без NaN і numpy: NaN → None, масиви → списки."""
    import math
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def keep_reset(task):
    (RESULTS / f"{task}.json").unlink(missing_ok=True)


def keep(task, name, obj):
    """Дані графіків, пораховані разом із метриками з тих самих передбачень → data/processed/results/<task>.json
    (сайт лише перекладає їх у свій JSON) і артефакт поточного запуску. Ніхто не перераховує модель заново."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS / f"{task}.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[name] = _plain(obj)
    f.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
    if ml.stack:
        ml.log_text(json.dumps(d[name], ensure_ascii=False), f"results/{name}.json")


def log_fig(fig, name):
    ml.log_figure(fig, f"plots/{name}.png")
    plt.close(fig)


def clean(d):
    """Лише скалярні фінітні метрики."""
    out = {}
    for k, v in d.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v == v and abs(v) != float("inf"):
            out[k] = v
    return out


def log_sklearn(model, input_example, registered_model_name=None):
    """sklearn-модель → ClearML OutputModel (joblib) поточного запуску; назва = registered_model_name."""
    import joblib
    tmp = Path(tempfile.mkdtemp()) / "model.joblib"
    joblib.dump(model, tmp)
    if not online():
        ml.run.task.upload_artifact(f"{registered_model_name or 'model'}.joblib", tmp)
        return None
    from clearml import OutputModel
    om = OutputModel(task=cm_task(ml.run.task), name=registered_model_name or ml.run.task.name, framework="ScikitLearn",
                     tags=[t for t in ml.run.task.get_tags() if t.startswith(("approach:", "role:"))])
    om.update_design(config_dict={"features": list(pd.DataFrame(input_example).columns), "class": type(model).__name__})
    om.update_weights(weights_filename=str(tmp), auto_delete_file=False)
    if registered_model_name:                         # фінальна модель гілки: опублікована = зареєстрована в Model Registry
        om.publish()
    return om


MODELS_PROJECT = f"{ROOT_PROJECT}/models"


def register_card(card, weights=None):
    """Переможець гілки (models/<код>/card.json) → ClearML Model Registry, проєкт Dadia/models, назва = код гілки.
    Модель без навчання (правило, алгоритм) теж реєструється: її параметри лежать у конфігурації моделі."""
    if not online():
        return None
    from clearml import OutputModel, Task
    t = Task.create(project_name=MODELS_PROJECT, task_name=f"register {card['code']}", task_type=Task.TaskTypes.custom, add_task_init_call=False)
    t.mark_started(force=True)
    om = OutputModel(task=t, name=f"Dadia {card['code']}", framework="ScikitLearn" if weights else None, comment=f"{card['task']}: {card['approach']}",
                     tags=[f"branch:{card['code']}", f"approach:{card['approach']}"[:120], "champion"])
    om.update_design(config_dict=json.loads(json.dumps(card, ensure_ascii=False, default=str)))
    src = Path(weights) if weights else Path(tempfile.mkdtemp()) / "card.json"
    if not weights:
        src.write_text(json.dumps(card, ensure_ascii=False, indent=1, default=str))
    om.update_weights(weights_filename=str(src), auto_delete_file=False)
    om.publish()
    t.mark_completed(force=True)
    print(f"ClearML Model Registry: {card['code']} ({card['approach']})", flush=True)
    return om.id


# ---------------------------------------------------------------- читання результатів (сторінка /science/, ноутбук)
def publish_branches(key, board, spec, page_md=None):
    """Гілки задачі → ClearML: проєкт Dadia/<гілка>, у ньому один запуск на алгоритм (найкраще налаштування за метрикою
    вибору), з метриками вибору й тесту, 95 % CI і параметрами. Усі налаштування алгоритму лежать у його запуску таблицею
    й графіком. Попередні запуски гілки архівуються: у проєкті видно лише поточний набір алгоритмів.
    Детальний журнал (пакет → підходи → налаштування) лишається в Dadia/журнал задач/<задача>."""
    import re
    from clearml import Task
    from . import pages
    from .report import BRANCHES, KIND
    mine = {c: b for c, b in BRANCHES.items() if b["key"] == key}
    if not mine or board.empty:
        return
    low = bool(spec.get("low"))
    algo = lambda n: re.split(r",| \{", str(n))[0].strip()
    board = board.assign(branch=[next((c for c, b in mine.items() if b.get("part") == g), next(iter(mine))) for g in board.group],
                         algo=[algo(n) for n in board.name])
    for code, rows in board.groupby("branch", sort=False):
        b = BRANCHES[code]
        proj = f"{ROOT_PROJECT}/{code.replace('_', ' ')} · {b['q'].rstrip('?')}"
        for t in Task.get_tasks(project_name=proj, task_filter={"system_tags": ["-archived"]}):
            t.set_archived(True)
        best = []
        for (g, a), q in rows.groupby(["group", "algo"], sort=False):
            k = "val" if q.val.notna().any() else "test"
            r = q.sort_values(k, ascending=low, na_position="last").iloc[0]
            champ = bool(q.champ.any())
            name = (f"{g} · " if g and not b.get("part") else "") + a + (" ★" if champ else "")
            t = Task.create(project_name=proj, task_name=name, task_type=Task.TaskTypes.training, add_task_init_call=False)
            t.mark_started(force=True)
            t.set_comment(f"{b['q']} {r.hyp}\nОбрано на метриці вибору; тест лише показує результат. Журнал: {JOURNAL}.")
            t.set_parameters({f"Config/{c}": str(v)[:300] for c, v in (r.raw or {}).items() if c not in ("hypothesis", "champion")})
            t.add_tags([f"branch:{code}", f"task:{key}", KIND[b["kind"]], *(["champion"] if champ else [])])
            lg = t.get_logger()
            for m, v in (("вибір", r.val), ("тест", r.test), ("CI від", r.lo), ("CI до", r.hi)):
                if v is not None and v == v:
                    lg.report_single_value(m, float(v))
            for c, v in (r.raw or {}).items():          # усі інші метрики алгоритму: точність, повнота, IoU, події …
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf"):
                    lg.report_single_value(c, float(v))
            mine = [(rn, f) for rn, fs in ml.figs.items() if a in rn for f in fs][:8]      # графіки, які задача намалювала для цього алгоритму
            for rn, (fn, path) in mine:
                lg.report_image(title="графіки алгоритму", series=f"{rn} · {fn}"[:120], iteration=0, local_path=str(path))
            if len(q) > 1:                               # усі налаштування цього алгоритму
                lg.report_table("налаштування", "усі", iteration=0, table_plot=pd.DataFrame([x for x in q.raw]).drop(columns=["hypothesis"], errors="ignore").head(300))
                ok = q[q.val.notna() & q.test.notna()]
                if len(ok):
                    lg.report_scatter2d("вибір проти тесту", "налаштування", iteration=0, scatter=ok[["val", "test"]].values.tolist(),
                                        xaxis=f"вибір: {spec['metric']}", yaxis=f"тест: {spec['metric']}", mode="markers")
            best.append((name, r.test, t, champ))
            t.flush(wait_for_uploads=True)
            t.mark_completed(force=True)
        # звіт гілки в Reports пишеться завжди, навіть якщо переможця немає (як у LM)
        if page_md:
            pages.clearml_report(proj, f"звіт гілки · {code.replace('_', ' ')}", page_md, ["звіт гілки"])
        # графіки гілки: у запуск-переможець, а без переможця в перший алгоритм гілки
        target = next((x for x in best if x[3]), best[0] if best else None)
        if target is not None:
            t = target[2]
            lg = t.get_logger()
            xs = [x for x in best if x[1] is not None and x[1] == x[1]]
            lg.report_histogram("алгоритми гілки", spec["metric"], values=[float(x[1]) for x in xs], iteration=0, xlabels=[x[0] for x in xs],
                                xaxis="алгоритм", yaxis=f"тест: {spec['metric']}")
            if page_md:
                pages.to_clearml(lg, page_md, "звіт задачі", f"task-{key}")
            for dk in [k for k in pages._FIGS if k.startswith("dataset-")]:   # датасет задачі: графіки
                if pages._PAGES.get(dk):
                    pages.to_clearml(lg, pages._PAGES[dk], f"датасет {dk[8:]}", dk)
            for fn, path in ml.figs.get(ml.last_top.name if ml.last_top is not None else "", [])[:8]:   # графіки всього пакета
                lg.report_image(title="графіки задачі", series=fn, iteration=0, local_path=str(path))
            t.flush(wait_for_uploads=True)
        print(f"ClearML: гілка {proj}: {len(best)} алгоритмів", flush=True)


def created_since(started):
    """Що прогін записав у ClearML після started: пакети задач, датасети, сирі набори, моделі, з посиланнями. Без --clearml None."""
    if not online():
        return None
    from clearml import Dataset, Task
    web = lambda t: t.get_output_log_web_page().replace("/output/log", "")
    after = lambda d: d is not None and (d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)) >= started
    out = {"Задачі": [], "Датасети": [], "Сирі набори": [], "Model Registry": []}
    out = {"Гілки: алгоритми": [], **out}
    for t in Task.get_tasks(project_name=ROOT_PROJECT, task_filter={"order_by": ["-last_update"]}):
        if t.parent or not after(t.data.created):
            continue
        pn = t.get_project_name()
        if t.name.startswith("benchmark"):
            out["Задачі"].append((f"журнал {pn.split('/')[-1]}: {t.name}", web(t)))
        elif t.name.startswith("register"):
            out["Model Registry"].append((t.name.replace("register ", ""), web(t)))
        elif pn.count("/") == 1 and not pn.endswith(("datasets", "sources", "models")) and "журнал" not in pn:
            out["Гілки: алгоритми"].append((f"{pn.split('/')[-1]}: {t.name}", web(t)))
    for d in Dataset.list_datasets(dataset_project=ROOT_PROJECT, only_completed=True):
        if after(d.get("created")):
            k = "Сирі набори" if d["project"].startswith(SOURCES_PROJECT) else "Датасети"
            out[k].append((f"{d['name']} версія {d.get('version')}", web(Task.get_task(task_id=d["id"]))))
    if out["Model Registry"]:
        out["Model Registry"].append(("усі моделі", "https://app.clear.ml/models"))
    return {k: sorted(v) for k, v in out.items()}


def summary_page(md):
    """Зміст прогону → розділ Reports у ClearML, проєкт Dadia (лише з --clearml)."""
    if not online():
        return
    from . import pages
    pages.clearml_report(ROOT_PROJECT, "зміст прогону " + dt.datetime.now().strftime("%Y-%m-%d %H:%M"), md, ["зміст прогону"])


def latest(exp_name):
    """Останній завершений пакет «benchmark …» проєкту задачі."""
    from clearml import Task
    ts = Task.get_tasks(project_name=_cm(_project(exp_name)), task_name="^benchmark", allow_archived=False,
                        task_filter={"status": ["completed"], "order_by": ["-last_update"], "parent": None})
    ts = [t for t in ts if not t.parent]
    if not ts:
        raise LookupError(f"немає завершених пакетів у {_cm(_project(exp_name))}")
    return max(ts, key=lambda t: t.data.last_update)


def children(task, role=None):
    from clearml import Task
    ts = Task.get_tasks(project_name=task.get_project_name(), task_filter={"parent": task.id})
    return [t for t in ts if role is None or t.get_user_properties(value_only=True).get("role") == role]


def artifact(task, name):
    return task.artifacts[name].get_local_copy()


def results(task):
    j = json.load(open(artifact(task, "results.json")))
    return pd.DataFrame(j["data"], columns=j["columns"])


def props(task):
    return task.get_user_properties(value_only=True)


def web_url():
    from clearml import Task
    try:
        return Task._get_app_server().rstrip("/")
    except Exception:
        return None


def count_runs():
    from clearml import Task
    return len(Task.get_tasks(project_name=ROOT_PROJECT, task_filter={"search_hidden": False}, allow_archived=False))
