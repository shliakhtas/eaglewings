"""Сторінки-звіти для Prefect і ClearML: у кожному місці потоку одна сторінка з текстом, таблицями й графіками.

    source(key)          сирий набір, щойно скачаний: звідки, ліцензія, що видно, розподіли, мапи (функції сторінки «Набори даних»)
    dataset(name)        датасет, щойно зібраний: опис, колонки, лейбли, зв'язок ознак, розподіли (функції вкладки «Датасети»)
    task(key, mod, res)  задача: постановка, як обирали лідера, лідер, усі підходи з CI
    summary(S)           зведення всіх задач

Текст і числа беруться тими самими функціями, що будують сторінку /science/ (scripts/10, 11, 12), тож звіт і сторінка
не розходяться. Без Prefect (ручний запуск, ноутбук) нічого не робиться. Звіт ніколи не валить розрахунок: помилка
побудови звіту лише пишеться в журнал.
"""
import base64
import importlib.util
import io
import json
import re
import sys
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import pflow
from .common import ROOT, LIB

SCRIPTS = LIB / "scripts"
WEB_SRC = ROOT / "data" / "processed" / "eda" / "maps"      # картинки мап кроку eda


# ---------------------------------------------------------------- дрібниці
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 80})
    return plt


_COLLECT = None        # список, куди page() збирає графіки сторінки для вкладки Plots у ClearML


def img(fig, alt="графік"):
    plt = _plt()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=80, bbox_inches="tight")
    if _COLLECT is not None:
        name = alt if alt != "графік" else (fig._suptitle.get_text() if fig._suptitle else "") or next((a.get_title() for a in fig.axes if a.get_title()), "графік")
        _COLLECT.append((f"{len(_COLLECT) + 1:02d} {name}"[:100], fig))
    else:
        plt.close(fig)
    return f"![{alt}](data:image/png;base64,{base64.b64encode(b.getvalue()).decode()})"


def cell(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "так" if v else ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.3g}" if abs(v) < 1e4 else f"{v:,.0f}".replace(",", " ")
    return str(v).replace("|", "/").replace("\n", " ")


def table(head, rows, limit=40):
    rows = list(rows)
    L = ["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)]
    L += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows[:limit]]
    if len(rows) > limit:
        L.append(f"\n_показано {limit} із {len(rows)} рядків_")
    return "\n".join(L)


@lru_cache
def _script(name):
    """scripts/<name>.py як модуль: ті самі функції, що будують сторінку."""
    spec = importlib.util.spec_from_file_location("_page_" + re.sub(r"\W", "_", name), SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(m)
    return m


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print(f"звіт: {getattr(fn, '__name__', fn)}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None


def _post(md, key, description):
    if pflow.on():
        pflow.artifact(md, key=key, description=description)


# ---------------------------------------------------------------- графіки з описів сторінки
def _hists(hs, title="Розподіли"):
    hs = [h for h in (hs or []) if h]
    if not hs:
        return ""
    plt = _plt()
    n = len(hs)
    c = min(n, 3)
    r = (n + c - 1) // c
    fig, axs = plt.subplots(r, c, figsize=(4.2 * c, 2.6 * r), squeeze=False)
    for ax, h in zip(axs.flat, hs):
        e = np.asarray(h["edges"], float)
        x = np.arange(len(e) - 1)
        if "pos" in h:
            neg, pos = np.asarray(h["neg"], float), np.asarray(h["pos"], float)
            ax.bar(x, neg / max(neg.sum(), 1), color="#8aa4c8", label="лейбл 0", width=.9)
            ax.bar(x, pos / max(pos.sum(), 1), color="#e8743b", alpha=.75, label="лейбл 1", width=.9)
            ax.set_ylabel("частка")
        else:
            ax.bar(x, h["all"], color="#5b8bd0", width=.9)
        ax.set_title(f"{h['n']}" + (f": {h['doc']}" if h.get("doc") else "") + (" (log)" if h.get("log") else ""), fontsize=8.5)
        ax.set_xticks([0, len(x) // 2, len(x) - 1])
        back = np.expm1 if h.get("log") and "med" not in h else (lambda v: v)     # у наборах краї в log1p, у датасетах уже ні
        ax.set_xticklabels([cell(float(back(v))) for v in (e[0], e[len(x) // 2], e[-1])])
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    if any("pos" in h for h in hs):
        axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    return f"### {title}\n\n{img(fig)}"


def _time(t, note=""):
    if not t or not t.get("rows"):
        return ""
    plt = _plt()
    rows = t["rows"]
    k = [r[0] for r in rows]
    a = np.array([r[1] for r in rows], float)
    b = np.array([r[2] for r in rows], float)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.bar(range(len(k)), a, color="#5b8bd0", label="усього")
    if b.sum() > 0:
        ax.bar(range(len(k)), b, color="#e8743b", label="лейбл 1")
        ax.legend(fontsize=7)
    step = max(1, len(k) // 14)
    ax.set_xticks(range(0, len(k), step))
    ax.set_xticklabels(k[::step], rotation=45, ha="right", fontsize=7.5)
    ax.set_title(note or f"за часом ({t.get('unit', '')})")
    return f"### Час\n\n{img(fig)}"


def _cats(cs):
    if not cs:
        return ""
    return "\n\n".join(f"**{c['n']}**{': ' + c['doc'] if c.get('doc') else ''}\n\n" + table(["значення", "рядків"], c["rows"], 14) for c in cs)


def _geo(g):
    if not g or not g.get("pts"):
        return ""
    plt = _plt()
    p = pd.DataFrame(g["pts"]).iloc[:, :3]
    p.columns = ["lat", "lon", "c"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (k, q) in enumerate(p.groupby("c")):
        ax.scatter(q.lon, q.lat, s=5, alpha=.6, label=str(k), color=plt.cm.tab10(i % 10))
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title(g.get("label", ""))
    ax.set_aspect(1 / np.cos(np.radians(p.lat.mean())))
    return f"### Де\n\n{img(fig)}"


def _map(m):
    """Одна мапа з 12_source_maps.py: растр, прямокутники, точки, контури, лінії."""
    plt = _plt()
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    r = m.get("raster")
    if r:
        f = WEB_SRC / Path(r["img"]).name
        if f.exists():
            (s, w), (n, e) = r["bounds"]
            ax.imshow(plt.imread(f), extent=[w, e, s, n], interpolation="nearest" if r.get("pixel") else "bilinear")
    for x in m.get("rects") or []:
        (s, w), (n, e) = x["b"]
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, lw=1.5, ec=x.get("color", "#333")))
        ax.text(w, n, x.get("label", ""), fontsize=7, va="bottom", color=x.get("color", "#333"))
    years = sorted({str(p[0]) for p in m.get("polys") or []})
    for p in m.get("polys") or []:
        xy = np.asarray(p[1], float)
        ax.plot(xy[:, 1], xy[:, 0], lw=.8, color=plt.cm.viridis(years.index(str(p[0])) / max(len(years) - 1, 1)))
    for k, xy in m.get("lines") or []:
        xy = np.asarray(xy, float)
        ax.plot(xy[:, 1], xy[:, 0], lw=.5 if k else .9, color="#d9a400" if k else "#3a7bd5")
    for p in m.get("pts") or []:
        ax.scatter(p[1], p[0], s=10, color=p[3] if len(p) > 3 else "#e8743b")
    fit = m.get("fit")
    if fit:
        (s, w), (n, e) = fit
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ax.get_ylim()))))
    ax.set_title(m.get("name", ""))
    sc = m.get("scale")
    note = m.get("note", "") + (f" Шкала: від {sc['lo']} до {sc['hi']}." if sc else "")
    return f"**{m.get('name', '')}.** {note}\n\n{img(fig)}"


# ---------------------------------------------------------------- 1. сирий набір
_PAGES = {}            # побудовані сторінки цього процесу: та сама сторінка йде і в Prefect, і в ClearML


_FIGS = {}


def page(key, build):
    global _COLLECT
    if key not in _PAGES:
        from . import tracking
        _COLLECT = [] if tracking.online() else None          # графіки тримаємо лише якщо їх треба віддати в ClearML
        try:
            _PAGES[key] = build()
        except Exception:
            print("звіт", key, "не побудовано:\n", traceback.format_exc()[-800:], flush=True)
            _PAGES[key] = None
        finally:
            if _COLLECT is not None:
                _FIGS[key] = _COLLECT
            _COLLECT = None
        if _PAGES[key]:
            save(key, _PAGES[key])
    return _PAGES[key]


REPORTS = ROOT / "artifacts" / "reports"


def save(key, md):
    """Та сама сторінка файлом artifacts/reports/<key>.html: її видно й без Prefect і ClearML."""
    try:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"{key.replace('_', '-')}.html").write_text(html(md, key))
    except Exception as e:
        print(f"звіт {key} не збережено файлом: {e}", flush=True)


def source(key):
    """EDA набору: результат у data/processed/eda/sources/<key>.json і .maps.json (їх лише перекладає сайт), сторінка
    в Prefect і файлом, з --clearml ще звіт у Reports."""
    _PAGES.pop(f"source-{key}", None)
    md = page(f"source-{key}", lambda: source_md(key))
    if md and pflow.on():
        _post(md, f"source-{key}", f"Набір даних: {key}")
    from . import tracking
    if md and tracking.online():
        _safe(clearml_report, tracking.SOURCES_PROJECT, f"набір · {key}", md, ["звіт набору"])
        vs = _safe(tracking._versions, tracking.SOURCES_PROJECT, key) or []            # графіки — у Plots останньої версії набору
        if vs:
            from clearml import Task
            _safe(to_clearml, Task.get_task(task_id=max(vs, key=tracking._num)["id"]).get_logger(), md, "EDA набору", f"source-{key}")


def source_md(key):
    """Сторінка про скачаний набір; функції scripts/11_source_eda.py і 12_source_maps.py."""
    if True:
        sys.path.insert(0, str(SCRIPTS))
        from data_lineage import LINEAGE, SRC
        m = SRC[key]
        E, M = _script("11_source_eda"), _script("12_source_maps")
        o = _safe(E.describe, key) or {}
        maps = _safe(M.MAPS[key]) if key in M.MAPS else None
        d = EDA_DIR / "sources"
        d.mkdir(parents=True, exist_ok=True)
        if o:
            (d / f"{key}.json").write_text(json.dumps(o, ensure_ascii=False, default=float, separators=(",", ":")))
        if key in M.MAPS:
            (d / f"{key}.maps.json").write_text(json.dumps(maps, ensure_ascii=False, default=float, separators=(",", ":")))
        maps = maps or []
        L = [f"# {m['name']}", "", f"_{m['who']}_. {m['what']}", "",
             table(["", ""], [["роздільність", m["res"]], ["покриття", m["cover"]], ["період", m["period"]], ["ліцензія", m["license"]],
                              ["джерело", m["url"] or "—"], ["як скачується", m["script"]], ["на диску", f"{E.du(m['path'])}, файлів {E.nfiles(m['path'])}"],
                              ["скільки", f"{cell(o.get('rows'))}"],
                              ["де використано", ", ".join(f"`{n}`" for n, v in LINEAGE.items() if key in v["src"] or key in v.get("also", [])) or "—"]]), ""]
        if o.get("short"):
            L += ["## Коротко", "", o["short"], ""]
        if o.get("seen"):
            L += ["## Що видно в даних", ""] + [f"- {s}" for s in o["seen"]] + [""]
        if o.get("table"):
            L += [f"## {o['table']['title']}", "", table(o["table"]["head"], o["table"]["rows"]), ""]
        L += [_hists(o.get("hist")), _time(o.get("time"), o.get("time_note", "")), _cats(o.get("cats")), _geo(o.get("geo"))]
        if maps:
            L += ["## Мапи", ""] + [_safe(_map, x) or "" for x in maps]
        if not o:
            L += ["", "_Опис цього набору будується з датасету, який збирає задача; він з'явиться після неї на сторінці «Набори даних»._"]
        return "\n\n".join(x for x in L if x is not None)


def downloaded(*keys):
    """Набір скачано: лише з --clearml нова версія в ClearML Dadia/sources. EDA і сторінки робить окремий крок eda."""
    from . import tracking
    sys.path.insert(0, str(SCRIPTS))
    from data_lineage import SRC
    for k in keys:
        try:
            tracking.log_source(k, SRC[k])
        except SystemExit:
            raise
        except Exception as e:
            print(f"ClearML: набір {k} не записано: {type(e).__name__}: {str(e)[:300]}", flush=True)
            pflow.failed()


# ---------------------------------------------------------------- 2. датасет
EDA_DIR = ROOT / "data" / "processed" / "eda"


def dataset(name):
    """EDA датасету: результат у data/processed/eda/datasets/<name>.json (його лише перекладає сайт), сторінка в Prefect
    і файлом, з --clearml ще звіт у Reports. Датасет міг змінитися, тож усе будуємо наново."""
    _PAGES.pop(f"dataset-{name}", None)
    md = page(f"dataset-{name}", lambda: dataset_md(name))
    if md and pflow.on():
        _post(md, f"dataset-{name.replace('_', '-')}", f"Датасет {name}")


def dataset_md(name):
    """Сторінка про датасет; функція build зі scripts/10_dataset_eda.py (вкладка «Датасети»)."""
    if True:
        E = _script("10_dataset_eda")
        if name not in E.CFG:
            return None
        o = E.build(name)
        (EDA_DIR / "datasets").mkdir(parents=True, exist_ok=True)
        (EDA_DIR / "datasets" / f"{name}.json").write_text(json.dumps(o, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, "item") else str(v), separators=(",", ":")))
        L = [f"# {o['code']} · {o['task']}: датасет `{name}`", "", o.get("description") or "", "",
             table(["", ""], [["версія", f"v{o['version']} (`{o['digest']}`)"], ["рядків × колонок", f"{o['rows']:,} × {o['cols']}".replace(",", " ")],
                              ["одиниця", o.get("unit")], ["дублікатів", o["dup"]], ["пропусків (середня частка)", o["miss"]],
                              ["навчальна частина для зв'язку ознак", o.get("tnote") or "—"],
                              ["з наборів", ", ".join(s["name"] for s in o["src"] + o["also"])], ["з датасетів", ", ".join(o["from_ds"]) or "—"]]), ""]
        if o.get("steps"):
            L += ["## Як зібрано", ""] + [f"{i + 1}. {s}" for i, s in enumerate(o["steps"])] + [""]
        if o.get("short"):
            L += ["## Коротко", "", o["short"], ""]
        if o.get("seen"):
            L += ["## Що видно в даних", ""] + [f"- {s}" for s in o["seen"]] + [""]
        if o.get("targets"):
            L += ["## Лейбли", "", table(["лейбл", "що це", "позитивних", "частка"], [[t["n"], t["doc"], t["pos"], f"{t['prev']:.2%}"] for t in o["targets"]]), ""]
        if o.get("assoc"):
            L += ["## Яка ознака окремо відділяє лейбл", "", "AUC однієї ознаки на навчальній частині; 0,5 це нічого, далі від 0,5 це сильніше. Тест тут не використовується.", "",
                  table(["ознака", "що це", "AUC", "медіана при 1", "медіана при 0"], [[a["n"], a["doc"], a["auc"], a["pos"], a["neg"]] for a in o["assoc"]], 16), ""]
        if o.get("bygroup"):
            g = o["bygroup"]
            L += [f"## За групою `{g['n']}`", "", table([g["n"], "рядків", "лейбл 1"], g["rows"], 30), ""]
        L += [_hists(o.get("hist"), "Розподіли ознак" + (" за лейблом" if o.get("target") else "")), _time(o.get("time"))]
        if o.get("corr"):
            plt = _plt()
            c = o["corr"]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(np.asarray(c["m"], float), cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(c["names"])))
            ax.set_xticklabels(c["names"], rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(c["names"])))
            ax.set_yticklabels(c["names"], fontsize=7)
            fig.colorbar(im, ax=ax, shrink=.8)
            ax.set_title("Кореляція Спірмена між ознаками")
            L += ["### Кореляції", "", img(fig)]
        L += ["## Колонки", "", table(["колонка", "роль", "що це", "тип", "пропуски", "мін", "медіана", "макс"],
                                     [[c["n"], c["role"], c["doc"], c["t"], c["miss"], c.get("min"), c.get("med"), c.get("max")] for c in o["columns"]], 60)]
        if o.get("cats"):
            L += ["## Категорії", "", _cats(o["cats"])]
        return "\n\n".join(x for x in L if x)


# ---------------------------------------------------------------- 3. задача: лідери
# Що показуємо для кожної задачі: колонка підходу, метрика вибору (валідація чи калібрування), метрика тесту, CI.
from . import branches as _BR
LEAD = _BR.lead()                                     # правила лідера з реєстру гілок ds/branches.py


def _ci(r, ci):
    if not ci:
        return None, None
    if isinstance(ci, tuple):
        return r.get(ci[0]), r.get(ci[1])
    v = r.get(ci)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return v
    if isinstance(v, str):
        x = re.findall(r"-?\d+(?:\.\d+)?", v)
        if len(x) == 2:
            return float(x[0]), float(x[1])
    return None, None


def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def board(key, res):
    """Таблиця лідерів задачі: рядки (група, підхід, вибір, тест, CI, лідер), відсортовані за метрикою вибору."""
    s = LEAD[key]
    d = pd.DataFrame(res)
    d = s["rows"](d) if "rows" in s else d
    out = []
    for _, r in d.iterrows():
        r = r.to_dict()
        lo, hi = _ci(r, s.get("ci"))
        g = s.get("group")
        g = " · ".join(str(r.get(x, "")) for x in g) if isinstance(g, tuple) else str(r.get(g, "")) if g else ""
        out.append(dict(group=g, name=str(r.get(s["name"])), val=_num(r.get(s["val"])) if s.get("val") else None,
                        test=_num(r.get(s["test"])), lo=_num(lo), hi=_num(hi), champ=bool(r.get("champion") is True or r.get("champion") == 1 or str(r.get("champion")).lower() == "true"),
                        hyp=str(r.get("hypothesis") or "")[:160], raw=r))
    b = pd.DataFrame(out)
    if b.empty:
        return b
    k = "val" if b.val.notna().any() else "test"
    b["_k"] = b[k].fillna(np.inf if s.get("low") else -np.inf)
    return b.sort_values(["group", "_k"], ascending=[True, bool(s.get("low"))]).drop(columns="_k").reset_index(drop=True)


def _board_chart(b, s, title):
    plt = _plt()
    gs = list(dict.fromkeys(b.group))
    fig, axs = plt.subplots(1, len(gs), figsize=(5.2 * len(gs), .32 * max(b.groupby("group").size()) + 1.4), squeeze=False)
    for ax, g in zip(axs[0], gs):
        q = b[b.group == g].head(14).iloc[::-1]
        y = np.arange(len(q))
        col = ["#e8743b" if c else "#8aa4c8" for c in q.champ]
        ax.barh(y, q.test.fillna(0), color=col)
        if q.lo.notna().any():
            ax.errorbar(q.test, y, xerr=[(q.test - q.lo).clip(lower=0).fillna(0), (q.hi - q.test).clip(lower=0).fillna(0)], fmt="none", ecolor="#333", lw=1, capsize=2)
        if q.val.notna().any():
            ax.scatter(q.val, y, marker="|", s=120, color="#2a6", label="метрика вибору", zorder=3)
            ax.legend(fontsize=7, loc="lower right")
        ax.set_yticks(y)
        ax.set_yticklabels([n[:32] for n in q.name], fontsize=7.5)
        ax.set_xlabel(s["metric"])
        ax.set_title(g or title, fontsize=9)
    fig.tight_layout()
    return img(fig, "лідери")


def task(key, mod, res):
    """Сторінка задачі → Prefect і, з --clearml, у пакет «benchmark» цієї задачі в ClearML."""
    base = key.split("_")[0]                           # t19_ld: своя сторінка, правила лідерів задачі t19
    md = page(f"task-{key}", lambda: task_md(base, mod, res))
    if not md:
        return
    if pflow.on():
        _post(md, f"task-{key}", f"{mod.EXPERIMENT}: лідери й усі підходи")
    from . import tracking
    if tracking.online() and tracking.ml.last_top is not None:
        _safe(to_clearml, tracking.ml.last_top.get_logger(), md, "звіт задачі", f"task-{key}", True)
    if tracking.online() and base in LEAD:
        _safe(tracking.publish_branches, base, board(base, res), LEAD[base], md)
    if tracking.online():
        _safe(clearml_report, tracking._cm(tracking.ml.project), f"звіт · {mod.EXPERIMENT}", md, ["звіт задачі"])


def task_md(key, mod, res):
    """Сторінка задачі: постановка (докстрінг модуля, той самий текст, що в описі проєкту ClearML), лідер, усі підходи."""
    if True:
        doc = (mod.__doc__ or "").strip()
        L = [f"# {mod.EXPERIMENT}", ""]
        if doc:
            doc = doc[:2500] + ("…" if len(doc) > 2500 else "")
            L += ["## Постановка", "", "  \n".join(x.strip() for x in doc.splitlines()), ""]      # рядки докстрінга лишаються рядками
        if key not in LEAD:
            L += ["## Результати", "", "Метрик тут немає: це демонстрація без розмітки.", "", table(list(pd.DataFrame(res).columns), pd.DataFrame(res).values.tolist())]
            return "\n".join(L)
        s = LEAD[key]
        b = board(key, res)
        L += ["## Як обрано лідера", "", s["how"] + ". Тест на вибір не впливає; негативні результати лишаються в таблиці.", ""]
        ch = b[b.champ]
        if len(ch):
            L += ["## Лідер", "", table(["група", "підхід", "вибір", f"тест: {s['metric']}", "95 % CI"],
                                       [[r.group or "—", f"**{r.name}**", r.val, r.test, f"{cell(r.lo)}–{cell(r.hi)}" if r.lo is not None and np.isfinite(r.lo) else "—"] for r in ch.itertuples()]), ""]
        L += ["## Усі підходи", "", "Помаранчевий це лідер; риска це метрика вибору; вуса це 95 % довірчий інтервал на тесті.", "", _board_chart(b, s, mod.EXPERIMENT), "",
              table(["група", "підхід", "вибір", "тест", "CI від", "CI до", "лідер", "гіпотеза"],
                    [[r.group or "—", r.name, r.val, r.test, r.lo, r.hi, r.champ, r.hyp] for r in b.itertuples()], 60)]
        d = pd.DataFrame(res)                         # усі метрики кожного алгоритму, не лише головна
        d = s["rows"](d) if "rows" in s else d
        num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]) and d[c].notna().any()]
        keys = [c for c in [*(s.get("group") if isinstance(s.get("group"), tuple) else (s.get("group"),)), s["name"]] if c and c in d]
        if num:
            L += ["", "## Усі метрики алгоритмів", "", "Кожен рядок це алгоритм або його налаштування, кожна колонка окрема метрика.", "",
                  table(keys + num, d[keys + num].values.tolist(), 80)]
        from . import tracking                        # графіки, які задача намалювала для алгоритмів і для всього пакета
        shown = []
        for rn, fs in tracking.ml.figs.items():
            for fn, path in fs[:1] if rn != getattr(tracking.ml.last_top, "name", None) else fs:
                if len(shown) < 14 and Path(path).exists():
                    shown.append(f"**{rn} · {fn}**\n\n![{fn}](data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()})")
        if shown:
            L += ["", "## Графіки задачі", ""] + shown
        return "\n".join(L)


# ---------------------------------------------------------------- ClearML: та сама сторінка як HTML
CSS = """body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1d2330;background:#fff}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #d5dae3;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f1f4f9}img{max-width:100%}code{background:#f1f4f9;padding:0 3px}h1{font-size:24px}h2{margin-top:28px}"""


def html(md, title=""):
    import markdown
    return f"<!doctype html><meta charset=utf-8><title>{title}</title><style>{CSS}</style>" + markdown.markdown(md, extensions=["tables"])


def to_clearml(logger, md, title, key=None, keep=False):
    """Графіки сторінки → вкладка Plots запуску чи датасету ClearML. Саму сторінку пише clearml_report у розділ Reports.
    keep: графіки лишаються для наступного місця (переможець гілки отримує графіки задачі й датасету)."""
    plt = _plt()
    figs = (_FIGS.get(key, []) if keep else _FIGS.pop(key, [])) if key else []
    for name, fig in figs:
        try:
            logger.report_matplotlib_figure(title=name, series=title, figure=fig, iteration=0, report_image=False)
        finally:
            if not keep:
                plt.close(fig)
    logger.flush()


def clearml_report(project, name, md, tags=()):
    """Сторінка → розділ Reports у ClearML (markdown із таблицями й графіками). Звіт із тією самою назвою в проєкті
    оновлюється, а не дублюється. Повертає посилання."""
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    from . import tracking
    if not tracking.online() or not md:
        return None
    c = APIClient()
    pid = Task.get_project_id(project)
    if pid is None:
        pid = c.projects.create(name=project).id
    send = lambda ep, **kw: c.session.send_request("reports", ep, json=kw).json()
    have = send("get_all_ex", project=[pid], name=f"^{re.escape(name)}$", only_fields=["id"]).get("data", {}).get("tasks", [])
    rid = have[0]["id"] if have else send("create", name=name, project=pid, tags=list(tags))["data"]["id"]
    out = send("update", task=rid, report=md)
    if out.get("meta", {}).get("result_code") != 200:
        print(f"ClearML: звіт {name} не записано: {out.get('meta', {}).get('result_msg')}", flush=True)
        return None
    url = f"{tracking.web_url() or 'https://app.clear.ml'}/reports/{pid}/{rid}"
    print(f"ClearML: звіт «{name}» → {url}", flush=True)
    return url


# ---------------------------------------------------------------- 4. зведення
def summary(S, keys=None):
    """Зведення всіх задач однією сторінкою (markdown); S це data/processed/experiments_summary.json."""
    rows = []
    for k, v in S.items():
        if keys and k not in keys or k not in LEAD:
            continue
        b = board(k, v["results"])
        for r in b[b.champ].itertuples():
            rows.append(dict(code=LEAD[k]["code"], key=k, exp=v["experiment"], group=r.group, name=r.name, metric=LEAD[k]["metric"], val=r.val, test=r.test, lo=r.lo, hi=r.hi, n=len(b)))
    if not rows:
        return None
    d = pd.DataFrame(rows)
    lab = lambda a, g, n: f"{g if '·' in a else (a + ' ' + g).strip()} · {n}"
    L = ["# Зведення задач", "", "Лідер кожної задачі, обраний на валідації або калібруванні; тест лише показує результат. "
         "XM і CG рахуються на симульованому потоці, кандидати на рубки лишаються кандидатами.", ""]
    q = d[d.test.between(0, 1)].iloc[::-1]
    if len(q):
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, .34 * len(q) + 1.2))
        y = np.arange(len(q))
        ax.barh(y, q.test, color="#e8743b")
        ax.errorbar(q.test, y, xerr=[(q.test - q.lo).clip(lower=0).fillna(0), (q.hi - q.test).clip(lower=0).fillna(0)], fmt="none", ecolor="#333", capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels([lab(a, g, n)[:46] for a, g, n in zip(q.code, q.group, q.name)], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("метрика тесту (своя для кожної задачі, див. таблицю)")
        L += [img(fig, "зведення"), ""]
    L += [table(["гілка", "лідер", "метрика", "вибір", "тест", "95 % CI", "підходів"],
                [[lab(r.code, r.group, "").rstrip(" ·"), f"**{r.name}**", r.metric, r.val, r.test, f"{cell(r.lo)}–{cell(r.hi)}" if r.lo is not None and np.isfinite(r.lo) else "—", r.n] for r in d.itertuples()], 80),
          "", "## Задачі", ""] + [f"- **{c}**: {e}" for c, e in dict.fromkeys(zip(d.code, d.exp))]
    return "\n".join(L)


# ---------------------------------------------------------------- 5. зміст прогону
def _ui():
    from prefect.settings import PREFECT_API_URL, PREFECT_UI_URL
    return (PREFECT_UI_URL.value() or (PREFECT_API_URL.value() or "http://127.0.0.1:4200/api").removesuffix("/api")).rstrip("/")


def _mins(a, b):
    m = (b - a).total_seconds() / 60
    return f"{m:.0f} хв" if m >= 1 else f"{(b - a).total_seconds():.0f} с"


def run_summary(name, run_id, started, journal, leaders=None, clearml=None, version_url=None):
    """Сторінка-зміст прогону: кроки по порядку зі станом і тривалістю, під кожним посилання на його сторінки-звіти,
    далі лідери задач і що записано в ClearML. Сторінку кроку визначаємо за часом: вона створена, поки крок ішов."""
    import datetime as dt
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.sorting import ArtifactSort
    ui = _ui()
    arts = {}
    for a in get_client(sync_client=True).read_artifacts(sort=ArtifactSort.CREATED_DESC, limit=200):
        if a.key and a.key != "summary" and a.created >= started and a.key not in arts:
            arts[a.key] = a
    end = dt.datetime.now(dt.timezone.utc)
    mark = {"зроблено": "✓", "впало": "✗", "пропущено": "–"}
    L = [f"# Зміст прогону {name}", "",
         table(["", ""], [["прогін", f"[{name}]({ui}/runs/flow-run/{run_id})"], ["почато", started.astimezone().strftime("%Y-%m-%d %H:%M")],
                          ["тривалість", _mins(started, end)],
                          ["кроків", f"{sum(j['state'] == 'зроблено' for j in journal)} зроблено, {sum(j['state'] == 'впало' for j in journal)} впало, {sum(j['state'] == 'пропущено' for j in journal)} пропущено"],
                          ["ClearML", "записано, див. нижче" if clearml else "не писали (без --clearml)"],
                          ["версія прогону", f"[що змінилось і журнал версій]({version_url})" if version_url else "—"]]), "",
         "## Покроково", "", "Кроки в тому порядку, у якому вони йшли. Під кожним його сторінки-звіти.", ""]
    step = None
    for i, j in enumerate(journal, 1):
        if j["step"] != step:
            step = j["step"]
            L += ["", f"### {step}", ""]
        why = f", бо впало: {j['why']}" if j.get("why") else ""
        L.append(f"{i}. **{j['name']}** {mark.get(j['state'], '')} {j['state']}{why}, {_mins(j['t0'], j['t1'])}")
        mine = sorted((a for a in arts.values() if j["t0"] <= a.created <= j["t1"] + dt.timedelta(seconds=5)), key=lambda a: a.created)
        L += [f"    - [{a.description or a.key}]({ui}/artifacts/artifact/{a.id})" for a in mine]
    if leaders:
        L += ["", "## Лідери задач цього прогону", ""] + [("#" + x if x.startswith("#") else x) for x in leaders.splitlines()[1:]]
    if clearml:
        L += ["", "## Що записано в ClearML", ""]
        for title, rows in clearml.items():
            if rows:
                L += [f"**{title}**", ""] + [f"- [{t}]({u})" for t, u in rows] + [""]
    return "\n".join(L)
