"""Документи з коду: схема архітектури й таблиця гілок → artifacts/. Запуск із теки eaglewings:

    eaglewings script build_docs                 # оновити згенеровані частини документів
    eaglewings script build_docs --check         # що змінилось у коді з часу, коли Claude востаннє переглядав прозу
    eaglewings script build_docs --mark-reviewed # проза переглянута й відповідає коду

Код генерує те, що можна взяти з коду й результатів, без рук:
- artifacts/architecture.html — розділи між мітками <!-- auto:<назва> --> … <!-- /auto:<назва> -->: кроки потоку
  (flows/pipeline.py), гілки (ds/branches.py), датасети (data/datasets/), таблиці асистента (assistant/agent.py);
- artifacts/branches.html — таблиця гілок: питання, метрика, кожен алгоритм з валідацією, тестом, CI й переможцем
  (data/processed/research_results.csv).
Пояснення поза мітками пише Claude. Щоб вони не відстали від коду, скрипт пам'ятає відбиток джерел прози
(artifacts/.docs_state.json): --check показує, що змінилось, і тоді Claude переписує прозу й робить --mark-reviewed.
Публікація документів назовні (артефакти claude.ai) не тут: її робить Claude лише після «так» людини.
"""
import ast
import hashlib
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
ART = ROOT / "artifacts"
ARCH = ART / "architecture.html"     # схема архітектури: необов'язкова копія в проєкті (сама схема фреймворку — у docs/ бібліотеки)
STYLE = ("<style>body{margin:0;background:#fff;color:#1a1a1a;font:16px/1.55 system-ui,sans-serif}.page{max-width:1100px;margin:0 auto;padding:24px 16px}"
         "table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #e3e3e3;padding:6px 8px;text-align:left;vertical-align:top}"
         ".muted,.s{color:#666}.s{display:block;font-size:13px}.lead{font-size:18px}"
         "@media (prefers-color-scheme:dark){body{background:#141414;color:#eee}th,td{border-color:#333}.muted,.s{color:#aaa}}</style>")
BRANCHES = ART / "branches.html"
STATE = ART / ".docs_state.json"
e = html.escape
# з чого пишеться проза схеми й скілів: якщо щось із цього змінилось, пояснення треба перечитати
PROSE_SOURCES = ["flows/pipeline.py", "ds/branches.py", "ds/regions.py", "ds/datasets.py", "assistant/agent.py", "assistant/server.py",
                 "scripts/03_signals.py", "scripts/08_app_extra.py", "scripts/app_recordings.py", "scripts/build_landing.py",
                 "web/landing/template.html", "web/template.html", "docs/LIMITATIONS.md",
                 ".claude/skills/new-study/SKILL.md", ".claude/skills/extend-study/SKILL.md", ".claude/skills/retarget-study/SKILL.md", ".claude/skills/study-docs/SKILL.md"]


def digest(rel):
    f = ROOT / rel
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else None


def table(head, rows):
    th = "".join(f"<th>{e(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def steps_part():
    from eaglewings.flows import pipeline as PL
    rows = []
    for i, (st, desc) in enumerate(PL.STEPS.items(), 1):
        jobs = [j for j in PL.JOBS if j[1] == st]
        what = "<br>".join(f"<code>{e(j[0])}</code> {e(j[2])}" for j in jobs)
        scripts = "<br>".join(e(" ".join(j[3])) for j in jobs)
        rows.append([f'<span class="n">{i}</span>', f"<b>{e(st)}</b><span class=\"s\">{e(desc)}</span>", what, f'<span class="s">{scripts}</span>'])
    return table(["", "Крок", "Задачі", "Скрипти"], rows)


def branches_part():
    from eaglewings.ds import branches as BR
    rows = []
    for k in BR.ORDER:
        r = BR.R.get(k) if hasattr(BR, "R") else None
        if not r:
            continue
        on = "так" if k in getattr(BR, "ACTIVE", BR.ORDER) else "вимкнена"
        rows.append([f"<b>{e(k)}</b>", e(r.get("name", k)), e(r.get("q", "")), e(r.get("group", "")),
                     "<br>".join(f"<code>{e(d)}</code>" for d in r.get("datasets", [])), on])
    return table(["Код", "Назва", "Питання", "Група", "Датасети", "Увімкнена"], rows)


def datasets_part():
    from eaglewings.ds import park
    packs = tuple(f"_{k}" for k in park.ALL)
    rows = []
    for d in sorted(p for p in (ROOT / "data" / "datasets").iterdir() if p.is_dir()):
        if d.name.endswith(packs):
            continue
        v = json.loads((d / "versions.json").read_text())[-1]
        m = json.loads((d / f"v{v['version']}" / "manifest.json").read_text())
        n_packs = sum(1 for k in park.ALL if (d.parent / f"{d.name}_{k}").exists())
        rows.append([f"<code>{e(d.name)}</code>", f"v{v['version']}", f"{m['rows']:,}".replace(",", " "), e(m.get("unit", "")),
                     f"{n_packs} парки, злиття" if n_packs else "—"])
    return table(["Датасет", "Версія", "Рядків", "Одиниця", "Пачки парків"], rows)


def assistant_part():
    """Таблиці асистента з DESCRIPTIONS у assistant/agent.py: читаємо код, а не імпортуємо (асистент живе в іншому середовищі)."""
    tree = ast.parse((LIB / "assistant" / "agent.py").read_text())
    desc = {}
    for node in ast.walk(tree):
        target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "DESCRIPTIONS" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple):
                    try:
                        desc[k.value] = ast.literal_eval(v.elts[0])
                    except ValueError:
                        desc[k.value] = ast.unparse(v.elts[0])[:200]
    rows = [[f"<code>{e(n)}</code>", e(d.split(". ")[0])] for n, d in desc.items()]
    return f'<p class="muted" style="font-size:14px">{len(rows)} таблиць.</p>' + table(["Таблиця", "Що в ній"], rows)


PARTS = dict(steps=steps_part, branches=branches_part, datasets=datasets_part, assistant=assistant_part)


def fill(s, name, body):
    a, b = f"<!-- auto:{name} -->", f"<!-- /auto:{name} -->"
    if a not in s:
        raise SystemExit(f"artifacts/architecture.html: немає мітки {a}")
    i, j = s.index(a) + len(a), s.index(b)
    return s[:i] + "\n" + body + "\n" + s[j:]


def branch_page():
    """artifacts/branches.html: стиль бере з architecture.html, вміст — з реєстру й research_results.csv."""
    from eaglewings.ds import branches as BR
    R = pd.read_csv(ROOT / "data" / "processed" / "research_results.csv")
    head = ARCH.read_text() if ARCH.exists() else ""
    style = head[head.index("<style>"):head.index("</style>") + 8] if "<style>" in head else STYLE
    secs = []
    for k in BR.ORDER:
        r = BR.R.get(k)
        if not r or k not in getattr(BR, "ACTIVE", BR.ORDER):
            continue
        g = R[R.branch == k]
        if not len(g):
            secs.append(f'<h2 id="{e(k)}">{e(r.get("name", k))}</h2><p class="muted">{e(r.get("q", ""))} Результатів ще немає.</p>')
            continue
        info = g.iloc[0]
        rows = []
        for x in g.itertuples():
            ci = "" if pd.isna(x.ci_lo) else f" ({x.ci_lo:.2f}–{x.ci_hi:.2f})"
            val = "" if pd.isna(x.val) else f"{x.val:.3f}"
            test = "" if pd.isna(x.test) else f"{x.test:.3f}{ci}"
            grp = f'<span class="s">{e(str(x.group))}</span>' if isinstance(x.group, str) and x.group else ""
            rows.append([("★ " if x.winner else "") + f"<b>{e(str(x.algorithm))}</b>{grp}", f'<span class="s">{e(str(x.hypothesis or ""))[:220]}</span>', val, test])
        secs.append(f'<h2 id="{e(k)}">{e(r.get("name", k))}</h2>'
                    f'<p>{e(r.get("q", ""))}</p><p class="muted" style="font-size:14px">{e(str(info.kind))} · метрика {e(str(info.metric))}'
                    f'{" (менше краще)" if str(info.lower_is_better) == "True" else ""} · переможця обирали: {e(str(info.how_chosen))}</p>'
                    + table(["Алгоритм", "Гіпотеза", "Валідація", "Тест (95 % CI)"], rows))
    toc = " · ".join(f'<a href="#{e(k)}">{e(k)}</a>' for k in BR.ORDER if BR.R.get(k) and k in getattr(BR, "ACTIVE", BR.ORDER))
    return (f'<!doctype html>\n<html lang="uk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Гілки EagleWings</title>{style}</head><body><div class="page">'
            f'<h1>Гілки EagleWings</h1><p class="lead">Кожне питання дослідження й усі алгоритми, що за нього змагались. ★ — переможець, обраний на валідації; '
            f'тест лише показує результат. Згенеровано з <code>ds/branches.py</code> і <code>data/processed/research_results.csv</code> (<code>scripts/build_docs.py</code>).</p>'
            f'<p class="muted" style="font-size:14px">{toc}</p>' + "".join(secs) + "</div></body></html>\n")


# ---------------------------------------------------------------- markdown-документи проєкту: docs/EXPERIMENTS.md, docs/ARCHITECTURE.md
# Код заповнює частини між <!-- auto:<назва> --> і <!-- /auto:<назва> -->; прозу поза мітками пише Claude (скіл study-docs).
DOCS = ROOT / "docs"


def mtable(head, rows):
    c = lambda v: str(v).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(c(v) for v in r) + " |" for r in rows])


def _ds_rows():
    out = {}
    for d in (ROOT / "data" / "datasets").glob("*/versions.json"):
        v = json.loads(d.read_text())[-1]
        m = json.loads((d.parent / f"v{v['version']}" / "manifest.json").read_text())
        out[d.parent.name] = m["rows"]
    return out


def _f(x, n=3):
    return "" if pd.isna(x) else f"{x:.{n}f}".replace(".", ",")


def md_experiments_summary():
    from eaglewings.ds import branches as BR
    R = pd.read_csv(ROOT / "data" / "processed" / "research_results.csv")
    rows_n = _ds_rows()
    rows = []
    for k in BR.ORDER:
        r = BR.R.get(k)
        if not r or k not in BR.ACTIVE:
            continue
        g = R[R.branch == k]
        ds = ", ".join(f"`{d}` · {rows_n[d]:,}".replace(",", " ") if d in rows_n else f"`{d}`" for d in r.get("datasets", []))
        if not len(g):
            rows.append([f"**{k}**", r.get("q", ""), "", "", "результатів ще немає", "", "", "", ds])
            continue
        W = g[g.winner.fillna(False).astype(bool)]
        w = W.iloc[0] if len(W) else g.iloc[0]
        metric = w.metric + (" (менше краще)" if str(w.lower_is_better) == "True" and "менше краще" not in str(w.metric) else "")
        if not len(W):                                 # гілка лише порівнює правила на тесті, без вибору на валідації
            best = g.sort_values("test", ascending=str(w.lower_is_better) == "True").iloc[0]
            rows.append([f"**{k}**", r.get("q", ""), w.kind, metric, f"без вибору: порівняння правил на тесті (найвищий тест — {best.algorithm}, {_f(best.test)})", "", "", w.how_chosen, ds])
            continue
        if len(W) > 1:                                 # переможець окремо для кожної групи (тип події, парк, датасет, лейбл)
            rows.append([f"**{k}**", r.get("q", ""), w.kind, metric, f"★ {len(W)} переможців, по одному на групу — див. [{k}](#{k.lower()})", "", "", w.how_chosen, ds])
            continue
        ci = "" if pd.isna(w.ci_lo) else f" ({_f(w.ci_lo, 2)}–{_f(w.ci_hi, 2)})"
        rows.append([f"**{k}**", r.get("q", ""), w.kind, metric,
                     f"★ {w.algorithm}" + (f" · {w.group}" if isinstance(w.group, str) and w.group else ""), _f(w.val), _f(w.test) + ci, w.how_chosen, ds])
    return mtable(["Гілка", "Питання", "Тип", "Метрика", "Переможець", "Валідація", "Тест (95 % CI)", "Як обирали", "Датасети · рядків"], rows)


def md_experiments_algorithms(limit=20):
    from eaglewings.ds import branches as BR
    R = pd.read_csv(ROOT / "data" / "processed" / "research_results.csv")
    out = []
    for k in BR.ORDER:
        r = BR.R.get(k)
        if not r or k not in BR.ACTIVE:
            continue
        g = R[R.branch == k].copy()
        if not len(g):
            continue
        g["w"] = g.winner.fillna(False).astype(bool)
        shown = g if len(g) <= limit else pd.concat([g[g.w], g[~g.w].sort_values("val", ascending=str(g.lower_is_better.iloc[0]) == "True").head(limit - int(g.w.sum()))])
        rows = [[("★ " if x.w else "") + str(x.algorithm), x.group if isinstance(x.group, str) else "", str(x.hypothesis or "")[:200], _f(x.val),
                 _f(x.test) + ("" if pd.isna(x.ci_lo) else f" ({_f(x.ci_lo, 2)}–{_f(x.ci_hi, 2)})")] for x in shown.itertuples()]
        more = f"\n\nПоказано {len(shown)} з {len(g)} варіантів (переможці й найкращі на валідації); усі — у `data/processed/research_results.csv`." if len(shown) < len(g) else ""
        out.append(f"<a id=\"{k.lower()}\"></a>\n\n### {r.get('name', k)}\n\n{r.get('q', '')} {r.get('why', '')}\n\n" + mtable(["Алгоритм", "Група", "Гіпотеза", "Валідація", "Тест (95 % CI)"], rows) + more)
    return "\n\n".join(out)


def md_steps():
    from eaglewings.flows import pipeline as PL
    rows = []
    for i, (st, desc) in enumerate(PL.STEPS.items(), 1):
        jobs = [j for j in PL.JOBS if j[1] == st]
        rows.append([i, f"**{st}** — {desc}", "<br>".join(f"`{j[0]}` {j[2]}" for j in jobs)])
    return mtable(["", "Крок", "Задачі"], rows)


def md_branches():
    from eaglewings.ds import branches as BR
    return mtable(["Код", "Назва", "Питання", "Група", "Датасети", "Увімкнена"],
                  [[f"**{k}**", BR.R[k].get("name", k), BR.R[k].get("q", ""), BR.R[k].get("group", ""), ", ".join(f"`{d}`" for d in BR.R[k].get("datasets", [])),
                    "так" if k in BR.ACTIVE else "ні"] for k in BR.ORDER if k in BR.R])


def md_datasets():
    rows = []
    for d in sorted((ROOT / "data" / "datasets").glob("*/versions.json")):
        v = json.loads(d.read_text())[-1]
        m = json.loads((d.parent / f"v{v['version']}" / "manifest.json").read_text())
        rows.append([f"`{d.parent.name}`", f"v{v['version']}", f"{m['rows']:,}".replace(",", " "), m.get("unit", ""), ", ".join(m.get("sources", []))])
    return mtable(["Датасет", "Версія", "Рядків", "Одиниця", "Джерела"], rows)


def md_assistant():
    tree = ast.parse((LIB / "assistant" / "agent.py").read_text())
    rows = []
    for node in ast.walk(tree):
        target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "DESCRIPTIONS" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple):
                    try:
                        d = ast.literal_eval(v.elts[0])
                    except ValueError:
                        d = ast.unparse(v.elts[0])[:200]
                    rows.append([f"`{k.value}`", str(d).split(". ")[0]])
    return mtable(["Таблиця", "Що в ній"], rows)


MD_DOCS = {"EXPERIMENTS.md": ("Експерименти", dict(summary=md_experiments_summary, algorithms=md_experiments_algorithms)),
           "ARCHITECTURE.md": ("Архітектура", dict(steps=md_steps, branches=md_branches, datasets=md_datasets, assistant=md_assistant))}


def fill_md():
    DOCS.mkdir(exist_ok=True)
    for name, (title, parts) in MD_DOCS.items():
        f = DOCS / name
        s = f.read_text() if f.exists() else f"# {title}\n\n" + "\n\n".join(f"<!-- auto:{p} -->\n<!-- /auto:{p} -->" for p in parts) + "\n"
        for p, fn in parts.items():
            a, b = f"<!-- auto:{p} -->", f"<!-- /auto:{p} -->"
            if a not in s:
                s += f"\n{a}\n{b}\n"
            try:
                body = fn()
            except FileNotFoundError as err:          # ще немає результатів чи датасетів
                body = f"_Ще немає даних: {err.filename}_"
            i, j = s.index(a) + len(a), s.index(b)
            s = s[:i] + "\n" + body + "\n" + s[j:]
        f.write_text(s)
        print(f"docs/{name}: частини", ", ".join(parts))


def check():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    seen = st.get("reviewed", {})
    changed = [f for f in PROSE_SOURCES if digest(f) != seen.get(f)]
    return changed


if __name__ == "__main__":
    if "--check" in sys.argv:
        ch = check()
        print("проза документів актуальна" if not ch else "змінилось з часу перегляду прози:\n  " + "\n  ".join(ch))
        sys.exit(0)
    if "--mark-reviewed" in sys.argv:
        STATE.write_text(json.dumps(dict(reviewed={f: digest(f) for f in PROSE_SOURCES}), ensure_ascii=False, indent=1))
        print("artifacts/.docs_state.json: проза позначена як переглянута")
        sys.exit(0)
    ART.mkdir(parents=True, exist_ok=True)
    if ARCH.exists():
        s = ARCH.read_text()
        for name, fn in PARTS.items():
            s = fill(s, name, fn())
        ARCH.write_text(s)
        print("artifacts/architecture.html: розділи", ", ".join(PARTS))
    else:
        print("artifacts/architecture.html немає в проєкті: схему не заповнюю (схема фреймворку — docs/architecture.html бібліотеки)")
    BRANCHES.write_text(branch_page())
    fill_md()
    ch = check()
    print("artifacts/branches.html")
    if ch:
        print("проза могла відстати від коду, змінилось:", ", ".join(ch), "→ Claude: перечитати й оновити пояснення, потім --mark-reviewed")
