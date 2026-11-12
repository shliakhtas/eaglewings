"""Сторінка дослідження з шаблону: web/science/template.html + розділ кожної гілки з реєстру ds/branches.py → web/science/index.html.

    eaglewings script build_science_page

Розділ гілки береться з web/science/parts/<якір>.html, якщо він є (там графіки, намальовані під цю гілку).
Гілка без свого фрагмента отримує розділ автоматично: постановка, як обрано лідера, лідер, усі алгоритми з
інтервалами й графіком, усі метрики. Його збирають ті самі функції, що й сторінки-звіти (ds/pages.py).
Меню, групи й типи гілок теж беруться з реєстру. Нічого не рахує: лише складає готові результати.
"""
import html
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from eaglewings.project import ROOT, LIB, template  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import branches as BR  # noqa: E402

WEB = ROOT / "web" / "science"
SUMMARY = ROOT / "data" / "processed" / "experiments_summary.json"
PARK_PAGES = {"a", "b", "c", "ar", "ld", "lm", "lg", "tp"}      # розділи гілок, де є метрики чи мапи по парках


def generic(code, b, S):
    """Розділ і картка меню для гілки без свого фрагмента: зі звіту гілки (ds/pages.task_md)."""
    import markdown
    from eaglewings.ds import pages
    key, lead = b["summary"], BR.lead().get(b["summary"])
    rows = S.get(key, {}).get("results", [])
    if b.get("part"):
        rows = [r for r in rows if r.get("part", "").startswith(b["part"])]
    mod = importlib.import_module(f"eaglewings.ds.tasks.{b['run'].split(':')[0]}")
    md = pages.task_md(key, SimpleNamespace(EXPERIMENT=f"{code} · {b.get('title') or b['q']}", __doc__=mod.__doc__), rows) if rows and lead else None
    body = markdown.markdown(md.split("\n", 1)[1], extensions=["tables"]) if md else "<p>Результатів ще немає: гілку додано, а потік її ще не рахував.</p>"
    title = b.get("title") or b["q"]
    sec = (f'<section id="{b["page"]}">\n  <div class="wrap">\n    <div class="task-hd"><span class="tk">{html.escape(code)}</span><div>'
           f'<span class="label">{html.escape(code)} · {html.escape(title)}</span><h2>{html.escape(b["q"])}</h2></div></div>\n'
           f'    <p class="lead">{html.escape(b["why"])}</p>\n    <div class="generic">{body}</div>\n  </div>\n</section>\n')
    value = "—"
    if rows and lead:
        bd = pages.board(key, rows)
        ch = bd[bd.champ]
        if len(ch) and ch.test.iloc[0] == ch.test.iloc[0]:
            value = f"{ch.test.iloc[0]:.2f}".replace(".", ",") + f"<small>{html.escape(lead['metric'])}</small>"
    card = [b["page"], code, title, b["q"], value, lead["how"] if lead else ""]
    return sec, card


def build():
    S = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    secs, cards, made = [], [], []
    for p in BR.page_branches():
        f = template(f"science/parts/{p['id']}.html")
        if f.exists():
            sec = f.read_text()
        else:
            sec, card = generic(p["code"], BR.R[p["code"]], S)
            cards.append(card)
            made.append(p["code"])
        if p["id"] in PARK_PAGES:                     # блок «По парках»: перемикач мап і метрики кожного парку (scripts/15_parks_page.py)
            i = sec.rstrip().rfind("</section>")
            sec = sec[:i] + f'  <div class="wrap"><div class="pk-block" data-page="{p["id"]}"></div></div>\n' + sec[i:]
        secs.append(sec)
    data = f"<script>window.PAGE_BRANCHES = {json.dumps(BR.page_branches(), ensure_ascii=False)}; window.PAGE_CARDS = {json.dumps(cards, ensure_ascii=False)};</script>\n"
    css = "<style>.generic table{border-collapse:collapse;margin:10px 0;font-size:13px;display:block;overflow-x:auto}.generic td,.generic th{border:1px solid var(--line,#333);padding:4px 8px}.generic img{max-width:100%;background:#fff;border-radius:8px}</style>\n"
    t = template("science/template.html").read_text()
    out = t.replace("<!-- BRANCHES -->", "\n".join(secs), 1).replace("<!-- PAGE_DATA -->", data + css, 1)
    (WEB / "index.html").write_text(out)
    print(f"web/science/index.html: {len(secs)} гілок, з них зібрано автоматично: {', '.join(made) or 'жодної'}")


if __name__ == "__main__":
    build()
