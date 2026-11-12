"""Дві мови сайту: копії сторінок у web/ua/ і web/en/ та скрипт перекладу в кожній. Запуск із теки eaglewings:

    eaglewings script i18n_pages            # після збирання сторінок (крок потоку i18n)

Оригінал кожної сторінки український, англійська — з словника web/i18n/en.json через web/assets/i18n.js. Копія
/en/<сторінка> — та сама сторінка з <base> на корінь сайту й мовою en; /ua/<сторінка> — так само з мовою ua.
Кореневі сторінки (/, /landing.html …) лишаються українськими й теж отримують перемикач.

Новий український текст проєкту (блоки лендінгу, назви парків) → <проєкт>/i18n/en.json через scripts/i18n_merge.py <файл {укр: англ}>;
текст шаблонів бібліотеки → eaglewings/templates/i18n/en.json (scripts/i18n_merge.py --lib).
Знайти тексти без перекладу: scripts/i18n_probe.html скопіювати в web/_i18n_probe.html і відкрити в headless Chrome
(…/_i18n_probe.html#page:/landing.html чи #app:<парк>) — він збирає всі українські рядки сторінки й станів застосунку.
"""
import re
import sys
from pathlib import Path

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
WEB = ROOT / "web"
PAGES = {"index.html": "", "landing.html": "", "app.html": "", "science/index.html": "science/"}


def inject(html, lang, base):
    html = re.sub(r'\s*<base href="[^"]*">', "", html)
    html = re.sub(r'\s*<script>window\.I18N_LANG=[^<]*</script>\s*<script src="[^"]*i18n\.js"></script>', "", html)
    tag = (f'<base href="{base}">' if base is not None else "") + \
          (f'<script>window.I18N_LANG="{lang}"</script>' if lang else "") + f'<script src="{"/assets/i18n.js"}"></script>'
    return re.sub(r"<head>", "<head>" + tag, html, count=1)


def build_assets():
    """Скрипт перекладу з бібліотеки й словник: переклад бібліотеки + переклад проєкту (i18n/en.json проєкту зверху)."""
    import json
    import shutil
    from eaglewings.project import TEMPLATES
    (WEB / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATES / "assets" / "i18n.js", WEB / "assets" / "i18n.js")
    d = json.loads((TEMPLATES / "i18n" / "en.json").read_text())
    own = ROOT / "i18n" / "en.json"
    if own.exists():
        o = json.loads(own.read_text())
        for part in ("exact", "patterns"):
            d[part].update(o.get(part, {}))
    (WEB / "i18n").mkdir(exist_ok=True)
    (WEB / "i18n" / "en.json").write_text(json.dumps(d, ensure_ascii=False, sort_keys=True))


def main():
    build_assets()
    for page, sub in PAGES.items():
        f = WEB / page
        if not f.exists():
            continue
        src = f.read_text()
        f.write_text(inject(src, None, None))                         # корінь: український, з перемикачем
        for lang in ("ua", "en"):
            out = WEB / lang / page
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(inject(src, lang, "/" + sub))
    for m in sorted(WEB.glob("map_live*.html")):                     # мапа моніторингу в iframe: мова з ?lang=
        m.write_text(inject(m.read_text(), None, None))
    print("web/ua/, web/en/:", ", ".join(PAGES))


if __name__ == "__main__":
    main()
