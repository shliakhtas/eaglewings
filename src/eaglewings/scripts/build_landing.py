"""Лендінг продукту з каркаса й блоків: web/landing/ → web/landing.html. Запуск із теки eaglewings:

    eaglewings script build_landing           # зібрати
    eaglewings script build_landing --check   # лише перевірити, що зібране збігається з web/landing.html

Будова (як сторінка дослідження з каркаса й розділів):
    web/landing/template.html      каркас: стилі, порядок блоків як мітки <!-- block:<назва> -->
    web/landing/blocks/<назва>.html  один блок: nav, hero, guard, happened, questions, data, who, cta, footer …
    web/landing/views.json         блок записів застосунку «views»: сцени, підписи й тексти

Блок views збирається з views.json: на кожну сцену запис web/assets/app/app_<сцена>.webp, який знімає з
реального застосунку scripts/app_recordings.py (ті самі назви сцен, що в його SCENES). Записи чергуються ліворуч і праворуч.
Мітка без файлу блоку прибирається, тож блок вимикається видаленням файлу; новий блок — це мітка в каркасі й файл.
Скрипт нічого не рахує: числа в блоках беруться з результатів дослідження і обмежені docs/LIMITATIONS.md.
"""
import html
import json
import re
import sys
from pathlib import Path

from eaglewings.project import ROOT, LIB, template  # noqa: E402  (тека проєкту й код бібліотеки)
SRC = ROOT / "web" / "landing"
OUT = ROOT / "web" / "landing.html"
APP = ROOT / "web" / "assets" / "app"
MARK = re.compile(r"^<!-- block:([\w-]+) -->$", re.M)


def views_block(cfg):
    """Блок записів застосунку: заголовок, на кожну сцену запис і текст, хвіст."""
    import hashlib
    ver = lambda sc: hashlib.sha1((APP / f"app_{sc}.webp").read_bytes()).hexdigest()[:8] if (APP / f"app_{sc}.webp").exists() else "0"   # новий запис — нова адреса, браузер не бере старий з кешу
    w, h = cfg.get("size", [1100, 575])
    if cfg.get("layout") == "scene":                  # розмітка «сцена»: запис і текст поруч, через одну навпаки (свій каркас проєкту)
        rows = [cfg["head"].rstrip("\n")]
        for i, v in enumerate(cfg["views"], 1):
            chips = "".join(f'<span class="chip">{c}</span>' for c in v.get("chips", []))
            rows += [f'      <div class="scene{" rev" if i % 2 == 0 else ""} rv">',
                     f'        <figure class="browser"><div class="bar"><i></i><i></i><i></i><span>{cfg["bar"]}</span></div>'
                     f'<img src="assets/app/app_{v["scene"]}.webp?v={ver(v["scene"])}" alt="{html.escape(v["alt"], quote=True)}" loading="lazy" width="{w}" height="{h}"></figure>',
                     f'        <div class="scene-t"><span class="label o">{i:02d} · {v["label"]}</span><h3>{v["title"]}</h3><p>{v["text"]}</p>'
                     + (f'<div class="chips">{chips}</div>' if chips else "") + "</div>",
                     "      </div>"]
        return "\n".join(rows + [cfg["tail"]]) + "\n"
    rows = [cfg["head"], '      <div class="views">']
    for i, v in enumerate(cfg["views"], 1):
        chips = "".join(f'<span class="chip">{c}</span>' for c in v.get("chips", []))
        rows += [f'        <div class="view{" rev" if i % 2 == 0 else ""}">',
                 f'          <div class="shot browser"><div class="bar"><i></i><i></i><i></i><span>{cfg["bar"]}</span></div>'
                 f'<img src="assets/app/app_{v["scene"]}.webp?v={ver(v["scene"])}" alt="{html.escape(v["alt"], quote=True)}" loading="lazy" width="{w}" height="{h}"></div>',
                 f'          <div><span class="label">{i:02d} · {v["label"]}</span><h3>{v["title"]}</h3><p>{v["text"]}</p>'
                 + (f'<div class="chips">{chips}</div>' if chips else "") + "</div>",
                 "        </div>"]
    rows += [cfg["tail"]]
    return "\n".join(rows) + "\n"


def build():
    t = template("landing.html").read_text()
    missing = []

    def block(m):
        name = m.group(1)
        if name == "views" and (SRC / "views.json").exists():
            cfg = json.loads((SRC / "views.json").read_text())
            missing.extend(v["scene"] for v in cfg["views"] if not (APP / f"app_{v['scene']}.webp").exists())
            return views_block(cfg).rstrip("\n")
        f = SRC / "blocks" / f"{name}.html"
        return f.read_text().rstrip("\n") if f.exists() else ""

    out = MARK.sub(block, t)
    out = re.sub(r"\n{3,}", "\n\n\n", out)
    return out, missing


def main():
    out, missing = build()
    if missing:
        print("немає записів застосунку для сцен:", ", ".join(missing), "— спершу scripts/app_recordings.py")
    if "--check" in sys.argv:
        same = OUT.exists() and OUT.read_text() == out
        print("збігається" if same else "відрізняється від web/landing.html")
        sys.exit(0 if same else 1)
    OUT.write_text(out)
    (ROOT / "web" / "index.html").write_text(out)           # лендінг продукту — головна сторінка проєкту
    print(f"{OUT.relative_to(ROOT)}: {len(out) // 1024} КБ, блоків {len(MARK.findall(template('landing.html').read_text()))}")
    sys.exit(3 if missing else 0)


if __name__ == "__main__":
    main()
