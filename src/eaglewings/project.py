"""Тека проєкту EagleWings і його конфіг.

Бібліотека (цей пакет) містить лише код і шаблони. Усе, що стосується конкретного дослідження — конфіг території
й гілок, дані, результати, сторінки, контент лендінгу, переклад, LIMITATIONS — лежить у теці проєкту:

    <проєкт>/
      eaglewings.toml          ознака теки проєкту (може бути порожнім)
      config/regions.py        REGION: територія, парки, джерела, роки
      config/branches.py       ACTIVE (необов'язково): які гілки рахувати; без нього — усі з реєстру
      config/media.py          PHOTOS, VIDEO (необов'язково): фото й відео для демо детекторів
      web/landing/blocks/      блоки лендінгу продукту, web/landing/views.json — записи застосунку
      web/i18n/en.json         переклад текстів проєкту (поверх перекладу бібліотеки)
      docs/LIMITATIONS.md      що стверджуємо
      data/ web/ models/ logs/ notebooks/ artifacts/   усе, що рахує й збирає потік

Тека проєкту: змінна EAGLEWINGS_PROJECT, інакше найближча тека з eaglewings.toml вгору від поточної.
"""
import importlib.util
import os
from pathlib import Path

LIB = Path(__file__).resolve().parent              # код бібліотеки
TEMPLATES = LIB / "templates"                       # шаблони сторінок за замовчуванням


def _find_root():
    env = os.environ.get("EAGLEWINGS_PROJECT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / "eaglewings.toml").exists():
            return d
    raise SystemExit("EagleWings: не знайдено теку проєкту. Запускайте з теки, де є eaglewings.toml, "
                     "або задайте EAGLEWINGS_PROJECT=<тека проєкту>. Новий проєкт: eaglewings init <тека>.")


ROOT = _find_root()
os.environ.setdefault("EAGLEWINGS_PROJECT", str(ROOT))   # дочірні процеси потоку бачать ту саму теку


def config(name, default=None):
    """Модуль config/<name>.py проєкту (або None, якщо його немає)."""
    f = ROOT / "config" / f"{name}.py"
    if not f.exists():
        return default
    spec = importlib.util.spec_from_file_location(f"eaglewings_project_config_{name}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def template(rel):
    """Шаблон: спершу з проєкту (<проєкт>/templates/<rel>), інакше з бібліотеки."""
    own = ROOT / "templates" / rel
    return own if own.exists() else TEMPLATES / rel
