"""Готові моделі й налаштування з іншого проєкту EagleWings: «модель як є», без підбору й перенавчання.

У config/regions.py проєкту:
    REGION["transfer"] = dict(project="/шлях/до/іншого/проєкту", name="Фракія", name_en="Thrace")

Модель береться з <проєкт>/models/<код>/ (card.json, model.joblib), налаштування зведення пожеж — з
<проєкт>/data/processed/matching_calibration.json. Перенесена модель нічого не вчить на цій території, тому
в виборі лідера не бере участі: вона стоїть поруч зі своїми підходами на тих самих даних, а результат — лише на тесті.
Без REGION["transfer"] усе тут повертає None, і гілки рахуються як раніше.
"""
import json
from pathlib import Path

from .regions import REGION

CFG = REGION.get("transfer") or {}
SRC = Path(CFG["project"]) if CFG.get("project") else None
NAME = CFG.get("name", "інший проєкт")
NAME_EN = CFG.get("name_en", "other project")


def on():
    return SRC is not None and SRC.exists()


def card(code):
    """card.json моделі-переможця гілки в проєкті-джерелі."""
    f = SRC / "models" / code / "card.json" if on() else None
    return json.loads(f.read_text()) if f and f.exists() else None


def model(code):
    """(модель, картка) — переможець гілки code у проєкті-джерелі, або (None, None)."""
    c = card(code)
    f = SRC / "models" / code / "model.joblib" if c else None
    if not f or not f.exists():
        return None, None
    import joblib
    return joblib.load(f), c


def matching():
    """Налаштування зведення теплових точок (FM) у проєкті-джерелі: km, days, champion."""
    f = SRC / "data" / "processed" / "matching_calibration.json" if on() else None
    return json.loads(f.read_text()) if f and f.exists() else None


def label(name):
    """Назва підходу «модель як є»: transfer_<код>."""
    return f"transfer_{name}"


def hyp():
    return f"Готова модель ({NAME}) без змін: без підбору й перенавчання"
