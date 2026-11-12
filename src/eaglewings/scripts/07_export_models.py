"""Моделі-переможці кожної гілки → models/<код>/ (card.json, model.joblib, якщо модель навчена). Запуск із теки eaglewings:

    eaglewings script 07_export_models
    eaglewings script 07_export_models --clearml     # ще й у ClearML Model Registry (Dadia/models)

Беремо підсумки data/processed/experiments_summary.json і навчені моделі з logs/runs (останній запуск).
Теки переможців перезаписуються; models/weights (ваги сторонніх детекторів) не чіпаємо.
"""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import tracking  # noqa: E402  (--clearml вмикає запис у ClearML)
S = json.loads((ROOT / "data" / "processed" / "experiments_summary.json").read_text())
OUT, RUNS = ROOT / "models", ROOT / "logs" / "runs"


def ds(name):
    d = ROOT / "data" / "datasets" / name
    v = json.load(open(d / "versions.json"))[-1]
    return f"{name}@v{v['version']} ({v['digest']})"


def joblib_of(name):
    fs = sorted(RUNS.glob(f"Dadia_*/*/{name}.joblib"))
    return fs[-1] if fs else None


def pick(key, cond):
    return next(r for r in S[key]["results"] if cond(r))


CARDS = []


def card(code, task, approach, selection, metric, test, datasets, extra=None, model=None, kind="виміряно"):
    d = OUT / code
    d.mkdir(parents=True, exist_ok=True)
    c = dict(code=code, task=task, kind=kind, approach=approach, selection=selection, metric=metric, test=test, datasets=datasets,
             created=dt.datetime.now().isoformat(timespec="minutes"), **(extra or {}))
    if model and joblib_of(model):
        shutil.copy(joblib_of(model), d / "model.joblib")
        c["files"] = ["card.json", "model.joblib"]
    else:
        c["files"] = ["card.json"]
        c["model"] = "правило або алгоритм без навчання: параметри в цій картці"
    (d / "card.json").write_text(json.dumps(c, ensure_ascii=False, indent=1, default=str))
    CARDS.append(c)
    tracking.register_card(c, d / "model.joblib" if (d / "model.joblib").exists() else None)


if __name__ == "__main__":
    for p in OUT.iterdir():                                                  # лише поточні переможці: старі теки прибираємо
        if p.is_dir() and p.name != "weights":
            shutil.rmtree(p)
    from eaglewings.ds import branches as BR
    if "t1" in S and not BR.mode("FD"):                                  # гілка увімкнена; у режимі signal (FD) моделі немає: це опис
        r = pick("t1", lambda r: r.get("champion"))
        card("FD", "Що згоріло у 2023 році", r["family"], "max F1 на калібрувальній половині блоків", "F1",
             dict(f1=r["test_f1"], ci=r["test_f1_ci"], precision=r["test_precision"], recall=r["test_recall"]), [ds("burned_area_pixels")], dict(params=r["best_config"]))
    if "t13" in S:                                                        # гілка увімкнена в config/branches.py
        r = pick("t13", lambda r: r.get("champion"))
        card("FM", "Які теплові точки це одна пожежа", r["approach"], "max BCubed F1 на 2021–2023", "BCubed F1",
             dict(bcubed_f1=r["bcubed_f1"], ci=[r.get("bcubed_f1_ci_lo"), r.get("bcubed_f1_ci_hi")]), [ds("matching_firms_effis")], dict(params=dict(km=r.get("km"), days=r.get("days"))))
    if "t2" in S:                                                        # гілка увімкнена в config/branches.py
        for tgt, nm in (("fire_day", "день із вогнем"), ("ignition", "нове займання")):
            r = pick("t2", lambda r, t=tgt: r["target"] == t and r.get("champion"))
            card(f"FG_day_{tgt}", f"Прогноз: {nm}", r["approach"], "max PR-AUC на 2022 (навчання 2021)", "PR-AUC",
                 dict(pr_auc=r["pr_auc"], ci=r.get("pr_auc_ci"), recall_at_10=r.get("recall_at_10")), [ds("fire_day_daily")], model=f"dadia_{tgt}")
    if "t3" in S and not BR.mode("FG_risk"):                             # у режимі transfer своєї моделі немає
        r = pick("t3", lambda r: r.get("champion"))
        card("FG_risk", "Які ділянки згорять", r["approach"], "max ROC-AUC на пожежі 2022", "ROC-AUC",
             dict(roc_auc=r["test_roc_auc"], ci=r.get("test_roc_auc_ci")), [ds("h3_risk_cells")], dict(note="Негативний результат: на тесті майже випадково."), model="dadia_fire_risk_h3")
    if "t5" in S and any(r.get("champion") for r in S["t5"]["results"]):   # без пожеж поширення не рахувалось
        r = pick("t5", lambda r: r.get("champion"))
        card("FG_spread", "Куди піде вогонь за добу", r["approach"], "max macro F1 на 27 пожежах, leave-one-fire-out", "F1 нової площі",
             dict(f1_24=r["test_f1_24"], f1_48=r["test_f1_48"], front_err_m=r.get("test_front_err_m_24")), [ds("fire_spread_cells")], model="dadia_fire_spread_24h")
    if "t19" in S:                                                        # гілка увімкнена в config/branches.py
        for part, code, task in (("LD", "LD", "Де зрубано ліс"), ("LG", "LG", "Де рубатимуть наступного року")):
            r = pick("t19", lambda r, p=part: r["part"] == p and r.get("champion"))
            card(code, task, r["approach"], "max метрики на валідації", "F1" if part == "LD" else "частка втрат в 11 % гексів",
                 dict(value=r["test_metric"], ci=r.get("ci")), [ds("logging_gr_pixels" if part == "LD" else "logging_gr_cells")])
    if "t21" in S and BR.mode("AR") == "logging":
        r = pick("t21", lambda r: r["part"] == "test" and r.get("champion"))
        card("AR", "Де рубки повторюються", r["approach"], "max PAI на валідаційних роках", "PAI", dict(pai=r["test_pai"], ci=[r.get("lo"), r.get("hi")]), [ds("ar_logging")],
             dict(params=dict(eps_km=r.get("eps"), window=r.get("win"))))
    elif "t21" in S:                                                      # гілка увімкнена в config/branches.py
        r = pick("t21", lambda r: r["part"] == "test" and r["approach"] == "ареалізація")
        card("AR", "Де пожежі повторюються", "DBSCAN + оболонка", "max PAI на 2023", "PAI", dict(pai=r["test_pai"], ci=[r.get("lo"), r.get("hi")]), [ds("ar_fires")], dict(params=dict(eps_km=r.get("eps"), window=r.get("win"))))
    if "t23" in S:                                                        # гілка увімкнена в config/branches.py
        r = pick("t23", lambda r: r.get("champion"))
        card("TP", "Де ставити вежі", f"{r['approach']}, {r['pool']} місць", "max частки пожеж на 2022", "частка пожеж, перша точка", dict(first=r["first"], ci=[r.get("first_lo"), r.get("first_hi")]),
             [ds("tp_fires"), ds("tp_candidates")], kind="сценарний аналіз")
    if "t24" in S:                                                        # гілка увімкнена в config/branches.py
        r = pick("t24", lambda r: r.get("champion"))
        card("CP", "Де ставити камери", f"{r['approach']}, {r['where']}", "max частки рубок на 2018–2020", "частка рубок", dict(log=r["log"], ci=[r.get("log_lo"), r.get("log_hi")]),
             [ds("cp_events")], kind="сценарний аналіз")
    if "t11" in S and BR.mode("SD") == "tracks":
        r = pick("t11", lambda r: r.get("champion"))
        card("SD", "Чи видно лісові колії на Sentinel-2", r["approach"], "max ROC AUC на відкладених блоках навчальних парків", "ROC AUC",
             dict(roc_auc=r["test_roc_auc"], ci=[r.get("ci_lo"), r.get("ci_hi")]), [ds("track_segments")])
    if "t26" in S:
        r = pick("t26", lambda r: r.get("champion"))
        card("RS", "Розв'язувач маршрутів на TOPTW", r["approach"], "min розрив на Solomon, ≤ 10 с", "розрив до найкращого відомого, %",
             dict(gap=r["test_gap"], ci=[r.get("gap_ci_lo"), r.get("gap_ci_hi")]), [ds("toptw_instances")])
    if "t27" in S:
        r = pick("t27", lambda r: r.get("champion"))
        card("RD", "Маршрути дня (симуляція)", r["approach"], "max частка вчасно на днях вибору", "частка обов'язкових зупинок вчасно",
             dict(on_time=r["test_on_time"], ci=[r.get("ci_lo"), r.get("ci_hi")]), [ds("route_instances")], kind="симуляція")
    lines = ["# Моделі-переможці", "", "Згенеровано `scripts/07_export_models.py` з останніх запусків. Кожна папка: `card.json` (підхід, правило вибору, метрика на тесті, датасет) і `model.joblib`, якщо модель навчена.",
             "Детектори SD і CD це готові моделі сторонніх авторів, їхні ваги тут не поширюються.", "",
             "| Код | Задача | Тип | Підхід | Метрика | Тест | Файли |", "|---|---|---|---|---|---|---|"]
    for c in CARDS:
        t = ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in c["test"].items() if v is not None and k != "ci")
        lines.append(f"| {c['code']} | {c['task']} | {c['kind']} | {c['approach']} | {c['metric']} | {t} | {', '.join(c['files'])} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(len(CARDS), "карток")
