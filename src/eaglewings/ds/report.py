"""Спільний код для ноутбуків: ті самі результати, що на сторінці досліджень, у вигляді таблиць і графіків.

Сторінку будують scripts/06_build_science.py і scripts/09_science_more.py з data/processed/experiments_summary.json.
Ноутбуки беруть той самий файл і ті самі задачі ds/tasks/*, тому числа збігаються з сайтом.
"""
import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from .branches import KIND, report_branches  # noqa: E402
BRANCHES = report_branches()                          # гілки з реєстру ds/branches.py


def summary():
    return json.loads((ROOT / "data" / "processed" / "experiments_summary.json").read_text())


def results(key, part=None):
    df = pd.DataFrame(summary()[key]["results"])
    return df[df.part == part].reset_index(drop=True) if part and "part" in df else df


def protocol(code):
    """Постановка, дані, протокол і метрики: докстрінг модуля задачі, той самий, що в журналі запусків."""
    return importlib.import_module(f"eaglewings.ds.tasks.{BRANCHES[code]['module']}").__doc__


def _ci(s):
    try:
        a, b = str(s).replace("–", "-").split("-")[:2]
        return float(a), float(b)
    except Exception:
        return np.nan, np.nan


def forest(df, label, value, lo=None, hi=None, ci=None, champ="champion", title="", xlabel=""):
    """Горизонтальний графік «підхід → метрика з інтервалом», як на сторінці."""
    d = df.copy()
    if ci is not None:
        d[["_lo", "_hi"]] = d[ci].apply(lambda s: pd.Series(_ci(s)))
        lo, hi = "_lo", "_hi"
    d = d.sort_values(value)
    fig, ax = plt.subplots(figsize=(8, .45 * len(d) + 1.2))
    col = ["#d95926" if c else "#3b82f6" for c in d.get(champ, pd.Series(False, index=d.index)).fillna(False)]
    ax.barh(d[label].astype(str), d[value], color=col)
    if lo:
        ax.errorbar(d[value], d[label].astype(str), xerr=[d[value] - d[lo], d[hi] - d[value]], fmt="none", ecolor="#333", capsize=3)
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    return fig


def show(code):
    """Головна таблиця й графік гілки: те саме, що в блоці «Результат» на сторінці."""
    b = BRANCHES[code]
    k = b["key"]
    if code == "FD":
        d = results(k)
        return d[["family", "calib_f1", "test_f1", "test_f1_ci", "test_f1_effis", "champion", "diff_base", "diff_lo", "diff_hi"]].round(3), forest(d, "family", "test_f1", ci="test_f1_ci", title="FD · F1 на тестових блоках", xlabel="F1")
    if code == "FM":
        d = results(k)
        return d[["approach", "val_bcubed_f1", "bcubed_f1", "bcubed_f1_ci_lo", "bcubed_f1_ci_hi", "bcubed_f1_strict", "champion", "diff_base"]].round(3), forest(d, "approach", "bcubed_f1", "bcubed_f1_ci_lo", "bcubed_f1_ci_hi", title="FM · BCubed F1, тест 2024–2025")
    if code == "FG_day":
        d = results(k).query("target == 'fire_day'")
        return d[["approach", "val_pr_auc", "pr_auc", "pr_auc_ci", "recall_at_10", "champion"]].round(3), forest(d, "approach", "pr_auc", ci="pr_auc_ci", title="FG · день пожежі, PR-AUC 2023–2025")
    if code == "FG_risk":
        d = results(k)
        return d[["approach", "val_roc_auc", "test_roc_auc", "test_roc_auc_ci", "test_roc_auc_hansen", "test_roc_auc_dndvi", "champion"]].round(3), forest(d, "approach", "test_roc_auc", ci="test_roc_auc_ci", title="FG · ризик, ROC-AUC на пожежі 2023")
    if code == "FG_spread":
        d = results(k)
        return d[["approach", "lofo_f1_24", "lofo_f1_24_ci", "test_f1_24", "test_f1_48", "champion"]].round(3), forest(d, "approach", "lofo_f1_24", ci="lofo_f1_24_ci", title="FG · поширення, F1 на 27 пожежах (LOFO)")
    if code == "AR":
        d = results(k, "test")
        return d[["approach", "val_pai", "test_pai", "lo", "hi", "test_capture"]].round(2), forest(d, "approach", "test_pai", "lo", "hi", champ=None, title="AR · PAI на 2024–2025")
    if code in ("LD", "LM", "LG"):
        d = results(k, b["part"])
        v = "precision" if code == "LM" else "test_metric"
        cols = [c for c in ("approach", "val_metric", "test_metric", "ci", "precision", "recall", "true_ha", "champion", "diff_hist", "diff_rand") if c in d]
        return d[cols].round(3), forest(d, "approach", v, ci="ci" if code != "LM" else None, title=f"{code}")
    if code in ("SD", "CD"):
        d = results(k)
        d["label"] = d.dataset + " · " + d.get("target", "").astype(str) + " · " + d.approach
        return d[[c for c in ("label", "thr", "ap50", "ap50_ci", "precision", "recall", "roc_auc") if c in d]].round(3), forest(d.dropna(subset=["ap50"]), "label", "ap50", ci="ap50_ci", title=f"{code} · AP@0.5")
    if code == "TP":
        d = results(k, "test").query("pool == 300")
        return d[["approach", "first", "first_lo", "first_hi", "any", "burn", "diff", "diff_lo", "diff_hi", "champion"]].round(3), forest(d, "approach", "first", "first_lo", "first_hi", title="TP · частка пожеж 2023–2025 у полі зору")
    if code == "CG":
        d = results(k, "locate")
        return d[["approach", "sigma", "k", "val_hit", "test_hit", "test_km2", "test_km", "n"]].round(3), forest(d.query("sigma == 3"), "approach", "test_km2", champ=None, title="CG · площа зони, похибка 3°", xlabel="км²")
    if code == "CP":
        d = results(k, "test")
        d["label"] = d.approach + " · " + d["where"]
        return d[["label", "log", "log_lo", "log_hi", "fire", "forest", "diff", "diff_lo", "diff_hi", "champion"]].round(3), forest(d, "label", "log", "log_lo", "log_hi", title="CP · частка рубок 2021–2025")
    if code == "XM":
        d = results(k)
        d = d[d.champion.fillna(False).astype(bool) | (d.approach == "без зведення")]
        return d[["kind", "approach", "test_f1", "champion"]].round(3), None
    if code == "AS":
        d = results(k)
        cols = ["approach", "val_score", "test_score", "ci_lo", "ci_hi", "test_grouping", "test_lifecycle", "test_dynamics", "test_regularity", "test_sources", "champion"]
        return d[cols].round(3), forest(d, "approach", "test_score", "ci_lo", "ci_hi", title="AS · середня точність рис зони (симуляція)")
    raise KeyError(code)


def by_year(code):
    """Прогресивний протокол: результат по роках для TP, CP і LG."""
    if code == "TP":
        return results("t23", "year").pivot_table(index="year", columns="approach", values="first").round(3)
    if code == "CP":
        return results("t24", "year").query("where == 'усі' or approach == 'random'").pivot_table(index="year", columns="approach", values="log").round(3)
    if code == "LG":
        return results("t19", "LG_year").pivot_table(index="year", columns="approach", values="test_metric").round(3)
    raise KeyError(code)


def dataset_eda(name):
    """Той самий EDA, що у вкладці «Датасети»: функція build зі scripts/10_dataset_eda.py."""
    import importlib.util
    import sys
    sys.path.insert(0, str(LIB / "scripts"))
    spec = importlib.util.spec_from_file_location("eda", LIB / "scripts" / "10_dataset_eda.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.build(name)
