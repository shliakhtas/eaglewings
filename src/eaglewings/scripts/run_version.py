"""Версії прогонів: числа кожного прогону потоку, різниця з попереднім і журнал усіх версій.

Наприкінці кожного прогону flows/pipeline.py записує нову версію. Окремо:

    eaglewings script run_version              # записати версію поточного стану
    eaglewings script run_version --clearml    # ще й у ClearML: датасет run_numbers і звіт у Reports

Версія це знімок після прогону: метрики вибору й тесту кожного алгоритму кожної гілки, 95 % CI, переможці,
датасети (версія, відбиток вмісту, рядки). Версія порівнюється з попередньою: що змінилось, було, стало, різниця.
Перша версія показує лише свої числа. Усе лежить у data/runs/: v<N>.json і history.json (журнал версій).
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
RUNS = ROOT / "data" / "runs"
TOL = .0005                                          # різниця, меншу за яку не вважаємо зміною (округлення)


def snapshot():
    """Числа поточного стану: усі алгоритми всіх гілок і всі датасети."""
    from eaglewings.ds.pages import LEAD, board
    s = json.loads((ROOT / "data" / "processed" / "experiments_summary.json").read_text())
    alg, champ = {}, {}
    CODE = dict(t2="FG_day", t3="FG_risk", t5="FG_spread", t19="")      # спільні коди в LEAD: три гілки FG; у рубок гілка це група LD, LM, LG
    for k, spec in LEAD.items():
        if k not in s:
            continue
        spec = dict(spec, code=CODE.get(k, spec["code"]))
        b = board(k, s[k]["results"])
        for r in b.itertuples():
            key = "/".join(x for x in (spec["code"].replace(" · ", "-"), r.group, r.name) if x)
            alg[key] = {m: (None if v is None or v != v else round(float(v), 4)) for m, v in (("val", r.val), ("test", r.test), ("lo", r.lo), ("hi", r.hi))}
            from eaglewings.ds import park as _pk
            for pk_ in (_pk.PARKS if len(_pk.PARKS) > 1 else []):       # тест окремо в кожному парку (колонки *_<парк> результатів)
                v_ = r.raw.get(f"{spec['test']}_{pk_}")
                if v_ is not None and v_ == v_:
                    alg[key][f"test_{pk_}"] = round(float(v_), 4)
            if r.champ:
                champ["/".join(x for x in (spec["code"].replace(" · ", "-"), r.group) if x)] = r.name
    ds = {}
    for f in sorted((ROOT / "data" / "datasets").glob("*/versions.json")):
        v = json.loads(f.read_text())[-1]
        ds[f.parent.name] = dict(version=v["version"], digest=v["digest"], rows=v["rows"], **({"changes": v["changes"]} if v.get("changes") else {}))
    return dict(algorithms=alg, champions=champ, datasets=ds)


def history():
    f = RUNS / "history.json"
    return json.loads(f.read_text()) if f.exists() else []


def diff(was, now):
    """Що змінилось між двома знімками."""
    wa, na = was["algorithms"], now["algorithms"]
    changed = []
    for k in sorted(set(wa) & set(na)):
        for m in ["val", "test"] + sorted(x for x in set(wa[k]) | set(na[k]) if x.startswith("test_")):
            a, b = wa[k].get(m), na[k].get(m)
            if (a is None) != (b is None) or (a is not None and abs(a - b) > TOL):
                changed.append((k, m, a, b))
    return dict(changed=changed, new=sorted(set(na) - set(wa)), lost=sorted(set(wa) - set(na)),
                champ=[(g, was["champions"].get(g), now["champions"].get(g)) for g in sorted(set(was["champions"]) | set(now["champions"]))
                       if was["champions"].get(g) != now["champions"].get(g)],
                ds_changed=[(k, was["datasets"][k], now["datasets"][k]) for k in sorted(set(was["datasets"]) & set(now["datasets"]))
                            if was["datasets"][k]["digest"] != now["datasets"][k]["digest"]],
                ds_new=sorted(set(now["datasets"]) - set(was["datasets"])), ds_lost=sorted(set(was["datasets"]) - set(now["datasets"])))


def f4(x):
    return "—" if x is None else f"{x:.4f}".rstrip("0").rstrip(".")


def page(v, now, d, hist):
    """Сторінка версії: різниця з попередньою (або числа першої версії) і журнал усіх версій."""
    L = [f"# Версія прогону {v['n']}", "", f"{v['date']} · прогін {v.get('run') or '—'} · кроки: {', '.join(v.get('steps') or []) or '—'}", ""]
    if d is None:
        L += ["Перша версія: порівнювати ні з чим, нижче її числа.", "", "## Переможці", "", "| Гілка | Переможець |", "|---|---|"]
        L += [f"| {g} | {a} |" for g, a in sorted(now["champions"].items())]
        L += ["", "## Алгоритми", "", "| Гілка / алгоритм | Вибір | Тест | 95 % CI |", "|---|---|---|---|"]
        L += [f"| `{k}` | {f4(x['val'])} | {f4(x['test'])} | {f4(x['lo'])}–{f4(x['hi'])} |" for k, x in sorted(now["algorithms"].items())]
        L += ["", "## Датасети", "", "| Датасет | Версія | Рядків | Відбиток |", "|---|---|---|---|"]
        L += [f"| `{k}` | v{x['version']} | {x['rows']:,} | `{x['digest']}` |".replace(",", " ") for k, x in sorted(now["datasets"].items())]
    else:
        n, p = sum(len(d[k]) for k in ("changed", "new", "lost", "champ", "ds_changed", "ds_new", "ds_lost")), v["n"] - 1
        L += [f"**Без змін порівняно з версією {p}.**" if not n else f"**Змін порівняно з версією {p}: {n}.**", ""]
        if d["champ"]:
            L += ["## Змінились переможці", "", "| Гілка | Було | Стало |", "|---|---|---|"] + [f"| {g} | {a or '—'} | {b or '—'} |" for g, a, b in d["champ"]] + [""]
        if d["changed"]:
            L += ["## Змінились числа", "", "| Гілка / алгоритм | Метрика | Було | Стало | Різниця |", "|---|---|---|---|---|"]
            L += [f"| `{k}` | {'вибір' if m == 'val' else 'тест' if m == 'test' else 'тест · ' + m[5:]} | {f4(a)} | {f4(b)} | {'—' if a is None or b is None else f'{b - a:+.4f}'} |" for k, m, a, b in d["changed"]] + [""]
        if d["new"] or d["lost"]:
            L += ["## Алгоритми, що з'явились або зникли", ""] + [f"- новий `{k}`: тест {f4(now['algorithms'][k]['test'])}" for k in d["new"]] + [f"- зник `{k}`" for k in d["lost"]] + [""]
        if d["ds_changed"]:
            L += ["## Змінились датасети", "", "| Датасет | Версія | Рядків було → стало | Що змінилось |", "|---|---|---|---|"]
            L += [f"| `{k}` | v{a['version']} → v{b['version']} | {a['rows']} → {b['rows']} |" + (f" {'; '.join(b['changes'])} |" if b.get("changes") else " |") for k, a, b in d["ds_changed"]] + [""]
        if d["ds_new"] or d["ds_lost"]:
            L += ["## Датасети, що з'явились або зникли", ""] + [f"- новий `{k}`" for k in d["ds_new"]] + [f"- зник `{k}`" for k in d["ds_lost"]] + [""]
    L += ["", "## Журнал версій", "", "| Версія | Дата | Прогін | Кроки | Змін |", "|---|---|---|---|---|"]
    L += [f"| {h['n']} | {h['date']} | {h.get('run') or '—'} | {', '.join(h.get('steps') or []) or '—'} | {'перша' if h.get('changes') is None else h['changes']} |" for h in reversed(hist)]
    return "\n".join(L)


KIND = dict(m="виміряно на реальних даних", s="сценарний аналіз", x="симуляція")
PART_METRIC = dict(LD="F1 за площею (га)", LM="F1 за площею (га)", LG="частка площі рубок у верхніх 11 % гексів прогнозу")   # спільний запис LEAD гілок рубок


def research_table():
    """Усі алгоритми всіх порахованих гілок однією таблицею для асистента застосунку (data/processed/research_results.csv):
    гілка, питання, тип, група, алгоритм, метрика, вибір, тест, 95 % CI, переможець, гіпотеза. Ті самі board(), що й у версії."""
    import pandas as pd
    from eaglewings.ds.branches import R
    from eaglewings.ds.pages import LEAD, board
    s = json.loads((ROOT / "data" / "processed" / "experiments_summary.json").read_text())
    rows = []
    for k, spec in LEAD.items():
        if k not in s:
            continue
        codes = [c for c, b in R.items() if b.get("summary") == k]
        b = board(k, s[k]["results"])
        for r in b.itertuples():
            code = r.group if r.group in codes else codes[0]
            br = R[code]
            rows.append(dict(branch=code, question=br["q"], kind=KIND.get(br["kind"], br["kind"]), group="" if r.group == code else r.group,
                             algorithm=r.name, metric=PART_METRIC.get(code, spec.get("metric")), lower_is_better=bool(spec.get("low")), how_chosen=spec.get("how"),
                             val=r.val, test=r.test, ci_lo=r.lo, ci_hi=r.hi, winner=bool(r.champ), hypothesis=r.hyp))
    out = ROOT / "data" / "processed" / "research_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def record(run=None, run_id=None, steps=None):
    """Записати нову версію: знімок, різниця з попередньою, журнал. → (сторінка markdown, версія)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    now, hist = snapshot(), history()
    prev = json.loads((RUNS / f"v{hist[-1]['n']}.json").read_text())["numbers"] if hist else None
    d = diff(prev, now) if prev else None
    from eaglewings import BUILT_WITH
    v = dict(n=(hist[-1]["n"] + 1) if hist else 1, built_with=BUILT_WITH, date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), run=run, run_id=run_id, steps=steps or [],
             changes=None if d is None else sum(len(d[k]) for k in ("changed", "new", "lost", "champ", "ds_changed", "ds_new", "ds_lost")))
    (RUNS / f"v{v['n']}.json").write_text(json.dumps(dict(version=v, numbers=now, diff=d), ensure_ascii=False, indent=1))
    hist.append(v)
    (RUNS / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1))
    md = page(v, now, d, hist)
    (RUNS / "latest.md").write_text(md + "\n")
    research_table()
    print(md)
    return md, v


def to_clearml(md, v):
    """Лише з --clearml: числа версії як новий файл датасету run_numbers (версія ClearML = нащадок попередньої) і звіт у Reports."""
    from eaglewings.ds import pages, tracking
    if not tracking.online():
        return
    f = RUNS / f"v{v['n']}.json"
    digest = hashlib.sha256(json.dumps(json.loads(f.read_text())["numbers"], sort_keys=True).encode()).hexdigest()[:16]
    tracking.clearml_dataset(tracking.DATASETS_PROJECT, "run_numbers", digest, [f, RUNS / "history.json"],
                             "Числа кожного прогону потоку: метрики всіх алгоритмів, переможці, датасети. З них рахується, що змінилось між прогонами.",
                             [f"run:{v['n']}"])
    pages.clearml_report(tracking.ROOT_PROJECT, "версії прогонів", md, ["версії прогонів"])


if __name__ == "__main__":
    md, v = record(steps=["вручну"])
    to_clearml(md, v)
