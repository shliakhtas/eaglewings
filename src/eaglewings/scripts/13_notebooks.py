"""Ноутбуки, що повторюють сторінку досліджень → notebooks/*.ipynb (зібрані й виконані). Запуск із кореня проєкту:

    .venv/bin/python scripts/13_notebooks.py            # зібрати й виконати
    .venv/bin/python scripts/13_notebooks.py --no-run   # лише зібрати

Код аналізу не дублюється: ноутбуки викликають ds/report.py, задачі ds/tasks/* і функції EDA зі scripts/10_dataset_eda.py.
"""
import sys
from pathlib import Path

import nbformat as nbf

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds.report import BRANCHES, KIND  # noqa: E402

NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)
SETUP = "import sys\nsys.path.insert(0, '..')\nimport pandas as pd\nfrom eaglewings.ds import report\npd.set_option('display.width', 200)\n%matplotlib inline"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def branch(i, code_, b):
    cells = [md(f"# {code_.replace('_', ' ')} · {b['q']}\n\n**Тип:** {KIND[b['kind']]}. **Навіщо:** {b['why']}\n\nЦей ноутбук повторює блок сторінки досліджень `web/science/#{b['page']}` і бере ті самі результати."),
             code(SETUP),
             md("## Постановка, дані, протокол і метрики\n\nТекст узято з модуля задачі, той самий записується в журнал кожного запуску."),
             code(f"print(report.protocol('{code_}'))"),
             md("## Результат\n\nОбраний підхід виділено. Вибір робився лише на валідації, числа тут з тесту."),
             code(f"table, fig = report.show('{code_}')\ntable")]
    if code_ in ("TP", "CP", "LG"):
        cells += [md("## По роках\n\nПрогресивний протокол: кожен рік бачить лише дані до нього."), code(f"report.by_year('{code_}')")]
    if code_ == "LG":
        cells += [md("## Частка втрат залежно від площі"), code("report.results('t19', 'LG_curve').pivot_table(index='approach', columns='share', values='test_metric').round(3)")]
    nb = nbf.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")))
    p = NB / f"{i:02d}_{code_}.ipynb"
    nbf.write(nb, p)
    return p


def overview():
    rows = "\n".join(f"| [{c.replace('_', ' ')}]({i:02d}_{c}.ipynb) | {b['q']} | {KIND[b['kind']]} |" for i, (c, b) in enumerate(BRANCHES.items(), 1))
    cells = [md("# НП Дадья: ноутбуки дослідження\n\nКожен ноутбук повторює блок сторінки `web/science/` і використовує той самий код: задачі `ds/tasks/`, підсумки `data/processed/experiments_summary.json`, спільні функції `ds/report.py`.\n\n| Гілка | Питання | Тип |\n|---|---|---|\n" + rows + "\n| [Датасети](90_datasets.ipynb) | EDA кожного датасету | |"),
             code(SETUP),
             md("## Усі задачі: обраний підхід на тесті"),
             code("rows = []\nfor c, b in report.BRANCHES.items():\n    t, _ = report.show(c)\n    ch = t[t['champion'].fillna(False).astype(bool)] if 'champion' in t else t.head(0)\n    rows.append(dict(гілка=c, тип=report.KIND[b['kind']], обрано=(ch.iloc[0, 0] if len(ch) else '—')))\npd.DataFrame(rows)")]
    nb = nbf.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")))
    nbf.write(nb, NB / "00_overview.ipynb")


def datasets_nb():
    cells = [md("# Датасети: EDA\n\nТой самий аналіз, що у вкладці «Датасети» сторінки: функція `build` зі `scripts/10_dataset_eda.py`."), code(SETUP),
             code("import json\nnames = [n for g in json.load(open('../web/science/eda.json'))['groups'] for n in g['items']]\nnames")]
    for n in [n for n in ("burned_area_pixels", "fire_day_daily", "h3_risk_cells", "logging_gr_cells") if (ROOT / "data" / "datasets" / n).exists()]:
        cells += [md(f"## `{n}`"), code(f"o = report.dataset_eda('{n}')\nprint(o['short'])\nfor s in o['seen']:\n    print('·', s)\npd.DataFrame(o['columns'])[['n', 'doc', 'role', 'miss', 'uniq', 'min', 'med', 'max']]")]
    cells += [md("Решта датасетів: `report.dataset_eda('<назва>')` для будь-якої назви зі списку вище.")]
    nb = nbf.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")))
    nbf.write(nb, NB / "90_datasets.ipynb")


if __name__ == "__main__":
    ps = [branch(i, c, b) for i, (c, b) in enumerate(BRANCHES.items(), 1)]
    overview()
    datasets_nb()
    ps = [NB / "00_overview.ipynb"] + ps + [NB / "90_datasets.ipynb"]
    if "--no-run" not in sys.argv:
        from nbclient import NotebookClient
        for p in ps:
            nb = nbf.read(p, as_version=4)
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
            nbf.write(nb, p)
            print("виконано", p.name, flush=True)
    print(len(ps), "ноутбуків")
