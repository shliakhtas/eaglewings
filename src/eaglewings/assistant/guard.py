"""Перевірка коду, який написала модель, перед виконанням.

Дозволено лише аналіз даних: pandas, numpy, matplotlib і стандартні помічники. Без файлів, мережі, змінних
середовища й інтроспекції. Порушення повертається моделі як помилка її коду, і PandasAI дає їй переписати.
"""
import ast

POLICY_MARK = "[policy]"

ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.ticker",
    "matplotlib.dates", "datetime", "math", "statistics", "re", "json",
    "collections", "itertools", "calendar", "decimal", "textwrap", "string",
}

FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "input", "breakpoint",
    "help", "memoryview", "exit", "quit", "__builtins__", "__loader__",
    "__spec__",
}

# pd.read_csv(...), df.to_sql(...), pd.io..., plt.imread(...)
FORBIDDEN_ATTRIBUTE_PREFIXES = ("read_", "to_sql", "to_pickle", "to_parquet", "to_feather", "to_hdf", "to_stata")
FORBIDDEN_ATTRIBUTES = {
    "io", "imread", "system", "popen", "environ", "load", "loads", "eval",
    "fromfile", "tofile", "loadtxt", "genfromtxt", "memmap", "save", "savez",
    "savetxt", "format_map",
}


class GuardError(Exception):
    """Код порушує правила: текст іде назад моделі, тому він каже, що робити натомість."""

    def __init__(self, message):
        super().__init__(f"{POLICY_MARK} {message}")


def check_code(code):
    """GuardError, якщо код виходить за межі аналізу даних."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise GuardError(f"Code does not parse: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise GuardError("Relative imports are not allowed.")
            _check_import(node.module or "")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise GuardError(f"`{node.id}` is not allowed - work only with execute_sql_query results.")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise GuardError("Strings containing `__` are not allowed.")       # "{0.__init__.__globals__}".format(f)
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if name.startswith("__") and name.endswith("__"):
                raise GuardError(f"Access to `{name}` is not allowed.")
            if name.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES) or name in FORBIDDEN_ATTRIBUTES:
                raise GuardError(f"`.{name}` is not allowed: data comes only from execute_sql_query, and nothing is read from or written to files.")


def _check_import(module):
    if module not in ALLOWED_IMPORTS and module.split(".")[0] not in {"pandas", "numpy"}:
        raise GuardError(f"Import of `{module}` is not allowed. Use only pandas, numpy, matplotlib, datetime, math and the standard data helpers.")
