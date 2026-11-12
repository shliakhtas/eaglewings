"""Додати переклади в web/i18n/en.json: python scripts/i18n_merge.py <файл.json з {укр: англ}>.
Рядок із числами, нормалізованими в {0}, {1}…, іде в patterns, решта — в exact."""
import json
import sys
from pathlib import Path

from eaglewings.project import ROOT  # noqa: E402
from eaglewings.project import TEMPLATES  # noqa: E402
D = (TEMPLATES / "i18n" / "en.json") if "--lib" in sys.argv else (ROOT / "i18n" / "en.json")
d = json.loads(D.read_text()) if D.exists() else {"exact": {}, "patterns": {}}
new = json.loads(Path([a for a in sys.argv[1:] if not a.startswith('--')][0]).read_text())
for k, v in new.items():
    (d["patterns"] if "{0}" in k else d["exact"])[k] = v
D.write_text(json.dumps(d, ensure_ascii=False, indent=0, sort_keys=True))
print(f"{D}: exact {len(d['exact'])}, patterns {len(d['patterns'])}")
