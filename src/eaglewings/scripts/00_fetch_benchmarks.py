"""Відкриті еталонні набори для перевірки детекторів → data/raw/benchmarks/ (не в git).

    eaglewings script 00_fetch_benchmarks

  dfire   D-Fire (Venâncio et al. 2022): фото диму й вогню з рамками, тестова частина                  → T11 детектори
  vedai   VEDAI (Razakarivony & Jurie 2016): аерофото ~12,5 см/пкс, сільська місцевість, рамки машин   → T11 детектори
  ena24   ENA24 (Yousif et al. 2019, LILA BC): фотопастки, лейбл кадру, тестова частина                → T11 детектори
  effis   EFFIS: офіційні периметри згарищ у рамці території з конфігу                                  → FM, FD
  toptw   TOPTW Righini–Salani (Solomon, Cordeau) + найкращі відомі результати (KU Leuven)            → RS розв'язувач
Качається лише те, що бере увімкнена гілка (config/branches.py). Вже завантажене пропускається. FIgLib (веб-камери, дим) закритий без дозволу власника, тому не використовується.
"""
import json
import subprocess
import zipfile
import sys
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
R = ROOT / "data" / "raw" / "benchmarks"
UA = {"User-Agent": "dadia-research/1.0"}
EFFIS = "https://maps.effis.emergency.copernicus.eu/effis"
from eaglewings.ds.regions import COUNTRY, EFFIS_BBOX, EFFIS_YEARS  # noqa: E402


HF = dict(dfire=("badsaarow/d-fire", ["data/test-*", "README.md"]), vedai=("ckyrkou/vedai", ["vedai.zip", "README.md"]),
          ena24=("mbiarreta/camera-trap-lila-ena24", ["data/test-*", "README.md"]))
KUL = "https://www.mech.kuleuven.be/en/mim/op"


def wanted():
    """Еталони, які бере хоч одна увімкнена гілка: CD — D-Fire і ENA24, SD (машини згори) — VEDAI, RS — TOPTW."""
    from eaglewings.ds import branches as BR
    out = []
    if BR.is_on("CD"):
        out += ["dfire", "ena24"]
    if BR.is_on("SD") and BR.mode("SD", "vehicles") == "vehicles":
        out.append("vedai")
    if BR.is_on("RS"):
        out.append("toptw")
    return out


def hf(names):
    for name in names:
        if name in HF:
            repo, pats = HF[name]
            snapshot_download(repo, repo_type="dataset", local_dir=R / name, allow_patterns=pats)


def toptw():
    """Задачі TOPTW (Righini & Salani; Solomon, Cordeau) і таблиця найкращих відомих результатів KU Leuven."""
    d = R / "toptw"
    (d / "inst").mkdir(parents=True, exist_ok=True)
    for n in ("instances/righiniTOPTW1", "instances/righiniTOPTW2", "instances/righiniTOPTW3", "ResultsTOPTW"):
        f = d / Path(n).name
        if not f.exists():
            r = requests.get(f"{KUL}/{n}", headers=UA, timeout=120)
            r.raise_for_status()
            f.write_bytes(r.content)
        if n.startswith("instances"):
            with zipfile.ZipFile(f) as z:
                for i in z.infolist():
                    if i.filename.endswith(".txt"):
                        (d / "inst" / Path(i.filename).name).write_bytes(z.read(i))


def effis():
    """Згарища EFFIS по роках; рік, який не вдалося скачати, пропускається з поясненням (→ код 3: дані неповні, потік іде далі)."""
    d = R / "effis"
    d.mkdir(exist_ok=True)
    bad = []
    for y in EFFIS_YEARS:
        f = d / f"effis_{y}_{COUNTRY.lower()}.geojson"
        if f.exists():
            continue
        try:
            _effis_year(y, f)
        except Exception as e:
            bad.append(y)
            print(f"НЕ ВДАЛОСЯ: EFFIS {y}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return bad


def _effis_year(y, f):
    r = requests.get(EFFIS, params=dict(service="WFS", version="1.1.0", request="GetFeature", typename=f"ms:modis.ba.poly.{y}",
                                        bbox=f"{EFFIS_BBOX},EPSG:4326", outputFormat="application/json; subtype=geojson"), headers=UA, timeout=300)
    r.raise_for_status()
    json.loads(r.content)
    f.write_bytes(r.content)


if __name__ == "__main__":
    R.mkdir(parents=True, exist_ok=True)
    from eaglewings.ds import pflow, pages
    need = wanted()
    hfn = [n for n in need if n in HF]
    if hfn:
        pflow.mark("benchmarks from Hugging Face: " + ", ".join(hfn))
        hf(hfn)
        pages.downloaded(*hfn)
    if "toptw" in need:
        pflow.mark("TOPTW instances and best known results")
        toptw()
        pages.downloaded("toptw")
    pflow.mark("EFFIS burnt areas")
    bad = effis()
    pages.downloaded("effis")
    print(subprocess.run(["du", "-sh", *map(str, sorted(R.iterdir()))], capture_output=True, text=True).stdout)
    if bad:
        print("Не скачано згарищ EFFIS за роки:", bad, flush=True)
        sys.exit(3)
