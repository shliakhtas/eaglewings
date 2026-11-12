"""NASA OPERA DIST-ALERT (порушення рослинності, HLS, ~30 м) для парку через відкритий NASA GIBS, без логіна Earthdata.

Сирі GeoTIFF (LP DAAC) вимагають акаунт, а GIBS віддає той самий шар VEG-DIST-STATUS як PNG. Палітра дискретна й публічна
(OPERA_Vegetation_Disturbance_Status.xml), браузерного змішування немає, тому колір однозначно декодується назад у статус.
Шар на дату — мозаїка останнього відомого статусу кожного пікселя; знімки кешуються в data/raw/nasa/carp/opera/.

Статус:  0 порушення немає · 1 перша детекція < 50 % · 2 попередньо < 50 % · 3 підтверджено < 50 % · 4 перша детекція ≥ 50 % · 5 попередньо ≥ 50 %
         6 підтверджено ≥ 50 % · 7 підтверджено < 50 %, завершено · 8 підтверджено ≥ 50 %, завершено · 255 немає даних
         (відсоток — падіння рослинного покриву відносно багаторічної норми)
"""
import io
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image

from . import common as K

WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
LAYER = "OPERA_L3_DIST-ALERT-HLS_Color_Index"
PALETTE = {(18, 18, 18): 0, (0, 85, 85): 1, (137, 127, 78): 2, (222, 224, 67): 3, (0, 136, 136): 4, (228, 135, 39): 5, (224, 27, 7): 6, (119, 119, 119): 7, (221, 221, 221): 8}
DIR = K.CACHE / "opera"
TILE = 1600
UA = {"User-Agent": "dadia-research/1.0"}


def _get(bbox, w, h, date):
    p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS=LAYER, CRS="EPSG:4326", BBOX=",".join(f"{v:.6f}" for v in bbox), WIDTH=w, HEIGHT=h, FORMAT="image/png", TRANSPARENT="TRUE", TIME=date)
    err = ""
    for k in range(6):                                  # сервіс інколи рве з'єднання: це не привід губити весь прогін
        try:
            r = requests.get(WMS, params=p, headers=UA, timeout=180)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            err = f"{r.status_code} {r.text[:150]}"
        except requests.RequestException as e:
            err = str(e)[:150]
        time.sleep(2 ** k)
    raise RuntimeError(f"GIBS {date}: {err}")


def decode(rgba):
    """RGBA → статус. Прозоре = немає даних; будь-який інший колір означав би змішування, тому це помилка."""
    out = np.full(rgba.shape[:2], 255, np.uint8)
    code = rgba[..., 0].astype(np.uint32) << 16 | rgba[..., 1].astype(np.uint32) << 8 | rgba[..., 2]
    op = rgba[..., 3] > 0
    seen = np.zeros(rgba.shape[:2], bool)
    for (r, g, b), v in PALETTE.items():
        m = op & (code == (r << 16 | g << 8 | b))
        out[m] = v
        seen |= m
    if (op & ~seen).any():
        raise ValueError(f"невідомі кольори: {(op & ~seen).sum()} пікселів")
    return out


def snapshot(date, key=None):
    """Статус на дату на сітці Hansen парку key (тайлами, бо сервіс обмежує розмір кадру); кеш у теці парку."""
    d = K._park.cache(key or K._park.KEY) / "opera"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{date}.npz"
    if f.exists():
        return np.load(f)["s"]
    B = K.base(key)
    out = np.full((B.H, B.W), 255, np.uint8)
    for r0 in range(0, B.H, TILE):
        for c0 in range(0, B.W, TILE):
            r1, c1 = min(r0 + TILE, B.H), min(c0 + TILE, B.W)
            w, e, n, s = B.TR.c + c0 * B.TR.a, B.TR.c + c1 * B.TR.a, B.TR.f + r0 * B.TR.e, B.TR.f + r1 * B.TR.e
            out[r0:r1, c0:c1] = decode(_get((s, w, n, e), c1 - c0, r1 - r0, date))
    np.savez_compressed(f, s=out)
    return out


def dates(year, start="06-01", end="10-15", step=10):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(f"{year}-{start}", f"{year}-{end}", freq=f"{step}D")]
