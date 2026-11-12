"""Мапи парків для першого екрана лендінгу продукту → web/assets/landing/park_<парк>.jpg і слайди в web/landing/blocks/hero.html.

    eaglewings script landing_parks

Для кожного парку з ds/regions.py: супутниковий знімок Esri World Imagery під рамку парку (тайли кешуються в теці парку),
поза межею парку притемнено, втрати лісу 2023–24 за Hansen помаранчевим, теплові точки FIRMS 2021–2025 жовтим.
Числа під мапою беруться з web/science/parks.json (той самий джерельний файл, що й сторінка дослідження «По парках»).
Нічого не рахує: лише малює вже пораховане.
"""
import io
import json
import math
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import park as PK  # noqa: E402

OUT = ROOT / "web" / "assets" / "landing"
HERO = ROOT / "web" / "landing" / "blocks" / "hero.html"
W, H = 1150, 1000                                    # пропорція панелі першого екрана (1,15)
TILE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ORANGE, YELLOW = (255, 107, 44), (255, 209, 102)


# роки на слайдах: за замовчуванням пожежа 2023 у Фракії; інший проєкт задає landing=dict(loss_years=…, firms_years=…) у config/regions.py
from eaglewings.ds.regions import REGION as _R  # noqa: E402
_L = _R["fetch"].get("landing", {})
_YR = lambda ys: f"{ys[0]}–{str(ys[-1])[2:]}" if len(ys) > 1 and ys[0] // 100 == ys[-1] // 100 else (str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}")
LOSS_Y = list(_L.get("loss_years", [2023, 2024]))
FIRMS_Y = list(_L.get("firms_years", [2023]))
LOSS_T, FIRMS_T = _YR(LOSS_Y), _YR(FIRMS_Y)
_FY = list(_R["fetch"]["firms"]["years"])
YEARS_T = f"{_FY[0]}–{_FY[-1]}"

def merc(lat, lon, z):
    """Градуси → пікселі Web Mercator на зумі z."""
    n = 256 * 2 ** z
    return (lon + 180) / 360 * n, (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n


def frame(bounds):
    """Рамка з пропорцією W:H довкола межі парку з полями, і зум, на якому вона ~W пікселів завширшки."""
    w, s, e, n = bounds
    for z in range(15, 6, -1):
        x0, y0 = merc(n, w, z)
        x1, y1 = merc(s, e, z)
        if (x1 - x0) * 1.12 <= W * 1.6 and (y1 - y0) * 1.12 <= H * 1.6:
            break
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.12, (y1 - y0) * 1.12
    if bw / bh > W / H:
        bh = bw * H / W
    else:
        bw = bh * W / H
    return z, cx - bw / 2, cy - bh / 2, bw, bh


def mosaic(key, z, X, Y, BW, BH):
    """Знімок Esri під рамку: тайли з кешу парку або з мережі."""
    cache = PK.cache(key) / "esri_tiles"
    cache.mkdir(parents=True, exist_ok=True)
    tx0, ty0, tx1, ty1 = int(X // 256), int(Y // 256), int((X + BW) // 256), int((Y + BH) // 256)
    img = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (12, 18, 16))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            f = cache / f"{z}_{tx}_{ty}.jpg"
            if not f.exists():
                r = requests.get(TILE.format(z=z, x=tx, y=ty), timeout=30, headers={"User-Agent": "research-map-renderer"})
                r.raise_for_status()
                f.write_bytes(r.content)
            img.paste(Image.open(f).convert("RGB"), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = X - tx0 * 256, Y - ty0 * 256
    return img.crop((round(ox), round(oy), round(ox + BW), round(oy + BH))).resize((W, H), Image.LANCZOS)


def rings(geom):
    g = geom["coordinates"]
    polys = g if geom["type"] == "MultiPolygon" else [g]
    return [p[0] for p in polys]


def render(key):
    P = PK.processed(key)
    pk = json.load(open(P / "park.geojson"))["features"][0]["geometry"]
    lons = [x for r in rings(pk) for x, _ in r]
    lats = [y for r in rings(pk) for _, y in r]
    z, X, Y, BW, BH = frame((min(lons), min(lats), max(lons), max(lats)))
    sx, sy = W / BW, H / BH
    px = lambda lat, lon: ((merc(lat, lon, z)[0] - X) * sx, (merc(lat, lon, z)[1] - Y) * sy)

    base = mosaic(key, z, X, Y, BW, BH)
    base = ImageEnhance.Color(base).enhance(.8)
    inside = Image.new("L", (W, H), 0)
    for r in rings(pk):
        ImageDraw.Draw(inside).polygon([px(la, lo) for lo, la in r], fill=255)
    dark = ImageEnhance.Brightness(base).enhance(.38)
    img = Image.composite(ImageEnhance.Brightness(base).enhance(.82), dark, inside).convert("RGBA")

    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    loss = [f for f in json.load(open(P / "hansen_loss.geojson"))["features"] if f["properties"]["year"] in LOSS_Y + [y - 2000 for y in LOSS_Y]]
    for f in loss:                                    # пікселі втрати лісу за роки LOSS_Y
        x, y = px(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])
        d.rectangle((x - 1, y - 1, x + 1, y + 1), fill=ORANGE + (150,))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    firms = json.load(open(P / "firms.geojson"))["features"]
    pts = [px(f["properties"]["latitude"], f["properties"]["longitude"]) for f in firms]
    for x, y in pts:
        g.ellipse((x - 7, y - 7, x + 7, y + 7), fill=YELLOW + (70,))
    glow = glow.filter(ImageFilter.GaussianBlur(5))
    for x, y in pts:                                  # теплові точки FIRMS
        d.ellipse((x - 2.2, y - 2.2, x + 2.2, y + 2.2), fill=YELLOW + (240,))
    edge = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    e = ImageDraw.Draw(edge)
    for r in rings(pk):                               # межа парку зі світлою обводкою
        xy = [px(la, lo) for lo, la in r]
        e.line(xy + xy[:1], fill=(0, 0, 0, 150), width=6, joint="curve")
        e.line(xy + xy[:1], fill=(236, 244, 238, 235), width=2, joint="curve")
    img = Image.alpha_composite(Image.alpha_composite(Image.alpha_composite(img, glow), lay), edge).convert("RGB")
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"park_{key}.jpg", quality=84, optimize=True)
    return len(loss), len(pts)


def num(v):
    return f"{round(v):,}".replace(",", " ")


def slides():
    """Слайди першого екрана: мапа й числа кожного парку з web/science/parks.json."""
    S = json.load(open(ROOT / "web" / "science" / "parks.json"))["parks"]
    out = []
    for k in PK.PARKS:
        s = S[k]
        loss = sum(s["hansen_loss_ha"].get(str(y), 0) for y in LOSS_Y)
        out.append(f'''          <figure class="slide" data-park="{k}">
            <div class="top"><i></i><i></i><i></i><span>{s["name"]} · втрати лісу {LOSS_T} і теплові точки</span></div>
            <div class="stack"><img src="assets/landing/park_{k}.jpg" alt="{s["name"]}: супутниковий знімок, втрати лісу {LOSS_T} і теплові точки FIRMS" loading="lazy"></div>
            <div class="kpis">
              <div class="kpi"><b>{num(loss)} га</b><span>втрат лісу {LOSS_T}</span></div>
              <div class="kpi"><b>{num(sum(s["firms_by_year"].get(str(y), 0) for y in FIRMS_Y))}</b><span>теплових точок {FIRMS_T}</span></div>
              <div class="kpi"><b>{num(s["area_ha"])} га</b><span>площа парку</span></div>
            </div>
          </figure>''')
    dots = "".join(f'<button class="dot" data-i="{i}" aria-label="{S[k]["short"]}"></button>' for i, k in enumerate(PK.PARKS))
    return ('      <div class="panel parks" aria-label="Парки дослідження: мапи й числа">\n        <div class="slides" tabindex="0" aria-roledescription="карусель">'
            '<button class="nav prev" aria-label="Попередній парк">‹</button><button class="nav next" aria-label="Наступний парк">›</button>\n        <div class="track">\n' + "\n".join(out) +
            f'\n        </div></div>\n        <div class="lg"><span><i style="background:#ff6b2c"></i>втрати лісу {LOSS_T} (Hansen)</span><span><i style="background:#ffd166;border-radius:50%"></i>теплові точки FIRMS {YEARS_T}</span></div>'
            f'\n        <div class="dots">{dots}</div>\n      </div>')


if __name__ == "__main__":
    for k in PK.PARKS:
        print(k, f"втрат {LOSS_T} пікселів, точок FIRMS:", render(k), flush=True)
    s = HERO.read_text()
    if "<!-- parks:start -->" not in s:                 # шапка без каруселі парків (проєкт обрав інший перший екран): мапи є, слайди не вставляємо
        print("web/landing/blocks/hero.html: міток parks:start / parks:end немає — слайди не вставлено")
        raise SystemExit(0)
    a, b = s.index("      <!-- parks:start -->"), s.index("<!-- parks:end -->")
    HERO.write_text(s[:a] + "      <!-- parks:start -->\n" + slides() + "\n      " + s[b:])
    print("web/landing/blocks/hero.html: слайди", ", ".join(PK.PARKS))
