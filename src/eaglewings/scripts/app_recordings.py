"""Анімовані записи застосунку для лендінгу: web/assets/app/app_<сцена>.webp і постер .jpg. Запуск із теки eaglewings:

    eaglewings script app_recordings               # сцени блоку записів лендінгу (web/landing/views.json)
    eaglewings script app_recordings camera chat   # вибрані

Скрипт сам піднімає сайт (http.server на web/), відкриває web/app.html у Chrome без вікна, у кожній сцені
натискає те, що натиснула б людина, знімає кадри й складає з них анімований WebP. Нічого не рахує: лише
записує те, що вже показує застосунок, тому записи завжди відповідають поточній версії застосунку.
"""
import asyncio
import base64
import contextlib
import functools
import http.server
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import websockets
from PIL import Image

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
WEB = ROOT / "web"
OUT = WEB / "assets" / "app"
W, H, FPS = 1600, 1000, 8
AREA = "#view-vision"                                         # робоча зона застосунку: розділи, панель і мапа поруч

# сцена: що натиснути перед записом, що натиснути під час запису, скільки секунд писати
SCENES = {
    "camera": dict(before=["[data-panel=p-monitor]", "[data-tab=v-video]"], during=[(0.2, "js:document.querySelector('#v-video video, video').play()")], secs=8),
    "sat": dict(before=["[data-panel=p-monitor]", "[data-tab=v-sat]"], during=[(1.0, "#satEvents [data-satev]"), (4.0, "#satGallery figure")], secs=6),
    "areals": dict(before=["[data-panel=p-monitor]", "[data-tab=v-areals]"],                 # історія змін: повзунок на різні дати, мапа й таблиця — стан того дня
                   during=[(0.5, "js:(() => { const s = document.getElementById('tSlider'); s.value = Math.round(s.max * 0.15); s.dispatchEvent(new Event('input')); })()"), (2.8, "js:(() => { const s = document.getElementById('tSlider'); s.value = Math.round(s.max * 0.45); s.dispatchEvent(new Event('input')); })()"), (5.1, "js:(() => { const s = document.getElementById('tSlider'); s.value = Math.round(s.max * 0.75); s.dispatchEvent(new Event('input')); })()"), (7.4, "js:(() => { const s = document.getElementById('tSlider'); s.value = Math.round(s.max * 1); s.dispatchEvent(new Event('input')); })()")], secs=9.5),
    "satsync": dict(before=["[data-panel=p-monitor]", "[data-tab=v-sat]"],               # клік у точку супутникового знімка → ця точка на мапі
                    during=[(0.4, "js:document.querySelectorAll('#satGallery figure')[0].click()"),
                            (2.4, "js:(() => { const i = document.getElementById('satBigImg'); i.scrollIntoView({block: 'center'}); const r = i.getBoundingClientRect(); i.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: r.left + r.width * 0.32, clientY: r.top + r.height * 0.38})); })()"),
                            (5.0, "js:(() => { const i = document.getElementById('satBigImg'); i.scrollIntoView({block: 'center'}); const r = i.getBoundingClientRect(); i.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: r.left + r.width * 0.68, clientY: r.top + r.height * 0.55})); })()"),
                            (7.6, "js:(() => { const i = document.getElementById('satBigImg'); i.scrollIntoView({block: 'center'}); const r = i.getBoundingClientRect(); i.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: r.left + r.width * 0.45, clientY: r.top + r.height * 0.78})); })()")], secs=10),
    "areal": dict(before=["[data-panel=p-monitor]", "[data-tab=v-areals]"],                # ареал розкривається: мапа ареалу, журнал, спостереження, історія
                  area="body", during=[(0.6, "#zoneBody [data-detail]"), (3.6, "js:document.querySelectorAll('#mSide .obs [data-k]')[0].click()"),
                          (5.6, "js:document.querySelectorAll('#mSide .obs [data-k]')[3].click()"), (7.4, "#mHist")], secs=10.5),
    "routes": dict(before=["[data-panel=p-routes]"],                                       # маршрут для кожної групи: своя мапа, клік по зупинці, повзунок днів
                   during=[(1.5, "#routesBody [data-rt]"), (3.5, "js:(s=>{s.value=Math.max(0,+s.max-6);s.oninput()})(document.querySelector('#rtSlider'))"),
                           (5.5, "js:(s=>{s.value=s.max;s.oninput()})(document.querySelector('#rtSlider'))")], secs=8),
    "tasks": dict(before=["[data-panel=p-tasks]"], during=[(1.5, "#taskChips .chip")], secs=5),   # усі задачі в одній таблиці, фільтри за типом
    "chat": dict(before=["[data-panel=p-chat]"], during=[(0.5, "#chatChips .chip"), (2.2, "#chatChips .chip:last-child")], secs=9),   # графік видно більшу частину запису
    "history": dict(before=["[data-panel=p-chat]"],                                           # попередні відповіді → мапа показує кожну
                    during=[(0.3, "#chatChips .chip:nth-child(1)"), (1.6, "#chatChips .chip:nth-child(4)"), (2.9, "#chatChips .chip:nth-child(2)"),
                            (4.6, "#chatLog .geo-btn[data-geo='0']"), (6.6, "#chatLog .geo-btn[data-geo='1']"), (8.6, "#chatLog .geo-btn[data-geo='2']")], secs=10.5),
    "analytics": dict(before=["[data-panel=p-analytics]"],                                  # пожежа 2023 по годинах, потім інші вкладки мапи результатів
                      during=[(0.8, "#anPlay"), (4.6, "#anMapTabs [data-k=risk]"), (7.2, "#anMapTabs [data-k=towers]"), (9.8, "#anMapTabs [data-k=burn]")], secs=12),
}


def chrome():
    for c in ("google-chrome", "chromium", "chromium-browser"):
        if shutil.which(c):
            return shutil.which(c)
    sys.exit("немає Chrome чи Chromium")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def site():
    """Локальний сервер для web/, щоб застосунок завантажив свої JSON."""
    port = free_port()
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), lambda *a: h(*a))
    srv.RequestHandlerClass.log_message = lambda *a: None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


class CDP:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        i = self.n
        await self.ws.send(json.dumps(dict(id=i, method=method, params=params)))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == i:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    async def js(self, code):
        r = await self.call("Runtime.evaluate", expression=code, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def click(self, sel):
        if sel.startswith("js:"):                                    # дія, яку не зробити кліком (наприклад, запустити відео)
            await self.js(f"(async () => {{ try {{ await {sel[3:]}; }} catch (e) {{}} return true; }})()")
            return True
        return await self.js(f"(() => {{ const e = document.querySelector({json.dumps(sel)}); if (!e) return false; e.scrollIntoView({{block:'nearest'}}); e.click(); return true; }})()")

    async def area(self, sel=AREA):
        """Прямокутник робочої зони застосунку на сторінці (у пікселях документа); sel — інша зона, напр. body для вікна поверх застосунку."""
        b = await self.js(f"(() => {{ const e = document.querySelector({json.dumps(sel)}); e.scrollIntoView({{block:'start'}}); const r = e.getBoundingClientRect(); return [r.left + scrollX, r.top + scrollY, r.width, Math.min(r.height, {H - 40})]; }})()")
        self.clip = dict(x=b[0], y=b[1], width=b[2], height=b[3], scale=.75)

    async def shot(self):
        r = await self.call("Page.captureScreenshot", format="jpeg", quality=82, clip=self.clip, captureBeyondViewport=False)   # True перекомпоновує сторінку й скидає прокрутку вкладених панелей (чат)
        return Image.open(io.BytesIO(base64.b64decode(r["data"]))).convert("RGB")


async def record(url, names):
    port = free_port()
    prof = tempfile.mkdtemp()
    p = subprocess.Popen([chrome(), "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", f"--remote-debugging-port={port}",
                          f"--user-data-dir={prof}", f"--window-size={W},{H}", "--autoplay-policy=no-user-gesture-required", "about:blank"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import urllib.request
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
                break
            except Exception:
                time.sleep(.2)
        async with websockets.connect(ws_url, max_size=None) as ws:
            c = CDP(ws)
            await c.call("Page.enable")
            await c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            OUT.mkdir(parents=True, exist_ok=True)
            for name in names:
                sc = SCENES[name]
                await c.call("Page.navigate", url=f"{url}/app.html")
                await asyncio.sleep(4)                                   # мапа, тайли й дані застосунку
                for sel in sc["before"]:
                    if not await c.click(sel):
                        print(f"{name}: не знайшов {sel}", flush=True)
                    await asyncio.sleep(.8)
                await c.area(sc.get("area", AREA))
                frames, t0, todo = [], time.time(), list(sc["during"])
                while time.time() - t0 < sc["secs"]:
                    el = time.time() - t0
                    while todo and el >= todo[0][0]:
                        sel = todo.pop(0)[1]
                        if not await c.click(sel):
                            print(f"{name}: не знайшов {sel}", flush=True)
                    frames.append(await c.shot())
                    await asyncio.sleep(max(0, 1 / FPS - .02))
                if not frames:
                    continue
                frames[0].save(OUT / f"app_{name}.webp", save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, quality=70, method=4)
                frames[min(len(frames) - 1, len(frames) * 2 // 3)].save(OUT / f"app_{name}_poster.jpg", quality=85)
                kb = (OUT / f"app_{name}.webp").stat().st_size // 1024
                print(f"app_{name}.webp: {len(frames)} кадрів, {kb} КБ", flush=True)
    finally:
        p.terminate()
        shutil.rmtree(prof, ignore_errors=True)


def landing_scenes():
    """Сцени блоку записів лендінгу (web/landing/views.json): їх і знімаємо, щоб блок і записи не розходились."""
    f = WEB / "landing" / "views.json"
    if not f.exists():
        return list(SCENES)
    want = [v["scene"] for v in json.loads(f.read_text())["views"]]
    unknown = [n for n in want if n not in SCENES]
    if unknown:
        sys.exit(f"у views.json є сцени без опису в SCENES: {', '.join(unknown)} — додайте, що натиснути в застосунку")
    return want


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if a in SCENES] or landing_scenes()
    with site() as url:
        asyncio.run(record(url, names))
