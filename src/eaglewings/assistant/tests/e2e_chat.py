"""Наскрізна перевірка чату застосунку в Chrome: ключ → живий режим → питання → відповідь із таблицею → об'єкти на мапі.

    ASSISTANT_TEST_KEY=sk-… python -m eaglewings.assistant.tests.e2e_chat [скріншот.png]

Потрібні запущені сайт (./run.sh, :8766) і асистент (./run_assistant.sh, :8770). Без ключа перевіряє режим готових питань.
З BAD_KEY=1 вводить вигаданий ключ і чекає червоний статус «ключ не працює» й пояснення в чаті.
Ключ береться лише зі змінної середовища й ніде не друкується.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("dg", Path(__file__).resolve().parents[2] / "scripts" / "app_recordings.py")
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)

SITE = os.environ.get("SITE_URL", "http://127.0.0.1:8766")
BAD = os.environ.get("BAD_KEY") == "1"
KEY = "sk-proj-invalid000" if BAD else os.environ.get("ASSISTANT_TEST_KEY", "")
Q = os.environ.get("ASSISTANT_TEST_Q", "Покажи гекси з критичним ризиком")


async def main(shot=None):
    import subprocess
    import shutil
    import tempfile
    import time
    import urllib.request
    import websockets
    port, prof = dg.free_port(), tempfile.mkdtemp()
    p = subprocess.Popen([dg.chrome(), "--headless=new", "--disable-gpu", "--no-sandbox", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                          f"--window-size={dg.W},{dg.H}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = False
    try:
        for _ in range(50):
            try:
                ws_url = next(t["webSocketDebuggerUrl"] for t in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read()) if t.get("type") == "page")
                break
            except Exception:
                time.sleep(.2)
        async with websockets.connect(ws_url, max_size=None) as ws:
            c = dg.CDP(ws)
            await c.call("Page.enable")
            await c.call("Emulation.setDeviceMetricsOverride", width=dg.W, height=dg.H, deviceScaleFactor=1, mobile=False)
            await c.call("Page.navigate", url=f"{SITE}/app.html")
            await asyncio.sleep(4)
            await c.click("[data-panel=p-chat]")
            await asyncio.sleep(.5)
            if KEY:
                await c.js(f"(() => {{ const i = document.querySelector('#chatKey'); i.value = {json.dumps(KEY)}; i.dispatchEvent(new Event('change')); return true; }})()")
                for _ in range(20):
                    await asyncio.sleep(.5)
                    mode = await c.js("(document.querySelector('#chatMode') || {}).textContent || ''") or ''
                    if "живі" in mode or "не запущено" in mode or "не працює" in mode:
                        break
                print("режим:", mode)
                if BAD:
                    last = await c.js("(() => { const m = [...document.querySelectorAll('#chatLog .msg.ai')].pop(); return m ? m.innerText : ''; })()") or ''
                    cls = await c.js("document.querySelector('#chatMode').className")
                    locked = await c.js("document.querySelector('#chatIn').disabled")
                    print("повідомлення:", last.replace("\n", " ")[:160], "· клас:", cls, "· поле вимкнене:", locked)
                    ok = "не працює" in mode and "bad" in cls and "Ключ не підійшов" in last and locked
                    if shot:
                        await c.area()
                        (await c.shot()).save(shot)
                    return ok
                if "живі" not in mode:
                    return False
                await c.js(f"(() => {{ const i = document.querySelector('#chatIn'); i.value = {json.dumps(Q)}; document.querySelector('#chatGo').click(); return true; }})()")
            else:
                await c.click("#chatChips .chip")
            for _ in range(120):
                await asyncio.sleep(1)
                if await c.js("!document.querySelector('#typing')"):
                    break
            last = await c.js("(() => { const m = [...document.querySelectorAll('#chatLog .msg.ai')].pop(); return m ? m.innerText : ''; })()")
            rows = await c.js("(() => { const m = [...document.querySelectorAll('#chatLog .msg.ai')].pop(); return m ? m.querySelectorAll('tbody tr').length : 0; })()")
            geo = await c.js("CHAT_GEO.length ? CHAT_GEO[CHAT_GEO.length - 1].geojson.features.length : 0")
            print("відповідь:", last.replace("\n", " ")[:160])
            print("рядків у таблиці:", rows, "· об'єктів на мапі:", geo)
            chart = await c.js("!!document.querySelector('#chatLog .msg.ai:last-child img')")
            ok = (bool(rows) or chart) and geo > 0
            if shot:
                await asyncio.sleep(2)
                await c.area()
                img = await c.shot()
                img.save(shot)
    finally:
        p.terminate()
        shutil.rmtree(prof, ignore_errors=True)
    print("OK" if ok else "НЕ ПРОЙШЛО")
    return ok


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None)) else 1)
