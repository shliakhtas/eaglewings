"""Готові відповіді чату застосунку → web/assets/app_answers.json. Рахуються зі стрічки подій і ареалів застосунку (scripts/03_signals.py), без LLM.

    eaglewings script 14_app_answers        # усі парки: web/assets/app_answers.json (основний), app_answers_<парк>.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import park as PK  # noqa: E402


def gj(ids=None, pts=None):
    if ids is not None:
        g = G.loc[[i for i in ids if i in G.index]]
        return json.loads(g[["geometry"]].assign(id=g.index).to_json())
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]}, "properties": {}} for r in pts.itertuples()]}


def chart(wk, en=False):
    """Графік сигналів по тижнях за типом → PNG як data URI (той самий формат, що в живого асистента)."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"fire": ("#ff6b2c", "fire" if en else "пожежа"), "logging": ("#e5c13a", "logging" if en else "вирубка"), "vehicle": ("#7fbf5f", "vehicle trip" if en else "проїзд")}
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=110, facecolor="#141c17")
    ax.set_facecolor("#141c17")
    for k, (c, n) in C.items():
        if k in wk:
            ax.plot(wk.index, wk[k], color=c, lw=2, marker="o", ms=3, label=n)
    ax.tick_params(colors="#8f9a8f", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#26332b")
    ax.grid(color="#26332b", lw=.6)
    ax.set_ylabel("signals per week" if en else "сигналів за тиждень", color="#8f9a8f", fontsize=8)
    ax.legend(frameon=False, labelcolor="#ece8dc", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def table(df, geo):
    n = len(df)
    df = df.head(50).astype(object).where(pd.notna(df.head(50)), None)
    return dict(text=f"Знайдено {n} рядків. Показано на мапі: {len(geo['features'])} об'єктів.", type="dataframe", table=df.to_dict("records"), rows=n, geojson=geo)


for key in PK.PARKS:                                   # той самий набір готових питань для кожного парку зі свого потоку
    P = PK.processed(key)
    O = json.load(open(P / "ops.json"))
    NOW = pd.Timestamp(O["now"])
    A = pd.DataFrame(O["areals"])
    F = pd.DataFrame(O["feed"])
    F["ts"] = pd.to_datetime(F.ts)
    G = gpd.read_file(P / "areals.geojson").set_index("id")
    exp = A[(A.kind == "logging") & A.dynamics.str.contains("розширю", na=False)][["id", "lat", "lon", "dyn_change", "area_ha"]]
    faded = A[A.lifecycle.str.startswith("згас", na=False)][["id", "type_name", "lifecycle", "regularity", "coverage", "signals", "area_ha", "lat", "lon"]]
    week = F[(F.source == "фотопастка") & (F.ts > NOW - pd.Timedelta(days=7)) & (F.ts <= NOW)]
    night = F[F.source.str.contains("гаряча лінія", na=False) & ((F.ts.dt.hour >= 18) | (F.ts.dt.hour < 8))][["ts", "lat", "lon", "areal", "text"]].assign(ts=lambda d: d.ts.astype(str))
    wk = F[(F.ts > NOW - pd.Timedelta(days=O["window_days"])) & (F.ts <= NOW)].assign(w=lambda d: d.ts.dt.to_period("W").dt.start_time)
    wk = wk.groupby(["w", "kind"]).size().unstack(fill_value=0)
    out = {
        "Покажи ареали вирубки, що розширюються": table(exp, gj(exp.id)),
        "Які ареали згасли?": table(faded, gj(faded.id)),
        "Скільки сигналів від фотопасток за тиждень?": dict(type="number", text=str(int(week["count"].fillna(1).sum()) if "count" in week else len(week)), geojson=gj(pts=week), geo_source=True),
        "Де були нічні дзвінки на гарячу лінію?": table(night, gj(pts=night)),
        "Як змінювались сигнали по тижнях за типом?": dict(type="chart", text=f"Сигнали за {O['window_days']} днів по тижнях: пожежі, вирубки й проїзди. Потік сценарний.",
                                                          chart=chart(wk), chart_en=chart(wk, en=True)),
    }
    if (P / "routes.json").exists():                    # маршрути груп і таски (гілка RD, симуляція)
        R = json.load(open(P / "routes.json"))
        T = pd.DataFrame(json.load(open(P / "tasks.json"))["tasks"])
        td = R["today"]
        st = pd.DataFrame([dict(team=tm["name"], order=i, eta=x["eta"], title=x["title"], type=x["type"], status=x["status"], lat=x["lat"], lon=x["lon"])
                           for tm in td["teams"] for i, x in enumerate(tm["stops"], 1)])
        rer = [tm for tm in td["teams"] if tm.get("rerouted")]
        out["Куди сьогодні їде кожна група?"] = dict(table(st, gj(pts=st)), text=f"План на {td['date']} (симуляція): " + "; ".join(
            f"{tm['name']} — {len(tm['stops'])} зупинок, {tm['km']} км, повернення {tm['end']}" for tm in td["teams"]) +
            (f". Маршрут {', '.join(t_['name'] for t_ in rer)} перебудовано о {rer[0]['at']} через нову подію." if rer else "."))
        tq = T[T.status == "new"][["date", "time", "type_ua", "title", "team", "areal", "lat", "lon"]]
        out["Які таски ще не виконані?"] = dict(table(tq, gj(pts=tq)), text=f"Невиконаних тасок: {len(tq)} (симуляція). За типом: " +
                                              ", ".join(f"{k} — {v}" for k, v in tq.type_ua.value_counts().items()) + ".")
    name = "app_answers.json" if key == PK.MAIN else f"app_answers_{key}.json"
    (ROOT / "web" / "assets" / name).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(key, {k: v.get("rows", v["text"]) for k, v in out.items()})
