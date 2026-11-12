"""Застосунок web/app.html з шаблону web/template.html і мапа моніторингу кожного парку (web/map_live.html для основного,
web/map_live_<парк>.html для решти). Дані всіх парків вшиті в сторінку; парк вибирається на сторінці (?park=<ключ>)."""
import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from folium import plugins

from eaglewings.project import ROOT, LIB, template  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds import park as PK  # noqa: E402

P = ROOT / "data" / "processed"                        # тека парку, який зараз збирається (main() перемикає)
V = P / "vision"                                       # фото й відео детекторів спільні для всіх парків
WEB = ROOT / "web"

# палітра лендингу
C = {"fire": "#ff6b2c", "logging": "#e5c13a", "vehicle": "#7fbf5f", "background": "#2a3a2e", "line": "#0e1411"}
LEVEL_OP = {"критична": 0.75, "висока": 0.55, "помірна": 0.35, "фонова": 0.12}


def base_map(park):
    c = park.geometry.iloc[0].centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=11, tiles=None, control_scale=True, prefer_canvas=False)   # SVG: кліки ловить лише фігура, полотно canvas не перекриває нижні шари
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Dark Gray", name="Темна схема", max_zoom=20, max_native_zoom=16).add_to(m)   # далі 16 — розтягуємо, а не сіра порожнеча
    # підписи (села, дороги, річки) поверх темної схеми — щоб на наближенні було видно, де ти
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Підписи", overlay=True, control=False, max_zoom=20, max_native_zoom=16, opacity=0.9).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Супутник", max_zoom=20, show=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Street Map", name="Вулична мапа", max_zoom=19, show=False).add_to(m)
    folium.GeoJson(park, name="Межа парку", style_function=lambda _: {"color": "#cfe8c4", "weight": 2, "fill": False, "dashArray": "6 4"}).add_to(m)
    return m


def areals_layer(m):
    """Суперареали в часі: знімки стану (Python, DBSCAN) кожні 2 дні; сторінка шле індекс знімка з повзунка.
    Кожен тип своїм кольором; пунктир — згаслий; дрібні точки — кластери за 14 днів до обраної дати."""
    tl = json.load(open(P / "timeline.json"))
    ops_ = json.load(open(P / "ops.json"))
    tl.update(feed=ops_["feed"], window_days=ops_["window_days"], contour_days=ops_["contour_days"])
    name = m.get_name()
    m.get_root().header.add_child(folium.Element("<style>.ar-lbl.compact .lbl-full{display:none}"
        ".eg-leg{background:rgba(14,20,17,.92);color:#ece8dc;border:1px solid #26332b;border-radius:8px;padding:8px 10px;font:12px/1.5 Inter,sans-serif}"
        ".eg-leg b{display:block;font:500 10px JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:.05em;color:#8f9a8f;margin:4px 0 2px}"
        ".eg-leg i{display:inline-block;width:14px;height:10px;margin-right:6px;vertical-align:middle;border-radius:2px}"
        ".eg-leg i.pt{width:10px;height:10px;border-radius:50%}"
        ".ar-lbl{font:11px/1.35 JetBrains Mono,monospace;color:#ece8dc;background:rgba(14,20,17,.9);border-radius:6px;padding:3px 7px;"
        "white-space:nowrap;transform:translate(-50%,calc(-100% - 9px));display:inline-block;cursor:pointer;position:relative}"
        ".ar-lbl:hover{filter:brightness(1.35)} .ar-lbl-wrap{background:none;border:0}"
        ".ar-leg{position:absolute;left:50%;bottom:-7px;margin-left:-6px;width:0;height:0;border:6px solid transparent;border-bottom:0;border-top-width:7px}</style>"))
    m.get_root().html.add_child(folium.Element("""<script>
var TL = %s;
window.addEventListener('load', function(){
  var map = %s, KC = {fire:'#ff6b2c', logging:'#e5c13a', vehicle:'#7fbf5f'},
      SC = {'супутник': '#60a5fa', 'камера': '#2dd4bf', 'соцмережі': '#c084fc', 'експерт': '#f472b6'};
  var leg = L.control({position: 'bottomleft'});
  leg.onAdd = function(){ var d = L.DomUtil.create('div', 'eg-leg');
    d.innerHTML = '<b>Ареал — тип</b>' + [['fire','пожежа'],['logging','вирубка'],['vehicle','проїзд']].map(function(x){
        return '<div><i style="border:2px solid ' + KC[x[0]] + ';background:' + KC[x[0]] + '44"></i>' + x[1] + '</div>'; }).join('') +
      '<div><i style="border:2px dashed #aaa"></i>згаслий</div>' +
      '<div><i class="pt" style="background:#aaa;border:2px solid #fff;width:8px;height:8px"></i>поодинока подія (точка)</div>' +
      '<div style="color:#8f9a8f;margin-top:3px">клік по ареалу — спостереження й дії</div>';
    return d; };
  leg.addTo(map);
  var arL = L.layerGroup().addTo(map), clL = L.layerGroup().addTo(map), lbL = L.layerGroup().addTo(map);
  // ареали й контури — нижні шари, точки — верхній (щоб їх завжди можна було навести/клікнути)
  map.createPane('areals').style.zIndex = 400; map.createPane('polys').style.zIndex = 410; map.createPane('pts').style.zIndex = 620;
  // SVG, не canvas: у canvas кожна панель — полотно на всю мапу, і верхнє перехоплює кліки по ареалах під ним
  var R_ = {areals: L.svg({pane: 'areals'}), pts: L.svg({pane: 'pts'})};
  function tip(a){ return '<b>' + a.id + ' · ' + a.type_name + '</b><br>Регулярність: ' + a.regularity + '<br>Джерела: ' + a.coverage +
      '<br>Життєвий цикл: ' + a.lifecycle + '<br>Динаміка: ' + a.dynamics + (a.dyn_change != null ? ' (' + (a.dyn_change > 0 ? '+' : '') + a.dyn_change + '%%)' : '') +
      '<br>Кластерів: ' + a.clusters + ' · сигналів: ' + a.signals + '<br>Період: ' + a.first + ' … ' + a.last +
      '<br>Впевненість: ' + a.confidence_level + '<br>Дії: ' + a.actions.join(', ') + ' — ' + a.rhythm +
      (a.recommendations.length ? '<br><span style="color:#ff6b2c">→ ' + a.recommendations.join('<br>→ ') + '</span>' : '') + '<br><i>клік — детально</i>'; }
  var labels = [];
  function open(a){ return function(){ parent.postMessage({type: 'areal', id: a.id}, '*'); }; }
  // На основній мапі — лише ареали. Ареал з однієї події (поодиноке спостереження) — точка кольору типу;
  // решта — контур. Окремі спостереження й супутникові контури — у вікні ареалу.
  function render(i){
    var s = TL.snapshots[i]; arL.clearLayers(); clL.clearLayers(); lbL.clearLayers(); labels = [];
    s.areals.slice().sort(function(a, b){ return b.area_ha - a.area_ha; }).forEach(function(a){
      var faded = a.lifecycle.indexOf('згас') === 0, single = a.clusters <= 1, anchor;
      if (single) {
        var mk = L.circleMarker([a.lat, a.lon], {pane: 'pts', renderer: R_.pts, radius: 8, color: '#ffffff', weight: 2, opacity: faded ? 0.6 : 1,
          fillColor: KC[a.kind], fillOpacity: faded ? 0.45 : 0.95}).bindTooltip(tip(a)).on('click', open(a)).addTo(arL);
        mk._areal = a.id;
        anchor = L.latLng(a.lat, a.lon);
      } else {
        var gj = L.geoJSON(a.geometry, {pane: 'areals', renderer: R_.areals, style: {color: KC[a.kind], fillColor: KC[a.kind], weight: 2.5,
          fillOpacity: faded ? 0.06 : 0.25, dashArray: faded ? '6 5' : null}}).bindTooltip(tip(a), {sticky: true}).on('click', open(a)).addTo(arL);
        gj._areal = a.id;
        var b = gj.getBounds(); anchor = L.latLng(b.getNorth(), b.getCenter().lng);
      }
      if (single) return;                       // поодинокі події — без підпису (характеристики в підказці, клік — вікно)
      var lm = L.marker(anchor, {interactive: true, zIndexOffset: 1000, icon: L.divIcon({className: 'ar-lbl-wrap', iconSize: [0, 0],
        html: '<div class="ar-lbl" style="border:1px solid ' + KC[a.kind] + (faded ? ';opacity:.65' : '') + '"><b style="color:' + KC[a.kind] + '">' + a.id +
        '</b><span class="lbl-full"> ' + a.regularity + '<br>' + a.coverage + '<br>' + a.lifecycle + ' · ' + a.dynamics + '</span>' +
        '<i class="ar-leg" style="border-top-color:' + KC[a.kind] + '"></i></div>'})}).on('click', open(a)).addTo(lbL);
      labels.push({m: lm, a: a, faded: faded});
    });
    declutter();
  }
  // розведення підписів: за важливістю; повний підпис, що перекриває інший підпис або чужий ареал, згортається до ID,
  // якщо й ID заважає — ховається
  var LV = {'критична': 0, 'висока': 1, 'помірна': 2};
  function declutter(){
    var placed = [], obst = [];
    arL.eachLayer(function(l){ var bb = l.getBounds ? l.getBounds() : L.latLngBounds([l.getLatLng(), l.getLatLng()]);
      var p1 = map.latLngToContainerPoint(bb.getNorthWest()), p2 = map.latLngToContainerPoint(bb.getSouthEast());
      obst.push({id: l.feature ? null : null, layer: l, x1: p1.x - 6, y1: p1.y - 6, x2: p2.x + 6, y2: p2.y + 6}); });
    function hit(r, own, labelsOnly){ return placed.concat(labelsOnly ? [] : obst.filter(function(o){ return o.layer !== own; })).some(function(o){
      return r.x1 < o.x2 && r.x2 > o.x1 && r.y1 < o.y2 && r.y2 > o.y1; }); }
    var mapR = map.getContainer().getBoundingClientRect();
    labels.slice().sort(function(p, q){ return (p.faded - q.faded) || (LV[p.a.level] - LV[q.a.level]); }).forEach(function(L_){
      var el = L_.m.getElement() && L_.m.getElement().querySelector('.ar-lbl'); if (!el) return;
      el.classList.remove('compact'); el.style.visibility = '';
      var own = null; arL.eachLayer(function(l){ if (l._areal === L_.a.id) own = l; });
      function rect(){ var r = el.getBoundingClientRect(); return {x1: r.left - mapR.left, y1: r.top - mapR.top, x2: r.right - mapR.left, y2: r.bottom - mapR.top}; }
      var r = rect();
      if (hit(r, own)) { el.classList.add('compact'); r = rect(); }
      if (hit(r, own, true)) { el.style.visibility = 'hidden'; return; }   // стислий чип може зачіпати чужий ареал, але не інший підпис
      placed.push(r);
    });
  }
  map.on('zoomend moveend', declutter);
  window.addEventListener('message', function(ev){ if (ev.data && ev.data.type === 'time') render(ev.data.idx); });
  render(TL.snapshots.length - 1);
  parent.postMessage({type: 'ready', map: 'areals'}, '*');
});
</script>""" % (json.dumps(tl, ensure_ascii=False), name)))


def build_map(mode, park, ar, cl, ops, firms, roads):
    """Мапа вікна 2: суперареали в часі + задачі реагування."""
    m = base_map(park)
    areals_layer(m)
    fg = folium.FeatureGroup(name="01 · Задачі реагування")
    for t in ops["response"]:
        folium.CircleMarker([t["lat"], t["lon"]], radius=11, color="#ffffff", weight=3, fill=True, fill_color=C[t["kind"]], fill_opacity=0.8,
                            tooltip=f'{t["id"]} · {t["title"]} · {t["priority"]} · впевненість {t["confidence_level"]}').add_to(fg)
    fg.add_to(m)
    if len(roads):
        folium.GeoJson(roads[roads.highway == "track"][["highway", "geometry"]], name="Лісові треки (OSM)", show=False,
                       style_function=lambda _: {"color": "#e5c13a", "weight": 1, "opacity": 0.5}).add_to(m)
    finish_map(m, mode)


KIND_C = {"fire": "#ff6b2c", "smoke": "#ff6b2c", "vehicle": "#22d3ee", "person": "#e5c13a"}


SRC_C = {"супутник": "#60a5fa", "камера": "#2dd4bf", "фотопастка": "#e2e8f0", "соцмережі": "#c084fc", "гаряча лінія / експерт": "#f472b6"}


def build_live_map(park, ops, video, mode="live"):
    """Мапа вікна 1 (екстрене реагування). Шари вмикаються/вимикаються кліком по легенді:
    камери (із сектором огляду), фотопастки, сигнали за 72 год, події, виявлення з супутника (центроїди), зони знімків."""
    m = base_map(park)
    now = pd.Timestamp(ops["now"])
    data = dict(devices=ops["devices"], response=ops["response"], scenes=ops["sat_scenes"],
                feed=[n for n in ops["feed"] if n["live"] and (now - pd.Timestamp(n["ts"])).total_seconds() <= ops["fresh_days"] * 86400],
                sat=[n for n in ops["feed"] if n["source"] == "супутник" and n["kind"] in ("fire", "logging") and (now - pd.Timestamp(n["ts"])).days <= 14])
    live_cl = {n["cluster"] for n in data["feed"]}
    data["members"] = {c: [dict(lat=n["lat"], lon=n["lon"], source=n["source"], text=n["text"], ts=n["ts"], kind=n["kind"], count=n["count"])
                           for n in ops["feed"] if n["cluster"] == c] for c in live_cl}
    data["events"] = {t["cluster"]: dict(title=t["title"], kind=t["kind"], confidence=t["confidence_level"], signals=t["signals"]) for t in ops["response"]}
    data["timeline"] = json.load(open(P / "timeline.json"))["snapshots"]
    data["routes"] = dict(teams=[], handoff=[], night=[], deferred=[])
    data["routes_tl"] = []
    m.get_root().header.add_child(folium.Element("<style>"
        ".eg-leg{background:rgba(14,20,17,.93);color:#ece8dc;border:1px solid #26332b;border-radius:8px;padding:8px 10px;font:12px/1.5 Inter,sans-serif;width:232px;box-sizing:border-box}"
        ".eg-leg .hd{font:500 10px JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:.05em;color:#ff6b2c;margin-bottom:4px}"
        ".eg-leg .lg-row{display:flex;flex-wrap:nowrap;align-items:center;gap:7px;padding:3px 4px;margin:0;border-radius:4px;cursor:pointer;user-select:none;white-space:nowrap}"
        ".eg-leg .lg-row:hover{background:#1b241e}"
        ".eg-leg .lg-row.off{opacity:.38;text-decoration:line-through}"
        ".eg-leg .lg-row>span:nth-child(2){flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis}"
        ".eg-leg .lg-row .eye{flex:none;width:16px;text-align:center;font-size:11px;color:#8f9a8f}"
        ".eg-leg .sw{display:inline-block;width:12px;height:12px;flex:none}"
        ".eg-rleg .lg-row{white-space:normal;align-items:flex-start;line-height:1.35}"
        ".lg-pin{flex:none;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50% 50% 50% 0;rotate:-45deg;border:2px solid #38bdf8;color:#0e1411;font:800 10px Inter,sans-serif}"
        ".lg-pin.done{opacity:.45;filter:saturate(.4)} .lg-pin.new{box-shadow:0 0 0 3px rgba(229,193,58,.45)}"
        ".eg-rleg .lg-key span{white-space:normal}"
        ".eg-leg .lg-key{margin-top:7px;padding-top:6px;border-top:1px solid #26332b;display:grid;grid-template-columns:1fr 1fr;gap:2px 10px;font-size:11.5px}"
        ".eg-leg .lg-key b{grid-column:1/-1;font:500 10px JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:.05em;color:#8f9a8f}"
        ".eg-leg .lg-key span{white-space:nowrap} .eg-leg .lg-key i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}"
        ".dev{font:600 10px JetBrains Mono,monospace;color:#0e1411;border-radius:5px;padding:2px 5px;white-space:nowrap;transform:translate(-50%,-50%);display:inline-block;border:1.5px solid #0e1411}"
        "@keyframes pulse{0%{transform:translate(-50%,-50%) scale(.6);opacity:.9}100%{transform:translate(-50%,-50%) scale(1.8);opacity:0}}"
        "@keyframes wave{0%{transform:translate(-50%,-50%) scale(.7);opacity:1}100%{transform:translate(-50%,-50%) scale(1.6);opacity:0}}"
        ".wave{width:26px;height:26px;border-radius:50%;border:2.5px solid #ff6b2c;transform:translate(-50%,-50%);animation:wave 1.5s ease-out infinite;pointer-events:none}"
        ".rt-n{min-width:22px;height:22px;padding:0 4px;border-radius:11px;color:#0e1411;font:700 11px/22px JetBrains Mono,monospace;text-align:center;transform:translate(-50%,-50%);border:2px solid #0e1411;box-sizing:border-box}"
        ".pin{position:relative;transform:translate(-50%,-100%);width:30px;height:30px;border-radius:50% 50% 50% 0;rotate:-45deg;background:var(--c);border:2.5px solid var(--t);box-shadow:0 2px 6px rgba(0,0,0,.5);translate:0 -4px}"
        ".pin .sym{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;rotate:45deg;font:800 15px Inter,sans-serif;color:#0e1411}"
        ".pin .no{position:absolute;left:24px;top:-12px;rotate:45deg;background:var(--t);color:#0e1411;border-radius:8px;padding:0 5px;height:16px;font:700 10px/16px JetBrains Mono,monospace;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,.5)}"
        ".pin.done{opacity:.45;filter:saturate(.4)}"
        "@keyframes pinp{0%{box-shadow:0 0 0 0 var(--c)}100%{box-shadow:0 0 0 14px transparent}}"
        ".pin.new{animation:pinp 1.3s infinite}"
        ".night-dot{position:absolute;width:10px;height:10px;border-radius:50%;background:var(--c);border:2px solid #ece8dc;transform:translate(-50%,-50%)}"
        ".night{position:absolute;left:10px;top:6px;background:rgba(14,20,17,.85);color:#ece8dc;border-left:3px solid var(--c);border-radius:4px;padding:1px 6px;font:600 10.5px Inter,sans-serif;white-space:nowrap}"
        ".rt-base{background:#ece8dc;color:#0e1411;font:700 11px JetBrains Mono,monospace;padding:3px 7px;border-radius:6px;transform:translate(-50%,-50%);white-space:nowrap;border:2px solid #0e1411}"
        ".scene-tip{background:rgba(14,20,17,.9);color:#fff;border:1px dashed #fff;font:11px JetBrains Mono,monospace}"
        "</style>"))
    m.get_root().html.add_child(folium.Element("""<script>
var LIVE = %s;
window.addEventListener('load', function(){
  var map = %s, SC = %s, KC = {fire: '#ff6b2c', logging: '#e5c13a', vehicle: '#7fbf5f'}, TUA = {fire: 'пожежа', logging: 'вирубка', vehicle: 'проїзд'};
  map.createPane('sect').style.zIndex = 400; map.createPane('pts').style.zIndex = 620;
  var R_ = {sect: L.svg({pane: 'sect'}), pts: L.svg({pane: 'pts'})};
  var G = {areals: L.layerGroup(), cam: L.layerGroup(), trap: L.layerGroup(), alarm: L.layerGroup(), sat: L.layerGroup(), zones: L.layerGroup()};
  // камери (+ сектор огляду) і фотопастки
  LIVE.devices.forEach(function(d){
    var cam = d.source === 'camera', g = cam ? G.cam : G.trap;
    if (cam) {
      var rk = d.video ? 2.2 : 0.5, pts = [[d.lat, d.lon]];
      for (var a = d.bearing - 30; a <= d.bearing + 30; a += 6) pts.push([d.lat + rk / 111 * Math.cos(a * Math.PI / 180), d.lon + rk / (111 * Math.cos(d.lat * Math.PI / 180)) * Math.sin(a * Math.PI / 180)]);
      L.polygon(pts, {pane: 'sect', renderer: R_.sect, color: '#0f766e', weight: 1, fillColor: '#2dd4bf', fillOpacity: 0.15, interactive: false}).addTo(g);
    }
    L.marker([d.lat, d.lon], {zIndexOffset: 500, icon: L.divIcon({iconSize: [0, 0], html: '<div class="dev" style="background:' + (cam ? '#2dd4bf' : '#e2e8f0') + '">' +
      (cam ? '◉ ' : '▣ ') + d.id + '</div>'})}).bindTooltip((cam ? (d.video ? 'Камера-вежа з відеострімом' : 'Камера') : 'Фотопастка') + ' ' + d.id)
      .on('click', function(){ parent.postMessage({type: 'device', id: d.id}, '*'); }).addTo(g);
  });
  // події (без окремих сигналів): хвиля, що блимає, у центрі сигналів події; якщо сигнали рознесені — тонке кільце навколо їхньої зони
  map.createPane('ev').style.zIndex = 610;
  var Rev = L.svg({pane: 'ev'}), waves = [];
  // події на обрану повзунком дату (знімки таймлайну): хвиля, що блимає; тонке кільце, якщо сигнали події рознесені
  function renderEvents(i){
    G.alarm.clearLayers(); waves = [];
    (LIVE.timeline[i].events || []).forEach(function(t){
      var c = L.latLng(t.lat, t.lon), r = t.r || 0;
      var tip = '<b>' + t.title + '</b><br>' + TUA[t.kind] + ' · ' + t.priority + ' · впевненість ' + t.confidence_level + '<br>' + t.detected +
                '<br>джерела: ' + t.sources.join(' + ') + ' · сигналів: ' + t.signals + (t.devices.length ? '<br>пристрій: ' + t.devices.join(', ') : '');
      if (r > 20) {
        r += 40;
        L.circle(c, {radius: r, pane: 'ev', renderer: Rev, color: KC[t.kind], weight: 1.5, fill: true, fillColor: KC[t.kind], fillOpacity: 0.08})
          .bindTooltip(tip).on('click', function(e){ L.DomEvent.stopPropagation(e); if (t.areal) parent.postMessage({type: 'areal', id: t.areal}, '*'); }).addTo(G.alarm);
      }
      L.circleMarker(c, {pane: 'ev', renderer: Rev, radius: 6, color: '#ffffff', weight: 1.5, fillColor: KC[t.kind], fillOpacity: 1})
        .bindTooltip(tip + (t.areal ? '<br><i>клік — ареал ' + t.areal + '</i>' : '')).on('click', function(e){ L.DomEvent.stopPropagation(e);
          if (t.areal) parent.postMessage({type: 'areal', id: t.areal}, '*'); else map.flyTo(c, 15, {duration: 0.7}); }).addTo(G.alarm);
      var w = L.marker(c, {interactive: false, icon: L.divIcon({iconSize: [0, 0], html: '<div class="wave" style="border-color:' + KC[t.kind] + '"></div>'})}).addTo(G.alarm);
      waves.push({m: w, c: c, r: r});
    });
    setTimeout(sizeWaves, 0);
  }
  function sizeWaves(){ waves.forEach(function(w){
    var el = w.m.getElement() && w.m.getElement().querySelector('.wave'); if (!el) return;
    var px = map.latLngToContainerPoint(w.c).distanceTo(map.latLngToContainerPoint(L.latLng(w.c.lat, w.c.lng + w.r / (111320 * Math.cos(w.c.lat * Math.PI / 180)))));
    var d = Math.max(30, 2 * px + 6); el.style.width = el.style.height = d + 'px'; }); }
  map.on('zoomend', sizeWaves); G.alarm.on('add', function(){ setTimeout(sizeWaves, 0); });
  // ареали в часі: знімки стану (Python, DBSCAN) — сторінка шле індекс з повзунка дат; клік — вікно ареалу
  map.createPane('areals').style.zIndex = 450;   // над зонами знімків і секторами камер, під подіями й точками
  var Rar = L.svg({pane: 'areals'}), arIdx = LIVE.timeline.length - 1, arLayers = {};
  function arTip(a){ return '<b>' + a.id + ' · ' + a.type_name + '</b><br>' + a.regularity + ' · ' + a.lifecycle + '<br>' + a.coverage + '<br>' + a.dynamics +
      (a.dyn_change != null ? ' (' + (a.dyn_change > 0 ? '+' : '') + a.dyn_change + '%%)' : '') + '<br><i>клік — детально</i>'; }
  function renderAreals(i){
    arIdx = i; G.areals.clearLayers(); arLayers = {};
    LIVE.timeline[i].areals.slice().sort(function(a, b){ return b.area_ha - a.area_ha; }).forEach(function(a){
      var faded = a.lifecycle.indexOf('згас') === 0, open = function(){ parent.postMessage({type: 'areal', id: a.id}, '*'); }, lay;
      if (a.clusters <= 1) lay = L.circleMarker([a.lat, a.lon], {pane: 'areals', renderer: Rar, radius: 8, color: '#fff', weight: 2, fillColor: KC[a.kind], fillOpacity: faded ? 0.45 : 0.9});
      else lay = L.geoJSON(a.geometry, {pane: 'areals', renderer: Rar, style: {color: KC[a.kind], fillColor: KC[a.kind], weight: 2.5, fillOpacity: faded ? 0.06 : 0.22, dashArray: faded ? '6 5' : null}});
      lay.bindTooltip(arTip(a), {sticky: true}).on('click', function(e){ L.DomEvent.stopPropagation(e); open(); }).addTo(G.areals);
      arLayers[a.id] = lay;
    });
  }
  renderAreals(arIdx);
  var arHl = null;
  function focusAreal(id){
    var lay = arLayers[id]; if (!lay) return;
    if (arHl) { map.removeLayer(arHl); arHl = null; }
    var b = lay.getBounds ? lay.getBounds() : L.latLngBounds([lay.getLatLng(), lay.getLatLng()]);
    arHl = L.rectangle(b.pad(0.25), {color: '#ffffff', weight: 2, dashArray: '6 5', fill: false, interactive: false}).addTo(map);
    map.flyToBounds(b.pad(0.6), {duration: 0.7, maxZoom: 15});
  }


  // виявлення з супутника (14 дн.) — статичні центроїди кольору типу
  LIVE.sat.forEach(function(n){
    L.circleMarker([n.lat, n.lon], {pane: 'pts', renderer: R_.pts, radius: 6, color: '#111827', weight: 2, fillColor: KC[n.kind], fillOpacity: 1})
      .bindTooltip('<b>' + n.text + '</b><br>' + n.ts + (n.area_ha ? ' · контур ' + n.area_ha + ' га' : '') + '<br>центроїд контуру, знайденого супутником')
      .on('click', function(){ showScene({poly: n.poly, title: n.text + ' · ' + n.ts.slice(5)}); }).addTo(G.sat);
  });
  // зони всіх знімків
  LIVE.scenes.forEach(function(S, i){
    L.rectangle(S.bounds, {color: '#ffffff', weight: 1.5, dashArray: '4 6', fill: false}).bindTooltip('ЗН-' + (i + 1) + ' · ' + S.title).addTo(G.zones);
  });
  ['areals', 'zones', 'cam', 'trap', 'alarm', 'sat'].forEach(function(k){ G[k].addTo(map); });
  renderEvents(LIVE.timeline.length - 1);   // зони знімків — сховані за замовчуванням

  // легенда = перемикач шарів
  var ROWS = [['areals', '<span class="sw" style="border:2px solid #e5c13a;background:#e5c13a44;border-radius:2px"></span>', 'ареали · 90 дн.'],
              ['cam', '<span class="dev" style="background:#2dd4bf;transform:none">◉ CAM</span>', 'камери'],
              ['trap', '<span class="dev" style="background:#e2e8f0;transform:none">▣ TRAP</span>', 'фотопастки'],
              ['alarm', '<span class="sw" style="border-radius:50%%;border:2.5px solid #ff6b2c"></span>', 'події (блимають)'],
              ['sat', '<span class="sw" style="border-radius:50%%;background:linear-gradient(90deg,#ff6b2c 50%%,#e5c13a 0);border:2px solid #111827"></span>', 'виявлено супутником'],
              ['zones', '<span class="sw" style="border:1.5px dashed #fff"></span>', 'зони знімків']];
  var leg = L.control({position: 'bottomleft'});
  leg.onAdd = function(){
    var d = L.DomUtil.create('div', 'eg-leg');
    d.innerHTML = '<div class="hd">шари · клік — показати / сховати</div>' + ROWS.map(function(r){
      return '<div class="lg-row' + (map.hasLayer(G[r[0]]) ? '' : ' off') + '" data-k="' + r[0] + '">' + r[1] + '<span>' + r[2] + '</span><span class="eye">' +
        (map.hasLayer(G[r[0]]) ? '👁' : '—') + '</span></div>'; }).join('') +
      '<div class="lg-key"><b>Подія / виявлення — тип</b>' + [['#ff6b2c', 'пожежа'], ['#e5c13a', 'вирубка'], ['#7fbf5f', 'проїзд']].map(function(x){
        return '<span><i style="background:' + x[0] + '"></i>' + x[1] + '</span>'; }).join('') + '</div>';
    L.DomEvent.disableClickPropagation(d);
    d.addEventListener('click', function(e){
      var r = e.target.closest('.lg-row'); if (!r) return; var g = G[r.dataset.k];
      if (map.hasLayer(g)) map.removeLayer(g); else g.addTo(map);
      var on = map.hasLayer(g); r.classList.toggle('off', !on); r.querySelector('.eye').textContent = on ? '👁' : '—';
    });
    return d;
  };
  leg.addTo(map);

  // обраний знімок / контур з вкладки «Супутник» — білим пунктиром
  var scene = null, chatLayer = null;
  function showScene(dt){
    if (scene) { map.removeLayer(scene); scene = null; }
    if (!dt || (!dt.bounds && !dt.poly && !dt.lat)) return;
    var st = {color: '#ffffff', weight: 2.5, dashArray: '8 6', fill: true, fillOpacity: 0.06, interactive: false};
    scene = L.featureGroup().addTo(map);
    if (dt.bounds) L.rectangle(dt.bounds, (dt.poly || dt.lat) ? Object.assign({}, st, {weight: 1.5, dashArray: '4 6', fillOpacity: 0.02}) : st).addTo(scene);
    if (dt.poly) L.polygon(dt.poly.map(function(p){ return [p[1], p[0]]; }), Object.assign({}, st, {fillOpacity: 0.18})).addTo(scene);
    else if (dt.lat && dt.bounds) L.circle([dt.lat, dt.lon], Object.assign({radius: 90}, st, {fillOpacity: 0.18})).addTo(scene);
    var lay = scene.getLayers();
    lay[lay.length - 1].bindTooltip(dt.title, {permanent: true, direction: 'top', className: 'scene-tip'}).openTooltip();
    map.flyToBounds(scene.getBounds().pad(dt.bounds && !dt.poly && !dt.lat ? 0.6 : 0.3), {duration: 0.7, maxZoom: 17});
  }
  window.addEventListener('message', function(ev){
    if (!ev.data) return;
    if (ev.data.type === 'time') { renderAreals(ev.data.idx); renderEvents(ev.data.idx); }
    if (ev.data.type === 'arealfocus') focusAreal(ev.data.id);
    if (ev.data.type === 'mode') setMode(ev.data.mode);
    if (ev.data.type === 'resize') map.invalidateSize();
    if (ev.data.type === 'chatgeo') {                // відповідь чату: GeoDataFrame → GeoJSON → полігони й точки (замінює попередню)
      if (mode !== 'chat') setMode('chat');
      if (chatLayer) map.removeLayer(chatLayer);
      chatLayer = L.geoJSON(ev.data.geojson, {
        style: {color: '#c084fc', weight: 2.5, dashArray: '6 4', fillColor: '#c084fc', fillOpacity: 0.18},
        pointToLayer: function(f, ll){ return L.circleMarker(ll, {radius: 7, color: '#fff', weight: 1.5, fillColor: '#c084fc', fillOpacity: 0.95}); },
        onEachFeature: function(f, l){ var p = f.properties || {}; l.bindTooltip(Object.keys(p).slice(0, 7).map(function(k){ return '<b>' + k + '</b>: ' + String(p[k]).slice(0, 60); }).join('<br>')); }
      }).addTo(map);
      if (chatLayer.getLayers().length) map.flyToBounds(chatLayer.getBounds().pad(0.8), {duration: 0.7, maxZoom: 12});   // з контекстом: межа парку, села
    }
    if (ev.data.type === 'layer') {                  // розділ сторінки вмикає/вимикає шар (напр. маршрути)
      var g = G[ev.data.k]; if (!g) return;
      if (ev.data.on && !map.hasLayer(g)) g.addTo(map); if (!ev.data.on && map.hasLayer(g)) map.removeLayer(g);
      var row = document.querySelector('.lg-row[data-k="' + ev.data.k + '"]');
      if (row) { row.classList.toggle('off', !ev.data.on); row.querySelector('.eye').textContent = ev.data.on ? '👁' : '—'; }
    }
    if (ev.data.type === 'scene') showScene(ev.data);
    if (ev.data.type === 'focus') {                 // елемент зі списку ліворуч — пунктиром кольору типу
      if (scene) { map.removeLayer(scene); scene = null; }
      scene = L.featureGroup().addTo(map);
      L.circle([ev.data.lat, ev.data.lon], {radius: 140, color: KC[ev.data.kind] || '#fff', weight: 2.5, dashArray: '6 5', fillColor: KC[ev.data.kind] || '#fff', fillOpacity: 0.1, interactive: false})
        .bindTooltip(ev.data.title, {permanent: true, direction: 'top', className: 'scene-tip'}).addTo(scene);
      map.flyTo([ev.data.lat, ev.data.lon], 15, {duration: 0.7});
    }
  });
  // зняти вибір: клік по порожньому місцю мапи або Esc — і повідомити сторінку, щоб зняла підсвітку в списку
  function clearScene(){ if (arHl) { map.removeLayer(arHl); arHl = null; } if (!scene) return; showScene(null); parent.postMessage({type: 'scene-cleared'}, '*'); }
  map.on('click', clearScene);
  var MON = ['areals', 'zones', 'cam', 'trap', 'alarm', 'sat'], monState = {}, mode = 'monitor';
  // режими мапи: monitor — шари моніторингу; chat — порожня мапа, лише відповіді агента.
  // Кожен режим пам'ятає свій вигляд (центр і масштаб); об'єкти відповіді чату ховаються при виході й повертаються при поверненні.
  var views = {};
  function setMode(m){
    if (m === mode) return;
    var prev = mode; mode = m;
    views[prev] = {c: map.getCenter(), z: map.getZoom()};
    if (prev === 'monitor') MON.forEach(function(k){ monState[k] = map.hasLayer(G[k]); if (monState[k]) map.removeLayer(G[k]); });
    if (prev === 'chat' && chatLayer) map.removeLayer(chatLayer);
    clearScene();
    leg.getContainer().style.display = m === 'monitor' ? '' : 'none';
    if (m === 'monitor') MON.forEach(function(k){ if (monState[k] !== false) G[k].addTo(map); });
    if (m === 'chat' && chatLayer) chatLayer.addTo(map);
    if (views[m]) map.setView(views[m].c, views[m].z, {animate: false});            // туди, де залишили
  }
  parent.postMessage({type: 'ready', map: 'live'}, '*');
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') clearScene(); });
});
</script>""" % (json.dumps(data, ensure_ascii=False, default=str), m.get_name(), json.dumps(SRC_C, ensure_ascii=False))))
    finish_map(m, mode)


def finish_map(m, mode):
    plugins.MousePosition(position="bottomleft", separator=", ", prefix="lat, lon:",
                          lat_formatter="function(n){return n.toFixed(5)}", lng_formatter="function(n){return n.toFixed(5)}").add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(primary_length_unit="kilometers", primary_area_unit="hectares").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    # приймач команд від сторінки: перелетіти до координат / накласти маску SamGeo
    name = m.get_name()
    m.get_root().html.add_child(folium.Element(f"""
<script>
window.addEventListener('load', function(){{
  var map = {name}, pin = null, ov = null;
  map.on('click', function(e){{ parent.postMessage({{type:'mapclick', lat:e.latlng.lat, lon:e.latlng.lng}}, '*'); }});
  window.addEventListener('message', function(ev){{
    var d = ev.data || {{}};
    if (d.type === 'fly') {{
      map.flyTo([d.lat, d.lon], d.zoom || 15, {{duration: 0.8}});
      if (pin) map.removeLayer(pin);
      pin = L.circleMarker([d.lat, d.lon], {{radius: 14, color: '#ffffff', weight: 3, fill: false}}).addTo(map);
      if (d.label) pin.bindPopup(d.label).openPopup();
    }}
    if (d.type === 'overlay') {{
      if (ov) map.removeLayer(ov);
      if (d.url) ov = L.imageOverlay(d.url, d.bounds, {{opacity: 0.8}}).addTo(map);
    }}
  }});
}});
</script>"""))
    m.save(WEB / f"map_{mode}.html")


def app_ops(ops):
    """Застосунок лише показує моніторинг, тому планові дані на сторінку не потрапляють."""
    ops["routes"] = dict(teams=[], handoff=[], night=[], deferred=[])
    ops["routes_tl"], ops["plan"], ops["response"] = [], [], []
    for a in ops["areals"]:
        for k in ("actions", "one_time", "systemic", "post", "team", "every_days", "rhythm"):
            a.pop(k, None)
        a["recommendations"] = []
    return ops


def build_park(key, photos, video):
    """Дані й мапа моніторингу одного парку."""
    global P
    P = PK.processed(key)
    park = gpd.read_file(P / "park.geojson")
    ops = app_ops(json.load(open(P / "ops.json")))
    mode = "live" if key == PK.MAIN else f"live_{key}"
    build_live_map(park, ops, video, mode)
    kpi = dict(signals=ops["signals"], signals_real=ops["signals_real"], clusters=ops["clusters"], areals=len(ops["areals"]),
               response=len(ops["response"]), plan=len(ops["plan"]))
    c = park.geometry.iloc[0].centroid
    tl = json.load(open(P / "timeline.json"))["snapshots"]
    for snap in tl:
        for a in snap.get("areals", []):
            for k in ("actions", "one_time", "systemic", "post", "team", "every_days", "rhythm"):
                a.pop(k, None)
            a["recommendations"] = []
    info = PK.ALL[key]
    return dict(key=key, name=info["name"], short=info["short"], main=key == PK.MAIN, map=f"map_{mode}.html",
                tl=tl, ops=ops, kpi=kpi, park_center=[c.y, c.x], extra=json.load(open(P / "app_extra.json")))


def main():
    photos = json.load(open(V / "photos.json"))
    video = json.load(open(V / "video.json"))
    parks = {k: build_park(k, photos, video) for k in PK.PARKS}
    data = dict(parks=parks, order=list(parks), main=PK.MAIN, photos=photos, video=video)
    tpl = template("app.html").read_text()
    (WEB / "app.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=str)))
    print("web/app.html:", {k: v["kpi"] for k, v in parks.items()})


if __name__ == "__main__":
    main()
