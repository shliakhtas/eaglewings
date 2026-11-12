"""Комп'ютерний зір: YOLO (детекція авто, людей, вогню, диму) + SamGeo LangSAM (сегментація за текстом).

- фото з парку      → YOLO-World (open-vocabulary: person, car, truck, fire, smoke)
- відеострім        → YOLO-World по кадрах, анотоване відео H.264
- супутникові знімки → YOLO11-OBB (DOTA: small/large vehicle) по тайлах + LangSAM за текстовими запитами
Результат: data/processed/vision/*.json, зображення/відео для сторінки, detections.geojson для ризик-моделі.
"""
import json
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, ExifTags

from eaglewings.project import ROOT, LIB  # noqa: E402  (тека проєкту й код бібліотеки)
from eaglewings.ds.regions import LAT0, COUNTRY, EFFIS_BBOX, EFFIS_YEARS, LOG_YEARS  # noqa: E402  (параметри території з конфігу)
P = ROOT / "data" / "processed"
MEDIA = ROOT / "data" / "raw" / "media"
V = P / "vision"
WEB = ROOT / "web" / "assets"
for d in (V, WEB):
    d.mkdir(parents=True, exist_ok=True)
NOW = pd.Timestamp("2026-10-19 11:40")

WORLD_CLASSES = ["person", "car", "truck", "motorcycle", "fire", "smoke"]
KIND = {"person": "person", "car": "vehicle", "truck": "vehicle", "motorcycle": "vehicle",
        "small vehicle": "vehicle", "large vehicle": "vehicle", "fire": "fire", "smoke": "smoke"}
COLORS = {"person": (255, 200, 0), "vehicle": (0, 200, 255), "fire": (0, 60, 255), "smoke": (180, 180, 180)}

# фото й відео для демо детекторів — з конфігу проєкту (<проєкт>/config/media.py: PHOTOS, VIDEO)
from eaglewings.project import config as _config  # noqa: E402
_media = _config("media")
PHOTOS = list(getattr(_media, "PHOTOS", []))
VIDEO = getattr(_media, "VIDEO", None)
from eaglewings.ds.regions import REGION as _REGION  # noqa: E402
_main = next(iter(_REGION["fetch"].get("parks", {}).values()), {})
CAM_LABEL = (VIDEO or {}).get("label") or _main.get("name_en", _REGION["key"]).upper()[:40]   # підпис у кадрі (латиницею: cv2 не пише кирилицю)


def fetch_media():
    """Скачати фото й відео з config/media.py, яких ще немає в data/raw/media/ (Wikimedia Commons, за url сторінки файлу)."""
    import time
    import urllib.parse
    import urllib.request
    MEDIA.mkdir(parents=True, exist_ok=True)
    for m in PHOTOS + ([VIDEO] if VIDEO else []):
        dst = MEDIA / m["file"]
        if dst.exists() and dst.stat().st_size > 0:
            continue
        name = urllib.parse.unquote(m["url"].rsplit("File:", 1)[-1])
        src = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(name)
        req = urllib.request.Request(src, headers={"User-Agent": "EagleWings/0.1 (research pipeline)"})
        for attempt in range(7):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception as e:  # 429 від Commons: почекати Retry-After (або дедалі довше) і повторити
                if attempt == 6:
                    raise SystemExit(f"медіа {m['file']}: не вдалося скачати {src}: {e}")
                wait = getattr(e, "headers", None) and e.headers.get("Retry-After")
                wait = int(wait) if wait and str(wait).isdigit() else 30 * (attempt + 1)
                print(f"медіа {m['file']}: {e}; чекаю {wait} с")
                time.sleep(wait)
        dst.write_bytes(data)
        print(f"медіа: {m['file']} ← {src} ({len(data) // 1024} КБ)")
        time.sleep(5)


fetch_media()
SAT_PROMPTS = ["tree", "road", "car", "building", "bare soil", "burned area"]


def draw(img, dets):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        c = COLORS[d["kind"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, max(2, img.shape[1] // 600))
        txt = f'{d["label"]} {d["conf"]:.2f}'
        fs = max(0.5, img.shape[1] / 2200)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), c, -1)
        cv2.putText(out, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def world_model():
    from ultralytics import YOLOWorld
    m = YOLOWorld("yolov8x-worldv2.pt")
    m.set_classes(WORLD_CLASSES)
    return m


def yolo_dets(res, names, min_conf):
    out = []
    for b in res.boxes:
        conf = float(b.conf)
        label = names[int(b.cls)]
        if conf >= min_conf.get(label, 0.25):
            out.append(dict(label=label, kind=KIND[label], conf=round(conf, 3), box=[round(float(v), 1) for v in b.xyxy[0]]))
    return out


PHOTO_CONF = {"person": 0.25, "car": 0.3, "truck": 0.3, "motorcycle": 0.3, "fire": 0.3, "smoke": 0.3}
VIDEO_CONF = {"person": 0.3, "car": 0.35, "truck": 0.35, "motorcycle": 0.35, "fire": 0.12, "smoke": 0.12}


def exif_date(path):
    try:
        ex = Image.open(path)._getexif() or {}
        tags = {ExifTags.TAGS.get(k): v for k, v in ex.items()}
        s = tags.get("DateTimeOriginal") or tags.get("DateTime")
        return str(pd.to_datetime(s, format="%Y:%m:%d %H:%M:%S")) if s else None
    except Exception:
        return None


def run_photos(model):
    out = []
    for ph in PHOTOS:
        img = cv2.imread(str(MEDIA / ph["file"]))
        res = model.predict(img, imgsz=1280, conf=0.1, verbose=False)[0]
        dets = yolo_dets(res, res.names, PHOTO_CONF)
        ann = draw(img, dets)
        h, w = ann.shape[:2]
        s = 1600 / max(h, w)
        if s < 1:
            ann = cv2.resize(ann, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            for d in dets:
                d["box_web"] = [v * s for v in d["box"]]
        name = "photo_" + Path(ph["file"]).stem + ".jpg"
        cv2.imwrite(str(WEB / name), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append({**ph, "img": "assets/" + name, "date": exif_date(MEDIA / ph["file"]),
                    "detections": dets, "counts": pd.Series([d["kind"] for d in dets]).value_counts().to_dict()})
        print(ph["file"], out[-1]["counts"])
    json.dump(out, open(V / "photos.json", "w"), ensure_ascii=False, indent=1)
    return out


FIRE_PROMPT = "smoke plume . fire . flame"


def dino_dets(sam, img_bgr, prompt=FIRE_PROMPT, box_th=0.25, text_th=0.2):
    """Детекція за текстовим запитом (GroundingDINO з samgeo) + жадібний NMS по IoU."""
    boxes, logits, phrases = sam.predict_dino(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), prompt, box_th, text_th)
    cands = sorted(zip(boxes.tolist(), logits.tolist(), phrases), key=lambda c: -c[1])
    keep = []

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
    for box, conf, ph in cands:
        if all(iou(box, k[0]) < 0.5 for k in keep):
            keep.append((box, conf, ph))
    out = []
    for box, conf, ph in keep:
        kind = "smoke" if "smoke" in ph else "fire"
        out.append(dict(label=ph or kind, kind=kind, conf=round(float(conf), 3), box=[round(float(v), 1) for v in box], model="GroundingDINO"))
    return out


def run_video(model, sam=None, stride_s=0.2, max_s=60):
    import imageio.v2 as imageio
    cap = cv2.VideoCapture(str(MEDIA / VIDEO["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(fps * max_s)
    step = max(1, int(round(fps * stride_s)))
    writer = imageio.get_writer(str(WEB / "stream_annotated.mp4"), fps=1 / stride_s, codec="libx264",
                                quality=7, macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok or i / fps > max_s:
            break
        if i % step == 0:
            res = model.predict(fr, imgsz=1280, conf=0.08, verbose=False)[0]
            dets = [d for d in yolo_dets(res, res.names, VIDEO_CONF) if d["kind"] in ("person", "vehicle")]
            if sam is not None:
                dets += dino_dets(sam, fr)
            ann = draw(fr, dets)
            t = i / fps
            ts = (NOW + pd.Timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(ann, f"CAM-02  {CAM_LABEL}  {ts}", (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            ann = cv2.resize(ann, (1280, int(ann.shape[0] * 1280 / ann.shape[1])))
            writer.append_data(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
            frames.append(dict(t=round(t, 2), ts=ts, detections=dets))
        i += 1
    writer.close()
    cnt = pd.Series([d["kind"] for f in frames for d in f["detections"]]).value_counts().to_dict()
    first_fire = next((f for f in frames if any(d["kind"] in ("fire", "smoke") for d in f["detections"])), None)
    meta = {**VIDEO, "video": "assets/stream_annotated.mp4", "fps_out": 1 / stride_s, "frames": frames,
            "counts": cnt, "first_fire": first_fire and first_fire["ts"], "duration_s": round(min(total / fps, max_s), 1)}
    json.dump(meta, open(V / "video.json", "w"), ensure_ascii=False)
    print("video", cnt, "first fire:", meta["first_fire"])
    return meta


def run_satellite_vehicles(aoi):
    from ultralytics import YOLO
    w = ROOT / "models" / "weights" / "yolo11x-obb.pt"
    model = YOLO(str(w) if w.exists() else "yolo11x-obb.pt")
    src = rasterio.open(P / f"sat_{aoi}.tif")
    img = np.transpose(src.read(), (1, 2, 0))[:, :, ::-1].copy()   # RGB→BGR
    H, W = img.shape[:2]
    # знімок ~0.5 м/пкс: легкове авто ≈ 8–10 пкс, тому тайли 512 пкс збільшуємо вдвічі перед YOLO-OBB (DOTA)
    T, OV, UP = 512, 96, 2
    dets = []
    for y in range(0, max(1, H - OV), T - OV):
        for x in range(0, max(1, W - OV), T - OV):
            tile = img[y:y + T, x:x + T]
            tile = cv2.resize(tile, (tile.shape[1] * UP, tile.shape[0] * UP), interpolation=cv2.INTER_CUBIC)
            r = model.predict(tile, imgsz=1024, conf=0.2, verbose=False)[0]
            if r.obb is None:
                continue
            for xywhr, c, cl in zip(r.obb.xywhr.cpu().numpy(), r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy()):
                label = r.names[int(cl)]
                if label not in ("small vehicle", "large vehicle"):
                    continue
                cx, cy, w, h, rot = xywhr / np.array([UP, UP, UP, UP, 1])
                poly = cv2.boxPoints(((cx + x, cy + y), (w, h), np.degrees(rot)))
                dets.append(dict(label=label, kind="vehicle", conf=round(float(c), 3), cx=float(cx + x), cy=float(cy + y),
                                 poly=poly.round(1).tolist()))
    # NMS між перекриттями тайлів: прибираємо дублікати ближче 3 м (≈7 px)
    dets.sort(key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        if all((d["cx"] - k["cx"]) ** 2 + (d["cy"] - k["cy"]) ** 2 > 49 for k in keep):
            keep.append(d)
    to_ll = pyproj.Transformer.from_crs(src.crs, 4326, always_xy=True)
    for d in keep:
        mx, my = src.transform * (d["cx"], d["cy"])
        d["lon"], d["lat"] = to_ll.transform(mx, my)
    ann = img.copy()
    for d in keep:
        cv2.polylines(ann, [np.array(d["poly"], np.int32)], True, (0, 230, 255), 2)
    cv2.imwrite(str(WEB / f"sat_{aoi}_vehicles.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(aoi, "vehicles:", len(keep))
    return keep, (W, H)


def run_langsam(aoi, sam):
    from segment import segment_tiled
    with rasterio.open(P / f"sat_{aoi}.tif") as src:
        rgb = np.transpose(src.read(), (1, 2, 0))[:, :, :3].copy()
        px_area = abs(src.transform.a * src.transform.e) * np.cos(np.radians(LAT0)) ** 2  # м² (Web Mercator → реальні)
    res = {}
    for prompt in SAT_PROMPTS:
        mask, n, _ = segment_tiled(sam, rgb, prompt)
        png = WEB / f"mask_{aoi}_{prompt.replace(' ', '_')}.png"
        if n:
            rgba = np.zeros((*mask.shape, 4), np.uint8)
            rgba[mask] = (*PROMPT_RGB[prompt], 150)
            Image.fromarray(rgba).save(png, optimize=True)
        res[prompt] = dict(count=int(n), area_ha=round(float(mask.sum() * px_area) / 1e4, 2), png=f"assets/{png.name}" if n else None)
        print(aoi, prompt, res[prompt])
    return res


PROMPT_RGB = {"tree": (34, 197, 94), "road": (250, 204, 21), "car": (56, 189, 248), "building": (244, 114, 182),
              "bare soil": (217, 119, 6), "burned area": (239, 68, 68)}


def run_satellite(sam=None):
    aois = json.load(open(P / "aois.json"))
    out = {}
    for aoi, meta in aois.items():
        vehicles, size = run_satellite_vehicles(aoi)
        Image.open(P / f"sat_{aoi}.jpg").save(WEB / f"sat_{aoi}.jpg", quality=85)
        bounds = json.load(open(P / f"sat_{aoi}.json"))["bounds_latlon"]
        out[aoi] = dict(title=meta["title"], img=f"assets/sat_{aoi}.jpg", img_veh=f"assets/sat_{aoi}_vehicles.jpg",
                        size=size, bounds=bounds, vehicles=vehicles,
                        prompts=run_langsam(aoi, sam) if sam else {})
    json.dump(out, open(V / "satellite.json", "w"), ensure_ascii=False)
    return out


def detections_geo(photos, video, sat):
    rows = []
    for ph in photos:
        date = ph["date"] or "2019-07-01"
        for d in ph["detections"]:
            rows.append(dict(kind=d["kind"], label=d["label"], conf=d["conf"], source="yolo_photo", date=date,
                             file=ph["img"], simulated=False, lat=ph["lat"], lon=ph["lon"]))
    # детекції з відео: проєкція в зону огляду камери (демо-прив'язка)
    if video["first_fire"]:
        b = np.radians(video["bearing"])
        lat = video["cam_lat"] + video["view_km"] / 111 * np.cos(b)
        lon = video["cam_lon"] + video["view_km"] / (111 * np.cos(np.radians(LAT0))) * np.sin(b)
        fires = [d for f in video["frames"] for d in f["detections"] if d["kind"] in ("fire", "smoke")]
        for k, d in enumerate(fires[:12]):
            rows.append(dict(kind=d["kind"], label=d["label"], conf=d["conf"], source="yolo_cam",
                             date=video["first_fire"], file=video["video"], simulated=True,
                             lat=lat + np.random.default_rng(k).normal(0, 0.0015), lon=lon + np.random.default_rng(k + 99).normal(0, 0.0015)))
    for aoi, s in sat.items():
        for v in s["vehicles"]:
            rows.append(dict(kind="vehicle", label=v["label"], conf=v["conf"], source="yolo_sat", date="2026-10-15",
                             file=s["img_veh"], simulated=False, lat=v["lat"], lon=v["lon"]))
    cols = ["kind", "label", "conf", "source", "date", "file", "simulated", "lat", "lon"]
    gdf = gpd.GeoDataFrame(rows or {c: [] for c in cols}, geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326)
    gdf.to_file(P / "detections.geojson", driver="GeoJSON")
    print("detections.geojson:", gdf.groupby(["source", "kind"]).size().to_dict() if len(gdf) else 0)



if __name__ == "__main__":
    steps = sys.argv[1:] or ["photos", "video", "sat"]
    from samgeo.text_sam import LangSAM
    sam = LangSAM() if ("video" in steps or "sat" in steps) else None
    if "photos" in steps or "video" in steps:
        wm = world_model()
    from eaglewings.ds import pflow
    pflow.mark("park photos: YOLO-World")
    photos = run_photos(wm) if "photos" in steps else json.load(open(V / "photos.json"))
    pflow.mark("camera video: smoke and people, YOLO-World + LangSAM")
    video = run_video(wm, sam) if "video" in steps else json.load(open(V / "video.json"))
    pflow.mark("area images: LangSAM masks, YOLO11x-OBB vehicles")
    if "sat" in steps:
        sat = run_satellite(sam)
    else:
        sat = json.load(open(V / "satellite.json"))
    pflow.mark("detections on map")
    detections_geo(photos, video, sat)
    pflow.mark("report: Wikimedia video and photos")
    from eaglewings.ds import pages
    pages.downloaded("wikimedia")
