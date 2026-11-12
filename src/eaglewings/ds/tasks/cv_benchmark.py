"""T11 (згори, супутник) і T18 (камери, фотопастки) · Детектори на відкритих еталонах: precision / recall замість «немає розмітки» в T4.

Це дві окремі задачі: run(part="aerial") — T11, VEDAI; run(part="ground") — T18, D-Fire і ENA24 (ds/tasks/cv_benchmark_ground.py).

Задача      скільки реально знаходять детектори і скільки дають хибних спрацювань. T4 цього порахувати не може: для парку
            немає розмітки. Тому міряємо на незалежних відкритих наборах, близьких за умовами, і чесно кажемо про різницю.
Дані        D-Fire (тест): 4306 фото, дим і вогонь, рамки.
            VEDAI: 2059 аерофото, переважно сільська місцевість, рамки машин; ніхто з моделей на ньому не вчився.
            Вихідні ~12,5 см/пкс штучно погіршуємо до 25 і 50 см/пкс: 50 см — це наші знімки Esri, де легкове авто ≈ 8–10 пкс.
            ENA24 (тест): 1337 кадрів фотопасток, 22 з машинами, решта тварини; лейбл кадру, без рамок.
Протокол    кадри випадково 50/50: на калібруванні обираємо поріг впевненості (max F1), на тесті — звіт. 95 % CI — bootstrap по кадрах.
Метрики     AP@0.5 — основна там, де є рамки (не залежить від вибору порога); precision / recall / F1 при порозі з калібрування;
            для ENA24 — recall кадрів із машинами й частка хибних кадрів (FPR) на тваринах.
Гіпотези
  yolo11x_obb       YOLO11x-OBB (DOTA): створена для вигляду згори; очікуємо перевагу на VEDAI, швидке падіння зі зниженням роздільності
  grounding_dino    GroundingDINO SwinB (через samgeo): детекція за текстом, без донавчання
Обмеження   YOLO-World і MegaDetector тут не міряються: їхніх ваг немає локально.
            D-Fire — переважно близькі кадри, а вежа бачить дим за кілометри; 12,5 → 50 см — лише симуляція зниження роздільності.
Вихід       data/processed/cv_calibration.json — виміряні recall / precision / розподіл впевненості.
"""
import glob
import inspect
import io
import json
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

from .. import pflow
from .. import datasets, metrics, tracking
from ..common import P, ROOT, SEED
from ..tracking import ml

EXPERIMENT = "T11 · CV benchmark (детектори на відкритих еталонах)"
B = ROOT / "data" / "raw" / "benchmarks"
W = ROOT / "models" / "weights"
GSD = {"vedai_12cm": 1, "vedai_25cm": 2, "vedai_50cm": 4}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================ дані
def dfire():
    """→ (id, PIL, {клас: [xyxy]}); у D-Fire клас 0 = дим, 1 = вогонь, рамки YOLO (cx cy w h, частки)."""
    for f in sorted(glob.glob(str(B / "dfire" / "data" / "*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=64):
            for r in b.to_pylist():
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                iw, ih = im.size
                gt = {"smoke": [], "fire": []}
                for line in (r["label"] or "").splitlines():
                    p = line.split()
                    if len(p) == 5:
                        c, x, y, w, h = int(p[0]), *map(float, p[1:])
                        gt[("smoke", "fire")[c]].append([(x - w / 2) * iw, (y - h / 2) * ih, (x + w / 2) * iw, (y + h / 2) * ih])
                yield r["filename"], im, gt


def vedai(scale):
    """→ (id, PIL, {vehicle: [xyxy]}): кадр зменшено в scale разів і збільшено назад (втрачена деталізація, той самий розмір кадру)."""
    with zipfile.ZipFile(B / "vedai" / "vedai.zip") as z:
        have = set(z.namelist())
        for n in sorted(x for x in have if x.endswith(".jpg") and "MACOSX" not in x):
            x = n.replace("/im/", "/boxes/").replace(".jpg", ".xml")
            if x not in have:
                continue
            gt = [[float(re.search(rf"<{k}>\s*([\d.]+)\s*</{k}>", b).group(1)) for k in ("xmin", "ymin", "xmax", "ymax")]
                  for b in re.findall(r"<bndbox>(.*?)</bndbox>", z.read(x).decode(), re.S)]
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            if scale > 1:
                w, h = im.size
                im = im.resize((w // scale, h // scale), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            yield n, im, {"vehicle": gt}


def ena24():
    """→ (id, PIL, {vehicle: чи є машина на кадрі}) (клас 17)."""
    i = 0
    for f in sorted(glob.glob(str(B / "ena24" / "data" / "*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=32):
            for r in b.to_pylist():
                i += 1
                yield f"ena24_{i}", Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"), {"vehicle": int(r["label"]) == 17}


# ================================================================ моделі: model(im) → {клас: [(conf, xyxy)]}
class OBB:
    def __init__(self):
        from ultralytics import YOLO
        self.m = YOLO(str(W / "yolo11x-obb.pt"))

    def __call__(self, im):
        r = self.m.predict(np.asarray(im)[..., ::-1], conf=.01, imgsz=1024, max_det=200, verbose=False, device=DEVICE)[0]
        return {"vehicle": [(cf, b) for c, cf, b in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxy.tolist())
                            if r.names[int(c)] in ("small vehicle", "large vehicle")]}


class GD:
    """GroundingDINO SwinB з локальних ваг samgeo; prompts: {текст запиту: наш клас}."""
    SAM = None

    def __init__(self, prompts):
        if GD.SAM is None:
            from samgeo.text_sam import LangSAM
            GD.SAM = LangSAM()
        self.prompts = prompts
        self.text = " . ".join(prompts)

    def __call__(self, im):
        boxes, logits, phrases = GD.SAM.predict_dino(im, self.text, .1, .1)
        d = {c: [] for c in set(self.prompts.values())}
        for b, s, ph in zip(boxes.tolist(), logits.tolist(), phrases):
            for k, c in self.prompts.items():
                if k in ph:
                    d[c].append((s, b))
                    break
        return d


def infer(model, it):
    ids, gts, preds = [], [], []
    for i, im, g in it:
        ids.append(i)
        gts.append(g)
        preds.append(model(im))
    return ids, gts, preds


# ================================================================ метрики
def boxes_eval(gts, preds, cls, n_boot, iou=.5):
    """Поріг із калібрування (max F1) → AP@0.5, precision, recall на тестовій половині."""
    per = [metrics.match_dets(p.get(cls, []), g[cls], iou) for g, p in zip(gts, preds)]
    cal = np.random.default_rng(SEED).permutation(len(gts)) % 2 == 0

    def flat(idx):
        return (np.array([c for i in idx for c, _ in per[i][0]], float), np.array([t for i in idx for _, t in per[i][0]], int), sum(per[i][1] for i in idx))

    c, t, n = flat(np.flatnonzero(cal))
    grid = np.unique(np.round(c, 2)) if len(c) else np.array([.5])
    thr = float(grid[int(np.argmax([2 * t[c >= g].sum() / max((c >= g).sum() + n, 1) for g in grid]))])
    te = np.flatnonzero(~cal)
    c, t, n = flat(te)
    keep = c >= thr
    prec, rec = t[keep].sum() / max(keep.sum(), 1), t[keep].sum() / max(n, 1)
    lo, hi = metrics.block_bootstrap(np.arange(len(te)), lambda i: metrics.average_precision(*flat(te[i])), n=n_boot)
    good = c[keep & (t == 1)]
    return dict(ap50=metrics.average_precision(c, t, n), ap50_ci=f"{lo:.3f}–{hi:.3f}", thr=thr, precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                n_gt=n, n_img=len(te), conf_q10=float(np.quantile(good, .1)) if len(good) else np.nan, conf_q90=float(np.quantile(good, .9)) if len(good) else np.nan,
                fp_per_img=float((keep & (t == 0)).sum() / max(len(te), 1)))


def _labels_dfire():
    for f in sorted(glob.glob(str(B / "dfire" / "data" / "*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["filename", "label"]):
            for r in b.to_pylist():
                yield r["filename"], sum(1 for line in (r["label"] or "").splitlines() if len(line.split()) == 5)


def _labels_vedai():
    with zipfile.ZipFile(B / "vedai" / "vedai.zip") as z:
        have = set(z.namelist())
        for n in sorted(x for x in have if x.endswith(".jpg") and "MACOSX" not in x):
            x = n.replace("/im/", "/boxes/").replace(".jpg", ".xml")
            if x in have:
                yield n, len(re.findall(r"<bndbox>(.*?)</bndbox>", z.read(x).decode(), re.S))


def _labels_ena24():
    i = 0
    for f in sorted(glob.glob(str(B / "ena24" / "data" / "*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["label"]):
            for r in b.to_pylist():
                i += 1
                yield f"ena24_{i}", int(int(r["label"]) == 17)


def build_aerial():
    """Функція збору cv_benchmark_aerial (datasets.get): кадри VEDAI для кожної роздільності й кількість рамок."""
    v = list(_labels_vedai())
    return save_frames(pd.DataFrame([dict(dataset=k, image=i, n_boxes=n) for k in GSD for i, n in v]), "aerial")


def build_ground():
    """Функція збору cv_benchmark_ground (datasets.get): кадри D-Fire і ENA24 з кількістю рамок."""
    rows = [dict(dataset="dfire", image=i, n_boxes=n) for i, n in _labels_dfire()]
    rows += [dict(dataset="ena24", image=i, n_boxes=n) for i, n in _labels_ena24()]
    return save_frames(pd.DataFrame(rows), "ground")


def save_frames(frames, part):
    """Два окремі еталони: знімки згори й фото з землі. Це різна зйомка, різні моделі й різні застереження."""
    common = dict(unit="кадр еталонного набору", targets=[], features=["dataset", "n_boxes"])
    if part == "aerial":
        return datasets.save(frames.reset_index(drop=True), "cv_benchmark_aerial", description="Знімки згори для детекції зі супутника (T11): VEDAI, ступені роздільності 12,5, 25 і 50 см, з кількістю рамок; зображення — scripts/00_fetch_benchmarks.py.",
                             params=dict(seed=SEED, gsd=GSD), sources=["VEDAI"], **common)
    return datasets.save(frames.reset_index(drop=True), "cv_benchmark_ground", description="Фото з землі для детекції з камер і фотопасток (T18): D-Fire (тест, дим і вогонь), ENA24 (тест, фотопастки), з кількістю рамок. Це кадри, відео в еталоні немає; зображення — scripts/00_fetch_benchmarks.py.",
                         params=dict(seed=SEED), sources=["D-Fire", "ENA24 / LILA BC"], **common)


def frames_eval(gts, preds, cls):
    """Лейбл кадру: оцінка = найвпевненіша детекція; ROC-AUC, recall і FPR при порозі 0,5."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([g[cls] for g in gts], int)
    s = np.array([max([c for c, _ in p.get(cls, [])], default=0.) for p in preds])
    return dict(ap_frame=average_precision_score(y, s), roc_auc=roc_auc_score(y, s), recall=float((s[y == 1] >= .5).mean()), fpr=float((s[y == 0] >= .5).mean()),
                n_pos=int(y.sum()), n_neg=int((1 - y).sum()), thr=.5)


# ================================================================ запуск
def run(n_boot=200, part="aerial"):
    exp = tracking.experiment(EXPERIMENT, __doc__)
    fire_p = {"smoke": "smoke", "fire": "fire"}
    veh_p = {"car": "vehicle", "truck": "vehicle", "vehicle": "vehicle"}
    pflow.mark('1. detectors on benchmarks')
    rows, frames = [], []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        ml.log_params(dict(protocol="50/50 калібрування / тест, поріг max F1 на калібруванні", n_boot=n_boot, iou=.5, device=DEVICE, gsd=str(GSD)))
        ml.log_text(inspect.getsource(inspect.getmodule(run)), "code/cv_benchmark.py")
        jobs = [("dfire", dfire, ["smoke", "fire"], "grounding_dino", lambda: GD(fire_p))] if part == "ground" else []
        for k, s in GSD.items() if part == "aerial" else ():
            jobs += [(k, lambda s=s: vedai(s), ["vehicle"], "yolo11x_obb", OBB), (k, lambda s=s: vedai(s), ["vehicle"], "grounding_dino", lambda: GD(veh_p))]
        for ds_, it, classes, name, mk in jobs:
            with ml.start_run(run_name=f"{ds_} · {name}", nested=True):
                ml.set_tags(dict(approach=name, target=",".join(classes), kind="detector", hypothesis=f"{name} на {ds_}", role="approach"))
                ids, gts, preds = infer(mk(), it())
                for c in classes:
                    r = boxes_eval(gts, preds, c, n_boot)
                    ml.log_metrics(tracking.clean({f"{c}_{k}": v for k, v in r.items()}))
                    rows.append(dict(dataset=ds_, target=c, approach=name, kind="boxes", **r))
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        if part == "ground":
            with ml.start_run(run_name="ena24 · grounding_dino", nested=True):
                ml.set_tags(dict(approach="grounding_dino", target="vehicle", kind="detector", hypothesis="GroundingDINO на фотопастках", role="approach"))
                ids, gts, preds = infer(GD(veh_p), ena24())
                r = frames_eval(gts, preds, "vehicle")
                ml.log_metrics(tracking.clean(r))
                rows.append(dict(dataset="ena24", target="vehicle", approach="grounding_dino", kind="frames", **r))
                print(rows[-1], flush=True)
        pflow.mark('2. thresholds and winners')
        res = pd.DataFrame(rows)
        res["champion"] = False
        for _, g in res[res.kind == "boxes"].groupby(["dataset", "target"]):
            res.loc[g.ap50.idxmax(), "champion"] = True
        pick = lambda d, t: res[(res.dataset == d) & (res.target == t) & res.champion].iloc[0]
        keys = dict(camera_smoke=("dfire", "smoke"), camera_fire=("dfire", "fire")) if part == "ground" else dict(sat_vehicle_12cm=("vedai_12cm", "vehicle"), sat_vehicle_25cm=("vedai_25cm", "vehicle"), sat_vehicle_50cm=("vedai_50cm", "vehicle"))
        f = P / "cv_calibration.json"
        cal = json.loads(f.read_text()) if f.exists() else {}            # друга задача дописує свої ключі, а не затирає чужі
        cal |= {k: {a: (None if pd.isna(b[a]) else float(b[a])) for a in ("precision", "recall", "conf_q10", "conf_q90", "fp_per_img", "ap50", "thr")} | dict(model=b.approach)
               for k, b in {k: pick(*v) for k, v in keys.items()}.items()}
        if part == "ground":
            e = res[res.dataset == "ena24"].iloc[0]
            cal["trap_vehicle"] = dict(model=e.approach, recall=float(e.recall), fpr=float(e.fpr), roc_auc=float(e.roc_auc), thr=.5)
        f.write_text(json.dumps(cal, ensure_ascii=False, indent=1))
        pflow.mark('3. frame dataset, save')
        dv = datasets.get("cv_benchmark_aerial" if part == "aerial" else "cv_benchmark_ground")
        tracking.log_dataset(dv, "evaluation", files=True)
        if part == "aerial":
            tracking.log_fig(_fig(res), "gsd_vehicle")
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id, dataset=dv.tag)


def _fig(res):
    fig, ax = plt.subplots(figsize=(7, 4))
    for a, g in res[res.dataset.str.startswith("vedai")].groupby("approach"):
        g = g.assign(cm=g.dataset.str.extract(r"(\d+)cm")[0].astype(int)).sort_values("cm")
        ax.plot(g.cm, g.ap50, "o-", label=a)
    ax.set(xlabel="см на піксель (штучне погіршення VEDAI)", ylabel="AP@0.5, машини", ylim=(0, 1), title="Детекція машин згори зі зниженням роздільності")
    ax.legend()
    fig.tight_layout()
    return fig
