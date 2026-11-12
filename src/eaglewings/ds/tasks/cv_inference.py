"""T4 · Комп'ютерний зір (інференс без лейблів).

Задача      знайти дим/вогонь, людей і транспорт на кадрах камери-вежі, фото з парку; маски LangSAM на знімках двох ділянок.
Моделі      претреновані, без донавчання: YOLO-World v2 (open-vocabulary: person/car/truck/fire/smoke),
            GroundingDINO + SAM (SamGeo LangSAM, сегментація за текстом), YOLO11x-OBB (DOTA: small/large vehicle) на двох ділянках.
Лейбли      НЕМАЄ розмічених даних для цього парку → precision/recall порахувати неможливо. Тому тут логуються лише
            конфігурації, кількість детекцій, розподіли впевненості.
План        розмітити ~300 кадрів камери (дим / не дим) і ~50 супутникових тайлів (bbox транспорту) у CVAT
            → датасет cv_labels@v1 → тоді метрики mAP@0.5 / precision@recall і порівняння моделей у цьому ж експерименті.
"""
import json

import pandas as pd

from .. import common, tracking
from ..tracking import ml

EXPERIMENT = "T4 · Computer vision inference (без лейблів)"
V = common.P / "vision"
FIG = common.ROOT / "notebooks" / "figures"


def run():
    exp = tracking.experiment(EXPERIMENT, __doc__)
    rows = []
    with ml.start_run(run_name=tracking.batch_name()) as parent:
        ml.set_tags(dict(labels="none", note="метрики якості неможливі без розмітки; див. опис експерименту"))

        vid = json.load(open(V / "video.json"))
        d = pd.DataFrame([dict(t=f["t"], **x) for f in vid["frames"] for x in f["detections"]])
        with ml.start_run(run_name="camera CAM-02 · YOLO-World + GroundingDINO", nested=True):
            ml.log_params(dict(models="yolov8x-worldv2 (person, vehicle); GroundingDINO (smoke/fire)", stride_s=0.2, frames=len(vid["frames"]),
                                   video=vid["file"], camera_bearing=vid["bearing"]))
            m = {f"detections_{k}": v for k, v in d.label.value_counts().items()} | {f"conf_median_{k}": v for k, v in d.groupby("label").conf.median().items()}
            m |= {"frames_with_smoke_share": d[d.kind.isin(["smoke", "fire"])].t.nunique() / len(vid["frames"])}
            ml.log_metrics(tracking.clean({k.replace(" ", "_"): v for k, v in m.items()}))
            ml.log_table(d.drop(columns="box").assign(box=d.box.astype(str)), "detections.json")
            for f in ("07a_stream.png", "07b_camera_zone.png"):
                if (FIG / f).exists():
                    ml.log_artifact(str(FIG / f), "plots")
            rows.append(dict(source="камера CAM-02", model="YOLO-World + GroundingDINO", **{k: round(v, 3) for k, v in m.items()}))

        ph = json.load(open(V / "photos.json"))
        dp = pd.DataFrame([dict(photo=p["file"], **x) for p in ph for x in p["detections"]])
        with ml.start_run(run_name="photos · YOLO-World", nested=True):
            ml.log_params(dict(model="yolov8x-worldv2", classes="person,car,truck,motorcycle,fire,smoke", photos=len(ph)))
            m = {f"detections_{k}": v for k, v in dp.kind.value_counts().items()}
            ml.log_metrics(tracking.clean(m))
            ml.log_table(dp.assign(box=dp.box.astype(str)), "detections.json")
            if (FIG / "07c_photos.png").exists():
                ml.log_artifact(str(FIG / "07c_photos.png"), "plots")
            rows.append(dict(source="фото з парку", model="YOLO-World", **m))

        sat = json.load(open(V / "satellite.json"))
        with ml.start_run(run_name="satellite · LangSAM", nested=True):
            ml.log_params(dict(imagery="Esri World Imagery z17–18", langsam_prompts=",".join(next(iter(sat.values()))["prompts"])))
            frac = [v["area_ha"] for s in sat.values() for k, v in s["prompts"].items() if k != "car"]
            m = dict(langsam_masks_over_80pct_share=sum(a > 0.8 * 175 for a in frac) / max(len(frac), 1))
            ml.log_metrics(tracking.clean(m))
            rows.append(dict(source="супутник", model="LangSAM", **m))
        res = pd.DataFrame(rows)
        ml.log_table(res, "results.json")
        return res, dict(experiment_id=exp.experiment_id, parent_run_id=parent.info.run_id)
