"""Тайлова сегментація за текстовим запитом (samgeo LangSAM = GroundingDINO + SAM) для великих знімків.

GroundingDINO працює на ~800 px, тож на цілому знімку 3000 px дрібні об'єкти зникають, а запит часто
"знаходить" рамку на весь кадр. Тому: тайли з перекриттям → відкидаємо рамки > MAX_BOX_FRAC тайла → SAM → зшиваємо.
"""
import numpy as np
import torch
from PIL import Image

TILE, OVERLAP, MAX_BOX_FRAC = 1024, 128, 0.4


def segment_tiled(sam, rgb, prompt, box_th=0.27, text_th=0.25, tile=TILE, overlap=OVERLAP, max_box_frac=MAX_BOX_FRAC):
    """rgb: HxWx3 uint8. Повертає (bool-маску HxW, кількість об'єктів, список рамок у пікселях повного знімка)."""
    H, W = rgb.shape[:2]
    full = np.zeros((H, W), bool)
    boxes_out = []
    step = tile - overlap
    for y in range(0, max(1, H - overlap), step):
        for x in range(0, max(1, W - overlap), step):
            t = rgb[y:y + tile, x:x + tile]
            th, tw = t.shape[:2]
            pil = Image.fromarray(t)
            boxes, logits, phrases = sam.predict_dino(pil, prompt, box_th, text_th)
            if len(boxes) == 0:
                continue
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = area < max_box_frac * th * tw
            boxes = boxes[keep]
            if len(boxes) == 0:
                continue
            masks = sam.predict_sam(pil, boxes)
            if masks.ndim == 4:
                masks = masks[:, 0]
            m = masks.any(0).cpu().numpy() if isinstance(masks, torch.Tensor) else np.any(masks, 0)
            full[y:y + th, x:x + tw] |= m
            for b in boxes.tolist():
                bb = [b[0] + x, b[1] + y, b[2] + x, b[3] + y]
                # дедуплікація об'єктів на перекритті тайлів
                if all(abs(bb[0] - o[0]) + abs(bb[1] - o[1]) > 12 for o in boxes_out):
                    boxes_out.append(bb)
    return full, len(boxes_out), boxes_out
