"""Метрики та довірчі інтервали.

Вибір метрик (обґрунтування — у docstring кожної задачі):
  * ранжування з рідкісним класом      → PR-AUC (average precision) основна; ROC-AUC допоміжна (нечутлива до дисбалансу)
  * операційний бюджет уваги           → recall@k / precision@k: частка подій у топ-k % днів або площі
  * ймовірнісні прогнози               → Brier score (калібрування)
  * сегментація площі                  → F1 (=Dice), IoU, commission / omission error (стандарт валідації burned-area)
Довірчі інтервали — блоковий bootstrap: дані автокорельовані в просторі (блоки H3) і в часі (14-денні блоки),
тож ресемплінг окремих рядків дав би занадто вузькі інтервали.
"""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ranking(y, s, ks=(0.1, 0.2)):
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    out = dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s), prevalence=y.mean(), n=len(y), positives=int(y.sum()))
    o = np.argsort(-s, kind="stable")
    for k in ks:
        n = max(1, int(round(len(s) * k)))
        top = y[o[:n]]
        out[f"recall_at_{int(k * 100)}"] = top.sum() / max(y.sum(), 1)
        out[f"precision_at_{int(k * 100)}"] = top.mean()
        out[f"lift_at_{int(k * 100)}"] = top.mean() / max(y.mean(), 1e-9)
    return out


def brier(y, p):
    return brier_score_loss(np.asarray(y).astype(int), np.clip(np.asarray(p, float), 0, 1))


def area(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return dict(f1=f1, iou=tp / max(tp + fp + fn, 1e-9), precision=p, recall=r, commission_error=1 - p, omission_error=1 - r,
                pred_ha=tp + fp, true_ha=tp + fn)


def block_bootstrap(groups, stat, n=500, seed=0):
    """95 % CI статистики stat(idx) при ресемплінгу цілих блоків з поверненням."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ug))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        try:
            vals.append(stat(idx))
        except ValueError:          # у ресемплі лише один клас
            pass
    return tuple(np.nanpercentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def paired_block_bootstrap(groups, stat_a, stat_b, n=1000, seed=0):
    """Різниця stat_a − stat_b на тих самих блоках: оцінка й 95 % CI парного бутстрепа."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ug))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        try:
            vals.append(stat_a(idx) - stat_b(idx))
        except ValueError:
            pass
    allidx = np.arange(len(groups))
    return float(stat_a(allidx) - stat_b(allidx)), *(tuple(np.nanpercentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan))


def box_iou(a, b):
    """IoU матриця рамок xyxy."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match_dets(dets, gts, iou=0.5):
    """Жадібне зіставлення на кадрі: dets — [(conf, xyxy)], gts — [xyxy]. → [(conf, tp)], кількість еталонних рамок."""
    if not len(dets):
        return [], len(gts)
    dets = sorted(dets, key=lambda d: -d[0])
    I = box_iou([d[1] for d in dets], gts) if len(gts) else np.zeros((len(dets), 0))
    used, out = set(), []
    for i, (c, _) in enumerate(dets):
        j = int(I[i].argmax()) if I.shape[1] else -1
        ok = j >= 0 and I[i, j] >= iou and j not in used
        if ok:
            used.add(j)
        out.append((c, int(ok)))
    return out, len(gts)


def average_precision(conf, tp, n_gt):
    """AP: площа під кривою precision–recall з монотонною обгорткою (all-point)."""
    if n_gt == 0 or len(conf) == 0:
        return 0.0
    o = np.argsort(-np.asarray(conf), kind="stable")
    t = np.asarray(tp, float)[o]
    tc, fc = np.cumsum(t), np.cumsum(1 - t)
    r, p = tc / n_gt, tc / np.maximum(tc + fc, 1e-9)
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0, r]) * p))


def thin(xs, ys, n=160):
    """Крива для графіка: не більше n точок, округлення до 4 знаків (так їх показує сайт)."""
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    r = lambda v: None if not np.isfinite(v) else round(float(v), 4)
    return [[r(xs[i]), r(ys[i])] for i in idx]
