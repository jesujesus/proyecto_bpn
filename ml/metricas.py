"""Utilidades de umbral."""
import numpy as np


def _acumulados(y, p):
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    ultimo = np.r_[ps[1:] != ps[:-1], True]       # fin de cada grupo de empates
    return ps, ultimo, tp / ys.sum(), tp / (tp + fp)


def umbral_max_precision(y, p, recall_obj):
    """Umbral de mayor precisión con Recall (fuera de pliegue) >= recall_obj."""
    ps, ultimo, rec, pre = _acumulados(np.asarray(y), np.asarray(p))
    idx = np.where(ultimo & (rec >= recall_obj))[0]
    return float(ps[idx[np.argmax(pre[idx])]])


def umbral_por_recall(y, p, recall_obj):
    """Mayor umbral con Recall >= recall_obj."""
    ps, ultimo, rec, _ = _acumulados(np.asarray(y), np.asarray(p))
    idx = np.where(ultimo & (rec >= recall_obj))[0]
    return float(ps[idx[0]])
