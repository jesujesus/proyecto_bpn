"""Análisis de la tesis BPN según el Método (C0-C3, contrastes HE1-HE3, subgrupos, SHAP, figuras).

Modelo principal SIN Ipress ni variables del parto; análisis de sensibilidad CON Ipress (solo CV).
Usa la partición 80/20 de data/processed/split.json.
Uso:  python ml/analisis_tesis.py            (completo, ~8 min)
      python ml/analisis_tesis.py --rapido   (prueba humo)
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter
from scipy.stats import friedmanchisquare, norm, shapiro, wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RAPIDO = "--rapido" in sys.argv
RAIZ = Path(__file__).resolve().parents[1]
PROY = RAIZ
OUT = RAIZ / "reports" / ("tesis" if not RAPIDO else "_humo")
FIG = OUT / "figuras"
SEMILLA = 42
REPS = 1 if RAPIDO else 10
REPS_TUN = 1 if RAPIDO else 3
NBOOT = 50 if RAPIDO else 2000
RECALL_OBJ = 0.70
N_JOBS = 4

NUM = ["Edad_Madre", "Num_embar_madre", "Hijos_vivo_madre", "nacmuer_abort_madre",
       "ES_ADOL", "MENOR18", "AVANZ", "PRIMIG", "GRAN", "PERDIDAS", "ESTCIV_IGN", "INT"]
CAT = ["Estado_Civil", "Nivel_Intrucción_Madre", "OCUPACION", "MULTIPLE", "DISTRITO", "SEGURO"]
BONITO = {"Edad_Madre": "Edad de la madre", "Num_embar_madre": "N.º de embarazos", "Hijos_vivo_madre": "Hijos vivos",
          "nacmuer_abort_madre": "Abortos / nacidos muertos", "ES_ADOL": "Adolescente (<20)", "MENOR18": "Menor de 18",
          "AVANZ": "Edad avanzada (≥35)", "PRIMIG": "Primer embarazo", "GRAN": "Gran multípara (≥4)",
          "PERDIDAS": "Antecedente de abortos", "ESTCIV_IGN": "Estado civil ignorado", "INT": "Embarazos sin resultado",
          "Estado_Civil": "Estado civil", "Nivel_Intrucción_Madre": "Instrucción", "OCUPACION": "Ocupación",
          "MULTIPLE": "Embarazo único/múltiple", "DISTRITO": "Distrito", "SEGURO": "Seguro", "Ipress": "IPRESS"}
MODELOS = {"C1": "Regresión Logística", "C2": "Random Forest", "C3": "XGBoost"}
GRILLAS = {"C1": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
           "C2": [{"min_samples_leaf": 5}, {"min_samples_leaf": 15}, {"min_samples_leaf": 30}],
           "C3": [{"max_depth": 2, "n_estimators": 200, "learning_rate": 0.03},
                  {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.03},
                  {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.02}]}


def coma(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def p_txt(p):
    return "< 0,001" if p < 0.001 else coma(p, 3)


# ------------------------------------------------------------------ datos
def cargar():
    ruta = PROY / "data" / "raw" / "Rodriguez_de_Mendoza_LIMPIO.xlsx"
    df = pd.read_excel(ruta, sheet_name="Datos_Completos", dtype={"IdUbigeoInei": str})
    df["IdUbigeoInei"] = df["IdUbigeoInei"].str.zfill(6)
    y = (df["PESO_NACIDO"] < 2500).astype(int).values
    sp = json.load(open(PROY / "data" / "processed" / "split.json"))
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    assert len(df) == 4444 and y.sum() == 254 and len(tr) == 3555 and len(te) == 889
    assert y[tr].sum() == 203 and y[te].sum() == 51
    return df, y, tr, te


def construir(df, con_ipress):
    d = pd.DataFrame(index=df.index)
    ed, emb = df["Edad_Madre"], df["Num_embar_madre"]
    viv, ab = df["Hijos_vivo_madre"], df["nacmuer_abort_madre"]
    d["Edad_Madre"], d["Num_embar_madre"], d["Hijos_vivo_madre"], d["nacmuer_abort_madre"] = ed, emb, viv, ab
    d["ES_ADOL"] = (ed < 20).astype(int)
    d["MENOR18"] = (ed < 18).astype(int)
    d["AVANZ"] = (ed >= 35).astype(int)
    d["PRIMIG"] = (emb == 1).astype(int)
    d["GRAN"] = (emb >= 4).astype(int)
    d["PERDIDAS"] = (ab > 0).astype(int)
    d["ESTCIV_IGN"] = (df["Estado_Civil"] == "IGNORADO").astype(int)
    d["INT"] = emb - viv - ab
    d["Estado_Civil"] = df["Estado_Civil"]
    d["Nivel_Intrucción_Madre"] = df["Nivel_Intrucción_Madre"]
    d["OCUPACION"] = np.where(df["DESC_OCUPACION"] == "AMA DE CASA", "AMA DE CASA", "OTRA")
    d["MULTIPLE"] = df["Tipo_Parto"].map({"UNICO": "UNICO", "DOBLE": "MULTIPLE", "TRIPLE": "MULTIPLE",
                                          "MAS DE TRES": "MULTIPLE"}).fillna("IGNORADO")
    d["DISTRITO"] = df["IdUbigeoInei"]
    d["SEGURO"] = df["Financiador_Parto"]
    if con_ipress:
        d["Ipress"] = df["Ipress"].astype(str)
    return d


# ------------------------------------------------------------------ pipelines
def pipeline(modelo, estrategia, params, cats, peso_pos):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), cats)])
    pesos = estrategia == "pesos"
    if modelo == "C0":
        clf = LogisticRegression(C=1.0, max_iter=2000)
    elif modelo == "C1":
        clf = LogisticRegression(C=params["C"], max_iter=2000, class_weight="balanced" if pesos else None)
    elif modelo == "C2":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=params["min_samples_leaf"], max_features="sqrt",
                                     class_weight="balanced_subsample" if pesos else None, random_state=SEMILLA, n_jobs=1)
    else:
        clf = XGBClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                            learning_rate=params["learning_rate"], subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                            reg_lambda=5.0, scale_pos_weight=peso_pos if pesos else 1.0, eval_metric="logloss",
                            tree_method="hist", n_jobs=1, random_state=SEMILLA, verbosity=0)
    pasos = [("prep", pre)]
    if estrategia == "smote":
        pasos.append(("smote", SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=SEMILLA)))
    pasos.append(("clf", clf))
    return Pipeline(pasos)


def _fold(pipe, X, y, tr, te):
    m = clone(pipe).fit(X.iloc[tr], y[tr])
    p = m.predict_proba(X.iloc[te])[:, 1]
    return te, p


def cv(pipe, X, y, splits):
    res = Parallel(n_jobs=N_JOBS)(delayed(_fold)(pipe, X, y, tr, te) for tr, te in splits)
    return res          # lista de (te, p) por pliegue


def umbral_max_precision(y, p, recall_obj=RECALL_OBJ):
    """Umbral de mayor precisión con Recall (fuera de pliegue) >= recall_obj."""
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    ultimo = np.r_[ps[1:] != ps[:-1], True]            # fin de cada grupo de empates
    rec, pre = tp / ys.sum(), tp / (tp + fp)
    ok = ultimo & (rec >= recall_obj)
    idx = np.where(ok)[0]
    mejor = idx[np.argmax(pre[idx])]
    return float(ps[mejor])


def metr(y, s, thr):
    y = np.asarray(y)
    pr = (np.asarray(s) >= thr).astype(int)
    tp, fp = int(((pr == 1) & (y == 1)).sum()), int(((pr == 1) & (y == 0)).sum())
    fn, tn = int(((pr == 0) & (y == 1)).sum()), int(((pr == 0) & (y == 0)).sum())
    d = lambda a, b: a / b if b else 0.0
    rec, pre = d(tp, tp + fn), d(tp, tp + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": rec, "precision": pre, "especificidad": d(tn, tn + fp),
            "f1": d(2 * pre * rec, pre + rec), "accuracy": d(tp + tn, len(y))}


def wilson(k, n, alfa=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z, p = norm.ppf(1 - alfa / 2), k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0, c - h), min(1, c + h))


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    aj, mx = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        mx = max(mx, (len(p) - r) * p[i])
        aj[i] = min(1.0, mx)
    return aj


# ------------------------------------------------------------------ DeLong pareado
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_pareado(y, p1, p2):
    y = np.asarray(y)
    orden = np.r_[np.where(y == 1)[0], np.where(y == 0)[0]]
    preds = np.vstack([p1, p2])[:, orden]
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(preds[r, :m]) for r in range(2)])
    ty = np.array([_midrank(preds[r, m:]) for r in range(2)])
    tz = np.array([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "z": float(z),
            "p_unilateral": float(1 - norm.cdf(z)), "p_bilateral": float(2 * (1 - norm.cdf(abs(z))))}


# ------------------------------------------------------------------ experimento CV
def experimento(df, y, tr, con_ipress, etiqueta):
    X = construir(df, con_ipress)
    cats = CAT + (["Ipress"] if con_ipress else [])
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
    peso_pos = float((ytr == 0).sum() / (ytr == 1).sum())
    t0 = time.time()
    # --- ajuste de hiperparámetros (CV 3x5) por (modelo, estrategia)
    spl_t = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPS_TUN, random_state=SEMILLA).split(Xtr, ytr))
    tuning, elegidos = [], {}
    for k in MODELOS:
        for est in ["pesos", "smote"]:
            mejor = (-1, None)
            for prm in GRILLAS[k]:
                res = cv(pipeline(k, est, prm, cats, peso_pos), Xtr, ytr, spl_t)
                a = np.mean([roc_auc_score(ytr[te], p) for te, p in res])
                tuning.append({"conjunto": etiqueta, "config": k, "modelo": MODELOS[k], "estrategia": est,
                               "params": json.dumps(prm), "auc_cv_3x5": a})
                if a > mejor[0]:
                    mejor = (a, prm)
            elegidos[(k, est)] = mejor[1]
            print(f"[{etiqueta}] tuning {k} {est}: {mejor[1]} AUC={mejor[0]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    # --- CV 10x5 con los mismos pliegues para todas las configuraciones
    spl = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPS, random_state=SEMILLA).split(Xtr, ytr))
    configs = {"C0": ("C0", "ninguna", {})}
    for k in MODELOS:
        for est in ["pesos", "smote"]:
            configs[f"{k}_{est}"] = (k, est, elegidos[(k, est)])
    cvres = {}
    for nombre, (k, est, prm) in configs.items():
        folds = cv(pipeline(k, est, prm, cats, peso_pos), Xtr, ytr, spl)
        oof = np.full((REPS, len(ytr)), np.nan)
        for i, (te, p) in enumerate(folds):
            oof[i // 5, te] = p
        pool_p, pool_y = oof.ravel(), np.tile(ytr, REPS)
        thr = 0.5 if nombre == "C0" else umbral_max_precision(pool_y, pool_p)
        auc = np.array([roc_auc_score(ytr[te], p) for te, p in folds])
        ap = np.array([average_precision_score(ytr[te], p) for te, p in folds])
        mf = [metr(ytr[te], p, thr) for te, p in folds]
        cvres[nombre] = {"k": k, "est": est, "params": prm, "thr": thr, "auc": auc, "ap": ap,
                         "f1": np.array([m["f1"] for m in mf]), "recall": np.array([m["recall"] for m in mf]),
                         "precision": np.array([m["precision"] for m in mf]), "oof_mean": np.nanmean(oof, axis=0)}
        print(f"[{etiqueta}] CV 10x5 {nombre}: AUC={auc.mean():.4f} AP={ap.mean():.4f} F1={cvres[nombre]['f1'].mean():.3f} "
              f"Recall={cvres[nombre]['recall'].mean():.3f} thr={thr:.4f} [{time.time()-t0:.0f}s]", flush=True)
    cand = [n for n in cvres if n != "C0"]
    sistema = max(cand, key=lambda n: cvres[n]["auc"].mean())
    # --- tablas CV
    resumen = pd.DataFrame([{"conjunto": etiqueta, "config": n, "modelo": "Regresión Logística (línea base)" if n == "C0" else MODELOS[v["k"]],
                             "estrategia": v["est"], "params": json.dumps(v["params"]), "umbral": v["thr"],
                             "auc_roc_media": v["auc"].mean(), "auc_roc_de": v["auc"].std(), "auc_pr_media": v["ap"].mean(),
                             "auc_pr_de": v["ap"].std(), "f1_media": v["f1"].mean(), "f1_de": v["f1"].std(),
                             "recall_media": v["recall"].mean(), "recall_de": v["recall"].std(),
                             "precision_media": v["precision"].mean(), "es_sistema": n == sistema}
                            for n, v in cvres.items()])
    porpl = pd.DataFrame({"conjunto": etiqueta, "pliegue": np.arange(len(spl))})
    for n, v in cvres.items():
        for m in ["auc", "ap", "f1", "recall"]:
            porpl[f"{n}__{m}"] = v[m]
    fr_rows, wx_rows = [], []
    nombres = list(cvres)
    for m, mn in [("auc", "AUC-ROC"), ("ap", "AUC-PR"), ("f1", "F1"), ("recall", "Recall")]:
        arr = [cvres[n][m] for n in nombres]
        try:
            f = friedmanchisquare(*arr)
            fr_rows.append({"conjunto": etiqueta, "metrica": mn, "chi2": f.statistic, "p": f.pvalue, "k_configs": len(nombres), "n_pliegues": len(arr[0])})
        except Exception:
            fr_rows.append({"conjunto": etiqueta, "metrica": mn, "chi2": np.nan, "p": np.nan, "k_configs": len(nombres), "n_pliegues": len(arr[0])})
        pares = list(combinations(nombres, 2))
        pv = []
        for a, b in pares:
            try:
                pv.append(wilcoxon(cvres[a][m], cvres[b][m]).pvalue)
            except ValueError:
                pv.append(1.0)
        ph = holm(pv)
        for (a, b), p, q in zip(pares, pv, ph):
            wx_rows.append({"conjunto": etiqueta, "metrica": mn, "a": a, "b": b,
                            "dif_media": cvres[a][m].mean() - cvres[b][m].mean(), "p": p, "p_holm": q})
    sh_rows = []
    for m, mn in [("auc", "AUC-ROC"), ("ap", "AUC-PR"), ("f1", "F1"), ("recall", "Recall")]:
        d = cvres[sistema][m] - cvres["C0"][m]
        try:
            s = shapiro(d)
            sh_rows.append({"conjunto": etiqueta, "metrica": mn, "comparacion": f"{sistema} - C0", "W": s.statistic, "p": s.pvalue,
                            "normal_alfa_0.05": bool(s.pvalue >= 0.05)})
        except Exception:
            sh_rows.append({"conjunto": etiqueta, "metrica": mn, "comparacion": f"{sistema} - C0", "W": np.nan, "p": np.nan, "normal_alfa_0.05": None})
    return {"X": X, "cats": cats, "cv": cvres, "sistema": sistema, "configs": configs, "peso_pos": peso_pos,
            "tablas": {"resumen": resumen, "porpliegue": porpl, "tuning": pd.DataFrame(tuning), "friedman": pd.DataFrame(fr_rows),
                       "wilcoxon": pd.DataFrame(wx_rows), "shapiro": pd.DataFrame(sh_rows)}}


# ------------------------------------------------------------------ prueba (una sola vez)
def boot_estratificado(y, s0, s1, thr0, thr1):
    rng = np.random.default_rng(SEMILLA)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    nombres = ["recall", "auc", "ap", "f1"]
    R = {f"{n}_{c}": [] for n in nombres for c in ("S", "C0")}
    for _ in range(NBOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        yy = y[i]
        for c, s, t in (("S", s1, thr1), ("C0", s0, thr0)):
            m = metr(yy, s[i], t)
            R[f"recall_{c}"].append(m["recall"])
            R[f"f1_{c}"].append(m["f1"])
            R[f"auc_{c}"].append(roc_auc_score(yy, s[i]))
            R[f"ap_{c}"].append(average_precision_score(yy, s[i]))
    return {k: np.array(v) for k, v in R.items()}


def prueba(df, y, tr, te, ex):
    X = ex["X"]
    Xtr, ytr, Xte, yte = X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), y[te]
    cfg, sis = ex["configs"], ex["sistema"]
    p0 = pipeline(*cfg["C0"], ex["cats"], ex["peso_pos"]).fit(Xtr, ytr)
    k, est, prm = cfg[sis]
    ps = pipeline(k, est, prm, ex["cats"], ex["peso_pos"]).fit(Xtr, ytr)
    s0, s1 = p0.predict_proba(Xte)[:, 1], ps.predict_proba(Xte)[:, 1]
    thr0, thr1 = 0.5, ex["cv"][sis]["thr"]
    B = boot_estratificado(yte, s0, s1, thr0, thr1)
    m0, m1 = metr(yte, s0, thr0), metr(yte, s1, thr1)
    pt = {"recall": (m0["recall"], m1["recall"]), "auc": (roc_auc_score(yte, s0), roc_auc_score(yte, s1)),
          "ap": (average_precision_score(yte, s0), average_precision_score(yte, s1)), "f1": (m0["f1"], m1["f1"])}
    filas, contr = [], []
    nom = {"recall": "Recall", "auc": "AUC-ROC", "ap": "AUC-PR", "f1": "F1"}
    for key in ["recall", "auc", "ap", "f1"]:
        for c, idx in (("C0", 0), (f"Sistema ({sis})", 1)):
            v = B[f"{key}_{'C0' if idx == 0 else 'S'}"]
            filas.append({"metrica": nom[key], "modelo": c, "valor": pt[key][idx], "ic95_inf": np.percentile(v, 2.5),
                          "ic95_sup": np.percentile(v, 97.5), "lim_inf_unilateral95": np.percentile(v, 5)})
        d = B[f"{key}_S"] - B[f"{key}_C0"]
        contr.append({"metrica": nom[key], "dif_sistema_menos_C0": pt[key][1] - pt[key][0], "ic95_inf": np.percentile(d, 2.5),
                      "ic95_sup": np.percentile(d, 97.5), "lim_inf_unilateral95": np.percentile(d, 5),
                      "p_bootstrap_unilateral": float((np.sum(d <= 0) + 1) / (len(d) + 1))})
    for n, m, thr in (("C0", m0, thr0), (f"Sistema ({sis})", m1, thr1)):
        for kk in ["precision", "especificidad", "accuracy"]:
            filas.append({"metrica": kk, "modelo": n, "valor": m[kk], "ic95_inf": np.nan, "ic95_sup": np.nan, "lim_inf_unilateral95": np.nan})
    lo, hi = wilson(m1["tp"], m1["tp"] + m1["fn"])
    dl = delong_pareado(yte, s1, s0)
    return {"s0": s0, "s1": s1, "thr0": thr0, "thr1": thr1, "m0": m0, "m1": m1, "B": B, "tabla": pd.DataFrame(filas),
            "contrastes": pd.DataFrame(contr), "delong": dl, "wilson_recall_S": (lo, hi), "yte": yte, "Xte": Xte,
            "pipe_sis": ps, "pipe_c0": p0, "Xtr": Xtr, "ytr": ytr, "test_idx": te, "pt": pt}


def hipotesis(ex, pr):
    sis, B, ct = ex["sistema"], pr["B"], pr["contrastes"].set_index("metrica")
    wx = ex["tablas"]["wilcoxon"]

    def holm_vs_c0(metrica):
        r = wx[(wx.metrica == metrica) & (((wx.a == sis) & (wx.b == "C0")) | ((wx.a == "C0") & (wx.b == sis)))].iloc[0]
        signo = r.dif_media if r.a == sis else -r.dif_media
        return float(r.p_holm), bool(signo > 0)
    rec_lo = float(np.percentile(B["recall_S"], 5))
    auc_lo = float(np.percentile(B["auc_S"], 5))
    he1 = bool(ct.loc["Recall", "lim_inf_unilateral95"] > 0 and rec_lo > 0.70)
    he2 = bool(pr["delong"]["p_unilateral"] < 0.05 and auc_lo > 0.75)
    ph_ap, fav_ap = holm_vs_c0("AUC-PR")
    ph_f1, fav_f1 = holm_vs_c0("F1")
    he3 = bool(ct.loc["AUC-PR", "lim_inf_unilateral95"] > 0 and ct.loc["F1", "lim_inf_unilateral95"] > 0
               and ph_ap < 0.05 and ph_f1 < 0.05 and fav_ap and fav_f1)
    return {"HE1": {"se_rechaza_H0": he1, "recall_S": pr["pt"]["recall"][1], "recall_S_lim_inf_unilateral95": rec_lo,
                    "dif_recall_lim_inf": float(ct.loc["Recall", "lim_inf_unilateral95"]), "criterio": "lím. inf. de ΔRecall > 0 y lím. inf. de Recall(S) > 0,70"},
            "HE2": {"se_rechaza_H0": he2, "auc_S": pr["pt"]["auc"][1], "auc_S_lim_inf_unilateral95": auc_lo,
                    "delong_p_unilateral": pr["delong"]["p_unilateral"], "criterio": "DeLong p < 0,05 y lím. inf. de AUC(S) > 0,75"},
            "HE3": {"se_rechaza_H0": he3, "dif_auc_pr_lim_inf": float(ct.loc["AUC-PR", "lim_inf_unilateral95"]),
                    "dif_f1_lim_inf": float(ct.loc["F1", "lim_inf_unilateral95"]), "wilcoxon_holm_auc_pr_p": ph_ap,
                    "wilcoxon_holm_f1_p": ph_f1, "criterio": "lím. inf. de ΔAUC-PR y ΔF1 > 0 (prueba) y Wilcoxon-Holm p < 0,05 a favor (CV)"},
            "HG": {"se_rechaza_H0": bool(he1 and he2 and he3), "criterio": "se rechaza H0 en HE1, HE2 y HE3"}}


# ------------------------------------------------------------------ subgrupos y SHAP
def subgrupos(df, te, pr):
    t = df.iloc[te].reset_index(drop=True)
    y, s0, s1 = pr["yte"], pr["s0"], pr["s1"]
    d0, d1 = s0 >= pr["thr0"], s1 >= pr["thr1"]
    seg = t["Financiador_Parto"].where(t["Financiador_Parto"].isin(["SIS", "ESSALUD"]), "OTROS")
    defs = [("Severidad", "Extremadamente bajo (<1 000 g)", t["PESO_NACIDO"] < 1000),
            ("Severidad", "Muy bajo (<1 500 g)", t["PESO_NACIDO"] < 1500),
            ("Severidad", "Bajo (1 500–2 499 g)", t["PESO_NACIDO"].between(1500, 2499)),
            ("Origen", "Prematuro (<37 semanas)", t["DUR_EMB_PARTO"] < 37),
            ("Origen", "A término (≥37 semanas)", t["DUR_EMB_PARTO"] >= 37)]
    defs += [("Distrito", f"{d}", t["IdUbigeoInei"] == d) for d in sorted(t["IdUbigeoInei"].unique())]
    defs += [("Seguro", s, seg == s) for s in ["SIS", "ESSALUD", "OTROS"]]
    f = []
    for dim, nombre, m in defs:
        m = m.values
        cas = int((m & (y == 1)).sum())
        r = {"dimension": dim, "subgrupo": nombre, "n_nacidos": int(m.sum()), "casos_BPN": cas}
        for lab, det in (("S", d1), ("C0", d0)):
            k = int((m & (y == 1) & det).sum())
            lo, hi = wilson(k, cas)
            r.update({f"detectados_{lab}": k, f"recall_{lab}": (k / cas if cas else np.nan), f"ic95_inf_{lab}": lo, f"ic95_sup_{lab}": hi})
        f.append(r)
    return pd.DataFrame(f)


def shap_sistema(ex, pr):
    ps = pr["pipe_sis"]
    prep, clf = ps.named_steps["prep"], ps.named_steps["clf"]
    Zte, Ztr = prep.transform(pr["Xte"]), prep.transform(pr["Xtr"])
    nombres = prep.get_feature_names_out()
    vars_ = NUM + ex["cats"]
    mapa = []
    for n in nombres:
        base = n.split("__", 1)[1]
        mapa.append(next((v for v in sorted(vars_, key=len, reverse=True) if base == v or base.startswith(v + "_")), base))
    mapa = np.array(mapa)
    if ex["configs"][ex["sistema"]][0] == "C1":
        expl = shap.LinearExplainer(clf, Ztr.mean(axis=0, keepdims=True))
    else:
        expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Zte)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    agg = pd.DataFrame(sv, columns=nombres).T.groupby(mapa).sum().T
    return agg, bool(np.isfinite(sv).all())


# ------------------------------------------------------------------ figuras
def fmt_ax(ax, x=True, y=True):
    f = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    if x:
        ax.xaxis.set_major_formatter(f)
    if y:
        ax.yaxis.set_major_formatter(f)
    ax.spines[["top", "right"]].set_visible(False)


def figuras(ex, pr, agg, df):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 100})
    AZ, RO, GR = "#1f6feb", "#d1242f", "#6e7781"
    sis = ex["sistema"]
    nombres = list(ex["cv"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bp = ax.boxplot([ex["cv"][n]["auc"] for n in nombres], tick_labels=[n.replace("_", "\n") for n in nombres], patch_artist=True)
    for b, n in zip(bp["boxes"], nombres):
        b.set(facecolor=RO if n == sis else ("#d0d7de" if n == "C0" else "#a5c8ff"), edgecolor="black")
    ax.axhline(0.5, ls="--", color=GR, lw=1)
    ax.set_ylabel("AUC-ROC (50 pliegues)")
    fmt_ax(ax, x=False)
    fig.tight_layout(); fig.savefig(FIG / "cv_cajas_auc.png", dpi=300); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, m, lab in zip(axs, (pr["m0"], pr["m1"]), ("C0 (línea base)", f"Sistema ({sis})")):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Normal", "BPN"]); ax.set_yticks([0, 1], ["Normal", "BPN"])
        ax.set_xlabel(f"Predicción\n{lab}"); ax.set_ylabel("Real")
    fig.tight_layout(); fig.savefig(FIG / "matrices_confusion_c0_vs_sistema.png", dpi=300); plt.close(fig)

    y = pr["yte"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for s, lab, c, thr in ((pr["s0"], "C0", GR, pr["thr0"]), (pr["s1"], f"Sistema ({sis})", RO, pr["thr1"])):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lab} (AUC = {coma(roc_auc_score(y, s), 3)})")
    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.8, label="Azar")
    ax.set_xlabel("1 − Especificidad"); ax.set_ylabel("Sensibilidad"); ax.legend(frameon=False, loc="lower right", fontsize=8)
    fmt_ax(ax); fig.tight_layout(); fig.savefig(FIG / "curvas_roc.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for s, lab, c in ((pr["s0"], "C0", GR), (pr["s1"], f"Sistema ({sis})", RO)):
        pre, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, pre, color=c, lw=2, label=f"{lab} (AUC-PR = {coma(average_precision_score(y, s), 3)})")
    ax.axhline(y.mean(), ls="--", color="black", lw=0.8, label=f"Prevalencia ({coma(y.mean(), 3)})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precisión"); ax.legend(frameon=False, fontsize=8)
    fmt_ax(ax); fig.tight_layout(); fig.savefig(FIG / "curvas_pr.png", dpi=300); plt.close(fig)

    imp = agg.abs().mean().sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.barh([BONITO.get(k, k) for k in imp.index], imp.values, color=AZ)
    ax.set_xlabel("Importancia SHAP media (|contribución|)"); fmt_ax(ax, y=False)
    fig.tight_layout(); fig.savefig(FIG / "shap_global.png", dpi=300); plt.close(fig)

    # caso local: verdadero positivo con mayor puntaje
    tp_idx = np.where((y == 1) & (pr["s1"] >= pr["thr1"]))[0]
    caso = int(tp_idx[np.argmax(pr["s1"][tp_idx])]) if len(tp_idx) else int(np.argmax(pr["s1"]))
    fila = agg.iloc[caso]
    top = fila.abs().sort_values(ascending=False).head(8).index[::-1]
    X = pr["Xte"].iloc[caso]
    etq = [f"{BONITO.get(v, v)} = {X[v]}" for v in top]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(etq, fila[top].values, color=[RO if v > 0 else AZ for v in fila[top].values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribución SHAP (rojo: aumenta el riesgo; azul: lo reduce)"); fmt_ax(ax, y=False)
    fig.tight_layout(); fig.savefig(FIG / "shap_local.png", dpi=300); plt.close(fig)
    loc = pd.DataFrame({"variable": top[::-1], "valor": [X[v] for v in top[::-1]], "contribucion_shap": fila[top[::-1]].values})
    loc.insert(0, "caso_prueba_pos", int(pr["test_idx"][caso]))
    loc.insert(1, "puntaje_modelo", float(pr["s1"][caso]))
    return loc


# ------------------------------------------------------------------ principal
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tablas").mkdir(exist_ok=True)
    df, y, tr, te = cargar()
    print(f"Datos OK: train {len(tr)} (BPN {y[tr].sum()}), test {len(te)} (BPN {y[te].sum()}) {'[MODO RAPIDO]' if RAPIDO else ''}", flush=True)
    ex = experimento(df, y, tr, False, "principal_sin_Ipress")
    ex_s = experimento(df, y, tr, True, "sensibilidad_con_Ipress")
    T = OUT / "tablas"
    for nombre in ["resumen", "porpliegue", "tuning", "friedman", "wilcoxon", "shapiro"]:
        pd.concat([ex["tablas"][nombre], ex_s["tablas"][nombre]]).to_csv(T / f"cv_{nombre}.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
    print("CV lista; evaluando en prueba (una sola vez, modelo principal)...", flush=True)
    pr = prueba(df, y, tr, te, ex)
    hip = hipotesis(ex, pr)
    sg = subgrupos(df, te, pr)
    agg, shap_ok = shap_sistema(ex, pr)
    imp = agg.abs().mean().sort_values(ascending=False)
    glob = pd.DataFrame({"variable": imp.index, "nombre": [BONITO.get(k, k) for k in imp.index], "shap_medio_abs": imp.values})
    loc = figuras(ex, pr, agg, df)
    kw = dict(index=False, encoding="utf-8-sig", sep=";", decimal=",")
    pr["tabla"].to_csv(T / "test_metricas.csv", **kw)
    pr["contrastes"].to_csv(T / "test_contrastes_bootstrap.csv", **kw)
    pd.DataFrame([{**pr["delong"], "auc_1": pr["delong"]["auc_1"], "comparacion": f"{ex['sistema']} vs C0"}]).to_csv(T / "test_delong.csv", **kw)
    sg.to_csv(T / "subgrupos.csv", **kw)
    glob.to_csv(T / "shap_global.csv", **kw)
    loc.to_csv(T / "shap_local.csv", **kw)
    pd.DataFrame([{"hipotesis": k, "se_rechaza_H0": v["se_rechaza_H0"], **{a: b for a, b in v.items() if a != "se_rechaza_H0"}} for k, v in hip.items()]).to_csv(T / "hipotesis.csv", **kw)
    escribir_informes(ex, ex_s, pr, hip, sg, glob, shap_ok, loc)
    print(f"Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


# ------------------------------------------------------------------ informes
def versiones():
    from importlib.metadata import version
    out = {"Python": sys.version.split()[0]}
    for p in ["pandas", "numpy", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "shap", "matplotlib", "joblib", "openpyxl"]:
        out[p] = version(p)
    return out


def escribir_informes(ex, ex_s, pr, hip, sg, glob, shap_ok, loc):
    sis, cv_, B, m0, m1 = ex["sistema"], ex["cv"], pr["B"], pr["m0"], pr["m1"]
    res = ex["tablas"]["resumen"].set_index("config")
    res_s = ex_s["tablas"]["resumen"].set_index("config")
    tb = pr["tabla"].set_index(["metrica", "modelo"])
    ct = pr["contrastes"].set_index("metrica")
    sn = f"Sistema ({sis})"
    L = ["# Resultados de la tesis BPN (modelo principal sin Ipress)", "",
         f"Partición: entrenamiento 3 555 (203 BPN), prueba 889 (51 BPN); misma semilla (42) y mismos índices que `proyecto_bpn`. "
         f"CV estratificada {REPS}×5 solo en entrenamiento; bootstrap pareado estratificado de {NBOOT} réplicas.", "",
         "## 1. Validación cruzada (entrenamiento, modelo principal)", "",
         "| Config. | Modelo | Estrategia | AUC-ROC | AUC-PR | F1 | Recall | Umbral |", "|---|---|---|---|---|---|---|---|"]
    for n, r in res.iterrows():
        marca = " **(sistema)**" if r.es_sistema else ""
        L.append(f"| {n}{marca} | {r.modelo} | {r.estrategia} | {coma(r.auc_roc_media)} ± {coma(r.auc_roc_de)} | {coma(r.auc_pr_media)} | "
                 f"{coma(r.f1_media)} | {coma(r.recall_media)} | {coma(r.umbral, 4)} |")
    fr = ex["tablas"]["friedman"]
    L += ["", "Friedman (7 configuraciones, 50 pliegues): " + "; ".join(f"{r.metrica}: χ² = {coma(r.chi2, 2)}, p = {p_txt(r.p)}" for r in fr.itertuples()) + ".", "",
          f"Sistema = {sis} ({res.loc[sis, 'modelo']}, {res.loc[sis, 'estrategia']}): mayor AUC-ROC medio entre C1-C3. "
          f"Umbral (mayor precisión con Recall OOF ≥ 0,70) = {coma(res.loc[sis, 'umbral'], 4)}; C0 conserva 0,5.", "",
          "Wilcoxon-Holm del sistema frente a C0 (50 pliegues): "]
    wx = ex["tablas"]["wilcoxon"]
    for mn in ["AUC-ROC", "AUC-PR", "F1", "Recall"]:
        r = wx[(wx.metrica == mn) & (((wx.a == sis) & (wx.b == "C0")) | ((wx.a == "C0") & (wx.b == sis)))].iloc[0]
        d = r.dif_media if r.a == sis else -r.dif_media
        L.append(f"- {mn}: Δ = {coma(d, 4)}, p Holm = {p_txt(r.p_holm)}")
    sh = ex["tablas"]["shapiro"]
    L += ["", "Shapiro-Wilk de las diferencias por pliegue (sistema − C0): " + "; ".join(
        f"{r.metrica}: W = {coma(r.W, 3)}, p = {p_txt(r.p)}" for r in sh.itertuples() if r.p == r.p) + ".", "",
          "## 2. Prueba (889 registros), C0 frente al sistema", "",
          "| Métrica | C0 | IC 95 % | Sistema | IC 95 % | Δ (S − C0) | IC 95 % de Δ | Lím. inf. unilateral Δ |", "|---|---|---|---|---|---|---|---|"]
    for mn in ["Recall", "AUC-ROC", "AUC-PR", "F1"]:
        a, b, c = tb.loc[(mn, "C0")], tb.loc[(mn, sn)], ct.loc[mn]
        L.append(f"| {mn} | {coma(a.valor)} | {coma(a.ic95_inf)}–{coma(a.ic95_sup)} | {coma(b.valor)} | {coma(b.ic95_inf)}–{coma(b.ic95_sup)} | "
                 f"{coma(c.dif_sistema_menos_C0)} | {coma(c.ic95_inf)}–{coma(c.ic95_sup)} | {coma(c.lim_inf_unilateral95)} |")
    lo, hi = pr["wilson_recall_S"]
    dl = pr["delong"]
    L += ["", f"- Matriz C0: VP {m0['tp']}, FP {m0['fp']}, VN {m0['tn']}, FN {m0['fn']} (precisión {coma(m0['precision'])}; especificidad {coma(m0['especificidad'])}).",
          f"- Matriz sistema: VP {m1['tp']}, FP {m1['fp']}, VN {m1['tn']}, FN {m1['fn']} (precisión {coma(m1['precision'])}; especificidad {coma(m1['especificidad'])}; exactitud {coma(m1['accuracy'])}).",
          f"- Recall del sistema, Wilson 95 %: {coma(lo)}–{coma(hi)}.",
          f"- DeLong pareado (AUC sistema − AUC C0): z = {coma(dl['z'], 3)}, p unilateral = {p_txt(dl['p_unilateral'])}, p bilateral = {p_txt(dl['p_bilateral'])}.",
          f"- Cobertura SHAP en prueba: {'100 %' if shap_ok else 'incompleta'}. Variables más influyentes: " + ", ".join(glob.nombre.head(5)) + ".", "",
          "## 3. Decisión por hipótesis (umbrales sin modificar)", "",
          "| Hipótesis | Criterio | Resultado | Decisión |", "|---|---|---|---|"]
    h1, h2, h3, hg = hip["HE1"], hip["HE2"], hip["HE3"], hip["HG"]
    dec = lambda b: "**Se rechaza H0**" if b else "**No se rechaza H0**"
    L += [f"| HE1 (Recall) | {h1['criterio']} | Recall(S) = {coma(h1['recall_S'])}; lím. inf. unilateral = {coma(h1['recall_S_lim_inf_unilateral95'])}; lím. inf. de ΔRecall = {coma(h1['dif_recall_lim_inf'])} | {dec(h1['se_rechaza_H0'])} |",
          f"| HE2 (AUC) | {h2['criterio']} | AUC(S) = {coma(h2['auc_S'])}; lím. inf. unilateral = {coma(h2['auc_S_lim_inf_unilateral95'])}; DeLong p = {p_txt(h2['delong_p_unilateral'])} | {dec(h2['se_rechaza_H0'])} |",
          f"| HE3 (AUC-PR y F1) | {h3['criterio']} | lím. inf. ΔAUC-PR = {coma(h3['dif_auc_pr_lim_inf'])}; lím. inf. ΔF1 = {coma(h3['dif_f1_lim_inf'])}; p Holm AUC-PR = {p_txt(h3['wilcoxon_holm_auc_pr_p'])}; p Holm F1 = {p_txt(h3['wilcoxon_holm_f1_p'])} | {dec(h3['se_rechaza_H0'])} |",
          "| HE4 (SUS) | media SUS > 68 | sin datos: no se inventaron | Pendiente |",
          f"| General | {hg['criterio']} | HE1: {'sí' if h1['se_rechaza_H0'] else 'no'}; HE2: {'sí' if h2['se_rechaza_H0'] else 'no'}; HE3: {'sí' if h3['se_rechaza_H0'] else 'no'} | {dec(hg['se_rechaza_H0'])} |", "",
          "Nota: C0 usa el umbral 0,5 con prevalencia de 5,7 %, por lo que marca muy pocos positivos; las comparaciones de Recall y F1 frente a C0 "
          "están influidas por ese umbral. Las comparaciones basadas en el AUC (ROC y PR) no dependen de él.", "",
          "## 4. Sensibilidad: modelo con Ipress (solo CV, sin evaluar en prueba)", "",
          "| Config. | AUC-ROC sin Ipress | AUC-ROC con Ipress | AUC-PR sin Ipress | AUC-PR con Ipress |", "|---|---|---|---|---|"]
    for n in res.index:
        L.append(f"| {n} | {coma(res.loc[n, 'auc_roc_media'])} | {coma(res_s.loc[n, 'auc_roc_media'])} | {coma(res.loc[n, 'auc_pr_media'])} | {coma(res_s.loc[n, 'auc_pr_media'])} |")
    sis_s = ex_s["sistema"]
    L += ["", f"Sistema en el análisis con Ipress: {sis_s} (AUC-ROC CV = {coma(res_s.loc[sis_s, 'auc_roc_media'])}). "
          "El detalle por pliegue, Friedman y Wilcoxon-Holm de este conjunto están en `tablas/cv_*.csv` (columna `conjunto`).", "",
          "## 5. Subgrupos (Recall con IC de Wilson; solo descriptivo)", "",
          "| Dimensión | Subgrupo | Nacidos | Casos BPN | Recall sistema (IC 95 %) | Recall C0 (IC 95 %) |", "|---|---|---|---|---|---|"]
    for r in sg.itertuples():
        f = lambda rec, a, b: "—" if rec != rec else f"{coma(rec, 2)} ({coma(a, 2)}–{coma(b, 2)})"
        L.append(f"| {r.dimension} | {r.subgrupo} | {r.n_nacidos} | {r.casos_BPN} | {f(r.recall_S, r.ic95_inf_S, r.ic95_sup_S)} | {f(r.recall_C0, r.ic95_inf_C0, r.ic95_sup_C0)} |")
    L += ["", "Muchos subgrupos (p. ej., distritos) tienen muy pocos casos de BPN: los intervalos son amplios y no permiten inferencia.", "",
          "## 6. Archivos", "", "`tablas/` (CSV, separador `;`, coma decimal), `figuras/` (PNG 300 ppp, sin título), `sus/` (script y plantilla)."]
    (OUT / "informe_resultados.md").write_text("\n".join(L), encoding="utf-8")

    v = versiones()
    C = ["# Cambios respecto del Método de la tesis (y decisiones tomadas)", "",
         "## Diferencias frente al texto del Método", "",
         "1. **Hiperparámetros:** no se hizo búsqueda en malla completa; se usó una malla pequeña de 3 configuraciones por (modelo, estrategia) con CV 3×5 sobre el entrenamiento (AUC-ROC), y luego CV 10×5 con las configuraciones elegidas.",
         "2. **Estrategia de desbalance:** para C1, C2 y C3 se ajustaron y guardaron ambas variantes (SMOTE y pesos de clase); el sistema es la de mayor AUC medio, por lo que puede no usar SMOTE. C0 no usa balanceo.",
         f"3. **Sistema elegido:** {sis} (modelo: {res.loc[sis, 'modelo']}, estrategia: {res.loc[sis, 'estrategia']}).",
         "4. **Umbral:** el de mayor precisión con Recall fuera de pliegue ≥ 0,70, calculado sobre las predicciones apiladas de las 10 repeticiones. En `proyecto_bpn` se usaba el mayor umbral con Recall ≥ 0,70 (regla distinta).",
         "5. **Sin selección de variables:** no se aplicó el procedimiento de no inferioridad del proyecto original, porque el Método no lo contempla.",
         "6. **Predictores:** edad, estado civil, instrucción, ocupación (recodificada a AMA DE CASA / OTRA, según el Anexo 3), embarazos, hijos vivos, abortos, único/múltiple (Tipo_Parto agrupado; IGNORADO como categoría propia), distrito (IdUbigeoInei) y seguro (Financiador_Parto). Derivadas: ES_ADOL, MENOR18, AVANZ, PRIMIG, GRAN, ESTCIV_IGN, PERDIDAS (solo abortos > 0) e INT (embarazos − vivos − abortos).",
         "7. **Excluidos:** Ipress (solo en la sensibilidad), hijos fallecidos, país de la madre, año y mes del parto, talla, sexo, condición, lugar, atiende, DUR_EMB_PARTO y peso. DUR_EMB_PARTO solo define subgrupos.",
         "8. **Datos faltantes:** imputación por mediana dentro de cada pliegue; categorías con menos de 5 registros se agrupan como 'infrecuentes'.",
         "9. **Bootstrap:** pareado y estratificado por clase (2 000 réplicas); el límite unilateral es el percentil 5. DeLong implementado manualmente (versión rápida de Sun y Xu).",
         "10. **Reglas de decisión usadas:** HE1: límite inferior de ΔRecall > 0 y de Recall(S) > 0,70. HE2: DeLong p < 0,05 (unilateral) y límite inferior de AUC(S) > 0,75. HE3: límites inferiores de ΔAUC-PR y ΔF1 > 0 y Wilcoxon-Holm p < 0,05 a favor del sistema (CV). General: HE1, HE2 y HE3 rechazadas. Umbrales de hipótesis no modificados.",
         "11. **Friedman/Wilcoxon-Holm:** sobre las 50 estimaciones de CV de las 7 configuraciones, para AUC-ROC, AUC-PR, F1 y Recall (Holm sobre los 21 pares por métrica). Las estimaciones de distintas repeticiones no son totalmente independientes. Shapiro-Wilk sobre las diferencias por pliegue (sistema − C0).",
         "12. **Prueba:** C0 y el sistema se evaluaron una vez cada uno. El análisis de sensibilidad con Ipress se evaluó solo en CV para no volver a usar la prueba.",
         "13. **Transparencia:** la partición de prueba ya se había usado antes en `proyecto_bpn` para un modelo exploratorio que incluía Ipress (AUC 0,638 en prueba). Este diseño sigue el Anexo 3, que ya excluía Ipress, pero debe declararse en el texto.",
         "14. **Subgrupos:** distrito y seguro incluyen cifras con muy pocos casos; seguro se agrupa en SIS, ESSALUD y OTROS.",
         "15. **SUS:** solo script y plantilla; no se generaron datos.", "",
         "## Versiones de las bibliotecas", ""]
    C += [f"- {k}: {val}" for k, val in v.items()]
    (OUT / "cambios_metodo.md").write_text("\n".join(C), encoding="utf-8")


if __name__ == "__main__":
    main()
