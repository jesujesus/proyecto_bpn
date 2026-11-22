"""Entrena el modelo desplegable (el sistema de la tesis: Random Forest con pesos de clase, SIN Ipress).

 1. Partición estratificada 80/20 (semilla 42). La prueba no se toca aquí.
 2. CV estratificada 10 x 5 SOLO en entrenamiento -> predicciones fuera de pliegue.
 3. Umbrales: riesgo MEDIO = mayor precisión con Recall >= 0,70; riesgo ALTO = Recall ~ 0,40.
 4. Ajuste final en todo el entrenamiento y guardado en models/modelo_bpn.joblib.

La comparación completa de modelos (C0-C3), las pruebas de hipótesis y las métricas en prueba están en
ml/analisis_tesis.py (resultados en reports/tesis/).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import config as C
from ml.features import COLUMNAS_CRUDAS, DISTRITOS, construir_entradas, crear_pipeline
from ml.metricas import umbral_max_precision, umbral_por_recall


def _pliegue(pipe, X, y, tr, te):
    m = clone(pipe).fit(X.iloc[tr], y[tr])
    return te, m.predict_proba(X.iloc[te])[:, 1]


def main():
    t0 = time.time()
    C.MODELOS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.CSV_LIMPIO, dtype={"IdUbigeoInei": str, "Ipress": str})
    y_all = df[C.TARGET].values
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE, stratify=y_all, random_state=C.SEMILLA)
    json.dump({"train": idx_tr.tolist(), "test": idx_te.tolist()}, open(C.SPLIT_JSON, "w"))
    X_all = construir_entradas(df)
    X, y = X_all.iloc[idx_tr].reset_index(drop=True), y_all[idx_tr]
    print(f"Entrenamiento: {len(idx_tr)} (BPN {y.sum()}) | Prueba reservada: {len(idx_te)}")

    pipe = crear_pipeline()
    splits = list(RepeatedStratifiedKFold(n_splits=C.CV_FOLDS, n_repeats=C.CV_REPETICIONES, random_state=C.SEMILLA).split(X, y))
    res = Parallel(n_jobs=4)(delayed(_pliegue)(pipe, X, y, tr, te) for tr, te in splits)
    oof = np.full((C.CV_REPETICIONES, len(y)), np.nan)
    aucs = []
    for i, (te, p) in enumerate(res):
        oof[i // C.CV_FOLDS, te] = p
        aucs.append(roc_auc_score(y[te], p))
    pool_p, pool_y = oof.ravel(), np.tile(y, C.CV_REPETICIONES)
    umbral = umbral_max_precision(pool_y, pool_p, C.RECALL_OBJETIVO)
    umbral_alto = umbral_por_recall(pool_y, pool_p, C.RECALL_ALTO)
    print(f"CV 10x5: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | umbral medio = {umbral:.4f} | umbral alto = {umbral_alto:.4f}")

    final = crear_pipeline().fit(X, y)
    bundle = {
        "pipeline": final, "modelo": "Random Forest (pesos de clase)", "version": "2.0", "params": C.RF_PARAMS,
        "umbral": umbral, "umbral_alto": umbral_alto, "tasa_base": float(y.mean()), "auc_cv": float(np.mean(aucs)),
        "opciones": {
            "distritos": DISTRITOS,
            "estado_civil": sorted(df["Estado_Civil"].dropna().unique().tolist()),
            "instruccion": sorted(df["Nivel_Intrucción_Madre"].dropna().unique().tolist()),
            "seguro": df["Financiador_Parto"].value_counts().index.tolist()},
        "columnas": COLUMNAS_CRUDAS}
    joblib.dump(bundle, C.MODELOS / "modelo_bpn.joblib")
    print(f"Modelo guardado en models/modelo_bpn.joblib [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
