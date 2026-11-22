"""Entradas del modelo (10 datos conocidos antes del parto) y pipeline de entrenamiento.

El modelo NO usa Ipress ni datos del parto (talla, sexo, condición, lugar, quién atiende, semanas de gestación).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

# Ubigeo INEI de los distritos de la provincia de Rodríguez de Mendoza (verificar con el INEI)
DISTRITOS = {"010601": "San Nicolás", "010602": "Chirimoto", "010603": "Cochamal", "010604": "Huambo",
             "010605": "Limabamba", "010606": "Longar", "010607": "Mariscal Benavides", "010608": "Milpuc",
             "010609": "Omia", "010610": "Santa Rosa", "010611": "Totora", "010612": "Vista Alegre"}

NUM = ["Edad_Madre", "Num_embar_madre", "Hijos_vivo_madre", "nacmuer_abort_madre",
       "ES_ADOL", "MENOR18", "AVANZ", "PRIMIG", "GRAN", "PERDIDAS", "ESTCIV_IGN", "INT"]
CAT = ["Estado_Civil", "Nivel_Intrucción_Madre", "OCUPACION", "MULTIPLE", "DISTRITO", "SEGURO"]

# Cada uno de los 10 datos del formulario agrupa las columnas del modelo que dependen de él
GRUPOS = {
    "edad": ["Edad_Madre", "ES_ADOL", "MENOR18", "AVANZ"],
    "embarazos": ["Num_embar_madre", "PRIMIG", "GRAN", "INT"],
    "hijos_vivos": ["Hijos_vivo_madre"],
    "abortos": ["nacmuer_abort_madre", "PERDIDAS"],
    "estado_civil": ["Estado_Civil", "ESTCIV_IGN"],
    "instruccion": ["Nivel_Intrucción_Madre"],
    "ocupacion": ["OCUPACION"],
    "multiple": ["MULTIPLE"],
    "distrito": ["DISTRITO"],
    "seguro": ["SEGURO"],
}
COLUMNAS_CRUDAS = ["Edad_Madre", "Num_embar_madre", "Hijos_vivo_madre", "nacmuer_abort_madre", "Estado_Civil",
                   "Nivel_Intrucción_Madre", "DESC_OCUPACION", "Tipo_Parto", "IdUbigeoInei", "Financiador_Parto"]


def construir_entradas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas del CNV en las entradas del modelo (con las variables derivadas)."""
    d = pd.DataFrame(index=df.index)
    ed, emb = pd.to_numeric(df["Edad_Madre"]), pd.to_numeric(df["Num_embar_madre"])
    viv, ab = pd.to_numeric(df["Hijos_vivo_madre"]), pd.to_numeric(df["nacmuer_abort_madre"])
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
    d["DISTRITO"] = df["IdUbigeoInei"].astype(str).str.zfill(6)
    d["SEGURO"] = df["Financiador_Parto"]
    return d[NUM + CAT]


def crear_pipeline(params=None):
    """Random Forest con pesos de clase; imputación y codificación dentro del Pipeline."""
    p = dict(C.RF_PARAMS, **(params or {}))
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), CAT)])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=p["min_samples_leaf"], max_features="sqrt",
                                class_weight="balanced_subsample", random_state=C.SEMILLA, n_jobs=1)
    return Pipeline([("prep", pre), ("clf", rf)])


def grupo_de_columna(nombres_transformados):
    """Mapea cada columna transformada al dato del formulario del que depende."""
    col_a_grupo = {c: g for g, cols in GRUPOS.items() for c in cols}
    out = []
    for n in nombres_transformados:
        base = n.split("__", 1)[1]
        cols = sorted(col_a_grupo, key=len, reverse=True)
        out.append(next((col_a_grupo[c] for c in cols if base == c or base.startswith(c + "_")), base))
    return np.array(out)
