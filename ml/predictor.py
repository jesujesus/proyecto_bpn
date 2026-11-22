"""Predictor reutilizable (API y app): puntaje de riesgo, nivel y explicación SHAP en lenguaje sencillo.

Entradas (nombres amigables):
  edad, estado_civil, instruccion, ocupacion ("AMA DE CASA" | "OTRA"), distrito (código ubigeo), seguro,
  embarazos (contando el actual), hijos_vivos (previos, sin el que va a nacer), abortos (previos),
  embarazo_multiple (bool).
En el CNV, "hijos vivos" incluye al recién nacido, por eso internamente se suma 1 (y se acota a 5 como en los datos).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import shap

import config as C
from ml.features import GRUPOS, construir_entradas, grupo_de_columna

NOMBRES_GRUPO = {"edad": "Edad", "embarazos": "Número de embarazos", "hijos_vivos": "Hijos vivos",
                 "abortos": "Abortos o nacidos muertos", "estado_civil": "Estado civil", "instruccion": "Instrucción",
                 "ocupacion": "Ocupación", "multiple": "Embarazo único o múltiple", "distrito": "Distrito", "seguro": "Seguro"}
ESTADO_CIVIL = {"SOLTERO": "soltera", "CASADO": "casada", "CONVIVIENTE": "conviviente", "DIVORCIADO": "divorciada",
                "VIUDO": "viuda", "IGNORADO": "no registrado"}
INSTRUCCION = {"NINGUN NIVEL/ILETRADO": "ningún nivel", "INCIAL/PRE-ESCOLAR": "inicial", "PRIMARIA INCOMPLETA": "primaria incompleta",
               "PRIMARIA COMPLETA": "primaria completa", "SECUNDARIA INCOMPLETA": "secundaria incompleta",
               "SECUNDARIA COMPLETA": "secundaria completa", "SUPERIOR NO UNIV. INCOMPLETA": "superior técnica incompleta",
               "SUPERIOR NO UNIV. COMPLETA": "superior técnica completa", "SUPERIOR UNIV. INCOMPLETA": "superior universitaria incompleta",
               "SUPERIOR UNIV. COMPLETA": "superior universitaria completa", "IGNORADO": "no registrado"}
MIN_CONTRIBUCION = 0.01       # contribuciones menores se consideran irrelevantes


def validar_datos(d: dict) -> list:
    """Devuelve una lista de mensajes claros (vacía si los datos son válidos)."""
    m = []
    if not 10 <= d["edad"] <= 55:
        m.append("La edad debe estar entre 10 y 55 años.")
    if d["embarazos"] < 1:
        m.append("El número de embarazos debe ser al menos 1 (cuente el embarazo actual).")
    if d["hijos_vivos"] < 0 or d["abortos"] < 0:
        m.append("Hijos vivos y abortos no pueden ser negativos.")
    if d["hijos_vivos"] + d["abortos"] > d["embarazos"] - 1 and not d.get("embarazo_multiple_previo", False):
        m.append(f"Los datos no cuadran: {d['hijos_vivos']} hijo(s) vivo(s) + {d['abortos']} aborto(s) suman "
                 f"{d['hijos_vivos'] + d['abortos']}, pero con {d['embarazos']} embarazo(s) (incluido el actual) "
                 f"el máximo posible es {max(d['embarazos'] - 1, 0)}. Revise el número de embarazos, hijos vivos o abortos.")
    return m


def _frase(grupo, d):
    if grupo == "edad":
        return f"edad de {d['edad']} años"
    if grupo == "embarazos":
        return "primer embarazo" if d["embarazos"] == 1 else f"{d['embarazos']} embarazos"
    if grupo == "hijos_vivos":
        n = d["hijos_vivos"]
        return "ningún hijo vivo previo" if n == 0 else f"{n} hijo{'s' if n > 1 else ''} vivo{'s' if n > 1 else ''} previo{'s' if n > 1 else ''}"
    if grupo == "abortos":
        n = d["abortos"]
        return "sin abortos ni nacidos muertos previos" if n == 0 else f"{n} aborto{'s' if n > 1 else ''} o nacido{'s' if n > 1 else ''} muerto{'s' if n > 1 else ''} previo{'s' if n > 1 else ''}"
    if grupo == "estado_civil":
        return f"estado civil: {ESTADO_CIVIL.get(d['estado_civil'], d['estado_civil'].lower())}"
    if grupo == "instruccion":
        return f"nivel de instrucción: {INSTRUCCION.get(d['instruccion'], d['instruccion'].lower())}"
    if grupo == "ocupacion":
        return "ama de casa" if d["ocupacion"] == "AMA DE CASA" else "otra ocupación"
    if grupo == "multiple":
        return "embarazo gemelar o múltiple" if d.get("embarazo_multiple") else "embarazo único"
    if grupo == "distrito":
        return f"residir en {C_DISTRITOS.get(d['distrito'], d['distrito'])}"
    return f"seguro: {d['seguro']}"


from ml.features import DISTRITOS as C_DISTRITOS  # noqa: E402


class Predictor:
    def __init__(self, ruta=None):
        self.b = joblib.load(ruta or (C.MODELOS / "modelo_bpn.joblib"))
        self.pipe = self.b["pipeline"]
        self.prep = self.pipe.named_steps["prep"]
        self.explainer = shap.TreeExplainer(self.pipe.named_steps["clf"])
        self.grupo_col = grupo_de_columna(self.prep.get_feature_names_out())
        self.total = 0
        self.con_shap = 0

    @staticmethod
    def a_columnas(d: dict) -> pd.DataFrame:
        """Convierte los datos del formulario a las columnas del CNV que usa el modelo."""
        return pd.DataFrame([{
            "Edad_Madre": d["edad"], "Num_embar_madre": min(d["embarazos"], 5),
            "Hijos_vivo_madre": min(d["hijos_vivos"] + 1, 5), "nacmuer_abort_madre": d["abortos"],
            "Estado_Civil": d["estado_civil"], "Nivel_Intrucción_Madre": d["instruccion"],
            "DESC_OCUPACION": d["ocupacion"], "Tipo_Parto": "DOBLE" if d.get("embarazo_multiple") else "UNICO",
            "IdUbigeoInei": d["distrito"], "Financiador_Parto": d["seguro"]}])

    def nivel(self, score):
        if score >= self.b["umbral_alto"]:
            return "ALTO"
        if score >= self.b["umbral"]:
            return "MEDIO"
        return "BAJO"

    def shap_por_dato(self, X):
        Z = self.prep.transform(X)
        sv = self.explainer.shap_values(Z)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        agg = pd.DataFrame(sv, columns=self.prep.get_feature_names_out()).T.groupby(self.grupo_col).sum().T
        return agg, np.isfinite(sv).all(axis=1)

    def predecir(self, d: dict):
        X = construir_entradas(self.a_columnas(d))
        score = float(self.pipe.predict_proba(X)[0, 1])
        aumentan, disminuyen, ok = [], [], False
        try:
            agg, mask = self.shap_por_dato(X)
            ok = bool(mask[0])
            fila = agg.iloc[0].sort_values()
            for g, v in fila[::-1].items():
                if v >= MIN_CONTRIBUCION and len(aumentan) < 4:
                    aumentan.append({"dato": NOMBRES_GRUPO[g], "texto": _frase(g, d), "contribucion": float(v)})
            for g, v in fila.items():
                if v <= -MIN_CONTRIBUCION and len(disminuyen) < 4:
                    disminuyen.append({"dato": NOMBRES_GRUPO[g], "texto": _frase(g, d), "contribucion": float(v)})
            detalle = [{"dato": NOMBRES_GRUPO[g], "texto": _frase(g, d), "contribucion": float(v)} for g, v in agg.iloc[0].items()]
        except Exception:
            ok, detalle = False, []
        self.total += 1
        self.con_shap += int(ok)
        return {"puntaje": round(score * 100, 1), "score": score, "nivel_riesgo": self.nivel(score),
                "supera_umbral": bool(score >= self.b["umbral"]), "umbral": float(self.b["umbral"]),
                "umbral_alto": float(self.b["umbral_alto"]), "modelo": self.b["modelo"], "explicacion_shap": ok,
                "factores_aumentan": aumentan, "factores_disminuyen": disminuyen, "detalle": detalle}

    @property
    def cobertura_shap(self):
        return self.con_shap / self.total if self.total else None
