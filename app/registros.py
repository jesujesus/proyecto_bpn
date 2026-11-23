"""Registro en archivos CSV (la "base de datos" de la app): predicciones y cuestionario SUS, en archivos separados.

 data/registros/predicciones.csv   una fila por evaluación de gestante
 data/registros/sus_respuestas.csv una fila por cuestionario SUS (anónimo)

Los CSV se guardan con UTF-8 con BOM, para que Excel muestre bien las tildes.
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C

SUS_ITEMS = [
    "Creo que me gustaría usar esta herramienta con frecuencia.",
    "Encontré la herramienta innecesariamente compleja.",
    "Pensé que la herramienta era fácil de usar.",
    "Creo que necesitaría el apoyo de personal experto para poder usar esta herramienta.",
    "Encontré que las diversas funciones de la herramienta estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en la herramienta.",
    "Imagino que la mayoría de las personas aprendería a usar esta herramienta muy rápidamente.",
    "Encontré la herramienta muy tediosa de usar.",
    "Me sentí muy seguro(a) usando la herramienta.",
    "Necesité aprender muchas cosas antes de poder empezar a usar esta herramienta.",
]
ESCALA = ["Totalmente en desacuerdo", "En desacuerdo", "Ni de acuerdo ni en desacuerdo", "De acuerdo", "Totalmente de acuerdo"]
PROFESIONES = ["Obstetra", "Enfermera(o)", "Médico(a)", "Otra"]
EXPERIENCIA = ["Menos de 5", "De 5 a 10", "Más de 10"]

CAMPOS_PRED = ["registrado_el", "edad", "estado_civil", "instruccion", "ocupacion", "distrito_codigo", "distrito", "seguro",
               "embarazos", "hijos_vivos_previos", "abortos", "embarazo_multiple", "puntaje_riesgo", "nivel_riesgo",
               "supera_umbral", "factores_aumentan", "factores_disminuyen", "explicacion_shap", "modelo"]
CAMPOS_SUS = (["cuestionario_n", "fecha", "registrado_el", "casos_registrados", "profesion", "profesion_otra", "experiencia",
               "usa_computadora"] + [f"p{i}" for i in range(1, 11)] + ["puntaje_sus", "interpretacion", "comentarios"])


def _ahora():
    return datetime.now().isoformat(timespec="seconds")


def _agregar(ruta: Path, campos, fila: dict) -> bool:
    """Agrega una fila al CSV. Devuelve False si no se pudo (p. ej., el archivo está abierto en Excel)."""
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not ruta.exists() or ruta.stat().st_size == 0
        with open(ruta, "a", newline="", encoding="utf-8-sig" if nuevo else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            if nuevo:
                w.writeheader()
            w.writerow(fila)
        return True
    except OSError:
        return False


def _leer(ruta: Path, campos) -> pd.DataFrame:
    try:
        if ruta.exists() and ruta.stat().st_size > 0:
            return pd.read_csv(ruta, encoding="utf-8-sig")
    except Exception:
        pass
    return pd.DataFrame(columns=campos)


# ------------------------------------------------------------------ predicciones
def guardar_prediccion(datos: dict, res: dict, distritos: dict) -> bool:
    fila = {"registrado_el": _ahora(), "edad": datos["edad"], "estado_civil": datos["estado_civil"],
            "instruccion": datos["instruccion"], "ocupacion": datos["ocupacion"], "distrito_codigo": datos["distrito"],
            "distrito": distritos.get(datos["distrito"], datos["distrito"]), "seguro": datos["seguro"],
            "embarazos": datos["embarazos"], "hijos_vivos_previos": datos["hijos_vivos"], "abortos": datos["abortos"],
            "embarazo_multiple": "Sí" if datos.get("embarazo_multiple") else "No", "puntaje_riesgo": res["puntaje"],
            "nivel_riesgo": res["nivel_riesgo"], "supera_umbral": "Sí" if res["supera_umbral"] else "No",
            "factores_aumentan": " | ".join(f["texto"] for f in res["factores_aumentan"]),
            "factores_disminuyen": " | ".join(f["texto"] for f in res["factores_disminuyen"]),
            "explicacion_shap": "Sí" if res["explicacion_shap"] else "No", "modelo": res["modelo"]}
    return _agregar(C.PRED_CSV, CAMPOS_PRED, fila)


def leer_predicciones() -> pd.DataFrame:
    return _leer(C.PRED_CSV, CAMPOS_PRED)


# ------------------------------------------------------------------ SUS
def puntaje_sus(resp):
    """SUS = 2,5 * [Σ(impares - 1) + Σ(5 - pares)], con 10 respuestas de 1 a 5."""
    if len(resp) != 10:
        raise ValueError("El SUS requiere 10 respuestas")
    impares = sum(resp[i] - 1 for i in range(0, 10, 2))   # ítems 1,3,5,7,9
    pares = sum(5 - resp[i] for i in range(1, 10, 2))     # ítems 2,4,6,8,10
    return 2.5 * (impares + pares)


def interpretar_sus(p):
    if p >= 85:
        return "Excelente"
    if p >= 73:
        return "Bueno"
    if p >= 68:
        return "Aceptable (promedio)"
    if p >= 51:
        return "Pobre"
    return "Muy pobre"


def leer_sus() -> pd.DataFrame:
    return _leer(C.SUS_CSV, CAMPOS_SUS)


def siguiente_sus() -> int:
    df = leer_sus()
    return int(df["cuestionario_n"].max()) + 1 if len(df) else 1


def guardar_sus(fecha, casos, profesion, profesion_otra, experiencia, usa_pc, resp, comentarios):
    """Guarda un cuestionario. Devuelve (numero, puntaje) o None si no se pudo escribir el archivo."""
    n = siguiente_sus()
    p = puntaje_sus(resp)
    fila = {"cuestionario_n": n, "fecha": fecha.strftime("%d/%m/%Y"), "registrado_el": _ahora(), "casos_registrados": casos,
            "profesion": profesion, "profesion_otra": profesion_otra.strip() if profesion == "Otra" else "",
            "experiencia": experiencia, "usa_computadora": usa_pc, "puntaje_sus": p, "interpretacion": interpretar_sus(p),
            "comentarios": comentarios.strip()}
    fila.update({f"p{i}": r for i, r in enumerate(resp, 1)})
    return (n, p) if _agregar(C.SUS_CSV, CAMPOS_SUS, fila) else None


def alfa_cronbach(df: pd.DataFrame):
    """Alfa de Cronbach con los ítems pares recodificados (requiere al menos 3 cuestionarios)."""
    if len(df) < 3:
        return None
    R = df[[f"p{i}" for i in range(1, 11)]].astype(float).values.copy()
    R[:, 1::2] = 6 - R[:, 1::2]
    var_total = R.sum(axis=1).var(ddof=1)
    return float(10 / 9 * (1 - R.var(axis=0, ddof=1).sum() / var_total)) if var_total > 0 else None
