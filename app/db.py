"""Almacenamiento SQLite (historial de evaluaciones, reporte y cuestionario SUS)."""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as C

SUS_ITEMS = [
    "Creo que me gustaría usar este sistema con frecuencia.",
    "Encontré el sistema innecesariamente complejo.",
    "Pensé que el sistema era fácil de usar.",
    "Creo que necesitaría el apoyo de un técnico para poder usar este sistema.",
    "Encontré que las diversas funciones del sistema estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en este sistema.",
    "Imagino que la mayoría de las personas aprenderían a usar este sistema muy rápidamente.",
    "Encontré el sistema muy incómodo de usar.",
    "Me sentí muy seguro/a usando el sistema.",
    "Necesité aprender muchas cosas antes de poder empezar a usar este sistema.",
]


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


def conectar():
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(C.DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS evaluaciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, entrada TEXT, puntaje REAL, nivel TEXT,
        shap_ok INTEGER, aumentan TEXT, disminuyen TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS sus (
        id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, participante TEXT, rol TEXT, respuestas TEXT,
        puntaje REAL, interpretacion TEXT, comentario TEXT)""")
    return con


def guardar_evaluacion(entrada, res):
    con = conectar()
    con.execute("INSERT INTO evaluaciones (fecha, entrada, puntaje, nivel, shap_ok, aumentan, disminuyen) VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), json.dumps(entrada, ensure_ascii=False), res["puntaje"],
                 res["nivel_riesgo"], int(res["explicacion_shap"]),
                 json.dumps([f["texto"] for f in res["factores_aumentan"]], ensure_ascii=False),
                 json.dumps([f["texto"] for f in res["factores_disminuyen"]], ensure_ascii=False)))
    con.commit()
    con.close()


def leer_historial():
    con = conectar()
    df = pd.read_sql_query("SELECT * FROM evaluaciones ORDER BY id DESC", con)
    con.close()
    return df


def guardar_sus(participante, rol, resp, comentario):
    p = puntaje_sus(resp)
    con = conectar()
    con.execute("INSERT INTO sus (fecha, participante, rol, respuestas, puntaje, interpretacion, comentario) VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), participante, rol, json.dumps(resp), p,
                 interpretar_sus(p), comentario))
    con.commit()
    con.close()
    return p


def leer_sus():
    con = conectar()
    df = pd.read_sql_query("SELECT id, fecha, participante, rol, puntaje, interpretacion, comentario FROM sus ORDER BY id DESC", con)
    con.close()
    return df
