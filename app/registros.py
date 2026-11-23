"""Registro de predicciones y cuestionarios SUS (tablas/archivos separados).

Destino de los datos:
  * Si hay una URL de base de datos (variable DATABASE_URL o `database_url` en los secretos de Streamlit):
    PostgreSQL (p. ej., Supabase) en las tablas `predicciones` y `sus_respuestas`.
    También se admite `sqlite:///ruta.db`, solo para pruebas.
  * Si no hay URL: archivos CSV locales en data/registros/ (predicciones.csv y sus_respuestas.csv, UTF-8 con BOM).
"""
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
TIPOS = {"edad": "INTEGER", "embarazos": "INTEGER", "hijos_vivos_previos": "INTEGER", "abortos": "INTEGER", "puntaje_riesgo": "REAL",
         "cuestionario_n": "INTEGER", "casos_registrados": "INTEGER", "puntaje_sus": "REAL",
         **{f"p{i}": "INTEGER" for i in range(1, 11)}}
T_PRED, T_SUS = "predicciones", "sus_respuestas"
ERROR_GUARDADO = ("No se pudo guardar el registro. Si se usan archivos CSV, ciérrelos en Excel; "
                  "si se usa la base de datos, revise la conexión e inténtelo de nuevo.")


def _ahora():
    return datetime.now().isoformat(timespec="seconds")


def _py(v):
    return v.item() if hasattr(v, "item") else v


# ------------------------------------------------------------------ destino
def _url():
    """URL de la base de datos: variable de entorno DATABASE_URL o secreto `DATABASE_URL`/`database_url` de Streamlit."""
    u = os.environ.get("DATABASE_URL")
    if not u:
        try:
            import streamlit as st
            u = st.secrets.get("DATABASE_URL") or st.secrets.get("database_url")
        except Exception:
            u = None
    return str(u).strip() if u else None


def inicializar():
    """Comprueba la conexión y crea las tablas si faltan. Devuelve 'csv', 'ok' o un texto de error (sin datos sensibles)."""
    u = _url()
    if not u:
        return "csv"
    try:
        _con(_BD(u)).close()
        return "ok"
    except Exception as e:
        return f"error ({type(e).__name__})"


def donde():
    """Texto breve para mostrar dónde se guardan los datos."""
    return "base de datos" if _url() else "data/registros/ (archivos CSV)"


class _BD:
    def __init__(self, url):
        self.url = url
        self.sqlite = url.startswith("sqlite")
        self.ph = "?" if self.sqlite else "%s"

    def conectar(self):
        if self.sqlite:
            import sqlite3
            p = Path(self.url.split("sqlite:///", 1)[1])
            p = p if p.is_absolute() else C.RAIZ / p
            p.parent.mkdir(parents=True, exist_ok=True)
            return sqlite3.connect(p)
        import psycopg2
        kw = {"connect_timeout": 10}
        if "sslmode" not in self.url and "localhost" not in self.url and "127.0.0.1" not in self.url:
            kw["sslmode"] = "require"
        return psycopg2.connect(self.url, **kw)

    def preparar(self, con):
        serial = "INTEGER PRIMARY KEY AUTOINCREMENT" if self.sqlite else "SERIAL PRIMARY KEY"
        cur = con.cursor()
        for tabla, campos in ((T_PRED, CAMPOS_PRED), (T_SUS, CAMPOS_SUS)):
            cols = ", ".join(f"{c} {TIPOS.get(c, 'TEXT')}" for c in campos)
            extra = ", UNIQUE (cuestionario_n)" if tabla == T_SUS else ""
            cur.execute(f"CREATE TABLE IF NOT EXISTS {tabla} (id {serial}, {cols}{extra})")
            if not self.sqlite:
                cur.execute(f"ALTER TABLE {tabla} ENABLE ROW LEVEL SECURITY")   # la API pública de Supabase no accede; la app sí (rol postgres)
        con.commit()


_PREPARADAS = set()


def _con(bd):
    con = bd.conectar()
    if bd.url not in _PREPARADAS:
        bd.preparar(con)
        _PREPARADAS.add(bd.url)
    return con


def _insertar_bd(bd, tabla, campos, fila, auto_n=False):
    """Inserta una fila. Con auto_n, cuestionario_n se asigna en la misma sentencia (MAX + 1). Devuelve cuestionario_n o True."""
    cols, marcas, params = [], [], []
    for c in campos:
        cols.append(c)
        if auto_n and c == "cuestionario_n":
            marcas.append(f"(SELECT COALESCE(MAX(cuestionario_n), 0) + 1 FROM {tabla})")
        else:
            marcas.append(bd.ph)
            params.append(_py(fila.get(c)))
    q = f"INSERT INTO {tabla} ({', '.join(cols)}) VALUES ({', '.join(marcas)})" + (" RETURNING cuestionario_n" if auto_n else "")
    con = _con(bd)
    try:
        cur = con.cursor()
        cur.execute(q, params)
        res = cur.fetchone()[0] if auto_n else True
        con.commit()
        return res
    finally:
        con.close()


def _leer_bd(bd, tabla, campos):
    con = _con(bd)
    try:
        cur = con.cursor()
        cur.execute(f"SELECT {', '.join(campos)} FROM {tabla} ORDER BY id")
        return pd.DataFrame(cur.fetchall(), columns=campos)
    finally:
        con.close()


# ------------------------------------------------------------------ CSV
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


def _leer_csv(ruta: Path, campos) -> pd.DataFrame:
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
    u = _url()
    if u:
        try:
            return bool(_insertar_bd(_BD(u), T_PRED, CAMPOS_PRED, fila))
        except Exception:
            return False
    return _agregar(C.PRED_CSV, CAMPOS_PRED, fila)


def leer_predicciones() -> pd.DataFrame:
    u = _url()
    if u:
        try:
            return _leer_bd(_BD(u), T_PRED, CAMPOS_PRED)
        except Exception:
            return pd.DataFrame(columns=CAMPOS_PRED)
    return _leer_csv(C.PRED_CSV, CAMPOS_PRED)


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
    u = _url()
    if u:
        try:
            return _leer_bd(_BD(u), T_SUS, CAMPOS_SUS)
        except Exception:
            return pd.DataFrame(columns=CAMPOS_SUS)
    return _leer_csv(C.SUS_CSV, CAMPOS_SUS)


def siguiente_sus() -> int:
    """Número que se mostrará en el formulario (el definitivo lo asigna el guardado)."""
    df = leer_sus()
    return int(df["cuestionario_n"].max()) + 1 if len(df) else 1


def guardar_sus(fecha, casos, profesion, profesion_otra, experiencia, usa_pc, resp, comentarios):
    """Guarda un cuestionario. Devuelve (numero, puntaje) o None si no se pudo guardar."""
    p = puntaje_sus(resp)
    fila = {"fecha": fecha.strftime("%d/%m/%Y"), "registrado_el": _ahora(), "casos_registrados": casos,
            "profesion": profesion, "profesion_otra": profesion_otra.strip() if profesion == "Otra" else "",
            "experiencia": experiencia, "usa_computadora": usa_pc, "puntaje_sus": p, "interpretacion": interpretar_sus(p),
            "comentarios": comentarios.strip()}
    fila.update({f"p{i}": r for i, r in enumerate(resp, 1)})
    u = _url()
    if u:
        bd = _BD(u)
        for _ in range(3):                                  # reintenta si dos personas guardan a la vez (UNIQUE en cuestionario_n)
            try:
                return _insertar_bd(bd, T_SUS, CAMPOS_SUS, fila, auto_n=True), p
            except Exception:
                continue
        return None
    n = siguiente_sus()
    fila["cuestionario_n"] = n
    return (n, p) if _agregar(C.SUS_CSV, CAMPOS_SUS, fila) else None


def alfa_cronbach(df: pd.DataFrame):
    """Alfa de Cronbach con los ítems pares recodificados (requiere al menos 3 cuestionarios)."""
    if len(df) < 3:
        return None
    R = df[[f"p{i}" for i in range(1, 11)]].astype(float).values.copy()
    R[:, 1::2] = 6 - R[:, 1::2]
    var_total = R.sum(axis=1).var(ddof=1)
    return float(10 / 9 * (1 - R.var(axis=0, ddof=1).sum() / var_total)) if var_total > 0 else None
