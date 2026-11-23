"""Genera DATOS SIMULADOS de demostración (10 predicciones y 10 cuestionarios SUS) en data/demo/.

Son datos inventados, solo para mostrar la app o probar el despliegue. NO son resultados de la tesis:
no deben usarse para contrastar HE4 ni mezclarse con los registros reales de data/registros/.
Cada fila lleva la columna `simulado` = Sí. Las predicciones las calcula el modelo real con entradas simuladas.

Uso:  python ml/generar_demo.py
"""
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import numpy as np

import config as C
import registros as reg
from ml.features import DISTRITOS
from ml.predictor import Predictor, validar_datos

SEMILLA = 7
DEMO = C.RAIZ / "data" / "demo"
rng = np.random.default_rng(SEMILLA)


def escribir(ruta, campos, filas):
    DEMO.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos + ["simulado"])
        w.writeheader()
        for fila in filas:
            w.writerow({**fila, "simulado": "Sí"})


def elegir(opciones, pesos):
    return opciones[rng.choice(len(opciones), p=np.array(pesos) / sum(pesos))]


def gestante(forzar_multiple=False):
    emb = int(elegir([1, 2, 3, 4, 5], [29, 28, 20, 11, 11]))
    abortos = int(elegir([0, 1, 2], [80, 17, 3]))
    abortos = min(abortos, emb - 1)
    vivos = int(rng.integers(0, emb - abortos)) if emb - abortos > 0 else 0
    return {
        "edad": int(np.clip(rng.normal(27, 7), 15, 44)),
        "estado_civil": elegir(["SOLTERO", "CASADO", "CONVIVIENTE", "IGNORADO"], [86, 6, 1, 7]),
        "instruccion": elegir(["SECUNDARIA COMPLETA", "SECUNDARIA INCOMPLETA", "PRIMARIA COMPLETA", "PRIMARIA INCOMPLETA",
                               "SUPERIOR NO UNIV. COMPLETA", "SUPERIOR UNIV. COMPLETA"], [24, 22, 21, 11, 10, 5]),
        "ocupacion": elegir(["AMA DE CASA", "OTRA"], [93, 7]),
        "distrito": elegir(list(DISTRITOS), [15, 11, 2, 6, 5, 5, 6, 2, 34, 2, 1, 10]),
        "seguro": elegir(["SIS", "ESSALUD"], [92, 8]),
        "embarazos": emb, "hijos_vivos": vivos, "abortos": abortos,
        "embarazo_multiple": bool(forzar_multiple)}


def predicciones(n=10):
    P = Predictor()
    t0 = datetime.now()
    filas = []
    for i in range(n):
        d = gestante(forzar_multiple=(i == 3))          # un caso gemelar, para mostrar un resultado de riesgo alto
        assert not validar_datos(d), d
        r = P.predecir(d)
        fila = {"registrado_el": (t0 + timedelta(minutes=3 * i)).isoformat(timespec="seconds"), "edad": d["edad"],
                "estado_civil": d["estado_civil"], "instruccion": d["instruccion"], "ocupacion": d["ocupacion"],
                "distrito_codigo": d["distrito"], "distrito": DISTRITOS[d["distrito"]], "seguro": d["seguro"],
                "embarazos": d["embarazos"], "hijos_vivos_previos": d["hijos_vivos"], "abortos": d["abortos"],
                "embarazo_multiple": "Sí" if d["embarazo_multiple"] else "No", "puntaje_riesgo": r["puntaje"],
                "nivel_riesgo": r["nivel_riesgo"], "supera_umbral": "Sí" if r["supera_umbral"] else "No",
                "factores_aumentan": " | ".join(f["texto"] for f in r["factores_aumentan"]),
                "factores_disminuyen": " | ".join(f["texto"] for f in r["factores_disminuyen"]),
                "explicacion_shap": "Sí" if r["explicacion_shap"] else "No", "modelo": r["modelo"]}
        filas.append(fila)
    escribir(DEMO / "predicciones_demo.csv", reg.CAMPOS_PRED, filas)
    return filas


COMENTARIOS = ["", "", "", "Fácil de usar", "Las explicaciones ayudan a entender el resultado",
               "Sería útil poder imprimir el resultado", "Al inicio no entendí qué era el puntaje", ""]


def sus(n=10):
    hoy = date.today()
    filas = []
    for i in range(n):
        satisf = float(np.clip(rng.normal(0.65, 0.17), 0.05, 1.0))       # satisfacción latente del participante
        resp = []
        for k in range(10):
            valor = float(np.clip(satisf + rng.normal(0, 0.15), 0, 1))
            r = int(round(1 + 4 * valor))
            resp.append(r if k % 2 == 0 else 6 - r)                       # ítems pares redactados en negativo
        prof = elegir(reg.PROFESIONES, [40, 30, 10, 20])
        p = reg.puntaje_sus(resp)
        fila = {"cuestionario_n": i + 1, "fecha": (hoy - timedelta(days=int(rng.integers(0, 3)))).strftime("%d/%m/%Y"),
                "registrado_el": (datetime.now() + timedelta(minutes=2 * i)).isoformat(timespec="seconds"),
                "casos_registrados": int(elegir([5, 4], [9, 1])), "profesion": prof,
                "profesion_otra": "Técnico(a) de enfermería" if prof == "Otra" else "",
                "experiencia": elegir(reg.EXPERIENCIA, [35, 40, 25]), "usa_computadora": elegir(["Sí", "No"], [85, 15]),
                "puntaje_sus": p, "interpretacion": reg.interpretar_sus(p),
                "comentarios": COMENTARIOS[int(rng.integers(0, len(COMENTARIOS)))]}
        fila.update({f"p{j}": r for j, r in enumerate(resp, 1)})
        filas.append(fila)
    escribir(DEMO / "sus_respuestas_demo.csv", reg.CAMPOS_SUS, filas)
    return filas


if __name__ == "__main__":
    pr, ss = predicciones(), sus()
    niveles = {n: sum(1 for f in pr if f["nivel_riesgo"] == n) for n in ("ALTO", "MEDIO", "BAJO")}
    puntajes = [f["puntaje_sus"] for f in ss]
    print(f"Predicciones simuladas: {len(pr)} {niveles}")
    print(f"SUS simulado: n={len(ss)}, media={np.mean(puntajes):.1f}, mín={min(puntajes):.1f}, máx={max(puntajes):.1f}")
    print(f"Archivos en {DEMO}")
