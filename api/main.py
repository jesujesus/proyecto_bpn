"""API FastAPI: /health, /opciones y /predict (puntaje, nivel de riesgo y factores en lenguaje sencillo)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.predictor import Predictor, validar_datos

app = FastAPI(title="API de riesgo de bajo peso al nacer (BPN)", version="2.0",
              description="Rodríguez de Mendoza (Amazonas). Solo usa 10 datos conocidos antes del parto. Resultado orientativo.")
_pred: Optional[Predictor] = None


def predictor() -> Predictor:
    global _pred
    if _pred is None:
        try:
            _pred = Predictor()
        except FileNotFoundError:
            raise HTTPException(503, "El modelo aún no está disponible. Ejecute primero el entrenamiento (ml/entrenar.py).")
    return _pred


class Gestante(BaseModel):
    edad: int = Field(..., examples=[27])
    estado_civil: str = Field(..., examples=["SOLTERO"])
    instruccion: str = Field(..., examples=["SECUNDARIA COMPLETA"])
    ocupacion: str = Field("AMA DE CASA", examples=["AMA DE CASA"], description="AMA DE CASA u OTRA")
    distrito: str = Field(..., examples=["010601"], description="Código ubigeo del distrito")
    seguro: str = Field(..., examples=["SIS"])
    embarazos: int = Field(..., examples=[2], description="Incluye el embarazo actual")
    hijos_vivos: int = Field(0, examples=[1], description="Hijos vivos previos (sin contar el que va a nacer)")
    abortos: int = Field(0, examples=[0], description="Abortos o nacidos muertos previos")
    embarazo_multiple: bool = False


@app.get("/health")
def health():
    try:
        p = predictor()
    except HTTPException as e:
        return {"estado": "sin_modelo", "detalle": e.detail}
    return {"estado": "ok", "modelo": p.b["modelo"], "version": p.b["version"], "umbral": p.b["umbral"],
            "predicciones": p.total, "cobertura_shap": p.cobertura_shap}


@app.get("/opciones")
def opciones():
    return predictor().b["opciones"]


@app.post("/predict")
def predict(g: Gestante):
    datos = g.model_dump()
    errores = validar_datos(datos)
    if errores:
        raise HTTPException(422, " ".join(errores))
    return predictor().predecir(datos)
