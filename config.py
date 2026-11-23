"""Configuración central del proyecto BPN (rutas y constantes)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"
REPORTES_TESIS = REPORTES / "tesis"
REGISTROS = RAIZ / "data" / "registros"          # "base de datos" en CSV: lo que guarda la app
PRED_CSV = REGISTROS / "predicciones.csv"        # una fila por evaluación de gestante
SUS_CSV = REGISTROS / "sus_respuestas.csv"       # una fila por cuestionario SUS

EXCEL_ORIGEN = "Rodriguez_de_Mendoza_LIMPIO.xlsx"
HOJA = "Datos_Completos"
CSV_LIMPIO = DATA_PROC / "dataset_limpio.csv"
SPLIT_JSON = DATA_PROC / "split.json"

SEMILLA = 42
UMBRAL_BPN = 2500          # g, criterio OMS
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPETICIONES = 10       # 10 x 5 = 50 evaluaciones
RECALL_OBJETIVO = 0.70     # umbral de "riesgo medio": Recall >= 0,70 en validación cruzada
RECALL_ALTO = 0.40         # umbral de "riesgo alto": Recall ~ 0,40 en validación cruzada
TARGET = "RIESGO_BAJO_PESO"

# Hiperparámetros del sistema elegido en el análisis de la tesis (Random Forest con pesos de clase)
RF_PARAMS = {"min_samples_leaf": 15}
