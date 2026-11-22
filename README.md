# Sistema de apoyo para estimar el riesgo de bajo peso al nacer (BPN)

Provincia de Rodríguez de Mendoza (Amazonas). Machine Learning local y gratuito, con datos abiertos del MINSA
(Certificado de Nacido Vivo, 2015 – mayo 2026; 4 444 registros, 254 con BPN = 5,72 %).
La app estima el riesgo con **10 datos conocidos antes del parto**, muestra un resultado con color y explica el porqué en frases.

> Herramienta de apoyo: no reemplaza el criterio clínico. Resultado orientativo.

## Estructura

```
proyecto_bpn/
├── config.py              rutas y constantes (semilla 42, CV 10x5, umbrales)
├── etl/limpieza.py        limpieza -> data/processed/dataset_limpio.csv + diccionario + reporte
├── ml/
│   ├── features.py        los 10 datos de entrada, variables derivadas y pipeline
│   ├── entrenar.py        CV 10x5, umbrales y modelo final -> models/modelo_bpn.joblib
│   ├── analisis_tesis.py  análisis completo de la tesis (C0-C3, hipótesis, subgrupos, SHAP) -> reports/tesis/
│   ├── metricas.py        utilidades de umbral
│   └── predictor.py       puntaje, nivel de riesgo y explicación SHAP en frases (API y app)
├── api/main.py            FastAPI: /health, /opciones, /predict
├── app/app.py, db.py      Streamlit + SQLite
├── data/  models/  reports/
├── requirements.txt
└── ejecutar_todo.ps1      ejecuta todo en orden
```

## Datos

Los datos no se incluyen en el repositorio (ver `data/README.md`). El modelo entrenado y los resultados de la tesis sí están,
por lo que la app funciona sin ellos; para reentrenar hay que colocar `Rodriguez_de_Mendoza_LIMPIO.xlsx` en `data/raw/`.

## Cómo ejecutarlo (Windows PowerShell)

```powershell
cd proyecto_bpn
powershell -ExecutionPolicy Bypass -File .\ejecutar_todo.ps1                 # ETL + modelo + abre API y app (~2 min)
powershell -ExecutionPolicy Bypass -File .\ejecutar_todo.ps1 -ConAnalisis    # además rehace el análisis de la tesis (~10 min más)
```

Paso a paso (con un entorno virtual `venv` creado, p. ej. `python -m venv venv`):

```powershell
python -m pip install -r requirements.txt
python etl\limpieza.py
python ml\entrenar.py
python ml\analisis_tesis.py            # opcional: tablas, figuras e informe en reports\tesis
python -m uvicorn api.main:app --port 8000          # ventana 1 (docs: http://127.0.0.1:8000/docs)
python -m streamlit run app\app.py                  # ventana 2 (http://localhost:8501)
```

La API es opcional: si está apagada, la app usa el modelo directamente (lo indica el menú lateral).
Si hay una API de una versión anterior encendida, la app la ignora y usa el modelo local.

## Los 10 datos del modelo

Madre: edad, estado civil, nivel de instrucción, ocupación (ama de casa u otra), distrito y tipo de seguro.
Antecedentes obstétricos: número de embarazos, hijos vivos, abortos o nacidos muertos y embarazo único o múltiple.

No usa Ipress ni datos del parto (talla, sexo, condición, lugar, quién atiende, semanas de gestación).
En el CNV, "hijos vivos" incluye al recién nacido; por eso el formulario pregunta por los hijos *previos* y el sistema suma 1.
Los nombres de los distritos (ubigeo 010601–010612) deben verificarse con el INEI.

## Pantallas de la app

1. **Inicio** (sin login) y botón "Nueva evaluación".
2. **Formulario** en dos bloques, con listas desplegables, validaciones con mensajes claros y botones "Calcular riesgo" y "Limpiar".
3. **Resultado:** 🔴 Riesgo alto · 🟡 Riesgo medio · 🟢 Riesgo bajo, "Puntaje de riesgo" (no es la probabilidad real),
   factores que aumentan o disminuyen el riesgo en frases, gráfico opcional en "Ver detalle", nota fija y botones "Volver" y "Nueva evaluación".
4. Menú lateral: **Historial y reporte** (descarga CSV y resumen), **Métricas del modelo** y **Cuestionario SUS** (puntaje = 2,5·[Σ(impares−1)+Σ(5−pares)]).

Los niveles salen de umbrales fijados en validación cruzada: medio = Recall 0,70 (con la mayor precisión posible); alto = Recall ≈ 0,40.

## Metodología (resumen)

* Partición estratificada 80/20 (`random_state=42`): 3 555 entrenamiento / 889 prueba (51 BPN). La prueba se usa solo en el análisis final.
* Validación cruzada estratificada 10 repeticiones × 5 pliegues solo en entrenamiento; imputación, escalado y SMOTE dentro del `Pipeline`.
* `ml/analisis_tesis.py` compara C0 (Regresión Logística sin balanceo, umbral 0,5) con C1–C3 (RL, RF y XGBoost, con SMOTE o pesos de clase),
  hace bootstrap pareado, DeLong, Friedman + Wilcoxon-Holm, subgrupos y SHAP, y escribe `reports/tesis/` (informe, tablas, figuras y `cambios_metodo.md`).
* Modelo desplegado: Random Forest con pesos de clase (mayor AUC medio en validación cruzada).

## Resultados (modelo principal, sin Ipress)

Ver `reports/tesis/informe_resultados.md`. En prueba: AUC-ROC 0,588 (IC 95 %: 0,497–0,671), Recall 0,784 y precisión 0,062 con el umbral de Recall 0,70.
Con Ipress (solo análisis de sensibilidad en validación cruzada) el AUC sube a ≈ 0,66, pero Ipress se excluye por registrarse en el parto.

## Limitaciones

* Capacidad predictiva modesta: sirve para priorizar el seguimiento, no para diagnosticar.
* Con el umbral de Recall 0,70 se marca como riesgo medio o alto a la mayoría de las gestantes y la precisión es baja (prevalencia 5,7 %).
* El puntaje no es una probabilidad calibrada. SHAP muestra asociaciones del modelo, no causas.
* La prueba tiene 51 casos de BPN: los intervalos son amplios y los subgrupos son solo descriptivos.
* El historial y el SUS se guardan en `data/app.db` (SQLite local).

## Autores y licencia

Acosta Vasquez, Lelys Daniel · Mamani Flores, Miguel Angel · Salvatierra de la Cruz, Jesus Jheison.
Asesora: Acuña Melendez, Maria Eudelia. Universidad Autónoma del Perú, Ingeniería de Sistemas.
Código bajo licencia MIT (ver `LICENSE`).
