# Datos

Los datos no se versionan en Git (se ignoran con `.gitignore`).

**Origen:** Certificado de Nacido Vivo (CNV), datos abiertos del Ministerio de Salud (MINSA), archivo
`DATOS_ABIERTOS_CNV_31052026`. Se filtraron los registros con `IdUbigeoInei` que empieza en `0106`
(provincia de Rodríguez de Mendoza, Amazonas) y se limpiaron, obteniendo `Rodriguez_de_Mendoza_LIMPIO.xlsx`
(hoja `Datos_Completos`, 4 444 registros).

**Cómo usarlos:**

1. Copie `Rodriguez_de_Mendoza_LIMPIO.xlsx` en `data/raw/`.
2. Ejecute `python etl/limpieza.py` (genera `data/processed/dataset_limpio.csv`) y luego `python ml/entrenar.py`
   (genera `data/processed/split.json` y `models/modelo_bpn.joblib`).

El modelo ya entrenado (`models/modelo_bpn.joblib`) y los resultados de la tesis (`reports/tesis/`) sí se incluyen,
por lo que la app y la API funcionan sin los datos.

> Aunque los datos de origen son públicos y anonimizados, los registros individuales de una provincia pequeña
> pueden ser reidentificables; por eso no se publican en este repositorio.

## Registros de la app

`data/registros/` guarda `predicciones.csv` y `sus_respuestas.csv` (creados por la app). No se versionan.
