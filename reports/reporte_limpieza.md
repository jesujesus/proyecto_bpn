# Reporte de limpieza (ETL)

- Archivo de origen: `Rodriguez_de_Mendoza_LIMPIO.xlsx` (hoja `Datos_Completos`)
- Filas finales: **4444**; BPN: **254** (5.72 %)

| Paso | Afectados | Detalle |
|---|---|---|
| Filas leídas | 4444 | 24 columnas |
| Hijos_fallec_madre: NaN -> 0 | 4282 | NaN significa 'ningún hijo fallecido' (antes se codificaba -1) |
| IdUbigeoInei como texto de 6 dígitos | 4444 | ej. 010609 (se conservan ceros a la izquierda) |
| Variable objetivo RIESGO_BAJO_PESO | 254 | casos BPN (<2500 g) de 4444 = 5.72 % |
| Num_embar_madre: valores faltantes | 43 | se imputan con la mediana DENTRO del Pipeline (sin fuga) |
| Hijos_vivo_madre: valores faltantes | 35 | se imputan con la mediana DENTRO del Pipeline (sin fuga) |
| nacmuer_abort_madre: valores faltantes | 469 | se imputan con la mediana DENTRO del Pipeline (sin fuga) |
| Filas duplicadas exactas | 0 | no se eliminan (cada fila es un nacido) |