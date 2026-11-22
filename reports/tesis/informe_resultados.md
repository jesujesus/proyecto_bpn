# Resultados de la tesis BPN (modelo principal sin Ipress)

Partición: entrenamiento 3 555 (203 BPN), prueba 889 (51 BPN); misma semilla (42) y mismos índices que `proyecto_bpn`. CV estratificada 10×5 solo en entrenamiento; bootstrap pareado estratificado de 2000 réplicas.

## 1. Validación cruzada (entrenamiento, modelo principal)

| Config. | Modelo | Estrategia | AUC-ROC | AUC-PR | F1 | Recall | Umbral |
|---|---|---|---|---|---|---|---|
| C0 | Regresión Logística (línea base) | ninguna | 0,545 ± 0,036 | 0,166 | 0,134 | 0,076 | 0,5000 |
| C1_pesos | Regresión Logística | pesos | 0,557 ± 0,034 | 0,161 | 0,110 | 0,705 | 0,4461 |
| C1_smote | Regresión Logística | smote | 0,561 ± 0,041 | 0,161 | 0,110 | 0,703 | 0,2841 |
| C2_pesos **(sistema)** | Random Forest | pesos | 0,566 ± 0,037 | 0,170 | 0,111 | 0,701 | 0,3391 |
| C2_smote | Random Forest | smote | 0,556 ± 0,040 | 0,162 | 0,112 | 0,705 | 0,1640 |
| C3_pesos | XGBoost | pesos | 0,557 ± 0,040 | 0,166 | 0,108 | 0,700 | 0,3692 |
| C3_smote | XGBoost | smote | 0,552 ± 0,040 | 0,155 | 0,111 | 0,700 | 0,1228 |

Friedman (7 configuraciones, 50 pliegues): AUC-ROC: χ² = 28,82, p = < 0,001; AUC-PR: χ² = 22,35, p = 0,001; F1: χ² = 20,74, p = 0,002; Recall: χ² = 120,07, p = < 0,001.

Sistema = C2_pesos (Random Forest, pesos): mayor AUC-ROC medio entre C1-C3. Umbral (mayor precisión con Recall OOF ≥ 0,70) = 0,3391; C0 conserva 0,5.

Wilcoxon-Holm del sistema frente a C0 (50 pliegues): 
- AUC-ROC: Δ = 0,0205, p Holm = < 0,001
- AUC-PR: Δ = 0,0038, p Holm = 1,000
- F1: Δ = -0,0229, p Holm = 0,028
- Recall: Δ = 0,6248, p Holm = < 0,001

Shapiro-Wilk de las diferencias por pliegue (sistema − C0): AUC-ROC: W = 0,984, p = 0,728; AUC-PR: W = 0,973, p = 0,308; F1: W = 0,955, p = 0,057; Recall: W = 0,955, p = 0,056.

## 2. Prueba (889 registros), C0 frente al sistema

| Métrica | C0 | IC 95 % | Sistema | IC 95 % | Δ (S − C0) | IC 95 % de Δ | Lím. inf. unilateral Δ |
|---|---|---|---|---|---|---|---|
| Recall | 0,078 | 0,020–0,157 | 0,784 | 0,667–0,902 | 0,706 | 0,588–0,824 | 0,608 |
| AUC-ROC | 0,653 | 0,573–0,726 | 0,588 | 0,497–0,671 | -0,065 | -0,126–-0,003 | -0,118 |
| AUC-PR | 0,181 | 0,103–0,297 | 0,164 | 0,089–0,281 | -0,018 | -0,066–0,036 | -0,057 |
| F1 | 0,138 | 0,035–0,262 | 0,115 | 0,099–0,131 | -0,022 | -0,144–0,088 | -0,123 |

- Matriz C0: VP 4, FP 3, VN 835, FN 47 (precisión 0,571; especificidad 0,996).
- Matriz sistema: VP 40, FP 602, VN 236, FN 11 (precisión 0,062; especificidad 0,282; exactitud 0,310).
- Recall del sistema, Wilson 95 %: 0,654–0,875.
- DeLong pareado (AUC sistema − AUC C0): z = -2,060, p unilateral = 0,980, p bilateral = 0,039.
- Cobertura SHAP en prueba: 100 %. Variables más influyentes: Instrucción, Edad de la madre, Embarazo único/múltiple, Distrito, N.º de embarazos.

## 3. Decisión por hipótesis (umbrales sin modificar)

| Hipótesis | Criterio | Resultado | Decisión |
|---|---|---|---|
| HE1 (Recall) | lím. inf. de ΔRecall > 0 y lím. inf. de Recall(S) > 0,70 | Recall(S) = 0,784; lím. inf. unilateral = 0,686; lím. inf. de ΔRecall = 0,608 | **No se rechaza H0** |
| HE2 (AUC) | DeLong p < 0,05 y lím. inf. de AUC(S) > 0,75 | AUC(S) = 0,588; lím. inf. unilateral = 0,515; DeLong p = 0,980 | **No se rechaza H0** |
| HE3 (AUC-PR y F1) | lím. inf. de ΔAUC-PR y ΔF1 > 0 (prueba) y Wilcoxon-Holm p < 0,05 a favor (CV) | lím. inf. ΔAUC-PR = -0,057; lím. inf. ΔF1 = -0,123; p Holm AUC-PR = 1,000; p Holm F1 = 0,028 | **No se rechaza H0** |
| HE4 (SUS) | media SUS > 68 | sin datos: no se inventaron | Pendiente |
| General | se rechaza H0 en HE1, HE2 y HE3 | HE1: no; HE2: no; HE3: no | **No se rechaza H0** |

Nota: C0 usa el umbral 0,5 con prevalencia de 5,7 %, por lo que marca muy pocos positivos; las comparaciones de Recall y F1 frente a C0 están influidas por ese umbral. Las comparaciones basadas en el AUC (ROC y PR) no dependen de él.

## 4. Sensibilidad: modelo con Ipress (solo CV, sin evaluar en prueba)

| Config. | AUC-ROC sin Ipress | AUC-ROC con Ipress | AUC-PR sin Ipress | AUC-PR con Ipress |
|---|---|---|---|---|
| C0 | 0,545 | 0,649 | 0,166 | 0,189 |
| C1_pesos | 0,557 | 0,644 | 0,161 | 0,183 |
| C1_smote | 0,561 | 0,641 | 0,161 | 0,188 |
| C2_pesos | 0,566 | 0,661 | 0,170 | 0,184 |
| C2_smote | 0,556 | 0,650 | 0,162 | 0,176 |
| C3_pesos | 0,557 | 0,651 | 0,166 | 0,179 |
| C3_smote | 0,552 | 0,644 | 0,155 | 0,166 |

Sistema en el análisis con Ipress: C2_pesos (AUC-ROC CV = 0,661). El detalle por pliegue, Friedman y Wilcoxon-Holm de este conjunto están en `tablas/cv_*.csv` (columna `conjunto`).

## 5. Subgrupos (Recall con IC de Wilson; solo descriptivo)

| Dimensión | Subgrupo | Nacidos | Casos BPN | Recall sistema (IC 95 %) | Recall C0 (IC 95 %) |
|---|---|---|---|---|---|
| Severidad | Extremadamente bajo (<1 000 g) | 2 | 2 | 0,50 (0,09–0,91) | 0,00 (0,00–0,66) |
| Severidad | Muy bajo (<1 500 g) | 4 | 4 | 0,75 (0,30–0,95) | 0,00 (0,00–0,49) |
| Severidad | Bajo (1 500–2 499 g) | 47 | 47 | 0,79 (0,65–0,88) | 0,09 (0,03–0,20) |
| Origen | Prematuro (<37 semanas) | 50 | 32 | 0,72 (0,55–0,84) | 0,12 (0,05–0,28) |
| Origen | A término (≥37 semanas) | 839 | 19 | 0,89 (0,69–0,97) | 0,00 (0,00–0,17) |
| Distrito | 010601 | 136 | 6 | 0,50 (0,19–0,81) | 0,00 (0,00–0,39) |
| Distrito | 010602 | 106 | 4 | 0,75 (0,30–0,95) | 0,25 (0,05–0,70) |
| Distrito | 010603 | 16 | 0 | — | — |
| Distrito | 010604 | 65 | 6 | 0,50 (0,19–0,81) | 0,00 (0,00–0,39) |
| Distrito | 010605 | 49 | 3 | 1,00 (0,44–1,00) | 0,00 (0,00–0,56) |
| Distrito | 010606 | 42 | 1 | 1,00 (0,21–1,00) | 0,00 (0,00–0,79) |
| Distrito | 010607 | 53 | 4 | 0,75 (0,30–0,95) | 0,25 (0,05–0,70) |
| Distrito | 010608 | 22 | 3 | 1,00 (0,44–1,00) | 0,00 (0,00–0,56) |
| Distrito | 010609 | 284 | 19 | 0,84 (0,62–0,94) | 0,11 (0,03–0,31) |
| Distrito | 010610 | 20 | 1 | 1,00 (0,21–1,00) | 0,00 (0,00–0,79) |
| Distrito | 010611 | 9 | 1 | 1,00 (0,21–1,00) | 0,00 (0,00–0,79) |
| Distrito | 010612 | 87 | 3 | 1,00 (0,44–1,00) | 0,00 (0,00–0,56) |
| Seguro | SIS | 809 | 44 | 0,82 (0,68–0,90) | 0,07 (0,02–0,18) |
| Seguro | ESSALUD | 57 | 6 | 0,67 (0,30–0,90) | 0,17 (0,03–0,56) |
| Seguro | OTROS | 23 | 1 | 0,00 (0,00–0,79) | 0,00 (0,00–0,79) |

Muchos subgrupos (p. ej., distritos) tienen muy pocos casos de BPN: los intervalos son amplios y no permiten inferencia.

## 6. Archivos

`tablas/` (CSV, separador `;`, coma decimal), `figuras/` (PNG 300 ppp, sin título), `sus/` (script y plantilla).