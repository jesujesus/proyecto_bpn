# Cambios respecto del Método de la tesis (y decisiones tomadas)

## Diferencias frente al texto del Método

1. **Hiperparámetros:** no se hizo búsqueda en malla completa; se usó una malla pequeña de 3 configuraciones por (modelo, estrategia) con CV 3×5 sobre el entrenamiento (AUC-ROC), y luego CV 10×5 con las configuraciones elegidas.
2. **Estrategia de desbalance:** para C1, C2 y C3 se ajustaron y guardaron ambas variantes (SMOTE y pesos de clase); el sistema es la de mayor AUC medio, por lo que puede no usar SMOTE. C0 no usa balanceo.
3. **Sistema elegido:** C2_pesos (modelo: Random Forest, estrategia: pesos).
4. **Umbral:** el de mayor precisión con Recall fuera de pliegue ≥ 0,70, calculado sobre las predicciones apiladas de las 10 repeticiones. En `proyecto_bpn` se usaba el mayor umbral con Recall ≥ 0,70 (regla distinta).
5. **Sin selección de variables:** no se aplicó el procedimiento de no inferioridad del proyecto original, porque el Método no lo contempla.
6. **Predictores:** edad, estado civil, instrucción, ocupación (recodificada a AMA DE CASA / OTRA, según el Anexo 3), embarazos, hijos vivos, abortos, único/múltiple (Tipo_Parto agrupado; IGNORADO como categoría propia), distrito (IdUbigeoInei) y seguro (Financiador_Parto). Derivadas: ES_ADOL, MENOR18, AVANZ, PRIMIG, GRAN, ESTCIV_IGN, PERDIDAS (solo abortos > 0) e INT (embarazos − vivos − abortos).
7. **Excluidos:** Ipress (solo en la sensibilidad), hijos fallecidos, país de la madre, año y mes del parto, talla, sexo, condición, lugar, atiende, DUR_EMB_PARTO y peso. DUR_EMB_PARTO solo define subgrupos.
8. **Datos faltantes:** imputación por mediana dentro de cada pliegue; categorías con menos de 5 registros se agrupan como 'infrecuentes'.
9. **Bootstrap:** pareado y estratificado por clase (2 000 réplicas); el límite unilateral es el percentil 5. DeLong implementado manualmente (versión rápida de Sun y Xu).
10. **Reglas de decisión usadas:** HE1: límite inferior de ΔRecall > 0 y de Recall(S) > 0,70. HE2: DeLong p < 0,05 (unilateral) y límite inferior de AUC(S) > 0,75. HE3: límites inferiores de ΔAUC-PR y ΔF1 > 0 y Wilcoxon-Holm p < 0,05 a favor del sistema (CV). General: HE1, HE2 y HE3 rechazadas. Umbrales de hipótesis no modificados.
11. **Friedman/Wilcoxon-Holm:** sobre las 50 estimaciones de CV de las 7 configuraciones, para AUC-ROC, AUC-PR, F1 y Recall (Holm sobre los 21 pares por métrica). Las estimaciones de distintas repeticiones no son totalmente independientes. Shapiro-Wilk sobre las diferencias por pliegue (sistema − C0).
12. **Prueba:** C0 y el sistema se evaluaron una vez cada uno. El análisis de sensibilidad con Ipress se evaluó solo en CV para no volver a usar la prueba.
13. **Transparencia:** la partición de prueba ya se había usado antes en `proyecto_bpn` para un modelo exploratorio que incluía Ipress (AUC 0,638 en prueba). Este diseño sigue el Anexo 3, que ya excluía Ipress, pero debe declararse en el texto.
14. **Subgrupos:** distrito y seguro incluyen cifras con muy pocos casos; seguro se agrupa en SIS, ESSALUD y OTROS.
15. **SUS:** solo script y plantilla; no se generaron datos.

## Versiones de las bibliotecas

- Python: 3.13.6
- pandas: 3.0.6
- numpy: 2.5.3
- scipy: 1.18.1
- scikit-learn: 1.9.1
- imbalanced-learn: 0.14.2
- xgboost: 3.4.1
- shap: 0.54.1
- matplotlib: 3.11.2
- joblib: 1.6.0
- openpyxl: 3.1.5