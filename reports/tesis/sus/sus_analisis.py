"""Análisis del SUS: puntaje, media, Shapiro-Wilk, t de Student unilateral frente a 68, Wilcoxon y alfa de Cronbach.
Uso:  python sus_analisis.py plantilla_sus.csv   (completar el CSV con respuestas 1-5 de personas reales)
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_1samp, wilcoxon

REF = 68
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "plantilla_sus.csv")
P = [f"p{i}" for i in range(1, 11)]
df = df.dropna(subset=P)
if len(df) < 3:
    sys.exit("Se necesitan al menos 3 respuestas completas (valores 1 a 5) para analizar.")
R = df[P].astype(float).values
if ((R < 1) | (R > 5)).any():
    sys.exit("Hay respuestas fuera de 1-5.")
df["puntaje_sus"] = 2.5 * ((R[:, 0::2] - 1).sum(axis=1) + (5 - R[:, 1::2]).sum(axis=1))   # impares: p1,p3,...; pares: p2,p4,...
x = df["puntaje_sus"].values
n = len(x)
print(f"n = {n}; media = {x.mean():.2f}; DE = {x.std(ddof=1):.2f}; mediana = {np.median(x):.1f}")
sw = shapiro(x)
print(f"Shapiro-Wilk: W = {sw.statistic:.3f}, p = {sw.pvalue:.4f}")
t = ttest_1samp(x, REF, alternative="greater")
print(f"t de Student unilateral (media > {REF}): t = {t.statistic:.3f}, p = {t.pvalue:.4f}")
try:
    w = wilcoxon(x - REF, alternative="greater")
    print(f"Wilcoxon unilateral (mediana > {REF}): p = {w.pvalue:.4f}")
except ValueError as e:
    print("Wilcoxon no calculable:", e)
print("Prueba a reportar:", "t de Student" if sw.pvalue >= 0.05 else "Wilcoxon")
# alfa de Cronbach con los ítems recodificados (todos en sentido positivo)
Rr = R.copy()
Rr[:, 1::2] = 6 - Rr[:, 1::2]
k = Rr.shape[1]
alfa = k / (k - 1) * (1 - Rr.var(axis=0, ddof=1).sum() / Rr.sum(axis=1).var(ddof=1))
print(f"Alfa de Cronbach = {alfa:.3f}")
df.to_csv("sus_con_puntajes.csv", index=False)
