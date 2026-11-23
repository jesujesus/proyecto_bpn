"""Análisis del SUS: media, DE, Shapiro-Wilk, t de Student unilateral frente a 68, Wilcoxon y alfa de Cronbach.

Lee data/registros/sus_respuestas.csv (lo que guarda la app).  Uso:
    python ml/sus_analisis.py [ruta_al_csv]
Solo analiza respuestas reales; no genera datos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_1samp, wilcoxon

import config as C

REF = 68
ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else C.SUS_CSV
if not ruta.exists():
    sys.exit(f"No encuentro {ruta}. Registre cuestionarios en la app (menú: Cuestionario SUS).")
df = pd.read_csv(ruta, encoding="utf-8-sig")
P = [f"p{i}" for i in range(1, 11)]
df = df.dropna(subset=P)
if len(df) < 3:
    sys.exit("Se necesitan al menos 3 cuestionarios completos para analizar.")
R = df[P].astype(float).values
if ((R < 1) | (R > 5)).any():
    sys.exit("Hay respuestas fuera de 1-5.")
x = 2.5 * ((R[:, 0::2] - 1).sum(axis=1) + (5 - R[:, 1::2]).sum(axis=1))    # ítems impares: p1,p3,...; pares: p2,p4,...
n = len(x)
print(f"n = {n}; media = {x.mean():.2f}; DE = {x.std(ddof=1):.2f}; mediana = {np.median(x):.1f}; mín-máx = {x.min():.1f}-{x.max():.1f}")
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
Rr = R.copy()
Rr[:, 1::2] = 6 - Rr[:, 1::2]                     # ítems pares recodificados (todos en sentido positivo)
alfa = 10 / 9 * (1 - Rr.var(axis=0, ddof=1).sum() / Rr.sum(axis=1).var(ddof=1))
print(f"Alfa de Cronbach = {alfa:.3f}")
