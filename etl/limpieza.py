"""ETL: lee el Excel limpio, corrige y genera el dataset CSV, el diccionario de datos y el reporte de limpieza."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as C
from ml.features import COLUMNAS_CRUDAS


def localizar_excel():
    for ruta in [C.DATA_RAW / C.EXCEL_ORIGEN, C.RAIZ.parent / C.EXCEL_ORIGEN]:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(f"No encuentro {C.EXCEL_ORIGEN} en data/raw ni en la carpeta superior.")


def main():
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    C.REPORTES.mkdir(parents=True, exist_ok=True)
    ruta = localizar_excel()
    print(f"Leyendo {ruta.name} (hoja {C.HOJA}) ...")
    df = pd.read_excel(ruta, sheet_name=C.HOJA, dtype={"IdUbigeoInei": str})
    rep = [("Filas leídas", len(df), f"{df.shape[1]} columnas")]

    n = int(df["Hijos_fallec_madre"].isna().sum())
    df["Hijos_fallec_madre"] = df["Hijos_fallec_madre"].fillna(0).astype(int)
    rep.append(("Hijos_fallec_madre: NaN -> 0", n, "NaN significa 'ningún hijo fallecido' (antes se codificaba -1)"))

    df["IdUbigeoInei"] = df["IdUbigeoInei"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    rep.append(("IdUbigeoInei como texto de 6 dígitos", len(df), "ej. 010609 (se conservan ceros a la izquierda)"))
    df["Ipress"] = df["Ipress"].astype(str).str.strip()

    df[C.TARGET] = (df["PESO_NACIDO"] < C.UMBRAL_BPN).astype(int)
    rep.append(("Variable objetivo RIESGO_BAJO_PESO", int(df[C.TARGET].sum()),
                f"casos BPN (<{C.UMBRAL_BPN} g) de {len(df)} = {df[C.TARGET].mean()*100:.2f} %"))
    for c in ["Num_embar_madre", "Hijos_vivo_madre", "nacmuer_abort_madre"]:
        rep.append((f"{c}: valores faltantes", int(df[c].isna().sum()),
                    "se imputan con la mediana DENTRO del Pipeline (sin fuga)"))
    rep.append(("Filas duplicadas exactas", int(df.duplicated().sum()), "no se eliminan (cada fila es un nacido)"))

    cols = ["FecNac_Año", "FecNac_Mes", "IdUbigeoInei", "Ipress", "Pais_Madre", "Hijos_fallec_madre",
            "DUR_EMB_PARTO", "PESO_NACIDO", C.TARGET]
    cols = list(dict.fromkeys(COLUMNAS_CRUDAS + cols))
    df = df[cols]
    df.to_csv(C.CSV_LIMPIO, index=False, encoding="utf-8")
    print(f"Dataset limpio: {C.CSV_LIMPIO}  ({df.shape[0]} filas x {df.shape[1]} columnas)")

    pd.DataFrame(rep, columns=["Paso", "Afectados", "Detalle"]).to_csv(
        C.REPORTES / "reporte_limpieza.csv", index=False, encoding="utf-8-sig")
    md = ["# Reporte de limpieza (ETL)", "",
          f"- Archivo de origen: `{ruta.name}` (hoja `{C.HOJA}`)",
          f"- Filas finales: **{len(df)}**; BPN: **{int(df[C.TARGET].sum())}** ({df[C.TARGET].mean()*100:.2f} %)", "",
          "| Paso | Afectados | Detalle |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rep]
    (C.REPORTES / "reporte_limpieza.md").write_text("\n".join(md), encoding="utf-8")

    entradas = set(COLUMNAS_CRUDAS)
    uso = {c: "Entrada del modelo" for c in entradas}
    uso.update({"DUR_EMB_PARTO": "Solo subgrupos OMS (NO es entrada)", "PESO_NACIDO": "Solo para construir el objetivo",
                C.TARGET: "Variable objetivo (1 = peso < 2500 g)", "Ipress": "NO es entrada (solo análisis de sensibilidad)",
                "IdUbigeoInei": "Entrada del modelo (distrito)", "Financiador_Parto": "Entrada del modelo (seguro)",
                "Tipo_Parto": "Entrada del modelo (único/múltiple)", "FecNac_Año": "Referencia (NO es entrada)",
                "FecNac_Mes": "Referencia (NO es entrada)", "Pais_Madre": "NO es entrada (99,6 % Perú)",
                "Hijos_fallec_madre": "NO es entrada (96,4 % sin dato = 0)"})
    desc = {
        "FecNac_Año": "Año de nacimiento", "FecNac_Mes": "Mes de nacimiento",
        "IdUbigeoInei": "Ubigeo INEI del distrito (texto de 6 dígitos)", "Ipress": "Código del establecimiento de salud",
        "Edad_Madre": "Edad de la madre en años", "Estado_Civil": "Estado civil de la madre",
        "Nivel_Intrucción_Madre": "Nivel de instrucción de la madre", "DESC_OCUPACION": "Ocupación de la madre",
        "Num_embar_madre": "Número de embarazos incluido el actual (>=5 se codifica como 5)",
        "Hijos_vivo_madre": "Hijos vivos incluido el recién nacido (>=5 se codifica como 5)",
        "Hijos_fallec_madre": "Hijos fallecidos (NaN original = 0)", "nacmuer_abort_madre": "Nacidos muertos / abortos previos",
        "DUR_EMB_PARTO": "Semanas de gestación al parto", "Pais_Madre": "País de origen de la madre",
        "Tipo_Parto": "Embarazo único o múltiple", "Financiador_Parto": "Seguro / financiador (SIS, EsSalud...)",
        "PESO_NACIDO": "Peso al nacer en gramos", C.TARGET: "1 si PESO_NACIDO < 2500 g, 0 si no"}
    dic = pd.DataFrame({
        "variable": df.columns, "descripcion": [desc.get(c, "") for c in df.columns],
        "tipo": [str(df[c].dtype) for c in df.columns], "uso": [uso.get(c, "") for c in df.columns],
        "n_faltantes": [int(df[c].isna().sum()) for c in df.columns],
        "n_distintos": [int(df[c].nunique()) for c in df.columns],
        "ejemplo": [str(df[c].dropna().iloc[0]) for c in df.columns]})
    dic.to_csv(C.REPORTES / "diccionario_datos.csv", index=False, encoding="utf-8-sig")
    print("Diccionario y reporte guardados en reports/")


if __name__ == "__main__":
    main()
