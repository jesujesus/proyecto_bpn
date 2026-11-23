"""Aplicación Streamlit: inicio, formulario, resultado, historial/reporte, métricas del modelo y cuestionario SUS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

import config as C
import registros as reg
from ml.predictor import Predictor, validar_datos

API_URL = "http://127.0.0.1:8000"
NIVELES = {"ALTO": ("#cf222e", "#ffebe9", "🔴 Riesgo alto"), "MEDIO": ("#9a6700", "#fff8c5", "🟡 Riesgo medio"),
           "BAJO": ("#1a7f37", "#dafbe1", "🟢 Riesgo bajo")}
ESTADO_CIVIL = {"SOLTERO": "Soltera", "CASADO": "Casada", "CONVIVIENTE": "Conviviente", "DIVORCIADO": "Divorciada",
                "VIUDO": "Viuda", "IGNORADO": "No sabe / no registrado"}
INSTRUCCION = {"NINGUN NIVEL/ILETRADO": "Ningún nivel", "INCIAL/PRE-ESCOLAR": "Inicial", "PRIMARIA INCOMPLETA": "Primaria incompleta",
               "PRIMARIA COMPLETA": "Primaria completa", "SECUNDARIA INCOMPLETA": "Secundaria incompleta",
               "SECUNDARIA COMPLETA": "Secundaria completa", "SUPERIOR NO UNIV. INCOMPLETA": "Superior técnica incompleta",
               "SUPERIOR NO UNIV. COMPLETA": "Superior técnica completa", "SUPERIOR UNIV. INCOMPLETA": "Superior universitaria incompleta",
               "SUPERIOR UNIV. COMPLETA": "Superior universitaria completa", "IGNORADO": "No sabe / no registrado"}
SEGURO = {"SIS": "SIS", "ESSALUD": "EsSalud", "PARTICULAR": "Particular (sin seguro)", "PRIVADOS": "Seguro privado",
          "SANIDAD PNP": "Sanidad PNP", "SANIDAD EP": "Sanidad del Ejército", "SANIDAD FAP": "Sanidad FAP", "IGNORADO": "No sabe"}
DEFECTOS = {"edad": 25, "estado_civil": "SOLTERO", "instruccion": "SECUNDARIA COMPLETA", "ocupacion": "AMA DE CASA",
            "distrito": "010601", "seguro": "SIS", "embarazos": 1, "hijos_vivos": 0, "abortos": 0, "multiple": "Único"}

st.set_page_config(page_title="Riesgo de bajo peso al nacer", page_icon="🤰", layout="centered")
st.markdown("""<style>
.tarjeta{border-radius:18px;padding:26px 28px;margin:8px 0 14px 0;border:3px solid;text-align:center}
.tarjeta h1{margin:0;font-size:2.6rem}.tarjeta p{margin:.3rem 0 0 0;font-size:1.1rem}
.nota{background:#f6f8fa;border-left:4px solid #8c959f;padding:10px 14px;border-radius:6px;color:#24292f;margin-top:14px}
.hero{text-align:center;padding:30px 0 10px 0}.hero h1{font-size:2.4rem}.hero p{font-size:1.15rem;color:#57606a}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- modelo / API
@st.cache_resource(show_spinner="Cargando el modelo...")
def predictor_local():
    return Predictor()


def api_viva():
    try:
        j = requests.get(f"{API_URL}/health", timeout=1.5).json()
        return j.get("estado") == "ok" and str(j.get("version", "")).startswith("2.")   # ignora APIs de versiones anteriores
    except Exception:
        return False


def calcular(datos, usar_api):
    if usar_api:
        r = requests.post(f"{API_URL}/predict", json=datos, timeout=30)
        r.raise_for_status()
        return r.json()
    return predictor_local().predecir(datos)


def reiniciar():
    for k, v in DEFECTOS.items():
        st.session_state[f"f_{k}"] = v
    st.session_state.pop("resultado", None)


def ir(pantalla, limpiar=False):
    if limpiar:
        reiniciar()
    st.session_state["pantalla"] = pantalla


if "pantalla" not in st.session_state:
    st.session_state["pantalla"] = "inicio"
    reiniciar()

try:
    pl = predictor_local()
    opc = pl.b["opciones"]
except Exception:
    st.error("El modelo aún no está disponible. Pida al responsable del sistema que ejecute el entrenamiento.")
    st.stop()
usar_api = api_viva()

# ----------------------------------------------------------------------------- menú lateral
with st.sidebar:
    st.header("Menú")
    seccion = st.radio("Ir a", ["🩺 Evaluación", "📜 Historial y reporte", "📊 Métricas del modelo", "📝 Cuestionario SUS"],
                       label_visibility="collapsed")
    st.divider()
    st.caption("Herramienta de apoyo. No reemplaza el criterio clínico.")
    st.caption(f"Modelo: {pl.b['modelo']}")
    st.caption("Conectado a la API" if usar_api else "Usando el modelo local")


# ----------------------------------------------------------------------------- 1. inicio
def pantalla_inicio():
    st.markdown("""<div class="hero"><h1>🤰 Riesgo de bajo peso al nacer</h1>
        <p>Herramienta de apoyo para estimar el riesgo de bajo peso al nacer.<br>No reemplaza el criterio clínico.</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    st.button("Nueva evaluación", type="primary", width="stretch", on_click=ir, args=("formulario", True))
    st.caption("Provincia de Rodríguez de Mendoza, Amazonas. No requiere usuario ni contraseña.")


# ----------------------------------------------------------------------------- 2. formulario
def pantalla_formulario():
    st.subheader("Nueva evaluación")
    st.markdown("##### Datos de la madre")
    c1, c2 = st.columns(2)
    c1.number_input("Edad (años)", 0, 99, key="f_edad")
    c2.selectbox("Estado civil", [k for k in ESTADO_CIVIL if k in opc["estado_civil"]], format_func=ESTADO_CIVIL.get, key="f_estado_civil")
    c1.selectbox("Nivel de instrucción", [k for k in INSTRUCCION if k in opc["instruccion"]], format_func=INSTRUCCION.get, key="f_instruccion")
    c2.radio("Ocupación", ["AMA DE CASA", "OTRA"], format_func=lambda v: "Ama de casa" if v == "AMA DE CASA" else "Otra ocupación",
             horizontal=True, key="f_ocupacion")
    c1.selectbox("Distrito de residencia", list(opc["distritos"]), format_func=lambda k: opc["distritos"][k], key="f_distrito")
    c2.selectbox("Tipo de seguro", [k for k in SEGURO if k in opc["seguro"]], format_func=SEGURO.get, key="f_seguro")

    st.markdown("##### Antecedentes obstétricos")
    c3, c4 = st.columns(2)
    c3.number_input("Número de embarazos (gestas), contando el actual", 0, 20, key="f_embarazos")
    c4.number_input("Hijos vivos que ya tiene (sin contar el que va a nacer)", 0, 20, key="f_hijos_vivos")
    c3.number_input("Abortos o nacidos muertos previos", 0, 20, key="f_abortos")
    c4.radio("Tipo de embarazo actual", ["Único", "Gemelar o múltiple"], horizontal=True, key="f_multiple")

    b1, b2 = st.columns([3, 1])
    calcular_ok = b1.button("Calcular riesgo", type="primary", width="stretch")
    b2.button("Limpiar", width="stretch", on_click=reiniciar)
    st.button("← Volver al inicio", on_click=ir, args=("inicio",))

    if calcular_ok:
        datos = {k: st.session_state[f"f_{k}"] for k in ["edad", "estado_civil", "instruccion", "ocupacion", "distrito", "seguro",
                                                          "embarazos", "hijos_vivos", "abortos"]}
        datos = {k: (int(v) if k in ("edad", "embarazos", "hijos_vivos", "abortos") else v) for k, v in datos.items()}
        datos["embarazo_multiple"] = st.session_state["f_multiple"] != "Único"
        errores = validar_datos(datos)
        if errores:
            for m in errores:
                st.error(m)
            return
        try:
            res = calcular(datos, usar_api)
        except Exception:
            st.error("No se pudo calcular el resultado en este momento. Inténtelo de nuevo; si el problema continúa, avise al responsable del sistema.")
            return
        st.session_state["resultado"] = {"res": res, "datos": datos,
                                         "guardado": reg.guardar_prediccion(datos, res, opc["distritos"])}
        ir("resultado")
        st.rerun()


# ----------------------------------------------------------------------------- 3. resultado
def pantalla_resultado():
    r = st.session_state.get("resultado")
    if not r:
        ir("formulario")
        st.rerun()
    res = r["res"]
    fg, bg, etiqueta = NIVELES[res["nivel_riesgo"]]
    st.markdown(f"""<div class="tarjeta" style="background:{bg};border-color:{fg};color:{fg}">
        <h1>{etiqueta}</h1><p>Puntaje de riesgo: <b>{res['puntaje']:.0f} %</b></p></div>""", unsafe_allow_html=True)
    st.caption("El puntaje es una medida interna del modelo para ordenar a las gestantes; no es la probabilidad real de que el bebé nazca con bajo peso.")

    st.subheader("¿Por qué este resultado?")
    if res["explicacion_shap"]:
        a, d = res["factores_aumentan"], res["factores_disminuyen"]
        st.markdown("**Factores que aumentan el riesgo:**")
        st.markdown("\n".join(f"- {f['texto'][0].upper() + f['texto'][1:]}" for f in a) if a else "- Ninguno destacado")
        st.markdown("**Factores que lo disminuyen:**")
        st.markdown("\n".join(f"- {f['texto'][0].upper() + f['texto'][1:]}" for f in d) if d else "- Ninguno destacado")
        with st.expander("Ver detalle"):
            det = pd.DataFrame(res["detalle"]).sort_values("contribucion")
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            ax.barh(list(det["texto"]), det["contribucion"] * 100, color=["#cf222e" if v > 0 else "#1f6feb" for v in det["contribucion"]])
            ax.axvline(0, color="black", lw=.8)
            ax.set_xlabel("← baja el riesgo        sube el riesgo →  (puntos de puntaje)")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No fue posible generar la explicación para este caso.")

    if not r.get("guardado", True):
        st.warning("El resultado no se pudo guardar en el historial (¿el archivo predicciones.csv está abierto en Excel?). Ciérrelo y vuelva a calcular.")
    st.markdown('<div class="nota">Resultado orientativo. Siga el protocolo de atención prenatal del establecimiento.</div>', unsafe_allow_html=True)
    st.write("")
    b1, b2 = st.columns(2)
    b1.button("← Volver", width="stretch", on_click=ir, args=("formulario",))
    b2.button("Nueva evaluación", type="primary", width="stretch", on_click=ir, args=("formulario", True))


# ----------------------------------------------------------------------------- historial y reporte
def seccion_historial():
    st.subheader("Historial y reporte")
    h = reg.leer_predicciones()
    if h.empty:
        st.info("Aún no hay evaluaciones guardadas.")
        return
    h = h.iloc[::-1].reset_index(drop=True)
    tabla = pd.DataFrame({
        "Fecha y hora": h["registrado_el"], "Edad": h["edad"], "Distrito": h["distrito"], "Seguro": h["seguro"].map(SEGURO).fillna(h["seguro"]),
        "Embarazos": h["embarazos"], "Riesgo": h["nivel_riesgo"].map({"ALTO": "Alto", "MEDIO": "Medio", "BAJO": "Bajo"}),
        "Puntaje (%)": h["puntaje_riesgo"].round(0),
        "Factor que más aumenta": h["factores_aumentan"].fillna("").apply(lambda t: t.split(" | ")[0] if t else "—")})
    k = st.columns(4)
    k[0].metric("Evaluaciones", len(h))
    for col, n in zip(k[1:], ["Alto", "Medio", "Bajo"]):
        col.metric(f"Riesgo {n.lower()}", int((tabla["Riesgo"] == n).sum()))
    st.caption(f"Puntaje promedio: {h['puntaje_riesgo'].mean():.0f} % · Archivo: data/registros/predicciones.csv")
    st.dataframe(tabla, hide_index=True, width="stretch")
    resumen = (f"# Reporte de evaluaciones\n\n- Total de evaluaciones: {len(h)}\n"
               + "".join(f"- Riesgo {n.lower()}: {int((tabla['Riesgo'] == n).sum())}\n" for n in ["Alto", "Medio", "Bajo"])
               + f"- Edad promedio: {tabla['Edad'].mean():.1f} años\n- Puntaje promedio: {h['puntaje_riesgo'].mean():.0f} %\n"
               f"- Período: {h['registrado_el'].min()} a {h['registrado_el'].max()}\n\nResultados orientativos; no reemplazan el criterio clínico.\n")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar predicciones (CSV completo)", h.iloc[::-1].to_csv(index=False).encode("utf-8-sig"), "predicciones.csv", "text/csv")
    c2.download_button("⬇️ Descargar resumen (texto)", resumen.encode("utf-8"), "resumen_evaluaciones.md", "text/markdown")


# ----------------------------------------------------------------------------- métricas
def leer_tabla(nombre):
    ruta = C.REPORTES_TESIS / "tablas" / nombre
    return pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig") if ruta.exists() else None


def seccion_metricas():
    st.subheader("Métricas del modelo")
    tm, hip, cvr = (leer_tabla(n) for n in ["test_metricas.csv", "hipotesis.csv", "cv_resumen.csv"])
    if tm is None:
        st.warning("Aún no hay métricas. Ejecute el análisis (ml/analisis_tesis.py).")
        return
    f3 = lambda v: f"{v:.3f}".replace(".", ",")
    sis = [m for m in tm["modelo"].unique() if m.startswith("Sistema")][0]
    g = lambda met: tm[(tm.metrica == met) & (tm.modelo == sis)].iloc[0]
    st.markdown(f"**Resultados en datos nuevos** (889 nacidos, 51 con bajo peso) — {sis}")
    c = st.columns(4)
    c[0].metric("AUC-ROC", f3(g("AUC-ROC").valor), help=f"IC 95 %: {f3(g('AUC-ROC').ic95_inf)} – {f3(g('AUC-ROC').ic95_sup)}")
    c[1].metric("Recall", f3(g("Recall").valor))
    c[2].metric("Precisión", f3(g("precision").valor))
    c[3].metric("AUC-PR", f3(g("AUC-PR").valor))
    st.info("El modelo discrimina de forma modesta: sirve para priorizar el seguimiento, no para diagnosticar.")
    for nombre, col in [("Curvas ROC", "curvas_roc.png"), ("Curvas Precisión-Recall", "curvas_pr.png"),
                        ("Matrices de confusión (C0 frente al sistema)", "matrices_confusion_c0_vs_sistema.png"),
                        ("Importancia de los datos (SHAP)", "shap_global.png"), ("Comparación de modelos en validación cruzada", "cv_cajas_auc.png")]:
        ruta = C.REPORTES_TESIS / "figuras" / col
        if ruta.exists():
            with st.expander(nombre):
                st.image(str(ruta))
    if cvr is not None:
        with st.expander("Tabla de validación cruzada 10×5"):
            st.dataframe(cvr[cvr["conjunto"] == "principal_sin_Ipress"][["config", "modelo", "estrategia", "auc_roc_media", "auc_pr_media", "f1_media", "recall_media"]].round(3),
                         hide_index=True, width="stretch")
    if hip is not None:
        with st.expander("Decisión por hipótesis"):
            st.dataframe(hip[["hipotesis", "se_rechaza_H0", "criterio"]], hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- SUS
def seccion_sus():
    st.markdown("<div style='text-align:center'><b>UNIVERSIDAD AUTÓNOMA DEL PERÚ</b><br>Escuela Profesional de Ingeniería de Sistemas</div>",
                unsafe_allow_html=True)
    st.subheader("Cuestionario de usabilidad del sistema (System Usability Scale, SUS)")
    st.caption("Investigación: Sistema predictivo basado en Machine Learning para la detección del bajo peso al nacer en un centro de "
               "atención primaria de la provincia de Rodríguez de Mendoza, Amazonas, 2026.")
    if msg := st.session_state.pop("sus_ok", None):
        st.success(msg)
    ver = st.session_state.setdefault("sus_form", 0)       # cambia al guardar, para limpiar el formulario
    with st.form(f"form_sus_{ver}"):
        c1, c2, c3 = st.columns(3)
        c1.text_input("Cuestionario N.º", value=str(reg.siguiente_sus()), disabled=True)
        fecha = c2.date_input("Fecha", value=date.today(), format="DD/MM/YYYY")
        casos = c3.selectbox("Casos registrados (de 5)", [0, 1, 2, 3, 4, 5], index=None, placeholder="Elija")
        st.markdown("**Datos del participante**")
        profesion = st.radio("Profesión", reg.PROFESIONES, horizontal=True, index=None)
        otra = st.text_input("Si marcó «Otra», especifique")
        experiencia = st.radio("Años de experiencia en atención prenatal", reg.EXPERIENCIA, horizontal=True, index=None)
        usa_pc = st.radio("¿Usa computadora en su trabajo diario?", ["Sí", "No"], horizontal=True, index=None)
        st.markdown("**Estimado(a) profesional de salud:** este cuestionario busca conocer su opinión sobre la facilidad de uso del "
                    "sistema predictivo de bajo peso al nacer que acaba de utilizar. Es anónimo: no escriba su nombre. "
                    "No hay respuestas correctas ni incorrectas.")
        st.markdown("**Instrucciones:** marque una sola opción por afirmación, según su grado de acuerdo. Responda todas las afirmaciones. "
                    "Si no está seguro(a), marque la opción 3.")
        st.caption(" · ".join(f"{i} = {t}" for i, t in enumerate(reg.ESCALA, 1)))
        resp = [st.radio(f"{i}. {t}", [1, 2, 3, 4, 5], index=None, horizontal=True, key=f"sus{ver}_{i}")
                for i, t in enumerate(reg.SUS_ITEMS, 1)]
        comentarios = st.text_area("Comentarios o sugerencias (opcional)")
        enviar = st.form_submit_button("Guardar cuestionario", type="primary")
    if enviar:
        faltan = [n for n, v in [("Casos registrados", casos), ("Profesión", profesion), ("Años de experiencia", experiencia),
                                 ("Uso de computadora", usa_pc)] if v is None]
        sin_resp = [str(i) for i, v in enumerate(resp, 1) if v is None]
        if sin_resp:
            faltan.append("afirmaciones " + ", ".join(sin_resp))
        if profesion == "Otra" and not otra.strip():
            faltan.append("especificar la profesión «Otra»")
        if faltan:
            st.error("Falta completar: " + "; ".join(faltan) + ".")
        else:
            r = reg.guardar_sus(fecha, casos, profesion, otra, experiencia, usa_pc, resp, comentarios)
            if r is None:
                st.error("No se pudo guardar el cuestionario (¿el archivo sus_respuestas.csv está abierto en Excel?). Ciérrelo e inténtelo de nuevo.")
            else:
                st.session_state["sus_ok"] = f"¡Gracias! Su cuestionario quedó registrado (N.º {r[0]})."
                st.session_state["sus_form"] = ver + 1
                st.rerun()

    r = reg.leer_sus()
    if not r.empty:
        st.divider()
        st.markdown("##### Resultados registrados")
        alfa = reg.alfa_cronbach(r)
        k = st.columns(4)
        k[0].metric("Cuestionarios", len(r))
        k[1].metric("SUS promedio", f"{r['puntaje_sus'].mean():.1f}", help="Una media mayor a 68 se considera aceptable")
        k[2].metric("Con SUS ≥ 68", int((r["puntaje_sus"] >= 68).sum()))
        k[3].metric("Alfa de Cronbach", "—" if alfa is None else f"{alfa:.2f}".replace(".", ","), help="Se calcula desde 3 cuestionarios")
        st.caption("Archivo: data/registros/sus_respuestas.csv")
        st.dataframe(r[["cuestionario_n", "fecha", "profesion", "experiencia", "usa_computadora", "casos_registrados", "puntaje_sus", "interpretacion"]]
                     .iloc[::-1], hide_index=True, width="stretch")
        st.download_button("⬇️ Descargar SUS (CSV completo)", r.to_csv(index=False).encode("utf-8-sig"), "sus_respuestas.csv", "text/csv")


# ----------------------------------------------------------------------------- enrutador
if seccion.startswith("🩺"):
    {"inicio": pantalla_inicio, "formulario": pantalla_formulario, "resultado": pantalla_resultado}[st.session_state["pantalla"]]()
elif seccion.startswith("📜"):
    seccion_historial()
elif seccion.startswith("📊"):
    seccion_metricas()
else:
    seccion_sus()
