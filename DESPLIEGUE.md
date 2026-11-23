# Despliegue temporal (Streamlit Community Cloud + Supabase)

Objetivo: que los participantes usen la app desde un enlace, y que las predicciones y los cuestionarios SUS se guarden en una base de datos
que no se pierde al reiniciar. Todo gratuito y se puede borrar al terminar.

```
Participante ──► Streamlit Cloud (app + modelo) ──► Supabase (PostgreSQL: predicciones, sus_respuestas)
```

Códigos: **código de acceso** (se entrega a los participantes) y **código de administrador** (solo para ustedes; permite ver el historial
de predicciones y los resultados del SUS). Se configuran en los *secretos*, nunca en el repositorio.

## 1. Crear la base de datos (Supabase)

1. Entre a https://supabase.com con su cuenta de GitHub y cree un proyecto (*New project*).
   - Nombre: `bpn-demo`. Región: la más cercana (South America, São Paulo).
   - **Database password:** use una clave **solo con letras y números** (evita problemas en la dirección de conexión) y guárdela.
2. Abra *Project Settings → Database → Connection string*, elija **Session pooler** y la pestaña **URI**. Copie la dirección y reemplace
   `[YOUR-PASSWORD]` por su clave. Se parece a:
   `postgresql://postgres.abcdefgh:SU_CLAVE@aws-0-sa-east-1.pooler.supabase.com:5432/postgres`
3. No hace falta crear tablas: la app las crea sola al primer guardado (`predicciones` y `sus_respuestas`).

## 2. Publicar la app (Streamlit Community Cloud)

1. Suba los cambios a GitHub (`git push`).
2. Entre a https://share.streamlit.io con GitHub y pulse *Create app*.
   - Repositorio: `jesujesus/proyecto_bpn`, rama `main`, archivo principal: `app/app.py`.
   - Si el repositorio es privado, Streamlit pedirá permiso para acceder a su GitHub.
3. En *Advanced settings*: **Python 3.13** y pegue en *Secrets* (con sus valores):
   ```toml
   database_url = "postgresql://postgres.abcdefgh:SU_CLAVE@aws-0-sa-east-1.pooler.supabase.com:5432/postgres"
   codigo_acceso = "un-codigo-para-participantes"
   codigo_admin = "un-codigo-solo-para-ustedes"
   aviso = "Evaluación de usabilidad: use solo casos simulados. No ingrese datos reales ni identificables de pacientes."
   ```
4. *Deploy*. La primera vez tarda varios minutos (instala las librerías). Al terminar tendrá un enlace `https://....streamlit.app`.

## 3. Comprobar que funciona

1. Abra el enlace, ingrese el código de acceso y registre una evaluación.
2. En *Historial y reporte* pida el código de administrador: debe verse la evaluación.
3. En Supabase (*Table Editor*) deben aparecer las tablas con la fila guardada.
4. Borre esa fila de prueba (Table Editor) antes de la aplicación real.

## 4. Ver y exportar los datos

- **En la app:** *Historial y reporte* y *Cuestionario SUS → Resultados* (código de administrador), con botones de descarga en CSV.
- **En Supabase:** *Table Editor → tabla → Export → CSV*.
- Para analizar el SUS: guarde el CSV como `data/registros/sus_respuestas.csv` y ejecute `python ml/sus_analisis.py`.

## 5. Al terminar

1. Exporte los CSV (los datos del SUS son parte de los resultados de la tesis).
2. Borre la app en Streamlit Cloud y el proyecto en Supabase.
3. Cambie los códigos si va a reutilizar algo.

## Cosas a tener en cuenta

- **Suspensión:** el plan gratuito de Streamlit duerme la app tras un tiempo sin uso (se despierta al abrir el enlace) y el de Supabase
  pausa el proyecto tras ~1 semana sin actividad (se reactiva desde su panel). Abra la app el día anterior a la aplicación.
- **Seguridad:** el enlace es público; el código de acceso evita usos ajenos pero no es una seguridad fuerte. Use códigos largos y no los publique.
  La tabla tiene activada la seguridad por filas, y la app se conecta con la clave de la base de datos (que está solo en los secretos).
- **Privacidad:** la app no pide nombre ni documento, pero no se deben ingresar datos reales de pacientes (el consentimiento prevé casos simulados).
- **Modelo y versiones:** el modelo guardado exige las mismas versiones de scikit-learn y SHAP que fijan `requirements.txt`; use Python 3.13.
- **Tesis:** ajuste los textos que dicen que todo es local (2.8, Tabla 3, Figura 12 y Anexo 8): la capa de datos es PostgreSQL (Supabase) y
  la aplicación está alojada en Streamlit Community Cloud.
- **Uso local:** sin `database_url`, la app guarda en CSV (`data/registros/`); sin códigos, queda abierta (modo de desarrollo).
