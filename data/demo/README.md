# Datos simulados (demostración)

`predicciones_demo.csv` y `sus_respuestas_demo.csv` contienen **datos inventados**, generados con `python ml/generar_demo.py`
(semilla fija). Sirven solo para mostrar la app o probar un despliegue.

- Las 10 predicciones las calcula el modelo real, pero con gestantes **simuladas**.
- Los 10 cuestionarios SUS **no son de personal de salud real**.
- Cada fila lleva `simulado = Sí`.

**No son resultados de la tesis.** No deben usarse para contrastar HE4 ni mezclarse con `data/registros/`, donde la app guarda los
registros reales (esos no se suben a GitHub).

Para verlos dentro de la app (solo en una copia de prueba), copie los archivos a `data/registros/` quitando `_demo` del nombre.
