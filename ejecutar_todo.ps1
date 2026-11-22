# Ejecuta el pipeline en orden: ETL -> entrenamiento del modelo -> (opcional) análisis de la tesis -> API + app.
# Uso:  .\ejecutar_todo.ps1                  (ETL + modelo y luego abre API + app; ~2 min)
#       .\ejecutar_todo.ps1 -ConAnalisis     (además rehace el análisis completo de la tesis; ~10 min más)
#       .\ejecutar_todo.ps1 -SinApp          (solo ETL + modelo, sin abrir la app)
param([switch]$ConAnalisis, [switch]$SinApp)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot

# Usa el venv de la carpeta superior (..\venv) o uno local (.venv); si no existe, el Python del sistema
$py = "python"
foreach ($c in @((Join-Path $PSScriptRoot ".venv\Scripts\python.exe"), (Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"))) {
    if (Test-Path $c) { $py = $c; break }
}

function Paso($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Correr($args_) { & $py @args_; if ($LASTEXITCODE -ne 0) { throw "Falló: $args_" } }

Paso "1/4 Instalando librerías"
Correr @("-m", "pip", "install", "-q", "-r", "requirements.txt")

Paso "2/4 ETL (limpieza, diccionario y reporte)"
Correr @("etl\limpieza.py")

Paso "3/4 Entrenamiento del modelo (validación cruzada 10x5 y umbrales)"
Correr @("ml\entrenar.py")

if ($ConAnalisis) {
    Paso "Análisis de la tesis (C0-C3, hipótesis, subgrupos, SHAP, figuras) -> reports\tesis"
    Correr @("ml\analisis_tesis.py")
}

if ($SinApp) { Write-Host "`nListo." -ForegroundColor Green; exit 0 }

Paso "4/4 Iniciando API (puerto 8000) y aplicación (puerto 8501)"
$api = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "api.main:app", "--port", "8000" -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 6
Write-Host "API: http://127.0.0.1:8000/docs   |   App: http://localhost:8501   (Ctrl+C para cerrar)" -ForegroundColor Green
try { & $py -m streamlit run app\app.py --server.port 8501 } finally { if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force } }
