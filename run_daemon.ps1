# ============================================================
# run_daemon.ps1 — Ejecución integral continua de Real Córdova
# ============================================================
$ROOT = $PSScriptRoot
$PYTHON = "$ROOT\venv\Scripts\python.exe"
$OLLAMA = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

# 1. Variables de entorno
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_BASE_URL = "http://localhost:11435"
$env:API_PORT = "8001"
$env:STREAMLIT_PORT = "8501"

Write-Host "============================================================"
Write-Host "  Iniciando Real Cordova AI Analytics (Completo)"
Write-Host "============================================================"

# Liberar puertos 8001 y 8501 si quedaron ocupados
try {
    $connections = Get-NetTCPConnection -LocalPort 8001, 8501 -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $connections) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
} catch {}

# 2. Ollama
$ollamaOk = $false
try {
    $null = Invoke-RestMethod "http://127.0.0.1:11435/api/tags" -TimeoutSec 3
    $ollamaOk = $true
    Write-Host "[OK] Ollama activo en http://127.0.0.1:11435"
} catch {}

if (-not $ollamaOk) {
    if (Test-Path $OLLAMA) {
        Write-Host "[..] Arrancando Ollama en 127.0.0.1:11435..."
        Start-Process $OLLAMA -ArgumentList "serve" -WindowStyle Minimized
        Start-Sleep -Seconds 3
    } else {
        Write-Host "[!] ollama.exe no encontrado en $OLLAMA. Intentando 'ollama' en PATH..."
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Minimized -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }
}

# 3. FastAPI (Uvicorn) en background
Write-Host "[..] Iniciando FastAPI en http://0.0.0.0:8001..."
$apiArgs = "-m uvicorn api.main:app --host 0.0.0.0 --port 8001 --workers 1"
$apiProc = Start-Process $PYTHON -ArgumentList $apiArgs -WorkingDirectory $ROOT -PassThru -WindowStyle Minimized

# Esperar a que FastAPI esté listo
$apiReady = $false
for ($i = 0; $i -lt 45; $i++) {
    try {
        $salud = Invoke-RestMethod "http://localhost:8001/salud" -TimeoutSec 3
        if ($salud) {
            $apiReady = $true
            Write-Host "[OK] FastAPI operativo: Ollama=$($salud.ollama_activo), PostgreSQL=$($salud.postgres_activo)"
            break
        }
    } catch {}
    Start-Sleep -Seconds 2
}

if (-not $apiReady) {
    Write-Host "[!] Advertencia: FastAPI tardando en responder. Continuando con Streamlit..."
}

# 4. Streamlit en foreground
Write-Host "[..] Iniciando Streamlit en http://localhost:8501..."
Write-Host "============================================================"
Write-Host "  SISTEMA EN EJECUCION CONTINUA"
Write-Host "  Frontend Streamlit: http://localhost:8501"
Write-Host "  Backend FastAPI   : http://localhost:8001"
Write-Host "  Documentacion API : http://localhost:8001/docs"
Write-Host "============================================================"

& $PYTHON -m streamlit run ui\app.py --server.port 8501 --server.headless true
