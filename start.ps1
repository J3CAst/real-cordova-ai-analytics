# ============================================================
# start.ps1 — Script de arranque oficial
# Real Córdova AI Analytics | Tesis SENATI EEST 2026
# Uso: .\start.ps1 [-SkipOllama] [-SkipUI]
# ============================================================

param(
    [switch]$SkipOllama,   # Si Ollama ya está corriendo
    [switch]$SkipUI        # Solo API, sin Streamlit
)

$ROOT = $PSScriptRoot
$PYTHON = "$ROOT\venv\Scripts\python.exe"
$UVICORN = "$ROOT\venv\Scripts\uvicorn.exe"
$STREAMLIT = "$ROOT\venv\Scripts\streamlit.exe"
$OLLAMA = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

# ── Colores ─────────────────────────────────────────────────
function Write-Ok   { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Fail { param($msg) Write-Host "  [!!] $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  [..] $msg" -ForegroundColor Cyan }
function Write-Head { param($msg) Write-Host "`n$msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkGreen
Write-Host "  Real Cordova AI Analytics — Arranque del Sistema" -ForegroundColor Green
Write-Host "  Tesis SENATI EEST 2026 | Acosta Cruz, Jean Steven" -ForegroundColor DarkGreen
Write-Host "============================================================`n" -ForegroundColor DarkGreen

# ── Verificar venv ──────────────────────────────────────────
Write-Head "1. Verificando entorno Python..."
if (-not (Test-Path $PYTHON)) {
    Write-Fail "venv no encontrado en $ROOT\venv"
    Write-Host "   Crea el venv con: python -m venv venv ; venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Ok "venv encontrado: $PYTHON"

# ── Cargar .env ─────────────────────────────────────────────
Write-Head "2. Cargando variables de entorno (.env)..."
$envFile = "$ROOT\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+?)\s*=\s*(.+)\s*$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
    Write-Ok ".env cargado"
} else {
    Write-Fail ".env no encontrado — usando valores por defecto"
}

$OLLAMA_URL = $env:OLLAMA_BASE_URL
if (-not $OLLAMA_URL) { $OLLAMA_URL = "http://localhost:11435" }
$API_PORT   = $env:API_PORT;    if (-not $API_PORT)   { $API_PORT   = "8000" }
$UI_PORT    = $env:STREAMLIT_PORT; if (-not $UI_PORT) { $UI_PORT    = "8501" }

# ── Helper: esperar un endpoint ─────────────────────────────
function Wait-Endpoint {
    param($url, $label, $timeoutSec = 60)
    $elapsed = 0
    Write-Info "Esperando $label en $url ..."
    while ($elapsed -lt $timeoutSec) {
        try {
            $r = Invoke-RestMethod $url -TimeoutSec 3 -ErrorAction Stop
            Write-Ok "$label listo ($elapsed s)"
            return $true
        } catch {
            Start-Sleep -Seconds 3
            $elapsed += 3
            Write-Host "     $elapsed s..." -NoNewline
            Write-Host "`r" -NoNewline
        }
    }
    Write-Fail "$label no respondió en $timeoutSec s"
    return $false
}

# ── Ollama ──────────────────────────────────────────────────
Write-Head "3. Iniciando Ollama (LLM inference)..."
if (-not $SkipOllama) {
    $ollamaRunning = $false
    try {
        Invoke-RestMethod "$OLLAMA_URL/api/tags" -TimeoutSec 3 | Out-Null
        $ollamaRunning = $true
        Write-Ok "Ollama ya estaba corriendo en $OLLAMA_URL"
    } catch { }

    if (-not $ollamaRunning) {
        if (Test-Path $OLLAMA) {
            $env:OLLAMA_HOST = "127.0.0.1:11435"
            Start-Process $OLLAMA -ArgumentList "serve" -WindowStyle Minimized
            $ok = Wait-Endpoint "$OLLAMA_URL/api/tags" "Ollama" 45
            if (-not $ok) { exit 1 }
        } else {
            Write-Fail "ollama.exe no encontrado en $OLLAMA"
            Write-Host "   Instala Ollama desde https://ollama.ai"
            exit 1
        }
    }
} else {
    Write-Info "SkipOllama: asumiendo que Ollama ya corre en $OLLAMA_URL"
}

# ── FastAPI ─────────────────────────────────────────────────
Write-Head "4. Iniciando FastAPI (backend API)..."
$apiRunning = $false
try {
    Invoke-RestMethod "http://localhost:$API_PORT/salud" -TimeoutSec 3 | Out-Null
    $apiRunning = $true
    Write-Ok "FastAPI ya estaba corriendo en puerto $API_PORT"
} catch { }

if (-not $apiRunning) {
    $apiArgs = "api.main:app", "--host", "0.0.0.0", "--port", $API_PORT, "--workers", "1"
    Start-Process $UVICORN -ArgumentList $apiArgs -WorkingDirectory $ROOT -WindowStyle Normal
    $ok = Wait-Endpoint "http://localhost:$API_PORT/salud" "FastAPI" 90
    if (-not $ok) { exit 1 }

    # Verificar que todos los servicios backend están OK
    $salud = Invoke-RestMethod "http://localhost:$API_PORT/salud" -TimeoutSec 5
    if ($salud.estado -ne "ok") {
        Write-Fail "FastAPI arrancó pero en estado '$($salud.estado)': $($salud.mensaje)"
    } else {
        Write-Ok "FastAPI OK — Ollama=$($salud.ollama_activo), PostgreSQL=$($salud.postgres_activo)"
    }
}

# ── Streamlit ────────────────────────────────────────────────
if (-not $SkipUI) {
    Write-Head "5. Iniciando Streamlit (interfaz de usuario)..."
    $uiRunning = $false
    try {
        Invoke-WebRequest "http://localhost:$UI_PORT" -UseBasicParsing -TimeoutSec 3 | Out-Null
        $uiRunning = $true
        Write-Ok "Streamlit ya estaba corriendo en puerto $UI_PORT"
    } catch { }

    if (-not $uiRunning) {
        $uiArgs = "run", "ui\app.py", "--server.port", $UI_PORT, "--server.headless", "true"
        Start-Process $STREAMLIT -ArgumentList $uiArgs -WorkingDirectory $ROOT -WindowStyle Minimized
        Start-Sleep -Seconds 6
        try {
            Invoke-WebRequest "http://localhost:$UI_PORT" -UseBasicParsing -TimeoutSec 5 | Out-Null
            Write-Ok "Streamlit listo en http://localhost:$UI_PORT"
        } catch {
            Write-Fail "Streamlit no respondió — verifica manualmente: http://localhost:$UI_PORT"
        }
    }
}

# ── Resumen final ────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkGreen
Write-Host "  SISTEMA LISTO" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor DarkGreen
Write-Host "  API Backend   : http://localhost:$API_PORT"
Write-Host "  API Docs      : http://localhost:$API_PORT/docs"
Write-Host "  Cola / Estado : http://localhost:$API_PORT/cola"
Write-Host "  Predicciones  : http://localhost:$API_PORT/prediccion/semana"
if (-not $SkipUI) {
    Write-Host "  UI Streamlit  : http://localhost:$UI_PORT"
}
Write-Host "============================================================`n" -ForegroundColor DarkGreen
