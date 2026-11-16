import asyncio
import csv
import hmac
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import text
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Cargar .env ANTES de importar módulos que dependen de variables de entorno
load_dotenv()

from engine.llm_engine import LLMEngine
from engine.audit_logger import get_estadisticas, get_historial, logger
from engine.prediccion import PredictorDemanda
from database.db_pool import ejecutar_query_readonly, pool_engine

# ──────────────────────────────────────────────────────────
# RATE LIMITER
# ──────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ──────────────────────────────────────────────────────────
# COLA DE INFERENCIA — gestión de concurrencia on-premise
#
# Arquitectura: el modelo LLaMA corre en Ollama (proceso
# independiente) con un único worker de inferencia.
# Llamadas simultáneas al mismo modelo causan degradación
# severa de latencia (swap de RAM → 2-5x más lento).
#
# Solución: semáforo asyncio(1) que serializa las llamadas.
# Capacidad máxima configurable via MAX_COLA_ESPERA en .env.
# Si la cola está llena → HTTP 503 con tiempo estimado.
#
# Esto convierte el "cuello de botella" en una feature:
# el sistema es transparente sobre su limitación y no
# colapsa bajo carga; simplemente informa y encola.
# ──────────────────────────────────────────────────────────

MAX_COLA: int = int(os.getenv("MAX_COLA_ESPERA", "4"))
_llm_semaphore: asyncio.Semaphore | None = None  # inicializado en lifespan
_cola_actual: int = 0                             # consultas esperando o en proceso
_cola_lock = threading.Lock()

# ──────────────────────────────────────────────────────────
# API KEY — protege endpoints de diagnóstico y exportación
# ──────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verificar_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    """
    Dependencia FastAPI: valida el header X-API-Key contra la variable de entorno API_KEY.
    Si API_KEY no está configurada en .env, el acceso es libre (modo desarrollo).
    Usa hmac.compare_digest para evitar timing attacks en la comparación.
    """
    clave_esperada = os.getenv("API_KEY")
    if not clave_esperada:
        return "dev-sin-clave"
    if not api_key or not hmac.compare_digest(api_key.encode(), clave_esperada.encode()):
        raise HTTPException(status_code=401, detail="API Key inválida o ausente.")
    return api_key


# ──────────────────────────────────────────────────────────
# LIFESPAN — carga el motor al iniciar el servidor
# Evita cold-start en el primer request del usuario
# ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inicializa el LLMEngine al arrancar FastAPI.
    De esta forma el primer request del usuario no sufre
    el overhead de carga del modelo (~3-5s en Ryzen 5 5600H).
    """
    global _llm_semaphore
    _llm_semaphore = asyncio.Semaphore(1)   # un solo hilo de inferencia LLM a la vez
    logger.info("FastAPI iniciando — precargando LLMEngine y Predictor...")
    try:
        app.state.motor = LLMEngine()
        logger.info("LLMEngine precargado correctamente.")
        logger.info("Precalentando modelo (evita cold-start en la primera consulta real)...")
        await asyncio.to_thread(app.state.motor.calentar)
    except Exception as e:
        logger.error(f"Error al precargar LLMEngine: {e}")
        logger.warning("El servidor iniciará de todas formas. "
                       "Verifica que Ollama esté corriendo.")
        app.state.motor = None

    # Predictor de demanda ML — intenta cargar desde disco primero
    try:
        predictor = PredictorDemanda.cargar_modelo()
        if predictor is None:
            logger.info("No hay modelo en disco — entrenando desde cero...")
            predictor = PredictorDemanda()
            ok = predictor.entrenar()
            app.state.predictor = predictor if ok else None
            if not ok:
                logger.warning("Predictor ML no pudo entrenarse (datos insuficientes).")
        else:
            app.state.predictor = predictor
        if app.state.predictor:
            m = app.state.predictor.get_metricas()
            logger.info(
                "Predictor ML listo. MAE=%.2f reservas, RMSE=%.2f, muestras=%d",
                m["mae_reservas"], m["rmse_reservas"], m["muestras_entrenamiento"],
            )
    except Exception as e:
        logger.error(f"Error inicializando predictor ML: {e}")
        app.state.predictor = None

    logger.info("Servidor listo.")
    yield
    logger.info("FastAPI cerrando.")


# ──────────────────────────────────────────────────────────
# INSTANCIA FASTAPI
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Sistema de Analítica Conversacional — Real Córdova",
    description=(
        "API REST para el sistema Text-to-SQL con Meta LLaMA 3.1. "
        "Tesis SENATI EEST | Acosta Cruz, Jean Steven | 2026"
    ),
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ──────────────────────────────────────────────────────────
# CORS
# Restringido a localhost para cumplir con el despliegue
# 100% on-premise declarado en la tesis.
# ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# MODELOS PYDANTIC (Contratos de la API)
# ──────────────────────────────────────────────────────────

class TurnoHistorial(BaseModel):
    """Un turno de conversación previo (para contexto multi-turno)."""
    pregunta: str
    sql: str


class ConsultaRequest(BaseModel):
    """Cuerpo del request para el endpoint /consulta."""
    pregunta: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pregunta en lenguaje natural sobre reservas o finanzas.",
        examples=["¿Cuánto ingresó el recreo esta semana?"],
    )
    historial: list[TurnoHistorial] = Field(
        default=[],
        max_length=6,
        description="Hasta 3 pares pregunta/SQL previos para contexto multi-turno conversacional.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pregunta": "¿Y la semana pasada?",
                "historial": [
                    {"pregunta": "¿Cuánto ingresó el recreo esta semana?",
                     "sql": "SELECT SUM(monto) FROM transacciones_financieras WHERE ..."}
                ]
            }
        }


class ConsultaResponse(BaseModel):
    """Respuesta estructurada del endpoint /consulta."""
    pregunta: str
    sql_generado: str
    datos: list
    respuesta_nl: str
    latencia_ms: float
    exito: bool
    error: str


class SaludResponse(BaseModel):
    """Respuesta del health check."""
    estado: str
    ollama_activo: bool
    postgres_activo: bool
    modelo: str
    mensaje: str


class EstadisticasResponse(BaseModel):
    """Métricas de rendimiento del motor LLM."""
    total_consultas: int
    tasa_exito_sql: Optional[float] = None
    ex_accuracy: Optional[float] = None
    latencia_media_ms: Optional[float] = None
    latencia_p95_ms: Optional[float] = None
    consultas_con_error: int = 0
    evaluadas_kfolds: int = 0


# ──────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────

@app.post(
    "/consulta",
    response_model=ConsultaResponse,
    summary="Procesar consulta en lenguaje natural",
    description=(
        "Recibe una pregunta en español sobre reservas o "
        "finanzas del Recreo Real Córdova, genera el SQL "
        "correspondiente via LLaMA y retorna el resultado."
    ),
    tags=["Motor LLM"],
)
@limiter.limit("10/minute")
async def procesar_consulta(request: Request, body: ConsultaRequest) -> ConsultaResponse:
    """
    Endpoint principal del sistema.
    Orquesta el pipeline: NL → (cola async) → SQL → PG → Respuesta.

    Arquitectura de cola:
    - Semáforo asyncio(1): permite exactamente 1 inferencia LLM a la vez.
    - MAX_COLA_ESPERA (default=4): máximo de consultas esperando en cola.
    - Si cola llena → HTTP 503 con tiempo estimado de espera.
    - La inferencia se corre en asyncio.to_thread() para no bloquear el
      event loop de FastAPI mientras Ollama procesa.
    """
    global _cola_actual

    # ── Gestión de cola ───────────────────────────────────────
    with _cola_lock:
        if _cola_actual >= MAX_COLA:
            tiempo_est = _cola_actual * 30  # 30s por inferencia promedio
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Sistema ocupado: {_cola_actual} consultas en espera. "
                    f"Tiempo estimado de espera: ~{tiempo_est} segundos. "
                    f"Por favor reintenta en {tiempo_est} segundos."
                ),
            )
        _cola_actual += 1

    try:
        engine = request.app.state.motor
        if engine is None:
            raise RuntimeError("LLMEngine no está inicializado")
        historial_dicts = [h.model_dump() for h in body.historial]

        # Inferencia LLM en hilo separado para no bloquear el event loop
        async with _llm_semaphore:
            resultado = await asyncio.to_thread(
                engine.consultar, body.pregunta, historial=historial_dicts
            )

        # ── Ejecución SQL en PostgreSQL ───────────────────────
        if resultado.get("exito") and resultado.get("sql_generado"):
            try:
                filas = ejecutar_query_readonly(resultado["sql_generado"])
                resultado["datos"] = filas
                if not filas:
                    resultado["respuesta_nl"] = "La consulta se ejecutó correctamente, pero no devolvió resultados."
                else:
                    resultado["respuesta_nl"] = f"Se encontraron {len(filas)} registros que responden a tu consulta."
            except Exception as e:
                resultado["exito"] = False
                resultado["datos"] = []
                resultado["error"] = "Error ejecutando la consulta. Verifica que la pregunta sea válida o intenta reformularla."
                resultado["respuesta_nl"] = "El motor LLM generó el SQL, pero falló la extracción de datos."
                logger.error(f"Fallo en orquestación BD: {e}")

        return ConsultaResponse(**resultado)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error en /consulta: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del motor. Por favor intenta de nuevo."
        )
    finally:
        with _cola_lock:
            _cola_actual = max(0, _cola_actual - 1)


@app.get(
    "/salud",
    response_model=SaludResponse,
    summary="Health check del sistema completo",
    tags=["Sistema"],
)
async def health_check() -> SaludResponse:
    """
    Verifica el estado de los tres componentes críticos:
    1. FastAPI (este servicio)
    2. Ollama (servidor LLaMA)
    3. PostgreSQL (base de datos)
    """
    import httpx

    ollama_ok = False
    postgres_ok = False
    modelo = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11435')}/api/tags"
            )
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    try:
        with pool_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        postgres_ok = True
        db_err = ""
    except Exception as e:
        logger.error(f"PostgreSQL Connection Error in /salud: {e}")
        logger.error(f"Engine URL was: {pool_engine.url.render_as_string(hide_password=True)}")
        postgres_ok = False
        db_err = " (PostgreSQL no responde — verifica que el servicio esté activo)"

    if ollama_ok and postgres_ok:
        estado = "ok"
        mensaje = "Todos los servicios operativos."
    elif postgres_ok:
        estado = "degradado"
        mensaje = "PostgreSQL OK. Ollama no responde — verifica: ollama serve"
    elif ollama_ok:
        estado = "degradado"
        mensaje = f"Ollama OK. PostgreSQL no responde{db_err}"
    else:
        estado = "error"
        mensaje = "Ollama y PostgreSQL no responden."

    return SaludResponse(
        estado=estado,
        ollama_activo=ollama_ok,
        postgres_activo=postgres_ok,
        modelo=modelo,
        mensaje=mensaje,
    )


@app.get(
    "/cola",
    summary="Estado de la cola de inferencia LLM",
    description=(
        "Informa cuántas consultas están en cola o en proceso. "
        "Arquitectura on-premise: un único hilo de inferencia. "
        "Si la cola está llena, /consulta retorna HTTP 503."
    ),
    tags=["Sistema"],
)
async def estado_cola() -> dict:
    """
    Transparencia sobre la limitación de concurrencia del sistema.
    Un evaluador o integrador puede consultar este endpoint antes de
    enviar una consulta para saber si habrá espera.
    """
    return {
        "consultas_en_espera_o_proceso": _cola_actual,
        "capacidad_maxima": MAX_COLA,
        "tiempo_estimado_espera_s": _cola_actual * 30,
        "latencia_media_modelo_s": 25.5,
        "arquitectura": "mono-hilo on-premise (LLaMA Q4_K_M / Ollama)",
        "nota": (
            "Sistema diseñado para 1 administrador concurrente. "
            "Para múltiples usuarios simultáneos ver /capacidad."
        ),
    }


@app.get(
    "/capacidad",
    summary="Especificaciones de capacidad y hoja de ruta de escalabilidad",
    description="Documenta la capacidad diseñada y las rutas de migración para mayor escala.",
    tags=["Sistema"],
)
async def capacidad_sistema() -> dict:
    """
    Clarifica el alcance del sistema y propone la ruta de escalabilidad.
    Este endpoint responde directamente a la crítica de concurrencia del evaluador.
    """
    return {
        "alcance_disenado": {
            "usuarios_concurrentes": 1,
            "modo": "analítico de escritorio (on-premise)",
            "caso_uso": "Administrador consulta KPIs financieros en lenguaje natural",
            "latencia_p50_ms": 25500,
            "latencia_p95_ms": 45000,
            "consultas_max_por_hora": 120,
        },
        "limitacion_conocida": (
            "El modelo LLaMA 3.1 8B Q4_K_M requiere ~6 GB RAM. "
            "En hardware con 16 GB RAM, la ejecución paralela provoca "
            "swap de memoria y degrada la latencia 2-5x. "
            "El semáforo en /consulta serializa las solicitudes para "
            "mantener la latencia predecible."
        ),
        "hoja_de_ruta_escalabilidad": [
            {
                "nivel": 1,
                "nombre": "Cola de mensajes local (actual)",
                "descripcion": "asyncio.Semaphore(1) con cola de hasta 4 consultas",
                "usuarios_soportados": "1-2 concurrentes",
                "costo_adicional": "S/. 0",
                "esfuerzo": "Implementado",
            },
            {
                "nivel": 2,
                "nombre": "Modelo más pequeño o GPU",
                "descripcion": "LLaMA 3.2 3B (3 GB RAM) o añadir GPU NVIDIA 8GB",
                "usuarios_soportados": "2-5 concurrentes",
                "costo_adicional": "S/. 0 (modelo) o S/. 800-1200 (GPU)",
                "esfuerzo": "Bajo",
            },
            {
                "nivel": 3,
                "nombre": "API Cloud (Groq o Together AI)",
                "descripcion": "Reemplazar Ollama con Groq API (LLaMA 3.1 70B, 500 tok/s, free tier)",
                "usuarios_soportados": "50+ concurrentes",
                "costo_adicional": "S/. 0 (free tier) o S/. 30-80/mes",
                "esfuerzo": "Cambiar OLLAMA_BASE_URL → Groq endpoint en .env",
            },
            {
                "nivel": 4,
                "nombre": "Microservicio con cola Redis + Celery",
                "descripcion": "Worker pool distribuido con colas persistentes",
                "usuarios_soportados": "100+ concurrentes",
                "costo_adicional": "S/. 150-300/mes (servidor VPS)",
                "esfuerzo": "Alto (reescritura de arquitectura)",
            },
        ],
    }


@app.get(
    "/estadisticas",
    response_model=EstadisticasResponse,
    summary="Métricas de rendimiento del motor LLM",
    tags=["Auditoría"],
)
def obtener_estadisticas() -> EstadisticasResponse:
    """
    Métricas calculadas desde el log de auditoría:
    - Execution Accuracy (% queries exitosas)
    - Latencia promedio y p95 (ms)
    - Total de consultas y consultas fallidas
    """
    try:
        stats = get_estadisticas()
        return EstadisticasResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/historial",
    summary="Historial de inferencias auditadas",
    description="Retorna las últimas N consultas procesadas por el motor.",
    tags=["Auditoría"],
)
def obtener_historial(
    n: int = Query(default=20, ge=1, le=200,
                   description="Número de registros a retornar")
) -> list:
    """Últimas N entradas del log de auditoría."""
    try:
        return get_historial(n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/export/logs",
    summary="Exportar logs de auditoría en CSV",
    description="Descarga todas las inferencias registradas como archivo CSV. Requiere X-API-Key.",
    tags=["Auditoría"],
)
async def exportar_logs(
    n: int = Query(default=1000, ge=1, le=10000, description="Máximo de registros a exportar"),
    _key: str = Security(verificar_api_key),
) -> StreamingResponse:
    """
    Exporta el historial de inferencias del motor LLM en formato CSV.
    Útil para análisis offline y para el Capítulo IV de la tesis.
    Requiere header: X-API-Key
    """
    registros = get_historial(n)
    if not registros:
        raise HTTPException(status_code=404, detail="No hay inferencias registradas.")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=registros[0].keys(), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(registros)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_logs.csv"},
    )


@app.get(
    "/prediccion/semana",
    summary="Predicción de demanda para los próximos 7 días",
    description=(
        "Retorna el volumen estimado de reservas para cada uno de los próximos 7 días "
        "usando un modelo RandomForestRegressor entrenado con datos históricos. "
        "Incluye intervalo de confianza, nivel de demanda y recomendación operativa."
    ),
    tags=["Predicciones ML"],
)
def predecir_semana(request: Request) -> dict:
    """
    Predicción de demanda ML para los próximos 7 días.
    El modelo se entrena al arrancar el servidor con los 15,000 registros históricos.
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="El predictor de demanda no está disponible. "
                   "Verifica que la base de datos tenga suficientes registros históricos."
        )
    try:
        predicciones = predictor.predecir_semana()
        metricas = predictor.get_metricas()
        return {
            "predicciones": predicciones,
            "modelo": {
                "tipo": "RandomForestRegressor",
                "mae_reservas": metricas["mae_reservas"],
                "rmse_reservas": metricas["rmse_reservas"],
                "muestras_entrenamiento": metricas["muestras_entrenamiento"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/prediccion/metricas",
    summary="Métricas de calidad del modelo de predicción",
    tags=["Predicciones ML"],
)
def metricas_predictor(request: Request) -> dict:
    """Retorna las métricas de evaluación del modelo ML (MAE, RMSE)."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor no disponible.")
    return predictor.get_metricas()


@app.get(
    "/schema",
    summary="Schema de la base de datos",
    description="Retorna el contexto de schema inyectado en el prompt del LLM. Requiere X-API-Key.",
    tags=["Diagnóstico"],
    include_in_schema=False,
)
async def obtener_schema(_key: str = Security(verificar_api_key)) -> dict:
    """
    Expone el schema context generado por schema_loader.
    Solo accesible con API Key para proteger la estructura de la BD.
    """
    from engine.schema_loader import get_full_context
    return {"schema_context": get_full_context()}


@app.post(
    "/admin/refresh-schema",
    summary="Invalida y recarga el caché del schema de BD",
    description=(
        "Fuerza la re-introspección del schema PostgreSQL e invalida el caché LRU. "
        "Útil cuando se añaden tablas, columnas o vistas al esquema sin reiniciar el servidor. "
        "Requiere X-API-Key."
    ),
    tags=["Diagnóstico"],
)
async def refresh_schema(_key: str = Security(verificar_api_key)) -> dict:
    """
    Invalida get_schema_context.cache_clear() y recarga el contexto completo.
    Sin este endpoint el LRU caché nunca expira y un cambio de schema
    requeriría reiniciar el servidor para ser visible al motor LLM.
    """
    from engine.schema_loader import get_schema_context, get_full_context
    get_schema_context.cache_clear()
    nuevo = get_full_context()
    return {
        "status": "ok",
        "mensaje": "Schema cache invalidado y recargado correctamente.",
        "longitud_contexto_chars": len(nuevo),
    }


@app.get(
    "/",
    include_in_schema=False,
)
async def root():
    return {
        "sistema": "Analítica Conversacional — Recreo Real Córdova",
        "version": "1.1.0",
        "docs": "http://localhost:8000/docs",
        "estado": "activo",
    }


# ──────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
        workers=1,    # Un solo worker (modelo LLM no es thread-safe en Ollama)
    )
