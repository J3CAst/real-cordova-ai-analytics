import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Cargar .env ANTES de importar módulos que dependen de variables de entorno
load_dotenv()

from engine.llm_engine import LLMEngine
from engine.audit_logger import get_estadisticas, get_historial, logger
from database.db_pool import ejecutar_query_readonly, pool_engine

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
    logger.info("FastAPI iniciando — precargando LLMEngine...")
    try:
        app.state.motor = LLMEngine()
        logger.info("LLMEngine precargado correctamente. Servidor listo.")
    except Exception as e:
        logger.error(f"Error al precargar LLMEngine: {e}")
        logger.warning("El servidor iniciará de todas formas. "
                       "Verifica que Ollama esté corriendo.")
        app.state.motor = None
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
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI en http://localhost:8000/docs
    redoc_url="/redoc",
)

# ──────────────────────────────────────────────────────────
# CORS
# Restringido a localhost para cumplir con el despliegue
# 100% on-premise declarado en la tesis.
# ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",    # Streamlit
        "http://127.0.0.1:8501",
        "http://localhost:3000",    # Dev front alternativo
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# MODELOS PYDANTIC (Contratos de la API)
# ──────────────────────────────────────────────────────────

class ConsultaRequest(BaseModel):
    """Cuerpo del request para el endpoint /consulta."""
    pregunta: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pregunta en lenguaje natural sobre reservas o finanzas.",
        examples=["¿Cuánto ingresó el recreo esta semana?"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pregunta": "¿Cuántas reservas llegaron por WhatsApp este mes?"
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
    estado: str       # "ok" | "degradado" | "error"
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
def procesar_consulta(request: ConsultaRequest) -> ConsultaResponse:
    """
    Endpoint principal del sistema.
    Orquesta el pipeline completo: NL → SQL → PG → Respuesta.
    """
    try:
        engine = app.state.motor
        if engine is None:
            raise RuntimeError("LLMEngine no está inicializado")
        resultado = engine.consultar(request.pregunta)
        
        # ──────────────────────────────────────────────────────
        # INYECCIÓN VÍA B: Orquestación de Ejecución SQL
        # ──────────────────────────────────────────────────────
        if resultado.get("exito") and resultado.get("sql_generado"):
            try:
                # Ejecutamos el SQL generado de forma segura
                filas = ejecutar_query_readonly(resultado["sql_generado"])
                resultado["datos"] = filas
                
                if not filas:
                    resultado["respuesta_nl"] = "La consulta se ejecutó correctamente, pero no devolvió resultados."
                else:
                    resultado["respuesta_nl"] = f"Se encontraron {len(filas)} registros que responden a tu consulta."
            except Exception as e:
                resultado["exito"] = False
                resultado["datos"] = []
                resultado["error"] = f"Error ejecutando SQL en DB: {str(e)}"
                resultado["respuesta_nl"] = "El motor LLM generó el SQL, pero falló la extracción de datos."
                logger.error(f"Fallo en orquestación BD: {e}")

        return ConsultaResponse(**resultado)

    except Exception as e:
        logger.exception(f"Error en /consulta: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del motor: {str(e)}"
        )


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
    import httpx

    ollama_ok = False
    postgres_ok = False
    modelo = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

    # Verificar Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags"
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
        logger.error(f"Engine URL was: {pool_engine.url}")
        postgres_ok = False
        db_err = f" (Err: {e} | URL: {pool_engine.url.render_as_string(hide_password=False)})"
    
    # Estado general
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
    "/estadisticas",
    response_model=EstadisticasResponse,
    summary="Métricas de rendimiento del motor LLM",
    description=(
        "Retorna las métricas agregadas de todas las "
        "inferencias registradas en logs/inferencias.csv. "
        "Fuente de datos para el Capítulo IV de la tesis."
    ),
    tags=["Auditoría"],
)
def obtener_estadisticas() -> EstadisticasResponse:
    """
    Métricas calculadas desde el CSV de auditoría:
    - Execution Accuracy (% queries exitosas)
    - Latencia promedio, máxima y mínima (ms)
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
    """
    Últimas N entradas del CSV de auditoría.
    Útil para monitoreo en tiempo real desde Streamlit.
    """
    try:
        return get_historial(n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/schema",
    summary="Schema de la base de datos",
    description="Retorna el contexto de schema inyectado en el prompt del LLM.",
    tags=["Diagnóstico"],
)
async def obtener_schema() -> dict:
    """
    Expone el schema context generado por schema_loader.
    Útil para verificar que el LLM está viendo el schema correcto.
    """
    from engine.schema_loader import get_full_context
    return {"schema_context": get_full_context()}


@app.get(
    "/",
    include_in_schema=False,
)
async def root():
    return {
        "sistema": "Analítica Conversacional — Recreo Real Córdova",
        "version": "1.0.0",
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
