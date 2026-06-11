import os
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "inferencias.jsonl"

# Logger estándar de Python para errores del propio módulo
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("audit_logger")


# ──────────────────────────────────────────────────────────
# ESTRUCTURA DE UN REGISTRO DE INFERENCIA
# ──────────────────────────────────────────────────────────

def _construir_registro(
    pregunta_nl: str,
    sql_generado: str,
    resultado_ok: bool,
    latencia_ms: float,
    filas_retorno: int = 0,
    error_msg: Optional[str] = None,
    sql_esperado: Optional[str] = None,   # Para validación K-Folds
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construye el dict que se serializa a JSONL.
    Incluye Execution Accuracy si se provee sql_esperado.
    """
    # Execution Accuracy (EX): 1 si el SQL generado es
    # funcionalmente equivalente al esperado, 0 si no.
    # En la validación K-Folds se compara resultado real vs esperado.
    ex_score: Optional[int] = None
    if sql_esperado is not None:
        # Comparación normalizada (sin espacios, minúsculas)
        sql_norm = _normalizar_sql(sql_generado)
        esperado_norm = _normalizar_sql(sql_esperado)
        ex_score = 1 if sql_norm == esperado_norm else 0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "pregunta_nl": pregunta_nl,
        "sql_generado": sql_generado,
        "sql_esperado": sql_esperado,
        "resultado_ok": resultado_ok,
        "filas_retorno": filas_retorno,
        "latencia_ms": round(latencia_ms, 2),
        "ex_score": ex_score,          # 1=correcto, 0=incorrecto, None=no evaluado
        "error_msg": error_msg,
    }


def _normalizar_sql(sql: str) -> str:
    """Normaliza SQL para comparación semántica básica."""
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)          # colapsar espacios
    sql = re.sub(r';\s*$', '', sql)         # quitar punto y coma final
    # quitar comillas en identificadores
    sql = re.sub(r'"(\w+)"', r'\1', sql)
    return sql


# ──────────────────────────────────────────────────────────
# FUNCIONES PÚBLICAS
# ──────────────────────────────────────────────────────────

def registrar_inferencia(
    pregunta_nl: str,
    sql_generado: str,
    resultado_ok: bool,
    latencia_ms: float,
    filas_retorno: int = 0,
    error_msg: Optional[str] = None,
    sql_esperado: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Registra una inferencia en el archivo JSONL de auditoría.

    Args:
        pregunta_nl   : Consulta en lenguaje natural del usuario.
        sql_generado  : SQL producido por el motor LLaMA.
        resultado_ok  : True si el SQL ejecutó sin errores en PG.
        latencia_ms   : Tiempo total de inferencia en milisegundos.
        filas_retorno : Número de filas retornadas por el SQL.
        error_msg     : Mensaje de error PostgreSQL si resultado_ok=False.
        sql_esperado  : SQL de referencia para cálculo de EX (K-Folds).
        session_id    : ID de sesión para agrupar turnos de conversación.

    Returns:
        Dict con el registro completo guardado.
    """
    registro = _construir_registro(
        pregunta_nl=pregunta_nl,
        sql_generado=sql_generado,
        resultado_ok=resultado_ok,
        latencia_ms=latencia_ms,
        filas_retorno=filas_retorno,
        error_msg=error_msg,
        sql_esperado=sql_esperado,
        session_id=session_id,
    )

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"No se pudo escribir en {LOG_FILE}: {e}")

    return registro


def calcular_metricas_globales() -> Dict[str, Any]:
    """
    Lee el archivo JSONL y calcula las métricas acumuladas
    del sistema. Usado para el panel de métricas de la UI
    y para el reporte de resultados de la tesis (Cap. IV).

    Métricas calculadas:
    - Total de consultas procesadas
    - Execution Accuracy promedio (EX%)
    - Tasa de éxito de ejecución SQL
    - Latencia media, p50, p95 (milisegundos)
    """
    if not LOG_FILE.exists():
        return {
            "total_consultas": 0,
            "ex_accuracy": None,
            "tasa_exito_sql": None,
            "latencia_media_ms": None,
            "latencia_p95_ms": None,
        }

    registros = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue

    if not registros:
        return {"total_consultas": 0}

    total = len(registros)
    exitos_sql = sum(1 for r in registros if r.get("resultado_ok"))
    latencias = [r["latencia_ms"] for r in registros if r.get("latencia_ms")]
    ex_scores = [r["ex_score"] for r in registros
                 if r.get("ex_score") is not None]

    latencias_sorted = sorted(latencias)
    p95_idx = int(len(latencias_sorted) * 0.95) - 1 if latencias else 0

    return {
        "total_consultas": total,
        "tasa_exito_sql": round(exitos_sql / total * 100, 1) if total else 0,
        "ex_accuracy": round(sum(ex_scores) / len(ex_scores) * 100, 1)
        if ex_scores else None,
        "latencia_media_ms": round(sum(latencias) / len(latencias), 1)
        if latencias else None,
        "latencia_p95_ms": round(latencias_sorted[p95_idx], 1)
        if latencias else None,
        "consultas_con_error": total - exitos_sql,
        "evaluadas_kfolds": len(ex_scores),
    }


def leer_ultimas_inferencias(n: int = 20) -> list:
    """
    Retorna las últimas N inferencias del log.
    Usado por la UI para mostrar historial de consultas.
    """
    if not LOG_FILE.exists():
        return []

    registros = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue

    return registros[-n:]


def get_estadisticas() -> Dict[str, Any]:
    """Compatibilidad con api.main: estadísticas agregadas de auditoría."""
    return calcular_metricas_globales()


def get_historial(n: int = 20) -> list:
    """Compatibilidad con api.main: últimas N inferencias."""
    return leer_ultimas_inferencias(n)


# ──────────────────────────────────────────────────────────
# CONTEXT MANAGER — medir latencia automáticamente
# ──────────────────────────────────────────────────────────

class MedirLatencia:
    """
    Context manager para medir el tiempo de inferencia.

    Uso:
        with MedirLatencia() as m:
            resultado = modelo.invoke(prompt)
        print(m.ms)   # latencia en milisegundos
    """

    def __init__(self):
        self.ms: float = 0.0
        self._inicio: float = 0.0

    def __enter__(self):
        self._inicio = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self._inicio) * 1_000
