import importlib
import os
import re
import time
from functools import lru_cache
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

ollama_module = None
try:
    ollama_module = importlib.import_module("langchain_ollama")
except ImportError:
    pass

OllamaLLM = None
if ollama_module is not None:
    OllamaLLM = getattr(ollama_module, "OllamaLLM", getattr(ollama_module, "Ollama", None))

from engine.schema_loader import get_full_context
from engine.audit_logger  import registrar_inferencia, logger

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1:8b-instruct-q4_K_M")
LLM_TEMPERATURE = 0.0
LLM_NUM_CTX     = 4096
LLM_NUM_THREAD  = 6
LLM_NUM_PREDICT = 256
LLM_TOP_P       = 0.9
LLM_TOP_K       = 50
LLM_REPEAT_PENALTY = 1.1
MAX_REINTENTOS  = 2


# ──────────────────────────────────────────────────────────
# FIX-01: Few-Shot ampliado — 13 ejemplos
# Los 6 nuevos ejemplos cubren exactamente los patrones
# que fallaron en la Iteración 1 del benchmark.
# ──────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
-- ════════════════════════════════════════════════════════
-- EJEMPLOS DE CONSULTAS (Few-Shot — 12 ejemplos)
-- ════════════════════════════════════════════════════════

-- [1] Conteo simple
-- Pregunta: ¿Cuántas reservas hay en total?
SELECT COUNT(*) AS total_reservas FROM reservas;

-- [2] Ingreso semanal — usar transacciones_financieras directamente
-- Pregunta: ¿Cuánto ingresó el recreo esta semana?
SELECT COALESCE(SUM(CASE WHEN tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN monto ELSE 0 END), 0) AS ingresos_semana
FROM transacciones_financieras
WHERE fecha_transaccion >= DATE_TRUNC('week', CURRENT_DATE);

-- [3] Ingreso de ayer — usar transacciones_financieras directamente
-- Pregunta: ¿Cuánto ingresó el recreo ayer?
SELECT COALESCE(SUM(CASE WHEN tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN monto ELSE 0 END), 0) AS ingresos_ayer
FROM transacciones_financieras
WHERE fecha_transaccion = CURRENT_DATE - 1;

-- [4] Cuadre de caja de hoy — usar transacciones_financieras y reservas
-- Pregunta: Dame el resumen financiero de hoy.
-- Pregunta: ¿Cuánto fue el cuadre de caja de hoy?
SELECT tf.fecha_transaccion AS fecha,
    COUNT(DISTINCT r.reserva_id) AS total_reservas,
    COALESCE(SUM(CASE WHEN tf.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN tf.monto ELSE 0 END), 0) AS total_ingresos,
    COALESCE(SUM(CASE WHEN tf.tipo = 'gasto_operativo' THEN ABS(tf.monto) ELSE 0 END), 0) AS total_gastos,
    COALESCE(SUM(CASE WHEN tf.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN tf.monto ELSE 0 END), 0)
      - COALESCE(SUM(CASE WHEN tf.tipo = 'gasto_operativo' THEN ABS(tf.monto) ELSE 0 END), 0)
      AS utilidad_neta
FROM transacciones_financieras tf
LEFT JOIN reservas r ON r.reserva_id = tf.reserva_id
WHERE tf.fecha_transaccion = CURRENT_DATE
GROUP BY tf.fecha_transaccion;

-- [5] Utilidad neta de ayer — usar transacciones_financieras
-- Pregunta: ¿Cuál fue la utilidad neta de ayer?
SELECT COALESCE(SUM(CASE WHEN tf.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN tf.monto ELSE 0 END), 0)
      - COALESCE(SUM(CASE WHEN tf.tipo = 'gasto_operativo' THEN ABS(tf.monto) ELSE 0 END), 0)
      AS utilidad_neta
FROM transacciones_financieras tf
WHERE tf.fecha_transaccion = CURRENT_DATE - 1;

-- [6] Ingreso promedio diario del mes — usar transacciones_financieras
-- Pregunta: ¿Cuál es el ingreso promedio diario del mes?
SELECT ROUND(AVG(daily_ingreso), 2) AS ingreso_promedio_diario
FROM (
    SELECT tf.fecha_transaccion AS fecha,
           COALESCE(SUM(CASE WHEN tf.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN tf.monto ELSE 0 END), 0) AS daily_ingreso
    FROM transacciones_financieras tf
    WHERE DATE_TRUNC('month', tf.fecha_transaccion) = DATE_TRUNC('month', CURRENT_DATE)
    GROUP BY tf.fecha_transaccion
) sub;

-- [7] Reservas por mes — usar transacciones_financieras para fechas
-- Pregunta: ¿Cuántas reservas hubo este mes?
SELECT COUNT(*) AS reservas_mes
FROM reservas
WHERE DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);

-- [8] Cliente con más reservas — JOIN con alias obligatorio + GROUP BY completo
-- Pregunta: ¿Cuántas reservas tiene el cliente más frecuente?
SELECT c.nombre_codigo, COUNT(r.reserva_id) AS total_reservas
FROM clientes c
JOIN reservas r ON r.cliente_id = c.cliente_id
GROUP BY c.cliente_id, c.nombre_codigo
ORDER BY total_reservas DESC
LIMIT 1;

-- [9] Total de personas por distrito — JOIN correcto con alias
-- Pregunta: ¿Cuál es el total de personas por distrito este mes?
SELECT c.distrito, SUM(r.numero_personas) AS total_personas
FROM reservas r
JOIN clientes c ON c.cliente_id = r.cliente_id
WHERE r.estado = 'completada'
  AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE)
GROUP BY c.distrito
ORDER BY total_personas DESC;

-- [10] Gasto de clientes corporativos — triple JOIN con alias
-- Pregunta: ¿Cuánto gastaron los clientes corporativos este mes?
SELECT COALESCE(SUM(t.monto), 0) AS gasto_corporativo
FROM reservas r
JOIN clientes c ON c.cliente_id = r.cliente_id
JOIN transacciones_financieras t ON t.reserva_id = r.reserva_id
WHERE c.segmento = 'corporativo'
  AND t.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional')
  AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);

-- [11] Ticket promedio por canal — JOIN items_reserva
-- Pregunta: ¿Cuál es el ticket promedio por canal?
SELECT r.canal, ROUND(AVG(ir.subtotal), 2) AS ticket_promedio
FROM reservas r
JOIN items_reserva ir ON ir.reserva_id = r.reserva_id
WHERE r.estado = 'completada'
GROUP BY r.canal
ORDER BY ticket_promedio DESC;

-- [12] Tasa de cancelación por canal — GROUP BY correcto
-- Pregunta: ¿Cuál es la tasa de cancelación por canal?
SELECT
    canal,
    ROUND(100.0 * SUM(CASE WHEN estado = 'cancelada' THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS tasa_cancelacion_pct
FROM reservas
GROUP BY canal
ORDER BY tasa_cancelacion_pct DESC;

-- [13] Reservas por temporada del detalle — alias vd obligatorio
-- Pregunta: ¿Cuántas reservas por temporada muestra el detalle este año?
SELECT
    vd.temporada,
    COUNT(*) AS total_reservas
FROM v_reservas_detalle vd
WHERE EXTRACT(YEAR FROM vd.fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY vd.temporada;
"""


def _build_system_prompt() -> str:
    schema_context = get_full_context()
    return f"""Eres un asistente de analítica de datos del Recreo Campestre Real Córdova (Lima, Perú).
Respondes SOLO preguntas sobre reservas y control financiero del establecimiento.

REGLAS ESTRICTAS:
1. Genera ÚNICAMENTE consultas SELECT. Nunca INSERT, UPDATE, DELETE ni DDL.
2. Usa SOLO las tablas y vistas del schema. No inventes nombres de tablas.
3. SIEMPRE usa alias de tabla en JOINs: r=reservas, c=clientes, t=transacciones_financieras, ir=items_reserva.
4. Si usas v_reservas_detalle, define alias vd y califica todas sus columnas: vd.fecha_reserva, vd.es_temporada_alta, vd.temporada, vd.canal, vd.estado.
5. En JOINs multi-tabla SIEMPRE califica las columnas con el alias: r.canal, c.segmento, t.monto.
6. En GROUP BY incluye TODAS las columnas del SELECT que no sean funciones de agregación.
7. La columna transacciones_financieras.tipo es un ENUM con valores: ingreso_reserva, ingreso_consumo, ingreso_adicional, gasto_operativo, devolucion.
   No uses LIKE sobre esa columna; compara con valores exactos o usa IN(...) según corresponda.
8. No uses CURRENT_YEAR ni expresiones no estándar para el año. Usa EXTRACT(YEAR FROM CURRENT_DATE) o DATE_TRUNC('year', CURRENT_DATE).
9. Para cuadre de caja o resumen diario: usar las tablas transacciones_financieras y reservas. No asumas que v_resumen_diario existe si no está listada en el schema.
10. Para consultas de ingresos, usar tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional'). Para gastos operativos, usar tipo = 'gasto_operativo'.
11. Para consultas de reservas con datos del cliente: usar v_reservas_detalle.
12. Los montos están en Soles peruanos (S/.). Zona horaria: America/Lima.
13. Razona paso a paso antes del SQL (Chain-of-Thought).
14. Responde EXACTAMENTE con este formato:

RAZONAMIENTO:
[1-3 líneas: qué tablas/vistas necesitas y por qué]

SQL:
[query SQL completa y ejecutable]

RESPUESTA:
[respuesta en español para el administrador]

{schema_context}

{FEW_SHOT_EXAMPLES}
"""


def _extraer_sql(texto: str) -> Optional[str]:
    match = re.search(
        r"SQL:\s*```(?:sql)?\s*(.*?)```",
        texto, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    match = re.search(
        r"SQL:\s*(SELECT\s+.*?);?\s*(?:RESPUESTA:|$)",
        texto, re.DOTALL | re.IGNORECASE
    )
    if match:
        sql = match.group(1).strip()
        return sql if sql.endswith(";") else sql + ";"

    match = re.search(r"(SELECT\s+.+?);", texto, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + ";"

    return None


def _extraer_respuesta_nl(texto: str) -> str:
    match = re.search(
        r"RESPUESTA:\s*(.*?)$", texto, re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else "Consulta procesada correctamente."


def _normalize_sql(sql: str) -> str:
    sql_normalizada = sql.strip()

    sql_normalizada = re.sub(
        r"((?:\b\w+\.)?)tipo\s*::\s*text\s*LIKE\s*'ingreso%'",
        r"\1tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional')",
        sql_normalizada,
        flags=re.IGNORECASE,
    )
    sql_normalizada = re.sub(
        r"((?:\b\w+\.)?)tipo\s*LIKE\s*'ingreso%'",
        r"\1tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional')",
        sql_normalizada,
        flags=re.IGNORECASE,
    )
    sql_normalizada = re.sub(
        r"\bCURRENT_YEAR\b",
        "EXTRACT(YEAR FROM CURRENT_DATE)",
        sql_normalizada,
        flags=re.IGNORECASE,
    )
    sql_normalizada = re.sub(
        r"CURRENT_DATE\s*::\s*INTEGER\s*/\s*100",
        "EXTRACT(YEAR FROM CURRENT_DATE)",
        sql_normalizada,
        flags=re.IGNORECASE,
    )

    def _normalize_v_reservas_detalle_aliases(sql_text: str) -> str:
        alias_names = []
        reserved = (
            "AS", "WHERE", "ON", "GROUP", "ORDER", "LIMIT", "HAVING",
            "UNION", "EXCEPT", "INTERSECT", "INNER", "LEFT", "RIGHT",
            "FULL", "CROSS", "NATURAL", "JOIN", "FROM"
        )

        def _replace_named_alias(match):
            keyword = match.group(1)
            alias = match.group(2)
            if alias and alias.upper() not in reserved and alias.lower() != "vd":
                alias_names.append(alias)
            return f"{keyword} v_reservas_detalle vd"

        normalized = re.sub(
            r"\b(FROM|JOIN)\s+v_reservas_detalle\s+(?:AS\s+)?(\w+)\b",
            _replace_named_alias,
            sql_text,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(
            r"\b(FROM|JOIN)\s+v_reservas_detalle\b(?!\s+\w)",
            lambda m: f"{m.group(1)} v_reservas_detalle vd",
            normalized,
            flags=re.IGNORECASE,
        )

        for alias in alias_names:
            normalized = re.sub(
                rf"\b{re.escape(alias)}\.",
                "vd.",
                normalized,
                flags=re.IGNORECASE,
            )

        normalized = re.sub(
            r"\bv_reservas_detalle\.(?=[A-Za-z_])",
            "vd.",
            normalized,
            flags=re.IGNORECASE,
        )

        return normalized

    sql_normalizada = _normalize_v_reservas_detalle_aliases(sql_normalizada)

    if "vd." in sql_normalizada:
        for bare_col in [
            "es_temporada_alta",
            "fecha_reserva",
            "estado",
            "canal",
            "temporada",
            "ticket_total",
            "segmento",
            "numero_personas",
            "dia_semana_nombre",
            "dia_semana_num",
            "mes_num",
            "mes_nombre",
        ]:
            sql_normalizada = re.sub(
                rf"(?<![\.\w]){bare_col}(?![\w\.])",
                f"vd.{bare_col}",
                sql_normalizada,
                flags=re.IGNORECASE,
            )

    return sql_normalizada


def _ejecutar_sql(sql: str, engine) -> tuple:
    try:
        with engine.connect() as conn:
            result   = conn.execute(text(sql))
            columnas = list(result.keys())
            filas    = [dict(zip(columnas, row)) for row in result.fetchall()]
        return True, filas, ""
    except SQLAlchemyError as e:
        return False, [], str(e)


class LLMEngine:
    def __init__(self):
        logger.info("Inicializando LLMEngine v2...")
        self.llm = OllamaLLM(
            model          = OLLAMA_MODEL,
            base_url       = OLLAMA_BASE_URL,
            temperature    = LLM_TEMPERATURE,
            num_ctx        = LLM_NUM_CTX,
            num_thread     = LLM_NUM_THREAD,
            num_predict    = LLM_NUM_PREDICT,
            top_p          = LLM_TOP_P,
            top_k          = LLM_TOP_K,
            repeat_penalty = LLM_REPEAT_PENALTY,
        )
        dsn = (
            f"postgresql+psycopg2://"
            f"{os.getenv('DB_USER','postgres')}:"
            f"{os.getenv('DB_PASS','postgres')}@"
            f"{os.getenv('DB_HOST','localhost')}:"
            f"{os.getenv('DB_PORT','5432')}/"
            f"{os.getenv('DB_NAME','real_cordova_db')}"
        )
        self.db_engine     = create_engine(dsn, pool_pre_ping=True)
        self.system_prompt = _build_system_prompt()
        logger.info("LLMEngine v2 listo.")

    def consultar(self, pregunta: str) -> dict:
        t_inicio  = time.perf_counter()
        sql_final = ""
        datos     = []
        respuesta = ""
        exito     = False
        error_msg = ""

        try:
            prompt = (
                f"{self.system_prompt}\n\n"
                f"PREGUNTA DEL ADMINISTRADOR:\n{pregunta}\n\n"
                f"Responde siguiendo exactamente el formato solicitado:"
            )
            logger.info(f"Consultando LLaMA v2: '{pregunta[:80]}'")
            raw = self.llm.invoke(prompt)

            sql_generado = _extraer_sql(raw)
            if not sql_generado:
                error_msg = "El modelo no generó SQL válido."
            else:
                sql_final    = _normalize_sql(sql_generado)
                ok, datos, err = _ejecutar_sql(sql_final, self.db_engine)

                if ok:
                    exito     = True
                    respuesta = _extraer_respuesta_nl(raw)
                else:
                    logger.warning(f"SQL falló: {err}. Self-correction...")
                    for intento in range(1, MAX_REINTENTOS + 1):
                        prompt_fix = (
                            f"{self.system_prompt}\n\n"
                            f"PREGUNTA DEL ADMINISTRADOR:\n{pregunta}\n\n"
                            f"El SQL generado falló:\n```sql\n{sql_final}\n```\n"
                            f"Error PostgreSQL: {err}\n\n"
                            f"IMPORTANTE: Revisa alias de tabla, columnas en GROUP BY "
                            f"y nombres exactos de vistas. Corrige y responde "
                            f"(intento {intento}/{MAX_REINTENTOS}):"
                        )
                        raw2       = self.llm.invoke(prompt_fix)
                        sql2       = _extraer_sql(raw2)
                        if sql2:
                            sql2 = _normalize_sql(sql2)
                            ok2, datos2, err2 = _ejecutar_sql(sql2, self.db_engine)
                            if ok2:
                                sql_final = sql2
                                datos     = datos2
                                respuesta = _extraer_respuesta_nl(raw2)
                                exito     = True
                                logger.info(f"Self-correction OK en intento {intento}")
                                break
                            err = err2
                    if not exito:
                        error_msg = f"Sin SQL válido tras {MAX_REINTENTOS} intentos. Último error: {err}"

        except Exception as e:
            error_msg = f"Error interno: {str(e)}"
            logger.exception(error_msg)

        latencia_ms = (time.perf_counter() - t_inicio) * 1000
        registrar_inferencia(
            pregunta_nl = pregunta,
            sql_generado = sql_final or "(sin SQL)",
            resultado_ok = exito,
            latencia_ms = latencia_ms,
            filas_retorno = len(datos),
            error_msg = error_msg,
        )

        return {
            "pregunta"    : pregunta,
            "sql_generado": sql_final,
            "datos"       : datos,
            "respuesta_nl": respuesta or error_msg,
            "latencia_ms" : round(latencia_ms, 2),
            "exito"       : exito,
            "error"       : error_msg,
        }


_engine_instance: Optional[LLMEngine] = None

def get_engine() -> LLMEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMEngine()
    return _engine_instance


if __name__ == "__main__":
    print("Test LLMEngine v2")
    engine = LLMEngine()
    tests  = [
        "Dame el resumen financiero de hoy.",
        "¿Cuántas reservas tiene el cliente más frecuente?",
        "¿Cuál fue la utilidad neta de ayer?",
        "¿Cuánto ingresó el recreo esta semana?",
    ]
    for p in tests:
        r = engine.consultar(p)
        estado = "✓" if r["exito"] else "✗"
        print(f"{estado} [{r['latencia_ms']:.0f}ms] {p}")
        if not r["exito"]:
            print(f"  ERROR: {r['error']}")