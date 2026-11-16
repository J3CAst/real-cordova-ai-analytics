import importlib
import logging as _bootstrap_log
import os
import re
import time
import unicodedata
from difflib import get_close_matches
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

OllamaLLM = None
try:
    _ollama_mod = importlib.import_module("langchain_ollama")
    OllamaLLM = getattr(_ollama_mod, "OllamaLLM", getattr(_ollama_mod, "Ollama", None))
except ImportError as _e:
    _bootstrap_log.getLogger(__name__).error(
        "langchain-ollama no está instalado o está roto. "
        "Instala con: pip install langchain-ollama. Error original: %s", _e
    )

from engine.schema_loader import get_full_context
from engine.audit_logger import registrar_inferencia, logger

load_dotenv()

# ──────────────────────────────────────────────────────────
# DETECCIÓN DE PROMPT INJECTION
# Filtra preguntas que intentan manipular las instrucciones
# del sistema o ejecutar comandos SQL maliciosos.
# ──────────────────────────────────────────────────────────
_PATRONES_INYECCION = [
    r"ignore\s+(all\s+)?previous",
    r"forget\s+your\s+instructions",
    r"you\s+are\s+now",
    r"act\s+as",
    r"system\s*:",
    r"union\s+select",
    r"drop\s+table",
    r"insert\s+into",
    r"delete\s+from",
    r"update\s+\w+\s+set",
]
_RE_INYECCION = re.compile(
    "|".join(_PATRONES_INYECCION),
    re.IGNORECASE,
)


def _detectar_inyeccion(texto: str) -> bool:
    return bool(_RE_INYECCION.search(texto))


# ──────────────────────────────────────────────────────────
# NORMALIZACIÓN DE ENTRADA (robustez ante datos ruidosos)
#
# Problema: el dataset sintético tiene texto perfecto.
# En producción real aparecen: faltas de ortografía, texto
# sin acentos (WhatsApp, teclado físico), abreviaturas
# informales y terminología local peruana.
#
# Solución: pipeline de normalización antes de enviar al LLM.
# El LLM recibe el texto limpio; el log registra el original.
# ──────────────────────────────────────────────────────────

# Vocabulario canónico de términos de negocio
_VOCAB_NEGOCIO = [
    "reservas", "ingresos", "egresos", "transacciones", "clientes",
    "temporada", "cancelaciones", "utilidad", "consumo", "mesas",
    "ocupacion", "semana", "mes", "ayer", "hoy", "mañana",
    "canal", "whatsapp", "presencial", "completada", "confirmada",
    "pendiente", "cancelada", "items", "ticket", "promedio",
    "temporada alta", "verano", "invierno", "recurrente",
]

# Correcciones ortográficas explícitas (sin acentos → con acentos)
_CORRECCIONES = {
    "cuanto": "cuánto",
    "cuantos": "cuántos",
    "cuantas": "cuántas",
    "cuando": "cuándo",
    "cuales": "cuáles",
    "cual": "cuál",
    "mas": "más",
    "maas": "más",
    "tambien": "también",
    "manana": "mañana",
    "proxima": "próxima",
    "ultimo": "último",
    "ultimos": "últimos",
    "ingreso": "ingresó",     # conjugación ambigua: normaliza al verbo
    "reservo": "reservó",
    "genero": "generó",
    "cancelo": "canceló",
    "pq": "porque",
    "xq": "porque",
    "q ": "que ",
    "tb ": "también ",
    "smna": "semana",
    "rsrv": "reserva",
    "rese": "reserva",
    "resrvas": "reservas",
}

_RE_ESPACIOS = re.compile(r"\s{2,}")
_RE_SIGNOS_ES = re.compile(r"[¿¡]")


def _normalizar_pregunta(texto: str) -> tuple[str, list[str]]:
    """
    Normaliza el texto de entrada del usuario para mejorar la robustez
    ante typos, texto sin acentos y abreviaturas informales.

    Retorna (texto_normalizado, lista_de_correcciones_aplicadas) para
    que el audit log registre las transformaciones.

    Pipeline:
    1. Normalización Unicode NFC (resuelve composición de caracteres)
    2. Strip + colapsar espacios múltiples
    3. Correcciones ortográficas explícitas (dict)
    4. Fuzzy match de términos de dominio (difflib, cutoff=0.82)
    """
    original = texto.strip()
    transformaciones: list[str] = []

    # 1. Unicode NFC: ã → ã (composición canónica)
    normalizado = unicodedata.normalize("NFC", original)

    # 2. Colapsar espacios
    normalizado = _RE_ESPACIOS.sub(" ", normalizado).strip()

    # 3. Correcciones ortográficas
    palabras = normalizado.split()
    palabras_out = []
    for p in palabras:
        p_lower = p.lower().rstrip("?,.")
        if p_lower in _CORRECCIONES:
            correcto = _CORRECCIONES[p_lower]
            # Preservar mayúscula inicial si la tenía
            if p[0].isupper():
                correcto = correcto.capitalize()
            transformaciones.append(f"'{p}' → '{correcto}'")
            palabras_out.append(correcto)
        else:
            palabras_out.append(p)
    normalizado = " ".join(palabras_out)

    # 4. Fuzzy match de términos del dominio
    palabras2 = normalizado.split()
    palabras_out2 = []
    for p in palabras2:
        p_clean = p.lower().rstrip("?,.")
        if len(p_clean) >= 5:  # solo palabras largas donde el typo es probable
            sugerencias = get_close_matches(p_clean, _VOCAB_NEGOCIO, n=1, cutoff=0.82)
            if sugerencias and sugerencias[0] != p_clean:
                transformaciones.append(f"'{p}' ≈ '{sugerencias[0]}' (fuzzy)")
                palabras_out2.append(sugerencias[0])
                continue
        palabras_out2.append(p)
    normalizado = " ".join(palabras_out2)

    return normalizado, transformaciones


# ──────────────────────────────────────────────────────────
# 1. OUTPUT PARSER: Extrae única y exclusivamente el SQL
# ──────────────────────────────────────────────────────────
class SQLCleanOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            return sql if sql.endswith(";") else sql + ";"

        match = re.search(r"(SELECT\s+.*?);", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip() + ";"

        return text.strip()


# ──────────────────────────────────────────────────────────
# 2. FEW-SHOT EXAMPLES (10 patrones)
# Cubren: agregaciones, filtros temporales, JOINs dobles y
# triples, vistas, TOP-N, comparación de períodos,
# GROUP BY compuesto y filtros de nulos.
# ──────────────────────────────────────────────────────────
EJEMPLOS_FEW_SHOT = [
    # 1. Conteo simple
    {
        "pregunta": "¿Cuántas reservas hay en total?",
        "sql": "SELECT COUNT(*) AS total_reservas FROM reservas;"
    },
    # 2. Suma de ingresos con filtro temporal
    {
        "pregunta": "¿Cuánto ingresó el recreo esta semana?",
        "sql": (
            "SELECT COALESCE(SUM(CASE WHEN tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') "
            "THEN monto ELSE 0 END), 0) AS ingresos_semana "
            "FROM transacciones_financieras "
            "WHERE fecha_transaccion >= DATE_TRUNC('week', CURRENT_DATE);"
        )
    },
    # 3. Utilidad neta de ayer
    {
        "pregunta": "¿Cuál fue la utilidad neta de ayer?",
        "sql": (
            "SELECT COALESCE(SUM(CASE WHEN tf.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') "
            "THEN tf.monto ELSE 0 END), 0) "
            "- COALESCE(SUM(CASE WHEN tf.tipo = 'gasto_operativo' THEN ABS(tf.monto) ELSE 0 END), 0) "
            "AS utilidad_neta "
            "FROM transacciones_financieras tf "
            "WHERE tf.fecha_transaccion = CURRENT_DATE - 1;"
        )
    },
    # 4. Ticket promedio por canal (JOIN doble)
    {
        "pregunta": "¿Cuál es el ticket promedio por canal?",
        "sql": (
            "SELECT r.canal, ROUND(AVG(ir.subtotal), 2) AS ticket_promedio "
            "FROM reservas r "
            "JOIN items_reserva ir ON ir.reserva_id = r.reserva_id "
            "WHERE r.estado = 'completada' "
            "GROUP BY r.canal "
            "ORDER BY ticket_promedio DESC;"
        )
    },
    # 5. Vista de detalle por temporada
    {
        "pregunta": "¿Cuántas reservas por temporada muestra el detalle este año?",
        "sql": (
            "SELECT vd.temporada, COUNT(*) AS total_reservas "
            "FROM v_reservas_detalle vd "
            "WHERE EXTRACT(YEAR FROM vd.fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE) "
            "GROUP BY vd.temporada;"
        )
    },
    # 6. Top-5 clientes con más reservas (JOIN triple + TOP-N)
    {
        "pregunta": "¿Cuáles son los 5 clientes con más reservas completadas?",
        "sql": (
            "SELECT c.nombre_codigo, COUNT(r.reserva_id) AS total_reservas "
            "FROM clientes c "
            "JOIN reservas r ON r.cliente_id = c.cliente_id "
            "WHERE r.estado = 'completada' "
            "GROUP BY c.cliente_id, c.nombre_codigo "
            "ORDER BY total_reservas DESC "
            "LIMIT 5;"
        )
    },
    # 7. Ticket promedio por segmento (JOIN triple)
    {
        "pregunta": "¿Cuál es el ticket promedio por segmento de cliente?",
        "sql": (
            "SELECT c.segmento, ROUND(AVG(ir.subtotal), 2) AS ticket_promedio "
            "FROM reservas r "
            "JOIN clientes c ON c.cliente_id = r.cliente_id "
            "JOIN items_reserva ir ON ir.reserva_id = r.reserva_id "
            "WHERE r.estado = 'completada' "
            "GROUP BY c.segmento "
            "ORDER BY ticket_promedio DESC;"
        )
    },
    # 8. Reservas sin consumo registrado (LEFT JOIN + IS NULL)
    {
        "pregunta": "¿Cuántas reservas completadas no tienen ítems de consumo registrados?",
        "sql": (
            "SELECT COUNT(*) AS reservas_sin_items "
            "FROM reservas r "
            "LEFT JOIN items_reserva ir ON ir.reserva_id = r.reserva_id "
            "WHERE r.estado = 'completada' AND ir.item_id IS NULL;"
        )
    },
    # 9. Comparación de ingresos mes actual vs. mes anterior
    {
        "pregunta": "¿Cuánto se ingresó este mes comparado con el mes pasado?",
        "sql": (
            "SELECT "
            "COALESCE(SUM(CASE WHEN DATE_TRUNC('month', fecha_transaccion) = DATE_TRUNC('month', CURRENT_DATE) "
            "THEN monto END), 0) AS ingresos_mes_actual, "
            "COALESCE(SUM(CASE WHEN DATE_TRUNC('month', fecha_transaccion) = DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' "
            "THEN monto END), 0) AS ingresos_mes_anterior "
            "FROM transacciones_financieras "
            "WHERE tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional');"
        )
    },
    # 10. GROUP BY compuesto: canal y temporada
    {
        "pregunta": "¿Cuántas reservas hubo por canal y temporada?",
        "sql": (
            "SELECT r.canal, "
            "CASE WHEN r.es_temporada_alta THEN 'verano' ELSE 'invierno' END AS temporada, "
            "COUNT(*) AS total_reservas "
            "FROM reservas r "
            "GROUP BY r.canal, r.es_temporada_alta "
            "ORDER BY r.canal, temporada;"
        )
    },
]

example_prompt = PromptTemplate(
    input_variables=["pregunta", "sql"],
    template="Pregunta: {pregunta}\nSQL:\n```sql\n{sql}\n```"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=EJEMPLOS_FEW_SHOT,
    example_prompt=example_prompt,
    prefix="""Eres un experto Arquitecto de Datos SQL.
Tu única tarea es traducir la pregunta del usuario a una consulta SQL válida para PostgreSQL.
NUNCA devuelvas explicaciones, saludos ni texto conversacional. SOLO devuelve el bloque de código SQL puro.

ESQUEMA DE BASE DE DATOS:
{esquema}""",
    suffix="Pregunta: {input}\nSQL:",
    input_variables=["esquema", "input"]
)


# ──────────────────────────────────────────────────────────
# MOTOR PRINCIPAL
# ──────────────────────────────────────────────────────────
class LLMEngine:
    def __init__(self, db_conexion_externa=None):
        logger.info("Inicializando LLMEngine v3 (Microservicio Text-to-SQL)...")
        if OllamaLLM is None:
            raise RuntimeError("langchain-ollama no está instalado.")

        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11435"),
            temperature=0.0,
            # El prompt real (schema + 10 few-shot + glosario) mide ~2850 tokens.
            # Con num_ctx=2048 el contexto se truncaba silenciosamente y el
            # modelo perdía la pregunta o el schema, cayendo en respuestas
            # incorrectas (ej. COUNT(*) FROM reservas para preguntas de ingresos).
            num_ctx=4096,
            num_thread=6,
            num_predict=150,
            keep_alive="24h"
        )

        self.db = db_conexion_externa
        self.chain = few_shot_prompt | self.llm | SQLCleanOutputParser()
        self.esquema = get_full_context()

    def consultar(self, pregunta: str, historial: list | None = None) -> Dict[str, Any]:
        """
        Ejecuta la cadena y devuelve exclusivamente el query en crudo.

        Pipeline:
        1. Normalización de entrada (typos, sin acentos, abreviaturas)
        2. Detección de prompt injection
        3. Inyección de contexto multi-turno en el prompt
        4. Inferencia LLM → SQLCleanOutputParser

        Args:
            pregunta: Consulta en lenguaje natural del usuario.
            historial: Últimas N interacciones [{pregunta, sql}] para contexto
                       multi-turno.
        """
        # ── Paso 1: Normalización de entrada ──────────────────
        pregunta_norm, transformaciones = _normalizar_pregunta(pregunta)
        if transformaciones:
            logger.info(
                "Normalización aplicada sobre '%s': %s",
                pregunta[:60], "; ".join(transformaciones)
            )

        # Detección de inyección sobre texto normalizado (más efectivo)
        if _detectar_inyeccion(pregunta_norm):
            msg = "La pregunta contiene patrones no permitidos. Formula una consulta sobre reservas o finanzas."
            registrar_inferencia(
                pregunta_nl=pregunta,
                sql_generado="(bloqueado — prompt injection detectado)",
                resultado_ok=False,
                latencia_ms=0.0,
                error_msg=msg,
            )
            return {
                "pregunta": pregunta,
                "sql_generado": "",
                "datos": [],
                "respuesta_nl": msg,
                "latencia_ms": 0.0,
                "exito": False,
                "error": msg,
            }

        t_inicio = time.perf_counter()
        exito = False
        sql_final = ""
        error_msg = ""

        try:
            logger.info("Generando SQL para: '%s'%s",
                        pregunta_norm[:80],
                        f" [norm: {', '.join(transformaciones)}]" if transformaciones else "")

            # Enriquecer el input con contexto de conversación previa (multi-turno)
            if historial:
                ctx_lineas = ["CONTEXTO DE CONVERSACIÓN PREVIA (últimas interacciones):"]
                for h in historial[-3:]:
                    preg = h.get("pregunta", "").strip()
                    sql_prev = h.get("sql", "").strip()
                    if preg and sql_prev:
                        ctx_lineas.append(f"  Pregunta: {preg}")
                        ctx_lineas.append(f"  SQL:      {sql_prev}")
                ctx_lineas.append("")
                ctx_lineas.append(f"PREGUNTA ACTUAL: {pregunta_norm}")
                input_efectivo = "\n".join(ctx_lineas)
            else:
                input_efectivo = pregunta_norm  # texto normalizado al LLM

            sql_final = self.chain.invoke({
                "input": input_efectivo,
                "esquema": self.esquema
            })
            if sql_final:
                exito = True
            else:
                error_msg = "El parser de LangChain no pudo extraer un SQL válido."
        except Exception as e:
            error_msg = f"Error interno: {str(e)}"
            logger.exception(error_msg)

        latencia_ms = (time.perf_counter() - t_inicio) * 1000

        registrar_inferencia(
            pregunta_nl=pregunta,           # original para el log
            sql_generado=sql_final or "(sin SQL)",
            resultado_ok=exito,
            latencia_ms=latencia_ms,
            filas_retorno=0,
            error_msg=error_msg,
        )

        return {
            "pregunta": pregunta,           # original al cliente
            "sql_generado": sql_final,
            "datos": [],
            "respuesta_nl": "SQL generado exitosamente." if exito else error_msg,
            "latencia_ms": round(latencia_ms, 2),
            "exito": exito,
            "error": error_msg,
        }

    def calentar(self) -> None:
        """
        Fuerza la carga del modelo en RAM y el JIT de Ollama con una
        inferencia trivial, sin pasar por el log de auditoría.

        Sin esto, la primera pregunta real de un usuario paga el costo
        de cargar el archivo GGUF (~5 GB) desde disco además de la
        inferencia normal, duplicando o triplicando la latencia percibida.
        """
        t_inicio = time.perf_counter()
        try:
            self.chain.invoke({
                "input": "¿Cuántas reservas hay en total?",
                "esquema": self.esquema,
            })
            logger.info(
                "Warm-up del modelo completado en %.1fs.",
                time.perf_counter() - t_inicio,
            )
        except Exception as e:
            logger.warning("Warm-up del modelo falló (no crítico): %s", e)


_engine_instance: Optional[LLMEngine] = None


def get_engine() -> LLMEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMEngine()
    return _engine_instance
