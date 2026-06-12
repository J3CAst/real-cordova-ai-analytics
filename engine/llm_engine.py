import os
import re
import time
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

ollama_module = None
try:
    import importlib
    ollama_module = importlib.import_module("langchain_ollama")
except ImportError:
    pass

OllamaLLM = None
if ollama_module is not None:
    OllamaLLM = getattr(ollama_module, "OllamaLLM", getattr(ollama_module, "Ollama", None))

from engine.schema_loader import get_full_context
from engine.audit_logger import registrar_inferencia, logger

load_dotenv()

# ──────────────────────────────────────────────────────────
# 1. OUTPUT PARSER: Extrae única y exclusivamente el SQL
# ──────────────────────────────────────────────────────────
class SQLCleanOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Extraer bloques de código Markdown si el modelo los genera
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            return sql if sql.endswith(";") else sql + ";"
        
        # Fallback: Extraer desde el SELECT hasta el punto y coma
        match = re.search(r"(SELECT\s+.*?);", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip() + ";"
            
        return text.strip()


# ──────────────────────────────────────────────────────────
# 2. PREPARACIÓN DEL FEW-SHOT
# ──────────────────────────────────────────────────────────
EJEMPLOS_FEW_SHOT = [
    {
        "pregunta": "¿Cuántas reservas hay en total?",
        "sql": "SELECT COUNT(*) AS total_reservas FROM reservas;"
    },
    {
        "pregunta": "¿Cuánto ingresó el recreo esta semana?",
        "sql": "SELECT COALESCE(SUM(CASE WHEN tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN monto ELSE 0 END), 0) AS ingresos_semana FROM transacciones_financieras WHERE fecha_transaccion >= DATE_TRUNC('week', CURRENT_DATE);"
    },
    {
        "pregunta": "¿Cuál fue la utilidad neta de ayer?",
        "sql": "SELECT COALESCE(SUM(CASE WHEN tf.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional') THEN tf.monto ELSE 0 END), 0) - COALESCE(SUM(CASE WHEN tf.tipo = 'gasto_operativo' THEN ABS(tf.monto) ELSE 0 END), 0) AS utilidad_neta FROM transacciones_financieras tf WHERE tf.fecha_transaccion = CURRENT_DATE - 1;"
    },
    {
        "pregunta": "¿Cuál es el ticket promedio por canal?",
        "sql": "SELECT r.canal, ROUND(AVG(ir.subtotal), 2) AS ticket_promedio FROM reservas r JOIN items_reserva ir ON ir.reserva_id = r.reserva_id WHERE r.estado = 'completada' GROUP BY r.canal ORDER BY ticket_promedio DESC;"
    },
    {
        "pregunta": "¿Cuántas reservas por temporada muestra el detalle este año?",
        "sql": "SELECT vd.temporada, COUNT(*) AS total_reservas FROM v_reservas_detalle vd WHERE EXTRACT(YEAR FROM vd.fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY vd.temporada;"
    }
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
        # 1. Ajuste de Hardware en Ollama (Regla 1)
        if OllamaLLM is None:
            raise RuntimeError("langchain-ollama no está instalado.")

        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11435"),
            temperature=0.0,
            num_ctx=2048,       # Límite estricto de contexto
            num_thread=6,       # Procesamiento optimizado
            num_predict=150,    # Previene respuestas verbosas y ahorra RAM
            keep_alive="24h"    # Mantiene el modelo en memoria permanente
        )
        
        # 2. Modularidad de Base de Datos (Regla 2)
        # El backend ya NO inicializa la BD. Recibe un inyector externo o asume que 
        # otro microservicio orquesta la ejecución.
        self.db = db_conexion_externa 
        
        # Ensamblaje de la cadena LangChain (Regla 3)
        self.chain = few_shot_prompt | self.llm | SQLCleanOutputParser()
        self.esquema = get_full_context()

    def consultar(self, pregunta: str) -> Dict[str, Any]:
        """
        Ejecuta la cadena y devuelve exclusivamente el query en crudo.
        Ya no ejecuta el query en la BD directamente.
        """
        t_inicio = time.perf_counter()
        exito = False
        sql_final = ""
        error_msg = ""
        
        try:
            logger.info(f"Generando SQL para: '{pregunta[:80]}'")
            sql_final = self.chain.invoke({
                "input": pregunta,
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
        
        # Opcional: seguimos loggeando la inferencia
        registrar_inferencia(
            pregunta_nl = pregunta,
            sql_generado = sql_final or "(sin SQL)",
            resultado_ok = exito,
            latencia_ms = latencia_ms,
            filas_retorno = 0, # Ya no tenemos filas
            error_msg = error_msg,
        )

        return {
            "pregunta": pregunta,
            "sql_generado": sql_final,
            "datos": [], # Siempre vacío por diseño
            "respuesta_nl": "SQL generado exitosamente." if exito else error_msg,
            "latencia_ms": round(latencia_ms, 2),
            "exito": exito,
            "error": error_msg,
        }

_engine_instance: Optional[LLMEngine] = None

def get_engine() -> LLMEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMEngine()
    return _engine_instance