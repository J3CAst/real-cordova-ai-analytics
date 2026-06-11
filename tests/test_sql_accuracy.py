import os
import csv
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import httpx
from sklearn.model_selection import KFold

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
K_FOLDS = 5
OUTPUT_DIR = Path(__file__).parent / "output"

# ──────────────────────────────────────────────────────────
# DATASET DE EVALUACIÓN
# 60 pares (pregunta NL → SQL de referencia)
# Estratificados por categoría y nivel de dificultad.
# Representan los patrones de consulta reales del
# administrador del Recreo Real Córdova.
# ──────────────────────────────────────────────────────────

DATASET_EVALUACION: List[Dict] = [

    # ── CATEGORÍA 1: Agregaciones Simples (10 consultas) ──
    {
        "id": "AGG_01", "categoria": "agregacion_simple", "dificultad": "baja",
        "pregunta": "¿Cuántas reservas hay en total?",
        "sql_referencia": "SELECT COUNT(*) AS total_reservas FROM reservas;"
    },
    {
        "id": "AGG_02", "categoria": "agregacion_simple", "dificultad": "baja",
        "pregunta": "¿Cuántos clientes están registrados?",
        "sql_referencia": "SELECT COUNT(*) AS total_clientes FROM clientes;"
    },
    {
        "id": "AGG_03", "categoria": "agregacion_simple", "dificultad": "baja",
        "pregunta": "¿Cuál es el ingreso total registrado?",
        "sql_referencia": "SELECT COALESCE(SUM(monto), 0) AS ingreso_total FROM transacciones_financieras WHERE monto > 0;"
    },
    {
        "id": "AGG_04", "categoria": "agregacion_simple", "dificultad": "baja",
        "pregunta": "¿Cuántas reservas se completaron?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_completadas FROM reservas WHERE estado = 'completada';"
    },
    {
        "id": "AGG_05", "categoria": "agregacion_simple", "dificultad": "baja",
        "pregunta": "¿Cuántas reservas están canceladas?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_canceladas FROM reservas WHERE estado = 'cancelada';"
    },
    {
        "id": "AGG_06", "categoria": "agregacion_simple", "dificultad": "media",
        "pregunta": "¿Cuál es el promedio de personas por reserva?",
        "sql_referencia": "SELECT ROUND(AVG(numero_personas), 2) AS promedio_personas FROM reservas WHERE estado = 'completada';"
    },
    {
        "id": "AGG_07", "categoria": "agregacion_simple", "dificultad": "media",
        "pregunta": "¿Cuántas reservas llegaron por WhatsApp?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_whatsapp FROM reservas WHERE canal = 'whatsapp';"
    },
    {
        "id": "AGG_08", "categoria": "agregacion_simple", "dificultad": "media",
        "pregunta": "¿Cuántos clientes son recurrentes?",
        "sql_referencia": "SELECT COUNT(*) AS clientes_recurrentes FROM clientes WHERE es_recurrente = TRUE;"
    },
    {
        "id": "AGG_09", "categoria": "agregacion_simple", "dificultad": "media",
        "pregunta": "¿Cuál es el gasto operativo total?",
        "sql_referencia": "SELECT COALESCE(SUM(ABS(monto)), 0) AS gasto_total FROM transacciones_financieras WHERE tipo = 'gasto_operativo';"
    },
    {
        "id": "AGG_10", "categoria": "agregacion_simple", "dificultad": "media",
        "pregunta": "¿Cuántas reservas de temporada alta hay?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_temporada_alta FROM reservas WHERE es_temporada_alta = TRUE;"
    },

    # ── CATEGORÍA 2: Filtros Temporales (12 consultas) ────
    {
        "id": "TMP_01", "categoria": "filtro_temporal", "dificultad": "baja",
        "pregunta": "¿Cuántas reservas hay para hoy?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_hoy FROM reservas WHERE fecha_reserva = CURRENT_DATE;"
    },
    {
        "id": "TMP_02", "categoria": "filtro_temporal", "dificultad": "baja",
        "pregunta": "¿Cuánto ingresó el recreo esta semana?",
        "sql_referencia": "SELECT COALESCE(SUM(monto), 0) AS ingresos_semana FROM transacciones_financieras WHERE tipo LIKE 'ingreso%' AND fecha_transaccion >= DATE_TRUNC('week', CURRENT_DATE);"
    },
    {
        "id": "TMP_03", "categoria": "filtro_temporal", "dificultad": "baja",
        "pregunta": "¿Cuántas reservas hubo este mes?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_mes FROM reservas WHERE DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "TMP_04", "categoria": "filtro_temporal", "dificultad": "media",
        "pregunta": "¿Cuánto se ingresó el mes pasado?",
        "sql_referencia": "SELECT COALESCE(SUM(monto), 0) AS ingresos_mes_pasado FROM transacciones_financieras WHERE tipo LIKE 'ingreso%' AND DATE_TRUNC('month', fecha_transaccion) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month');"
    },
    {
        "id": "TMP_05", "categoria": "filtro_temporal", "dificultad": "media",
        "pregunta": "¿Cuántas reservas hubo el fin de semana pasado?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_fds FROM reservas WHERE EXTRACT(DOW FROM fecha_reserva) IN (0, 6) AND fecha_reserva >= CURRENT_DATE - INTERVAL '7 days';"
    },
    {
        "id": "TMP_06", "categoria": "filtro_temporal", "dificultad": "media",
        "pregunta": "¿Cuántas reservas por WhatsApp llegaron este mes?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_whatsapp_mes FROM reservas WHERE canal = 'whatsapp' AND DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "TMP_07", "categoria": "filtro_temporal", "dificultad": "media",
        "pregunta": "¿Cuánto se gastó en operaciones esta semana?",
        "sql_referencia": "SELECT COALESCE(SUM(ABS(monto)), 0) AS gastos_semana FROM transacciones_financieras WHERE tipo = 'gasto_operativo' AND fecha_transaccion >= DATE_TRUNC('week', CURRENT_DATE);"
    },
    {
        "id": "TMP_08", "categoria": "filtro_temporal", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas hubo en temporada alta este año?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_verano FROM reservas WHERE es_temporada_alta = TRUE AND EXTRACT(YEAR FROM fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE);"
    },
    {
        "id": "TMP_09", "categoria": "filtro_temporal", "dificultad": "alta",
        "pregunta": "¿Cuál fue el ingreso del último trimestre?",
        "sql_referencia": "SELECT COALESCE(SUM(monto), 0) AS ingresos_trimestre FROM transacciones_financieras WHERE tipo LIKE 'ingreso%' AND fecha_transaccion >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months') AND fecha_transaccion < DATE_TRUNC('quarter', CURRENT_DATE);"
    },
    {
        "id": "TMP_10", "categoria": "filtro_temporal", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas se hicieron los sábados este mes?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_sabado FROM reservas WHERE EXTRACT(DOW FROM fecha_reserva) = 6 AND DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "TMP_11", "categoria": "filtro_temporal", "dificultad": "alta",
        "pregunta": "¿Cuánto ingresó el recreo ayer?",
        "sql_referencia": "SELECT COALESCE(SUM(monto), 0) AS ingresos_ayer FROM transacciones_financieras WHERE tipo LIKE 'ingreso%' AND fecha_transaccion = CURRENT_DATE - 1;"
    },
    {
        "id": "TMP_12", "categoria": "filtro_temporal", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas canceladas hubo en los últimos 30 días?",
        "sql_referencia": "SELECT COUNT(*) AS canceladas_30d FROM reservas WHERE estado = 'cancelada' AND fecha_reserva >= CURRENT_DATE - INTERVAL '30 days';"
    },

    # ── CATEGORÍA 3: JOINs entre tablas (10 consultas) ────
    {
        "id": "JOI_01", "categoria": "join", "dificultad": "media",
        "pregunta": "¿Cuántas reservas completadas tienen transacción registrada?",
        "sql_referencia": "SELECT COUNT(DISTINCT r.reserva_id) AS reservas_con_transaccion FROM reservas r INNER JOIN transacciones_financieras t ON t.reserva_id = r.reserva_id WHERE r.estado = 'completada';"
    },
    {
        "id": "JOI_02", "categoria": "join", "dificultad": "media",
        "pregunta": "¿Cuántas reservas tiene el cliente más frecuente?",
        "sql_referencia": "SELECT c.nombre_codigo, COUNT(r.reserva_id) AS total_reservas FROM clientes c JOIN reservas r ON r.cliente_id = c.cliente_id GROUP BY c.cliente_id, c.nombre_codigo ORDER BY total_reservas DESC LIMIT 1;"
    },
    {
        "id": "JOI_03", "categoria": "join", "dificultad": "media",
        "pregunta": "¿Cuánto ingresó en promedio por reserva completada?",
        "sql_referencia": "SELECT ROUND(AVG(t.monto), 2) AS ingreso_promedio_reserva FROM transacciones_financieras t JOIN reservas r ON r.reserva_id = t.reserva_id WHERE t.tipo = 'ingreso_reserva' AND r.estado = 'completada';"
    },
    {
        "id": "JOI_04", "categoria": "join", "dificultad": "media",
        "pregunta": "¿Cuántas reservas por segmento de cliente hay este mes?",
        "sql_referencia": "SELECT c.segmento, COUNT(r.reserva_id) AS total FROM clientes c JOIN reservas r ON r.cliente_id = c.cliente_id WHERE DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE) GROUP BY c.segmento ORDER BY total DESC;"
    },
    {
        "id": "JOI_05", "categoria": "join", "dificultad": "alta",
        "pregunta": "¿Cuál es el ingreso total por canal de reserva este mes?",
        "sql_referencia": "SELECT r.canal, COALESCE(SUM(t.monto), 0) AS ingreso_total FROM reservas r JOIN transacciones_financieras t ON t.reserva_id = r.reserva_id WHERE t.tipo LIKE 'ingreso%' AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE) GROUP BY r.canal ORDER BY ingreso_total DESC;"
    },
    {
        "id": "JOI_06", "categoria": "join", "dificultad": "alta",
        "pregunta": "¿Cuántos clientes recurrentes hicieron reservas este mes?",
        "sql_referencia": "SELECT COUNT(DISTINCT c.cliente_id) AS clientes_recurrentes_activos FROM clientes c JOIN reservas r ON r.cliente_id = c.cliente_id WHERE c.es_recurrente = TRUE AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "JOI_07", "categoria": "join", "dificultad": "alta",
        "pregunta": "¿Cuál es el total de personas por distrito este mes?",
        "sql_referencia": "SELECT c.distrito, SUM(r.numero_personas) AS total_personas FROM clientes c JOIN reservas r ON r.cliente_id = c.cliente_id WHERE r.estado = 'completada' AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE) GROUP BY c.distrito ORDER BY total_personas DESC;"
    },
    {
        "id": "JOI_08", "categoria": "join", "dificultad": "alta",
        "pregunta": "¿Cuánto gastaron los clientes corporativos este mes?",
        "sql_referencia": "SELECT COALESCE(SUM(t.monto), 0) AS gasto_corporativo FROM clientes c JOIN reservas r ON r.cliente_id = c.cliente_id JOIN transacciones_financieras t ON t.reserva_id = r.reserva_id WHERE c.segmento = 'corporativo' AND t.tipo LIKE 'ingreso%' AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "JOI_09", "categoria": "join", "dificultad": "alta",
        "pregunta": "¿Cuál es el ticket promedio por canal de reserva?",
        "sql_referencia": "SELECT r.canal, ROUND(AVG(ir.subtotal), 2) AS ticket_promedio FROM reservas r JOIN items_reserva ir ON ir.reserva_id = r.reserva_id WHERE r.estado = 'completada' GROUP BY r.canal ORDER BY ticket_promedio DESC;"
    },
    {
        "id": "JOI_10", "categoria": "join", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas no show hubo por mes este año?",
        "sql_referencia": "SELECT TO_CHAR(fecha_reserva, 'YYYY-MM') AS mes, COUNT(*) AS no_shows FROM reservas WHERE estado = 'no_show' AND EXTRACT(YEAR FROM fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY mes ORDER BY mes;"
    },

    # ── CATEGORÍA 4: Consultas sobre Vistas (8 consultas) ─
    {
        "id": "VIS_01", "categoria": "vista", "dificultad": "baja",
        "pregunta": "Dame el resumen financiero de hoy.",
        "sql_referencia": "SELECT * FROM v_resumen_diario WHERE fecha = CURRENT_DATE;"
    },
    {
        "id": "VIS_02", "categoria": "vista", "dificultad": "baja",
        "pregunta": "¿Cuál fue la utilidad neta de ayer?",
        "sql_referencia": "SELECT utilidad_neta FROM v_resumen_diario WHERE fecha = CURRENT_DATE - 1;"
    },
    {
        "id": "VIS_03", "categoria": "vista", "dificultad": "media",
        "pregunta": "¿Cuántas reservas completadas hubo esta semana según el detalle?",
        "sql_referencia": "SELECT COUNT(*) AS completadas_semana FROM v_reservas_detalle WHERE estado = 'completada' AND fecha_reserva >= DATE_TRUNC('week', CURRENT_DATE);"
    },
    {
        "id": "VIS_04", "categoria": "vista", "dificultad": "media",
        "pregunta": "¿Cuál es el ticket promedio este mes según el detalle?",
        "sql_referencia": "SELECT ROUND(AVG(ticket_total), 2) AS ticket_promedio FROM v_reservas_detalle WHERE estado = 'completada' AND DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "VIS_05", "categoria": "vista", "dificultad": "media",
        "pregunta": "¿Cuántas reservas por temporada muestra el detalle este año?",
        "sql_referencia": "SELECT temporada, COUNT(*) AS total FROM v_reservas_detalle WHERE EXTRACT(YEAR FROM fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY temporada;"
    },
    {
        "id": "VIS_06", "categoria": "vista", "dificultad": "media",
        "pregunta": "¿Qué día de la semana tiene más reservas según el detalle?",
        "sql_referencia": "SELECT dia_semana_nombre, COUNT(*) AS total FROM v_reservas_detalle GROUP BY dia_semana_nombre, dia_semana_num ORDER BY total DESC LIMIT 1;"
    },
    {
        "id": "VIS_07", "categoria": "vista", "dificultad": "alta",
        "pregunta": "¿Cuál fue el día con mayor utilidad neta este mes?",
        "sql_referencia": "SELECT fecha, utilidad_neta FROM v_resumen_diario WHERE DATE_TRUNC('month', fecha) = DATE_TRUNC('month', CURRENT_DATE) ORDER BY utilidad_neta DESC LIMIT 1;"
    },
    {
        "id": "VIS_08", "categoria": "vista", "dificultad": "alta",
        "pregunta": "¿Cuál es el ingreso promedio diario del mes según el resumen?",
        "sql_referencia": "SELECT ROUND(AVG(total_ingresos), 2) AS ingreso_promedio_diario FROM v_resumen_diario WHERE DATE_TRUNC('month', fecha) = DATE_TRUNC('month', CURRENT_DATE);"
    },

    # ── CATEGORÍA 5: Consultas Compuestas (10 consultas) ──
    {
        "id": "CMP_01", "categoria": "compuesta", "dificultad": "media",
        "pregunta": "¿Cuántas reservas por canal hubo este mes?",
        "sql_referencia": "SELECT canal, COUNT(*) AS total FROM reservas WHERE DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE) GROUP BY canal ORDER BY total DESC;"
    },
    {
        "id": "CMP_02", "categoria": "compuesta", "dificultad": "media",
        "pregunta": "¿Cuántas reservas por estado hay en total?",
        "sql_referencia": "SELECT estado, COUNT(*) AS total FROM reservas GROUP BY estado ORDER BY total DESC;"
    },
    {
        "id": "CMP_03", "categoria": "compuesta", "dificultad": "media",
        "pregunta": "¿Cuál es el ingreso por método de pago este mes?",
        "sql_referencia": "SELECT metodo_pago, COALESCE(SUM(monto), 0) AS total FROM transacciones_financieras WHERE tipo LIKE 'ingreso%' AND DATE_TRUNC('month', fecha_transaccion) = DATE_TRUNC('month', CURRENT_DATE) GROUP BY metodo_pago ORDER BY total DESC;"
    },
    {
        "id": "CMP_04", "categoria": "compuesta", "dificultad": "alta",
        "pregunta": "¿Cuáles son los 5 días con más reservas este año?",
        "sql_referencia": "SELECT fecha_reserva, COUNT(*) AS total FROM reservas WHERE EXTRACT(YEAR FROM fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY fecha_reserva ORDER BY total DESC LIMIT 5;"
    },
    {
        "id": "CMP_05", "categoria": "compuesta", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas por mes hubo este año?",
        "sql_referencia": "SELECT TO_CHAR(fecha_reserva, 'YYYY-MM') AS mes, COUNT(*) AS total FROM reservas WHERE EXTRACT(YEAR FROM fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY mes ORDER BY mes;"
    },
    {
        "id": "CMP_06", "categoria": "compuesta", "dificultad": "alta",
        "pregunta": "¿Cuál es la tasa de cancelación por canal?",
        "sql_referencia": "SELECT canal, ROUND(100.0 * SUM(CASE WHEN estado = 'cancelada' THEN 1 ELSE 0 END) / COUNT(*), 2) AS tasa_cancelacion_pct FROM reservas GROUP BY canal ORDER BY tasa_cancelacion_pct DESC;"
    },
    {
        "id": "CMP_07", "categoria": "compuesta", "dificultad": "alta",
        "pregunta": "¿Cuál es el ingreso mensual promedio por temporada?",
        "sql_referencia": "SELECT r.es_temporada_alta, ROUND(AVG(t.monto), 2) AS ingreso_promedio FROM transacciones_financieras t JOIN reservas r ON r.reserva_id = t.reserva_id WHERE t.tipo LIKE 'ingreso%' GROUP BY r.es_temporada_alta;"
    },
    {
        "id": "CMP_08", "categoria": "compuesta", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas no conciliadas hay?",
        "sql_referencia": "SELECT COUNT(*) AS no_conciliadas FROM transacciones_financieras WHERE es_conciliado = FALSE;"
    },
    {
        "id": "CMP_09", "categoria": "compuesta", "dificultad": "alta",
        "pregunta": "¿Cuál es la categoría de ítem que más ingreso genera?",
        "sql_referencia": "SELECT categoria, ROUND(SUM(subtotal), 2) AS ingreso_total FROM items_reserva GROUP BY categoria ORDER BY ingreso_total DESC LIMIT 1;"
    },
    {
        "id": "CMP_10", "categoria": "compuesta", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas por segmento y temporada hay en total?",
        "sql_referencia": "SELECT c.segmento, CASE WHEN r.es_temporada_alta THEN 'verano' ELSE 'invierno' END AS temporada, COUNT(*) AS total FROM reservas r JOIN clientes c ON c.cliente_id = r.cliente_id GROUP BY c.segmento, r.es_temporada_alta ORDER BY total DESC;"
    },

    # ── CATEGORÍA 6: Dominio Específico (10 consultas) ────
    {
        "id": "DOM_01", "categoria": "dominio", "dificultad": "media",
        "pregunta": "¿Cuántas reservas perdidas (no show) hubo este mes?",
        "sql_referencia": "SELECT COUNT(*) AS no_shows_mes FROM reservas WHERE estado = 'no_show' AND DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "DOM_02", "categoria": "dominio", "dificultad": "media",
        "pregunta": "¿Cuál es la tasa de pérdida de reservas este mes?",
        "sql_referencia": "SELECT ROUND(100.0 * SUM(CASE WHEN estado IN ('cancelada','no_show') THEN 1 ELSE 0 END) / COUNT(*), 2) AS tasa_perdida_pct FROM reservas WHERE DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "DOM_03", "categoria": "dominio", "dificultad": "media",
        "pregunta": "¿Cuántas reservas familiares llegaron por WhatsApp este mes?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_familiar_whatsapp FROM reservas r JOIN clientes c ON c.cliente_id = r.cliente_id WHERE c.segmento = 'familiar' AND r.canal = 'whatsapp' AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "DOM_04", "categoria": "dominio", "dificultad": "media",
        "pregunta": "¿Cuánto fue el cuadre de caja de hoy?",
        "sql_referencia": "SELECT total_ingresos, total_gastos, utilidad_neta FROM v_resumen_diario WHERE fecha = CURRENT_DATE;"
    },
    {
        "id": "DOM_05", "categoria": "dominio", "dificultad": "media",
        "pregunta": "¿Cuántas reservas de grupos vinieron este mes?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_grupos FROM reservas r JOIN clientes c ON c.cliente_id = r.cliente_id WHERE c.segmento = 'grupo' AND DATE_TRUNC('month', r.fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "DOM_06", "categoria": "dominio", "dificultad": "alta",
        "pregunta": "¿Cuál fue el ingreso por Yape y Plin este semana?",
        "sql_referencia": "SELECT metodo_pago, COALESCE(SUM(monto), 0) AS total FROM transacciones_financieras WHERE metodo_pago IN ('yape','plin') AND tipo LIKE 'ingreso%' AND fecha_transaccion >= DATE_TRUNC('week', CURRENT_DATE) GROUP BY metodo_pago;"
    },
    {
        "id": "DOM_07", "categoria": "dominio", "dificultad": "alta",
        "pregunta": "¿Cuántas devoluciones se hicieron este mes?",
        "sql_referencia": "SELECT COUNT(*) AS devoluciones_mes, COALESCE(SUM(ABS(monto)), 0) AS monto_devuelto FROM transacciones_financieras WHERE tipo = 'devolucion' AND DATE_TRUNC('month', fecha_transaccion) = DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "id": "DOM_08", "categoria": "dominio", "dificultad": "alta",
        "pregunta": "¿Cuántas reservas confirmadas hay para la próxima semana?",
        "sql_referencia": "SELECT COUNT(*) AS reservas_proxima_semana FROM reservas WHERE estado IN ('confirmada','pendiente') AND fecha_reserva BETWEEN CURRENT_DATE + 1 AND CURRENT_DATE + 7;"
    },
    {
        "id": "DOM_09", "categoria": "dominio", "dificultad": "alta",
        "pregunta": "¿Cuál es el porcentaje de reservas por canal este año?",
        "sql_referencia": "SELECT canal, COUNT(*) AS total, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS porcentaje FROM reservas WHERE EXTRACT(YEAR FROM fecha_reserva) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY canal ORDER BY total DESC;"
    },
    {
        "id": "DOM_10", "categoria": "dominio", "dificultad": "alta",
        "pregunta": "¿Cuántas personas en total visitaron el recreo este mes?",
        "sql_referencia": "SELECT COALESCE(SUM(numero_personas), 0) AS total_visitantes FROM reservas WHERE estado = 'completada' AND DATE_TRUNC('month', fecha_reserva) = DATE_TRUNC('month', CURRENT_DATE);"
    },
]


# ──────────────────────────────────────────────────────────
# FUNCIÓN DE EVALUACIÓN
# ──────────────────────────────────────────────────────────

RETRYABLE_ERROR_PHRASES = (
    "timed out",
    "timeout",
    "10054",
    "connection reset",
    "connection aborted",
    "connection refused",
    "name or service not known",
    "connection error",
    "temporarily unavailable",
)


def _is_retryable_error(error_text: str) -> bool:
    lower_text = error_text.lower()
    return any(token in lower_text for token in RETRYABLE_ERROR_PHRASES)


def evaluar_consulta(item: Dict, max_retries: int = 1, delay_seconds: float = 3.0) -> Dict:
    """
    Evalúa una consulta individual enviándola al motor LLM
    via la API FastAPI y registrando el resultado.

    Execution Accuracy (EX):
    - 1.0 si la API retorna exito=True (SQL ejecutó sin error)
    - 0.0 si la API retorna exito=False

    Returns
    -------
    dict con todos los campos del item + métricas de evaluación
    """
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.post(
                f"{API_BASE}/consulta",
                json={"pregunta": item["pregunta"]},
                timeout=240.0,
            )
            r.raise_for_status()
            resp = r.json()

            return {
                **item,
                "sql_generado": resp.get("sql_generado", ""),
                "exito": resp.get("exito", False),
                "execution_accuracy": 1.0 if resp.get("exito") else 0.0,
                "latencia_ms": resp.get("latencia_ms", 0),
                "n_filas": len(resp.get("datos", [])),
                "error": resp.get("error", ""),
            }

        except Exception as e:
            error_text = str(e)
            if attempt >= max_retries or not _is_retryable_error(error_text):
                return {
                    **item,
                    "sql_generado": "",
                    "exito": False,
                    "execution_accuracy": 0.0,
                    "latencia_ms": 0.0,
                    "n_filas": 0,
                    "error": error_text,
                }

            print(f"      retry {attempt}/{max_retries} due to error: {error_text}")
            time.sleep(delay_seconds)


# ──────────────────────────────────────────────────────────
# VALIDACIÓN K-FOLDS
# ──────────────────────────────────────────────────────────

def ejecutar_kfolds(dataset: List[Dict], k: int = 5) -> Dict:
    """
    Ejecuta la validación cruzada K-Folds sobre el dataset
    de evaluación completo.

    Proceso:
    1. Divide el dataset en K particiones iguales
    2. En cada fold: evalúa el subset de test (N/K consultas)
    3. Calcula EX, latencia y tasa de error por fold
    4. Agrega estadísticas sobre los K folds

    Parameters
    ----------
    dataset : Lista de pares (pregunta, sql_referencia)
    k       : Número de folds (K=5 según metodología)

    Returns
    -------
    dict con resultados detallados por fold y agregados
    """
    import numpy as np

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))

    resultados_por_fold = []
    todos_resultados = []

    print(f"\n{'='*65}")
    print(f"  VALIDACIÓN CRUZADA K-FOLDS (K={k})")
    print(f"  Dataset: {len(dataset)} consultas | Modelo: LLaMA 3.1 8B")
    print(f"  Objetivo Execution Accuracy: > 85%")
    print(f"{'='*65}\n")

    for fold_num, (_, test_idx) in enumerate(kf.split(indices), start=1):
        fold_subset = [dataset[i] for i in test_idx]
        n_test = len(fold_subset)

        print(f"  Fold {fold_num}/{k} — evaluando {n_test} consultas...")

        fold_resultados = []
        for i, item in enumerate(fold_subset, start=1):
            print(
                f"    [{i:02d}/{n_test}] {item['id']} — {item['pregunta'][:55]}...", end="", flush=True)
            resultado = evaluar_consulta(item)
            fold_resultados.append(resultado)
            todos_resultados.append(resultado)

            estado = "✓" if resultado["exito"] else "✗"
            lat = resultado["latencia_ms"]
            print(f" {estado} ({lat:.0f}ms)")

        # Métricas del fold
        ex_fold = statistics.mean(r["execution_accuracy"] for r in fold_resultados)
        lat_list = [r["latencia_ms"] for r in fold_resultados if r["latencia_ms"] > 0]
        lat_fold = statistics.mean(lat_list) if lat_list else 0.0

        resultados_por_fold.append({
            "fold": fold_num,
            "n_consultas": n_test,
            "execution_accuracy": round(ex_fold, 4),
            "latencia_promedio": round(lat_fold, 2),
            "consultas_ok": sum(1 for r in fold_resultados if r["exito"]),
            "consultas_error": sum(1 for r in fold_resultados if not r["exito"]),
        })

        print(
            f"    → Fold {fold_num} EX: {ex_fold:.1%} | Latencia: {lat_fold:.0f}ms\n")

    # ── Estadísticas agregadas ───────────────────────────
    ex_vals = [f["execution_accuracy"] for f in resultados_por_fold]
    lat_vals = [f["latencia_promedio"] for f in resultados_por_fold]

    ex_media = statistics.mean(ex_vals)
    ex_std = statistics.stdev(ex_vals) if len(ex_vals) > 1 else 0.0
    ex_min = min(ex_vals)
    ex_max = max(ex_vals)

    lat_media = statistics.mean(lat_vals)
    lat_std = statistics.stdev(lat_vals) if len(lat_vals) > 1 else 0.0

    # EX por categoría
    ex_por_categoria = {}
    for cat in set(r["categoria"] for r in todos_resultados):
        items_cat = [r for r in todos_resultados if r["categoria"] == cat]
        ex_cat = statistics.mean(r["execution_accuracy"] for r in items_cat)
        ex_por_categoria[cat] = round(ex_cat, 4)

    # EX por dificultad
    ex_por_dificultad = {}
    for dif in ["baja", "media", "alta"]:
        items_dif = [r for r in todos_resultados if r["dificultad"] == dif]
        if items_dif:
            ex_dif = statistics.mean(r["execution_accuracy"]
                                     for r in items_dif)
            ex_por_dificultad[dif] = round(ex_dif, 4)

    return {
        "k": k,
        "n_total": len(todos_resultados),
        "ex_media": round(ex_media, 4),
        "ex_std": round(ex_std, 4),
        "ex_min": round(ex_min, 4),
        "ex_max": round(ex_max, 4),
        "latencia_media_ms": round(lat_media, 2),
        "latencia_std_ms": round(lat_std, 2),
        "hipotesis_aceptada": ex_media >= 0.85,
        "resultados_por_fold": resultados_por_fold,
        "ex_por_categoria": ex_por_categoria,
        "ex_por_dificultad": ex_por_dificultad,
        "todos_resultados": todos_resultados,
    }


# ──────────────────────────────────────────────────────────
# GENERACIÓN DE REPORTE
# ──────────────────────────────────────────────────────────

def generar_reporte(resultado: Dict) -> str:
    """
    Genera el reporte textual del benchmark listo para
    copiar en el Capítulo IV del informe de tesis.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "─" * 65

    lineas = [
        "=" * 65,
        "  REPORTE DE BENCHMARK — SISTEMA REAL CÓRDOVA",
        "  Tesis SENATI EEST | Acosta Cruz, Jean Steven | 2026",
        f"  Generado: {ts}",
        "=" * 65,
        "",
        "1. PARÁMETROS DE EVALUACIÓN",
        sep,
        f"   Método          : Validación Cruzada K-Folds (K={resultado['k']})",
        f"   Dataset         : {resultado['n_total']} consultas",
        f"   Modelo          : Meta LLaMA 3.1 8B GGUF Q4_K_M (Ollama)",
        f"   Hardware        : ASUS Vivobook M1603QA (Ryzen 5 5600H, 16GB)",
        f"   Métrica         : Execution Accuracy (EX) — Estándar BIRD",
        f"   Hipótesis H₁   : EX > 85%",
        "",
        "2. RESULTADOS POR FOLD",
        sep,
        f"   {'Fold':<6} {'N':<5} {'EX (%)':>8} {'Lat. Prom (ms)':>15} {'OK':>5} {'Error':>7}",
        f"   {'─'*6} {'─'*5} {'─'*8} {'─'*15} {'─'*5} {'─'*7}",
    ]

    for f in resultado["resultados_por_fold"]:
        lineas.append(
            f"   {f['fold']:<6} {f['n_consultas']:<5} "
            f"{f['execution_accuracy']*100:>7.1f}% "
            f"{f['latencia_promedio']:>14.0f} "
            f"{f['consultas_ok']:>5} "
            f"{f['consultas_error']:>7}"
        )

    lineas += [
        f"   {'─'*6} {'─'*5} {'─'*8} {'─'*15} {'─'*5} {'─'*7}",
        f"   {'MEDIA':<6} {resultado['n_total']:<5} "
        f"{resultado['ex_media']*100:>7.1f}% "
        f"{resultado['latencia_media_ms']:>14.0f}",
        f"   {'STD':<6} {'':5} "
        f"{resultado['ex_std']*100:>7.2f}% "
        f"{resultado['latencia_std_ms']:>14.0f}",
        "",
        "3. EXECUTION ACCURACY POR CATEGORÍA",
        sep,
    ]

    cat_labels = {
        "agregacion_simple": "Agregaciones simples",
        "filtro_temporal": "Filtros temporales",
        "join": "JOINs entre tablas",
        "vista": "Consultas sobre vistas",
        "compuesta": "Consultas compuestas",
        "dominio": "Dominio específico",
    }
    for cat, ex in sorted(resultado["ex_por_categoria"].items()):
        label = cat_labels.get(cat, cat)
        barra = "█" * int(ex * 20)
        lineas.append(f"   {label:<25} {ex*100:>6.1f}%  {barra}")

    lineas += [
        "",
        "4. EXECUTION ACCURACY POR DIFICULTAD",
        sep,
    ]
    for dif, ex in resultado["ex_por_dificultad"].items():
        barra = "█" * int(ex * 20)
        lineas.append(f"   {dif.capitalize():<10} {ex*100:>6.1f}%  {barra}")

    lineas += [
        "",
        "5. CONCLUSIÓN ESTADÍSTICA",
        sep,
        f"   EX Media (K=5)  : {resultado['ex_media']*100:.2f}%",
        f"   Desv. Estándar  : ±{resultado['ex_std']*100:.2f}%",
        f"   Intervalo 95%   : [{(resultado['ex_media']-2*resultado['ex_std'])*100:.1f}%, "
        f"{(resultado['ex_media']+2*resultado['ex_std'])*100:.1f}%]",
        f"   Umbral H₁       : 85.00%",
        "",
        f"   DECISIÓN: {'✅ H₁ ACEPTADA — EX > 85%' if resultado['hipotesis_aceptada'] else '❌ H₀ NO RECHAZADA — EX ≤ 85%'}",
        f"   El sistema {'cumple' if resultado['hipotesis_aceptada'] else 'NO cumple'} el criterio de precisión declarado en la metodología.",
        "",
        "=" * 65,
    ]

    return "\n".join(lineas)


# ──────────────────────────────────────────────────────────
# EXPORTAR RESULTADOS
# ──────────────────────────────────────────────────────────

def exportar_resultados(resultado: Dict, reporte: str) -> None:
    """Guarda CSV de resultados individuales y reporte TXT."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV con todos los resultados individuales
    csv_path = OUTPUT_DIR / "resultados_kfolds.csv"
    campos = [
        "id", "categoria", "dificultad", "pregunta",
        "sql_generado", "exito", "execution_accuracy",
        "latencia_ms", "n_filas", "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(resultado["todos_resultados"])

    # Reporte TXT
    txt_path = OUTPUT_DIR / "reporte_benchmark.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(reporte)

    # JSON con estadísticas agregadas (sin todos_resultados para legibilidad)
    json_path = OUTPUT_DIR / "estadisticas_kfolds.json"
    stats = {k: v for k, v in resultado.items() if k != "todos_resultados"}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  📁 Resultados exportados en {OUTPUT_DIR}/")
    print(f"     ✓ {csv_path.name}")
    print(f"     ✓ {txt_path.name}")
    print(f"     ✓ {json_path.name}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  BENCHMARK DE VALIDACIÓN — SISTEMA REAL CÓRDOVA")
    print("  Tesis SENATI EEST | Acosta Cruz, Jean Steven | 2026")
    print("=" * 65)
    print(f"\n  Dataset   : {len(DATASET_EVALUACION)} consultas de prueba")
    print(f"  K-Folds   : K={K_FOLDS}")
    print(f"  Objetivo  : Execution Accuracy > 85%\n")
    print("  Asegúrate de que Ollama y FastAPI estén corriendo:")
    print("  → ollama serve")
    print("  → uvicorn api.main:app --port 8000\n")

    input("  Presiona ENTER para iniciar la evaluación...")

    t_inicio = time.perf_counter()
    resultado = ejecutar_kfolds(DATASET_EVALUACION, K_FOLDS)
    elapsed = time.perf_counter() - t_inicio

    reporte = generar_reporte(resultado)
    print("\n" + reporte)

    exportar_resultados(resultado, reporte)

    print(f"\n  Tiempo total de evaluación: {elapsed/60:.1f} minutos")
    print(f"\n  Siguiente paso:")
    print(f"  → Copiar 'reporte_benchmark.txt' al Capítulo IV del informe")
    print(f"  → Adjuntar 'resultados_kfolds.csv' como Anexo: Instrumento 3\n")


if __name__ == "__main__":
    main()
