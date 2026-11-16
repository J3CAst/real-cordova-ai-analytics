import os
from functools import lru_cache
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

load_dotenv()


def _get_engine():
    dsn = (
        f"postgresql+psycopg2://"
        f"{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASS', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'real_cordova_db')}"
    )
    return create_engine(dsn, pool_pre_ping=True)


# ──────────────────────────────────────────────────────────
# FIX-01: Definición explícita de vistas en el contexto
# En v1 el modelo intentaba usar las vistas pero fallaba
# con "no existe la relación" porque el schema_context
# las listaba con columnas resumidas sin su estructura
# completa. En v2 se definen con todas las columnas
# exactas tal como las devuelve PostgreSQL.
# ──────────────────────────────────────────────────────────

VISTA_DEFINICIONES = {
    "v_resumen_diario": """
-- ════════════════════════════════════════════════════════
-- VISTA: v_resumen_diario
-- Uso: cuadre de caja diario, ingresos/gastos por fecha
-- COLUMNAS EXACTAS:
--   fecha              DATE      → fecha del día
--   total_reservas     BIGINT    → reservas del día
--   reservas_completadas BIGINT  → reservas completadas
--   reservas_canceladas  BIGINT  → reservas canceladas
--   total_ingresos     NUMERIC   → suma de ingresos S/.
--   total_gastos       NUMERIC   → suma de gastos S/.
--   utilidad_neta      NUMERIC   → ingresos - gastos S/.
-- EJEMPLO DE USO:
--   SELECT * FROM v_resumen_diario WHERE fecha = CURRENT_DATE;
--   SELECT utilidad_neta FROM v_resumen_diario WHERE fecha = CURRENT_DATE - 1;
--   SELECT AVG(total_ingresos) FROM v_resumen_diario
--     WHERE DATE_TRUNC('month', fecha) = DATE_TRUNC('month', CURRENT_DATE);
""",
    "v_reservas_detalle": """
-- ════════════════════════════════════════════════════════
-- VISTA: v_reservas_detalle
-- Uso: consultas conversacionales sobre reservas con contexto de cliente
-- COLUMNAS EXACTAS:
--   reserva_id         INTEGER
--   fecha_reserva      DATE
--   fecha_solicitud    DATE
--   cliente            VARCHAR   → nombre del cliente
--   segmento           VARCHAR   → 'familiar','corporativo','pareja','grupo'
--   distrito           VARCHAR
--   canal              VARCHAR   → 'whatsapp','llamada','presencial','web'
--   numero_personas    SMALLINT
--   numero_mesas       SMALLINT
--   estado             VARCHAR
--   es_temporada_alta  BOOLEAN
--   temporada          VARCHAR   → 'verano' o 'invierno'
--   dia_semana_num     FLOAT     → 0=Dom, 1=Lun, ... 6=Sab
--   dia_semana_nombre  TEXT
--   mes_num            FLOAT
--   mes_nombre         TEXT
--   ticket_total       NUMERIC   → suma de consumo de la reserva
-- EJEMPLO DE USO:
--   SELECT segmento, COUNT(*) FROM v_reservas_detalle GROUP BY segmento;
--   SELECT AVG(ticket_total) FROM v_reservas_detalle WHERE estado='completada';
--   SELECT dia_semana_nombre, COUNT(*) FROM v_reservas_detalle
--     GROUP BY dia_semana_nombre, dia_semana_num ORDER BY COUNT(*) DESC;
""",
}

# ──────────────────────────────────────────────────────────
# FIX-03: Glosario ampliado con reglas de JOIN explícitas
# En v1 los JOINs multi-tabla fallaban por columnas
# ambiguas. En v2 se especifican los alias y prefijos
# correctos para cada combinación de tablas frecuente.
# ──────────────────────────────────────────────────────────

ESTADISTICAS_DATOS = """
-- ════════════════════════════════════════════════════════
-- ESTADÍSTICAS DEL DATASET (referencia para el LLM)
-- Generado con 15,000 reservas sintéticas (2024-01-01 a 2026-06-30)
-- ════════════════════════════════════════════════════════
-- clientes          : ~1,200 registros | es_recurrente ~30% TRUE
-- reservas          : ~15,000 registros | 60% es_temporada_alta=TRUE
--   estados         : completada ~55%, cancelada ~20%, confirmada ~15%, no_show ~7%, pendiente ~3%
--   canales         : whatsapp 65%, llamada 20%, presencial 10%, web 5%
--   fecha_reserva   : rango 2024-01-01 → 2026-06-30
-- items_reserva     : ~50,000 registros | avg ~3-4 ítems por reserva completada
--   categorías      : entrada, gastronomia, bebida, instalacion, extra
-- transacciones_financieras: ~20,000 registros
--   tipos ingreso   : ingreso_reserva, ingreso_consumo, ingreso_adicional
--   tipo gasto      : gasto_operativo
--   métodos de pago : efectivo 40%, yape 30%, plin 15%, transferencia 10%, tarjeta 5%
-- TEMPORADA ALTA    : es_temporada_alta=TRUE → diciembre a marzo (verano peruano)
-- TEMPORADA BAJA    : es_temporada_alta=FALSE → junio a setiembre (invierno peruano)
"""

GLOSARIO_NEGOCIO = """
-- ════════════════════════════════════════════════════════
-- GLOSARIO DEL DOMINIO (Recreo Campestre Real Córdova)
-- ════════════════════════════════════════════════════════

-- TÉRMINOS DE NEGOCIO:
-- "reserva confirmada"      → estado IN ('confirmada','completada')
-- "reserva completada"      → estado = 'completada'
-- "reserva cancelada"       → estado = 'cancelada'
-- "no show / perdida"       → estado = 'no_show'
-- "temporada alta / verano" → es_temporada_alta = TRUE
-- "temporada baja/invierno" → es_temporada_alta = FALSE
-- "fin de semana"           → EXTRACT(DOW FROM fecha_reserva) IN (0,6)
-- "ingreso / entrada"       → transacciones_financieras.tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional')
-- "gasto / egreso"          → transacciones_financieras.tipo = 'gasto_operativo'
-- "utilidad / ganancia"     → total_ingresos - total_gastos
-- "ticket promedio"         → AVG(items_reserva.subtotal agrupado por reserva)
-- "canal WhatsApp"          → canal = 'whatsapp'
-- "cliente frecuente"       → clientes.es_recurrente = TRUE
-- "cuadre de caja diario"   → SELECT * FROM v_resumen_diario WHERE fecha = X
-- "reporte de reservas"     → usar v_reservas_detalle

-- REGLAS DE JOIN (FIX-03 — usar siempre alias de tabla):
-- reservas + clientes:
--   FROM reservas r JOIN clientes c ON c.cliente_id = r.cliente_id
-- reservas + transacciones_financieras:
--   FROM reservas r JOIN transacciones_financieras t ON t.reserva_id = r.reserva_id
-- reservas + items_reserva:
--   FROM reservas r JOIN items_reserva ir ON ir.reserva_id = r.reserva_id
-- reservas + clientes + transacciones (triple JOIN):
--   FROM reservas r
--   JOIN clientes c ON c.cliente_id = r.cliente_id
--   JOIN transacciones_financieras t ON t.reserva_id = r.reserva_id
-- NUNCA usar columnas sin prefijo en JOINs multi-tabla.
-- SIEMPRE calificar: r.canal, c.segmento, t.monto, ir.subtotal

-- REGLAS DE GROUP BY:
-- Al usar MAX(), MIN(), SUM(), AVG() con otras columnas:
--   TODAS las columnas del SELECT no agregadas deben estar en GROUP BY.
-- Ejemplo correcto:
--   SELECT c.nombre_codigo, COUNT(*) FROM clientes c
--   JOIN reservas r ON r.cliente_id = c.cliente_id
--   GROUP BY c.cliente_id, c.nombre_codigo   ← incluir todas las cols de SELECT
"""


@lru_cache(maxsize=1)
def get_schema_context() -> str:
    """
    Lee el schema de tablas base desde PostgreSQL e incorpora
    las definiciones explícitas de vistas (FIX-01).
    """
    engine = _get_engine()
    inspector = inspect(engine)

    tablas = [
        "clientes",
        "reservas",
        "items_reserva",
        "transacciones_financieras",
    ]

    lineas = [
        "-- SCHEMA: real_cordova_db — Recreo Campestre Real Córdova",
        "-- Moneda: Soles peruanos (S/.) | Zona horaria: America/Lima",
        "",
    ]

    for tabla in tablas:
        columnas = inspector.get_columns(tabla)
        lineas.append(f"CREATE TABLE {tabla} (")
        for col in columnas:
            tipo = str(col["type"])
            nullable = "" if col["nullable"] else " NOT NULL"
            lineas.append(f"    {col['name']} {tipo}{nullable},")
        lineas[-1] = lineas[-1].rstrip(",")
        lineas.append(");")
        lineas.append("")

    view_names = inspector.get_view_names()
    if view_names:
        lineas.append("-- VISTAS DISPONIBLES en la base de datos:")
        lineas.append(f"--   {', '.join(sorted(view_names))}")
        lineas.append("")
        for view in view_names:
            definition = VISTA_DEFINICIONES.get(view)
            if definition:
                lineas.append(definition)
            else:
                lineas.append(f"-- VISTA detectada en la DB: {view} (sin definición explícita en el prompt)")
                lineas.append("")

    return "\n".join(lineas)


def get_full_context() -> str:
    """Schema completo + estadísticas del dataset + glosario ampliado (v3)."""
    return get_schema_context() + "\n" + ESTADISTICAS_DATOS + "\n" + GLOSARIO_NEGOCIO


if __name__ == "__main__":
    print(get_full_context())
