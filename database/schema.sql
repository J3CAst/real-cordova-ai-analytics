-- ----------------------------------------------------------
-- 0. LIMPIEZA (Idempotente - seguro para re-ejecutar)
-- ----------------------------------------------------------
DROP TABLE IF EXISTS transacciones_financieras CASCADE;
DROP TABLE IF EXISTS items_reserva CASCADE;
DROP TABLE IF EXISTS reservas CASCADE;
DROP TABLE IF EXISTS clientes CASCADE;
DROP TYPE IF EXISTS estado_reserva_tipo;
DROP TYPE IF EXISTS metodo_pago_tipo;
DROP TYPE IF EXISTS canal_reserva_tipo;
DROP TYPE IF EXISTS tipo_transaccion_tipo;
-- ----------------------------------------------------------
-- 1. TIPOS ENUMERADOS
--    Garantizan integridad de dominio sin necesidad de
--    tablas de lookup adicionales, reduciendo JOINs en
--    consultas Text-to-SQL.
-- ----------------------------------------------------------
CREATE TYPE estado_reserva_tipo AS ENUM (
    'pendiente',
    'confirmada',
    'cancelada',
    'completada',
    'no_show'
);
CREATE TYPE metodo_pago_tipo AS ENUM (
    'efectivo',
    'yape',
    'plin',
    'transferencia',
    'tarjeta'
);
CREATE TYPE canal_reserva_tipo AS ENUM (
    'whatsapp',
    'llamada',
    'presencial',
    'web'
);
CREATE TYPE tipo_transaccion_tipo AS ENUM (
    'ingreso_reserva',
    'ingreso_consumo',
    'ingreso_adicional',
    'gasto_operativo',
    'devolucion'
);
-- ----------------------------------------------------------
-- 2. TABLA: clientes
--    Almacena perfiles sintéticos anonimizados.
--    Principio: Privacidad por Diseño (Ley N° 29733).
--    Nombres generados con Faker - ningún dato real.
-- ----------------------------------------------------------
CREATE TABLE clientes (
    cliente_id SERIAL PRIMARY KEY,
    nombre_codigo VARCHAR(60) NOT NULL,
    -- Ej: "CLI_00001" o nombre Faker
    distrito VARCHAR(50) NOT NULL,
    segmento VARCHAR(20) NOT NULL -- 'familiar','corporativo','pareja','grupo'
    CHECK (
        segmento IN ('familiar', 'corporativo', 'pareja', 'grupo')
    ),
    es_recurrente BOOLEAN NOT NULL DEFAULT FALSE,
    fecha_registro DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE clientes IS 'Perfiles sintéticos anonimizados - sin datos personales reales (Ley N°29733)';
COMMENT ON COLUMN clientes.nombre_codigo IS 'Código o nombre generado por Faker - no corresponde a persona real';
COMMENT ON COLUMN clientes.segmento IS 'Segmento de cliente: familiar, corporativo, pareja, grupo';
-- ----------------------------------------------------------
-- 3. TABLA: reservas
--    Núcleo del sistema. Registra cada intención/confirmación
--    de visita al recreo.
--    λ (tasa de llegada) modelada con dist. Poisson en el
--    script de generación.
-- ----------------------------------------------------------
CREATE TABLE reservas (
    reserva_id SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL REFERENCES clientes(cliente_id) ON DELETE RESTRICT,
    fecha_reserva DATE NOT NULL,
    -- Fecha de la visita
    fecha_solicitud TIMESTAMPTZ NOT NULL,
    -- Cuándo se hizo la reserva
    canal canal_reserva_tipo NOT NULL,
    numero_personas SMALLINT NOT NULL CHECK (
        numero_personas BETWEEN 1 AND 200
    ),
    numero_mesas SMALLINT NOT NULL CHECK (
        numero_mesas BETWEEN 1 AND 40
    ),
    estado estado_reserva_tipo NOT NULL DEFAULT 'pendiente',
    motivo_cancelacion VARCHAR(120),
    -- NULL si no fue cancelada
    hora_llegada TIME,
    -- NULL hasta que llega
    hora_salida TIME,
    -- NULL hasta que sale
    es_temporada_alta BOOLEAN NOT NULL,
    -- TRUE = verano (60%), FALSE = invierno (40%)
    observaciones TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE reservas IS 'Registro de reservas - núcleo transaccional del sistema';
COMMENT ON COLUMN reservas.es_temporada_alta IS 'Estratificación: 60% verano (dic-mar), 40% invierno (jun-sep)';
COMMENT ON COLUMN reservas.numero_personas IS 'Afluencia modelada con distribución Poisson (λ según temporada)';
-- ----------------------------------------------------------
-- 4. TABLA: items_reserva
--    Detalle de consumo por reserva (servicios contratados).
--    Permite calcular ticket promedio y RevPAR adaptado.
-- ----------------------------------------------------------
CREATE TABLE items_reserva (
    item_id SERIAL PRIMARY KEY,
    reserva_id INTEGER NOT NULL REFERENCES reservas(reserva_id) ON DELETE CASCADE,
    descripcion VARCHAR(80) NOT NULL,
    -- 'Ingreso adulto', 'Parrilla', 'Bebida', etc.
    categoria VARCHAR(30) NOT NULL CHECK (
        categoria IN (
            'entrada',
            'gastronomia',
            'bebida',
            'instalacion',
            'extra'
        )
    ),
    cantidad SMALLINT NOT NULL CHECK (cantidad > 0),
    precio_unitario NUMERIC(8, 2) NOT NULL CHECK (precio_unitario >= 0),
    subtotal NUMERIC(10, 2) GENERATED ALWAYS AS (cantidad * precio_unitario) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE items_reserva IS 'Detalle de servicios consumidos por reserva';
COMMENT ON COLUMN items_reserva.subtotal IS 'Columna calculada automáticamente: cantidad × precio_unitario';
-- ----------------------------------------------------------
-- 5. TABLA: transacciones_financieras
--    Control de caja: ingresos y egresos diarios.
--    Corresponde a la variable dependiente "error financiero":
--    objetivo reducir del 15% a menos del 3%.
-- ----------------------------------------------------------
CREATE TABLE transacciones_financieras (
    transaccion_id SERIAL PRIMARY KEY,
    reserva_id INTEGER REFERENCES reservas(reserva_id) ON DELETE
    SET NULL,
        fecha_transaccion DATE NOT NULL,
        tipo tipo_transaccion_tipo NOT NULL,
        concepto VARCHAR(120) NOT NULL,
        monto NUMERIC(10, 2) NOT NULL,
        -- Positivo = ingreso, negativo = egreso
        metodo_pago metodo_pago_tipo NOT NULL,
        es_conciliado BOOLEAN NOT NULL DEFAULT FALSE,
        numero_comprobante VARCHAR(30),
        notas TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE transacciones_financieras IS 'Libro de caja digital - reemplaza cuaderno físico manual';
COMMENT ON COLUMN transacciones_financieras.monto IS 'Monto en S/. Positivo=ingreso, negativo=gasto/devolución';
COMMENT ON COLUMN transacciones_financieras.es_conciliado IS 'Pre-test=FALSE (manual), post-test=TRUE (sistema automatizado)';
-- ----------------------------------------------------------
-- 6. ÍNDICES B-TREE
--    Optimizan los patrones de consulta más frecuentes del
--    sistema Text-to-SQL:
--    - Filtros por fecha (reportes diarios/semanales/mensuales)
--    - JOINs reservas ↔ transacciones
--    - Filtros por estado y canal
-- ----------------------------------------------------------
-- clientes
CREATE INDEX idx_clientes_segmento ON clientes (segmento);
CREATE INDEX idx_clientes_recurrente ON clientes (es_recurrente);
-- reservas
CREATE INDEX idx_reservas_fecha ON reservas (fecha_reserva);
CREATE INDEX idx_reservas_cliente ON reservas (cliente_id);
CREATE INDEX idx_reservas_estado ON reservas (estado);
CREATE INDEX idx_reservas_canal ON reservas (canal);
CREATE INDEX idx_reservas_temporada ON reservas (es_temporada_alta);
CREATE INDEX idx_reservas_solicitud ON reservas (fecha_solicitud DESC);
-- Índice compuesto para reportes de ocupación por período
CREATE INDEX idx_reservas_fecha_estado ON reservas (fecha_reserva, estado);
-- items_reserva
CREATE INDEX idx_items_reserva_id ON items_reserva (reserva_id);
CREATE INDEX idx_items_categoria ON items_reserva (categoria);
-- transacciones_financieras
CREATE INDEX idx_transac_fecha ON transacciones_financieras (fecha_transaccion);
CREATE INDEX idx_transac_reserva ON transacciones_financieras (reserva_id);
CREATE INDEX idx_transac_tipo ON transacciones_financieras (tipo);
CREATE INDEX idx_transac_conciliado ON transacciones_financieras (es_conciliado);
-- Índice compuesto para cuadre de caja diario
CREATE INDEX idx_transac_fecha_tipo ON transacciones_financieras (fecha_transaccion, tipo);
-- ----------------------------------------------------------
-- 7. FUNCIÓN: actualizar updated_at automáticamente
-- ----------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_set_updated_at() RETURNS TRIGGER LANGUAGE plpgsql AS $$ BEGIN NEW.updated_at = NOW();
RETURN NEW;
END;
$$;
CREATE TRIGGER trg_reservas_updated_at BEFORE
UPDATE ON reservas FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();
-- ----------------------------------------------------------
-- 8. VISTA: v_resumen_diario
--    Usada por el motor Text-to-SQL para responder consultas
--    frecuentes sobre ingresos del día sin JOINs manuales.
-- ----------------------------------------------------------
CREATE OR REPLACE VIEW v_resumen_diario AS
SELECT tf.fecha_transaccion AS fecha,
    COUNT(DISTINCT r.reserva_id) AS total_reservas,
    SUM(
        CASE
            WHEN r.estado = 'completada' THEN 1
            ELSE 0
        END
    ) AS reservas_completadas,
    SUM(
        CASE
            WHEN r.estado = 'cancelada' THEN 1
            ELSE 0
        END
    ) AS reservas_canceladas,
    COALESCE(
        SUM(
            CASE
                WHEN tf.tipo LIKE 'ingreso%' THEN tf.monto
                ELSE 0
            END
        ),
        0
    ) AS total_ingresos,
    COALESCE(
        SUM(
            CASE
                WHEN tf.tipo = 'gasto_operativo' THEN ABS(tf.monto)
                ELSE 0
            END
        ),
        0
    ) AS total_gastos,
    COALESCE(
        SUM(
            CASE
                WHEN tf.tipo LIKE 'ingreso%' THEN tf.monto
                ELSE 0
            END
        ),
        0
    ) - COALESCE(
        SUM(
            CASE
                WHEN tf.tipo = 'gasto_operativo' THEN ABS(tf.monto)
                ELSE 0
            END
        ),
        0
    ) AS utilidad_neta
FROM transacciones_financieras tf
    LEFT JOIN reservas r ON r.reserva_id = tf.reserva_id
GROUP BY tf.fecha_transaccion
ORDER BY tf.fecha_transaccion DESC;
COMMENT ON VIEW v_resumen_diario IS 'Vista de cuadre de caja diario - reemplaza conciliación manual de 150 min';
-- ----------------------------------------------------------
-- 9. VISTA: v_reservas_detalle
--    JOIN desnormalizado para consultas conversacionales
--    del tipo: "¿Cuántas reservas hubo este fin de semana?"
-- ----------------------------------------------------------
CREATE OR REPLACE VIEW v_reservas_detalle AS
SELECT r.reserva_id,
    r.fecha_reserva,
    r.fecha_solicitud::DATE AS fecha_solicitud,
    c.nombre_codigo AS cliente,
    c.segmento,
    c.distrito,
    r.canal,
    r.numero_personas,
    r.numero_mesas,
    r.estado,
    r.es_temporada_alta,
    CASE
        WHEN r.es_temporada_alta THEN 'verano'
        ELSE 'invierno'
    END AS temporada,
    EXTRACT(
        DOW
        FROM r.fecha_reserva
    ) AS dia_semana_num,
    -- 0=Dom, 6=Sab
    TO_CHAR(r.fecha_reserva, 'Day') AS dia_semana_nombre,
    EXTRACT(
        MONTH
        FROM r.fecha_reserva
    ) AS mes_num,
    TO_CHAR(r.fecha_reserva, 'Month') AS mes_nombre,
    COALESCE(
        (
            SELECT SUM(ir.subtotal)
            FROM items_reserva ir
            WHERE ir.reserva_id = r.reserva_id
        ),
        0
    ) AS ticket_total
FROM reservas r
    JOIN clientes c ON c.cliente_id = r.cliente_id;
COMMENT ON VIEW v_reservas_detalle IS 'Vista desnormalizada para consultas conversacionales Text-to-SQL';
-- ----------------------------------------------------------
-- 10. TABLA: llm_audit_logs (Métricas de Tesis)
--     Almacena la latencia, accuracy y f1-score de las 
--     traducciones Text-to-SQL del modelo Meta LLaMA 3.1.
-- ----------------------------------------------------------
CREATE TABLE llm_audit_logs (
    log_id SERIAL PRIMARY KEY,
    pregunta_usuario TEXT NOT NULL,
    sql_generado TEXT NOT NULL,
    sql_esperado TEXT,
    execution_accuracy NUMERIC(5,4) NOT NULL DEFAULT 0.0000,
    f1_score NUMERIC(5,4) NOT NULL DEFAULT 0.0000,
    latencia_ms INTEGER NOT NULL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    modelo VARCHAR(50) DEFAULT 'llama3.1:8b',
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE llm_audit_logs IS 'Registro de métricas de validación del LLM para la tesis';

-- ----------------------------------------------------------
-- 11. VERIFICACIÓN FINAL
-- ----------------------------------------------------------
SELECT schemaname,
    tablename,
    pg_size_pretty(
        pg_total_relation_size(schemaname || '.' || tablename)
    ) AS tamaño
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;