import os
import sys
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "real_cordova_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

DATA_DIR = os.path.join(os.path.dirname(__file__), "output")

# Batch size para inserciones masivas (optimizado para 16GB RAM)
BATCH_SIZE = 1_000


# ──────────────────────────────────────────────────────────
# CONEXIÓN
# ──────────────────────────────────────────────────────────

def crear_engine():
    """
    Crea el engine SQLAlchemy con pool de conexiones
    optimizado para un solo proceso local.
    """
    dsn = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(
        dsn,
        pool_size=2,
        max_overflow=0,
        echo=False,                # True para debug SQL
        connect_args={"options": "-c timezone=America/Lima"}
    )
    # Prueba de conexión
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
        print(f"  Conectado: {version.split(',')[0]}")
    return engine


# ──────────────────────────────────────────────────────────
# TRANSFORMACIONES (T del ETL)
# ──────────────────────────────────────────────────────────

def transformar_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos para la tabla clientes."""
    df = df.copy()
    df["cliente_id"] = df["cliente_id"].astype(int)
    df["nombre_codigo"] = df["nombre_codigo"].str.strip().str[:60]
    df["distrito"] = df["distrito"].str.strip().str[:50]
    df["segmento"] = df["segmento"].str.strip()
    df["es_recurrente"] = df["es_recurrente"].astype(bool)
    df["fecha_registro"] = pd.to_datetime(df["fecha_registro"]).dt.date

    # Validar dominio del enum segmento
    dominios = {"familiar", "corporativo", "pareja", "grupo"}
    invalidos = set(df["segmento"].unique()) - dominios
    if invalidos:
        raise ValueError(f"Segmentos inválidos detectados: {invalidos}")

    return df


def transformar_reservas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos para la tabla reservas."""
    df = df.copy()
    df["reserva_id"] = df["reserva_id"].astype(int)
    df["cliente_id"] = df["cliente_id"].astype(int)
    df["fecha_reserva"] = pd.to_datetime(df["fecha_reserva"]).dt.date
    df["fecha_solicitud"] = pd.to_datetime(df["fecha_solicitud"])
    df["numero_personas"] = df["numero_personas"].astype(int).clip(1, 200)
    df["numero_mesas"] = df["numero_mesas"].astype(int).clip(1, 40)
    df["estado"] = df["estado"].str.strip()
    df["es_temporada_alta"] = df["es_temporada_alta"].astype(bool)

    # Nulos permitidos
    df["motivo_cancelacion"] = df["motivo_cancelacion"].where(
        df["motivo_cancelacion"].notna(), None
    )
    df["hora_llegada"] = df["hora_llegada"].where(
        df["hora_llegada"].notna(), None
    )
    df["hora_salida"] = df["hora_salida"].where(
        df["hora_salida"].notna(), None
    )
    df["observaciones"] = None

    # Validar enums
    estados_validos = {"pendiente", "confirmada",
                       "cancelada", "completada", "no_show"}
    canales_validos = {"whatsapp", "llamada", "presencial", "web"}
    assert set(df["estado"].unique()).issubset(estados_validos)
    assert set(df["canal"].unique()).issubset(canales_validos)

    return df


def transformar_items(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos para la tabla items_reserva."""
    df = df.copy()
    df["item_id"] = df["item_id"].astype(int)
    df["reserva_id"] = df["reserva_id"].astype(int)
    df["descripcion"] = df["descripcion"].str.strip().str[:80]
    df["categoria"] = df["categoria"].str.strip()
    df["cantidad"] = df["cantidad"].astype(int).clip(1, 9999)
    df["precio_unitario"] = df["precio_unitario"].astype(float).round(2)
    df["precio_unitario"] = df["precio_unitario"].clip(lower=0.0)

    # Eliminar la columna subtotal si existe (es GENERATED ALWAYS en PG)
    if "subtotal" in df.columns:
        df = df.drop(columns=["subtotal"])

    categorias_validas = {"entrada", "gastronomia",
                          "bebida", "instalacion", "extra"}
    assert set(df["categoria"].unique()).issubset(categorias_validas)

    return df


def transformar_transacciones(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos para la tabla transacciones_financieras."""
    df = df.copy()
    df["transaccion_id"] = df["transaccion_id"].astype(int)
    df["fecha_transaccion"] = pd.to_datetime(df["fecha_transaccion"]).dt.date
    df["concepto"] = df["concepto"].str.strip().str[:120]
    df["monto"] = df["monto"].astype(float).round(2)
    df["es_conciliado"] = df["es_conciliado"].astype(bool)

    # FK nullable
    df["reserva_id"] = pd.to_numeric(
        df["reserva_id"], errors="coerce"
    ).where(df["reserva_id"].notna(), None)

    # Nulos opcionales
    df["numero_comprobante"] = df["numero_comprobante"].where(
        df["numero_comprobante"].notna(), None
    )
    df["notas"] = df["notas"].where(df["notas"].notna(), None)

    tipos_validos = {
        "ingreso_reserva", "ingreso_consumo", "ingreso_adicional",
        "gasto_operativo", "devolucion"
    }
    metodos_validos = {"efectivo", "yape", "plin", "transferencia", "tarjeta"}
    assert set(df["tipo"].unique()).issubset(tipos_validos)
    assert set(df["metodo_pago"].unique()).issubset(metodos_validos)

    return df


# ──────────────────────────────────────────────────────────
# CARGA (L del ETL)
# ──────────────────────────────────────────────────────────

def cargar_tabla(
    engine,
    df: pd.DataFrame,
    tabla: str,
    columnas_excluir: list = None
) -> None:
    """
    Carga un DataFrame en PostgreSQL usando to_sql con
    chunksize para eficiencia de memoria en 16GB RAM.
    Usa 'append' con los IDs explícitos del CSV.
    """
    if columnas_excluir:
        df = df.drop(columns=columnas_excluir, errors="ignore")

    n_filas = len(df)
    print(f"  Cargando {tabla:<35} {n_filas:>7,} filas...", end="", flush=True)
    t0 = time.perf_counter()

    df.to_sql(
        name=tabla,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=BATCH_SIZE,
        method="multi",    # INSERT múltiple, eficiente sin COPY especial
    )

    elapsed = time.perf_counter() - t0
    print(f" ✓  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────
# SECUENCIA DE RESET (por si se re-ejecuta)
# ──────────────────────────────────────────────────────────

def limpiar_tablas(engine) -> None:
    """
    Trunca las tablas en orden inverso de FK para permitir
    re-ejecución idempotente del pipeline.
    """
    print("\n  Limpiando tablas existentes (TRUNCATE CASCADE)...")
    with engine.begin() as conn:
        conn.execute(text(
            "TRUNCATE TABLE transacciones_financieras, "
            "items_reserva, reservas, clientes RESTART IDENTITY CASCADE;"
        ))
    print("  Tablas truncadas. ✓")


# ──────────────────────────────────────────────────────────
# VERIFICACIÓN POST-CARGA
# ──────────────────────────────────────────────────────────

def verificar_carga(engine) -> None:
    """
    Ejecuta conteos y validaciones en PG para confirmar
    que la carga fue íntegra.
    """
    print("\n── VERIFICACIÓN POST-CARGA ─────────────────────────")
    queries = {
        "clientes": "SELECT COUNT(*) FROM clientes",
        "reservas": "SELECT COUNT(*) FROM reservas",
        "items_reserva": "SELECT COUNT(*) FROM items_reserva",
        "transacciones_financieras": "SELECT COUNT(*) FROM transacciones_financieras",
        "vista v_resumen_diario": "SELECT COUNT(*) FROM v_resumen_diario",
        "vista v_reservas_detalle": "SELECT COUNT(*) FROM v_reservas_detalle",
    }

    totales = {}
    with engine.connect() as conn:
        for nombre, sql in queries.items():
            try:
                count = conn.execute(text(sql)).scalar()
            except SQLAlchemyError as e:
                err = str(e.__cause__ or e).lower()
                if any(substr in err for substr in [
                    'undefined table',
                    'undefined relation',
                    'does not exist',
                    'no existe la relación',
                    'no existe la tabla',
                    'relation "v_resumen_diario" does not exist',
                    'relation "v_reservas_detalle" does not exist'
                ]):
                    conn.rollback()
                    print(f"  {nombre:<35}: {'NO EXISTE':>8}")
                    totales[nombre] = None
                    continue
                raise
            totales[nombre] = count
            print(f"  {nombre:<35}: {count:>8,} registros")

        if totales.get("vista v_resumen_diario") is None or totales.get("vista v_reservas_detalle") is None:
            print("\n  Nota: algunas vistas no existen. Ejecuta database/schema.sql si deseas crearlas.")

        # Verificar integridad: ingresos totales
        total_ing = conn.execute(text(
            "SELECT COALESCE(SUM(monto),0) FROM transacciones_financieras WHERE monto > 0"
        )).scalar()
        total_gas = conn.execute(text(
            "SELECT COALESCE(ABS(SUM(monto)),0) FROM transacciones_financieras WHERE monto < 0"
        )).scalar()
        print(
            f"\n  Ingresos totales en DB     : S/ {float(total_ing):>12,.2f}")
        print(f"  Egresos totales en DB      : S/ {float(total_gas):>12,.2f}")
        print(
            f"  Utilidad neta simulada     : S/ {float(total_ing - total_gas):>12,.2f}")

    assert totales["reservas"] == 15_000, \
        f"Se esperaban 15,000 reservas, hay {totales['reservas']}"
    print("\n  ✅ Carga verificada. Base de datos lista para el motor LLM.")


# ──────────────────────────────────────────────────────────
# ACTUALIZAR SECUENCIAS (importante después de carga masiva
# con IDs explícitos)
# ──────────────────────────────────────────────────────────

def actualizar_secuencias(engine) -> None:
    """
    Sincroniza las secuencias SERIAL de PostgreSQL con el
    máximo ID cargado, para evitar conflictos en inserciones
    futuras desde la aplicación.
    """
    print("\n  Actualizando secuencias SERIAL...")
    sqls = [
        "SELECT setval('clientes_cliente_id_seq', MAX(cliente_id)) FROM clientes",
        "SELECT setval('reservas_reserva_id_seq', MAX(reserva_id)) FROM reservas",
        "SELECT setval('items_reserva_item_id_seq', MAX(item_id)) FROM items_reserva",
        "SELECT setval('transacciones_financieras_transaccion_id_seq', MAX(transaccion_id)) FROM transacciones_financieras",
    ]
    with engine.begin() as conn:
        for sql in sqls:
            conn.execute(text(sql))
    print("  Secuencias actualizadas. ✓")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  PIPELINE ETL — CARGA A POSTGRESQL 17")
    print("  Tesis SENATI EEST | Acosta Cruz, Jean Steven")
    print("=" * 60)

    # 1. Conexión
    print(f"\n  Conectando a {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    try:
        engine = crear_engine()
    except Exception as e:
        print(f"\n  ❌ Error de conexión: {e}")
        print("  Verifica que PostgreSQL esté corriendo y que .env sea correcto.")
        sys.exit(1)

    # 2. Limpiar tablas (idempotente)
    limpiar_tablas(engine)

    # 3. EXTRACCIÓN — leer CSV
    print("\n── EXTRACCIÓN (lectura de CSV) ──────────────────────")
    archivos = {
        "clientes": "clientes.csv",
        "reservas": "reservas.csv",
        "items_reserva": "items_reserva.csv",
        "transacciones_financieras": "transacciones_financieras.csv",
    }

    dataframes = {}
    for tabla, archivo in archivos.items():
        ruta = os.path.join(DATA_DIR, archivo)
        if not os.path.exists(ruta):
            print(f"  ❌ No se encontró: {ruta}")
            print("  Ejecuta primero: python data/generate_dataset.py")
            sys.exit(1)
        df = pd.read_csv(ruta, low_memory=False)
        print(f"  ✓ {archivo:<40} {len(df):>7,} filas leídas")
        dataframes[tabla] = df

    # 4. TRANSFORMACIÓN
    print("\n── TRANSFORMACIÓN (normalización de tipos) ──────────")
    try:
        dataframes["clientes"] = transformar_clientes(dataframes["clientes"])
        dataframes["reservas"] = transformar_reservas(dataframes["reservas"])
        dataframes["items_reserva"] = transformar_items(
            dataframes["items_reserva"])
        dataframes["transacciones_financieras"] = transformar_transacciones(
            dataframes["transacciones_financieras"])
        print("  Todas las transformaciones completadas. ✓")
    except (AssertionError, ValueError) as e:
        print(f"  ❌ Error de validación: {e}")
        sys.exit(1)

    # 5. CARGA — respetar orden de FK
    print("\n── CARGA (INSERT a PostgreSQL 17) ───────────────────")
    t_total = time.perf_counter()

    # Orden obligatorio: clientes → reservas → items → transacciones
    orden = [
        ("clientes",                  []),
        ("reservas",                  []),
        ("items_reserva",             []),
        ("transacciones_financieras", []),
    ]

    try:
        for tabla, excluir in orden:
            cargar_tabla(engine, dataframes[tabla], tabla, excluir)
    except SQLAlchemyError as e:
        print(f"\n  ❌ Error en carga: {e}")
        sys.exit(1)

    elapsed_total = time.perf_counter() - t_total
    print(f"\n  Carga total completada en {elapsed_total:.1f}s")

    # 6. Actualizar secuencias
    actualizar_secuencias(engine)

    # 7. Verificación
    verificar_carga(engine)

    print("\n" + "=" * 60)
    print("  Pipeline ETL finalizado exitosamente.")
    print("  Siguiente paso:")
    print("  → python engine/llm_engine.py (test de inferencia)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
