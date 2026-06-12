import os
from typing import List, Dict, Any
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("db_pool")

from sqlalchemy.engine import URL

# Construimos la URL de conexión de forma segura (codifica automáticamente el password)
dsn_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASS', 'postgres'),
    host=os.getenv('DB_HOST', 'localhost'),
    port=int(os.getenv('DB_PORT', '5432')),
    database=os.getenv('DB_NAME', 'real_cordova_db')
)

# ──────────────────────────────────────────────────────────
# POOL DE CONEXIONES (Vía B - Inyección de Dependencias)
# ──────────────────────────────────────────────────────────
# Mantiene conexiones abiertas para evitar el overhead de 
# TCP handshake en cada consulta de FastAPI a PostgreSQL.
pool_engine = create_engine(
    dsn_url, 
    pool_size=5, 
    max_overflow=10, 
    pool_pre_ping=True
)

def ejecutar_query_readonly(sql: str) -> List[Dict[str, Any]]:
    """
    Toma un SQL puro, verifica que sea de solo lectura (seguridad),
    lo ejecuta mediante el pool de conexiones y retorna diccionarios listos para JSON.
    """
    sql_upper = sql.upper().strip()
    
    # Validación estricta de seguridad contra inyecciones y DDL/DML
    operaciones_prohibidas = [
        "INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", 
        "TRUNCATE ", "GRANT ", "REVOKE ", "COMMIT", "ROLLBACK"
    ]
    if any(op in sql_upper for op in operaciones_prohibidas):
        raise ValueError("Violación de seguridad: El motor intentó ejecutar una operación de escritura o modificación (DML/DDL). Solo se permite SELECT.")

    try:
        with pool_engine.connect() as conn:
            resultado = conn.execute(text(sql))
            # SQLAlchemy mappings a dicts nativos
            return [dict(row) for row in resultado.mappings()]
    except Exception as e:
        logger.error(f"Error en ejecución de base de datos: {e}")
        raise e
