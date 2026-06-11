import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
user = os.getenv('DB_USER')
password = os.getenv('DB_PASS')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
dbname = os.getenv('DB_NAME')
print('user', repr(user))
print('password', repr(password))
print('host', repr(host))
print('port', repr(port))
print('dbname', repr(dbname))
dsn = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
print('dsn repr', repr(dsn))
print('dsn', dsn)
engine = create_engine(dsn, pool_pre_ping=True)
with engine.connect() as conn:
    print('sqlalchemy connected')
