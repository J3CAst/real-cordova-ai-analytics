import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
print('cwd', os.getcwd())
for key in ['DB_USER','DB_PASS','DB_HOST','DB_PORT','DB_NAME','OLLAMA_BASE_URL','OLLAMA_MODEL']:
    print(key, repr(os.getenv(key)))

print('CONNECTING')
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASS'),
)
print('CONNECTED')
conn.close()
