import re
from engine.llm_engine import _normalize_sql

sql = "SELECT * FROM v_reservas_detalle WHERE tipo LIKE 'ingreso%';"
print('Original:', sql)

pattern = re.compile(r"((?:\b\w+\.)?)tipo\s*::\s*text\s*LIKE\s*'ingreso%'", flags=re.IGNORECASE)
print('find tipo::text LIKE:', pattern.findall(sql))

pattern2 = re.compile(r"((?:\b\w+\.)?)tipo\s*LIKE\s*'ingreso%'", flags=re.IGNORECASE)
print('find tipo LIKE:', pattern2.findall(sql))

res = pattern.sub(r"\1tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional')", sql)
print('after pattern1:', res)
res2 = pattern2.sub(r"\1tipo IN ('ingreso_reserva','ingreso_consumo','ingreso_adicional')", sql)
print('after pattern2:', res2)
print('normalized full:', _normalize_sql(sql))
