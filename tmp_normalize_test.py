from engine.llm_engine import _normalize_sql

inputs = [
    "SELECT * FROM v_reservas_detalle WHERE tipo LIKE 'ingreso%';",
    "SELECT * FROM v_reservas_detalle vd WHERE CURRENT_YEAR = 2024;",
    "SELECT * FROM v_reservas_detalle AS v WHERE v.tipo::text LIKE 'ingreso%';",
    "SELECT * FROM v_reservas_detalle v WHERE v.es_temporada_alta = TRUE;",
]

for sql in inputs:
    print('INPUT :', sql)
    print('OUTPUT:', _normalize_sql(sql))
    print('---')
