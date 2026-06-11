from engine.llm_engine import _normalize_sql

sql = "SELECT * FROM v_reservas_detalle WHERE tipo LIKE 'ingreso%';"
print('input repr:', repr(sql))
print('output repr:', repr(_normalize_sql(sql)))
print('contains WHERE:', 'WHERE' in _normalize_sql(sql))
print('normalized:', _normalize_sql(sql))
