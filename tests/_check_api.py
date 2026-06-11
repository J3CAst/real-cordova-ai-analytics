import httpx

try:
    r = httpx.post('http://127.0.0.1:8000/consulta', json={'pregunta':'¿Cuántas reservas hay en total?'}, timeout=10.0)
    print('status', r.status_code)
    print(r.text)
except Exception as e:
    print('error', e)
