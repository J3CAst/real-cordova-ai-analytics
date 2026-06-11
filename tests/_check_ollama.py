import httpx

urls = [
    'http://127.0.0.1:11435/api/tags',
    'http://localhost:11435/api/tags',
    'http://127.0.0.1:11435/',
    'http://localhost:11435/',
]

for url in urls:
    try:
        resp = httpx.get(url, timeout=10.0)
        print(url)
        print('  status:', resp.status_code)
        print('  body:', resp.text[:400])
    except Exception as err:
        print(url)
        print('  error:', repr(err))
