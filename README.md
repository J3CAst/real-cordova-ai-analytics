# Sistema de Analítica Conversacional — Recreo Campestre Real Córdova

**Tesis de grado | Ingeniería de Ciencia de Datos e IA — SENATI EEST**
**Autor:** Acosta Cruz, Jean Steven | **Ciclo:** 2026-I

---

## Stack Tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| LLM | Meta LLaMA 3.1 GGUF Q4_K_M | 8B params |
| Inferencia local | Ollama | latest |
| Base de datos | PostgreSQL | 17 |
| Backend | FastAPI + LangChain | 0.111 / 0.2 |
| Frontend | Streamlit | 1.36 |
| Lenguaje | Python | 3.11 |
| Hardware | ASUS Vivobook M1603QA | Ryzen 5 5600H / 16GB |
| Costo total | S/ 0.00 | 100% Open Source |

---

## Setup Paso a Paso

### 1. Prerrequisitos

```bash
# Verificar Python 3.11
python --version

# Verificar PostgreSQL 17
psql --version

# Verificar Ollama
ollama --version
```

### 2. Clonar e instalar dependencias

```bash
git clone https://github.com/[tu-usuario]/real-cordova-ai-analytics.git
cd real-cordova-ai-analytics

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# Instalar dependencias de Node.js (Microservicio de BD)
npm install
```

### 3. Configurar entorno

```bash
cp .env.example .env
# Editar .env con tu password de PostgreSQL
```

### 4. Crear base de datos PostgreSQL

```bash
# Abrir psql como superusuario
psql -U postgres

-- Dentro de psql:
CREATE DATABASE real_cordova_db ENCODING 'UTF8';
\c real_cordova_db
\i database/schema.sql
\q
```

### 5. Descargar modelo LLaMA

```bash
# ~4.7 GB de descarga (solo la primera vez)
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 6. Generar dataset sintético

```bash
python data/generate_dataset.py
# → Genera 15,000 registros en data/output/
```

### 7. Cargar datos en PostgreSQL

```bash
python data/load_to_postgres.py
# → ETL completo + verificación de integridad
```

### 8. Ejecutar el sistema completo

```bash
# Terminal 1: Ollama (debe estar corriendo)
ollama serve

# Terminal 2: Microservicio Node (Inicialización y Auditoría)
npm start

# Terminal 3: FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 4: Streamlit UI
streamlit run ui/app.py --server.port 8501
```

Abrir en el navegador: **http://localhost:8501**

---

## Estructura del Proyecto

```
real-cordova-ai-analytics/
├── database/
│   └── schema.sql              ← DDL PostgreSQL 17 (tablas + índices B-Tree)
├── data/
│   ├── generate_dataset.py     ← Simulación estocástica (15,000 registros)
│   ├── load_to_postgres.py     ← Pipeline ETL (Pandas + SQLAlchemy)
│   └── output/                 ← CSV generados (gitignored)
├── engine/
│   ├── llm_engine.py           ← LangChain + Ollama Text-to-SQL
│   ├── schema_loader.py        ← Schema linking dinámico
│   └── audit_logger.py         ← Módulo de auditoría (logs de inferencia)
├── api/
│   └── main.py                 ← FastAPI endpoints
├── ui/
│   └── app.py                  ← Interfaz Streamlit conversacional
├── tests/
│   └── test_sql_accuracy.py    ← Validación K-Folds + Execution Accuracy
├── init.js                     ← Microservicio Node.js para Base de Datos
├── package.json                ← Dependencias de Node.js
├── Modelfile                   ← Configuración Ollama (SHA256 fijo)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Métricas Objetivo (Post-test)

| Indicador | Pre-test | Objetivo | Métrica |
|---|---|---|---|
| Tiempo conciliación diaria | 150 min | < 15 min | Cronómetro |
| Error en reportes financieros | 15% | < 3% | Δ% error |
| Pérdida de reservas | 30% | < 10% | Tasa pérdida |
| Execution Accuracy Text-to-SQL | — | > 85% | EX benchmark |
| Latencia de inferencia | — | < 3s | ms promedio |

---

## Privacidad y Ética

- Dataset 100% sintético (Faker locale `es_PE`) — Ley N° 29733
- Modelo ejecutado on-premise — los datos nunca salen del equipo local
- SHA256 del modelo fijado en `Modelfile` para reproducibilidad