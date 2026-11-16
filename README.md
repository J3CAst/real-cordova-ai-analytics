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
git clone https://github.com/J3CAst/real-cordova-ai-analytics.git
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
| Execution Accuracy Text-to-SQL | — | > 85% | EX benchmark (K-Folds, K=5) |
| Semantic Accuracy (Answer Accuracy) | — | — | SA benchmark (comparación de result sets) |
| Latencia de inferencia | — | < 3s | ms promedio |

---

## Estructura del Proyecto (v1.2)

```
real-cordova-ai-analytics/
├── database/
│   └── schema.sql              ← DDL PostgreSQL 17 (tablas + índices B-Tree)
├── data/
│   ├── generate_dataset.py     ← Simulación estocástica (15,000 registros)
│   ├── load_to_postgres.py     ← Pipeline ETL (Pandas + SQLAlchemy)
│   └── output/                 ← CSV generados (gitignored)
├── engine/
│   ├── llm_engine.py           ← LangChain + Ollama Text-to-SQL (10 few-shot examples)
│   ├── schema_loader.py        ← Schema linking dinámico (LRU caché)
│   ├── prediccion.py           ← RandomForest 5-features + persistencia joblib
│   └── audit_logger.py         ← Módulo de auditoría (logs de inferencia)
├── api/
│   └── main.py                 ← FastAPI endpoints (+ /admin/refresh-schema)
├── models/
│   └── predictor_demanda.joblib  ← Modelo ML serializado (generado en runtime)
├── ui/
│   └── app.py                  ← Interfaz Streamlit conversacional (5 tabs)
├── tests/
│   └── test_sql_accuracy.py    ← K-Folds + EX clásica + EX semántica (SA)
├── init.js                     ← Microservicio Node.js para Base de Datos
├── package.json                ← Dependencias de Node.js
├── Modelfile                   ← Configuración Ollama (SHA256 fijo)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Privacidad y Ética

- Dataset 100% sintético (Faker locale `es_PE`) — Ley N° 29733
- Modelo ejecutado on-premise — los datos nunca salen del equipo local
- SHA256 del modelo fijado en `Modelfile` para reproducibilidad

---

## Registro de Mejoras (v1.2.0)

### Correcciones de bugs
| Módulo | Problema | Solución |
|---|---|---|
| `engine/prediccion.py:95` | `mean_squared_error(squared=False)` deprecado en scikit-learn ≥ 1.5 | Migrado a `root_mean_squared_error()` |
| `database/db_pool.py:47` | `"WITH "` en blacklist bloqueaba CTEs válidos generados por el LLM | Eliminado; protección real: `statement_timeout=10s` + blacklist DDL/DML |
| `ui/app.py:769` | `pred.get("is_fin_de_semana")` (clave inexistente) silenciaba el badge | Corregido a `pred.get("es_fin_de_semana", False)` |

### Modelo ML (`engine/prediccion.py`)
- **+1 feature `es_fin_de_semana`**: vector pasa de 4 a 5 dimensiones, captura el patrón diferencial fin de semana vs. días hábiles
- **Persistencia joblib** (`models/predictor_demanda.joblib`): el lifespan de FastAPI carga el modelo desde disco en <1s en lugar de re-entrenar ~15s en cada reinicio

### Seguridad y arquitectura (`api/main.py`)
- **`hmac.compare_digest()`**: comparación de API Key resistente a timing attacks
- **`POST /admin/refresh-schema`**: invalida el LRU caché y re-introspecciona el schema PostgreSQL sin reiniciar el servidor

### Integridad del benchmark (`tests/test_sql_accuracy.py`)
- **EX Semántica / Answer Accuracy**: ejecuta el SQL de referencia directamente en PostgreSQL y compara los result sets contra los del SQL generado. Reporte muestra columnas EX + SA por fold y por categoría.
- **Corrección de data leakage**: 5 preguntas del `DATASET_EVALUACION` coincidían exactamente con los `EJEMPLOS_FEW_SHOT` del prompt (el LLM los recibe en contexto en cada inferencia). Sustituidas por preguntas de misma categoría/dificultad pero sin solapamiento — el EX% ahora mide generalización real.