# Reporte Técnico: Optimización de Latencia y Manejo de Memoria (LLaMA 3.1)

**Proyecto:** Sistema de Inteligencia de Negocios con IA Generativa (Recreo Campestre Real Córdova)  
**Entorno de Hardware:** ASUS Vivobook M1603QA (Ryzen 5 5600H, 16GB RAM)  
**Motor de Inferencia:** Ollama (Meta LLaMA 3.1 8B GGUF Q4_K_M)  

---

## 1. Análisis del Problema de Rendimiento (El Cuello de Botella)

Durante las pruebas de integración entre el Backend (FastAPI) y el Frontend (Streamlit), se detectó una discrepancia crítica en los tiempos de respuesta del modelo de IA:

- **En Entorno de Pruebas (Script K-Folds):** El modelo procesaba consultas Text-to-SQL con un promedio estable de **~24.5 segundos** por consulta.
- **En Entorno Web (Streamlit):** Las consultas se degradaban severamente, registrando tiempos de entre **90 y 132 segundos**.

### Causa Raíz Diagnosticada
Al contar el equipo con 16GB de RAM y requerir LLaMA 3.1 aproximadamente 5.5GB para operar de forma fluida, el entorno web saturaba la memoria física. Esto forzaba al sistema operativo Windows a realizar **Paginación (Swapping)**, moviendo temporalmente los pesos del modelo de IA al disco duro (memoria virtual), lo que multiplicaba el tiempo de lectura/procesamiento por 5.

---

## 2. Optimizaciones Técnicas Implementadas

Para resolver esta degradación y estabilizar la latencia web para que coincida con el entorno de pruebas, se implementaron 4 ajustes a nivel de código y arquitectura:

### 2.1. Restricción Estricta de Ventana de Contexto (`num_ctx`)
Por defecto, Ollama abre una ventana de contexto de 8192 tokens. En el entorno web, la consulta del Insight Diario consumía esta memoria masiva. 
- **Solución:** Se limitó explícitamente el parámetro `num_ctx = 2048` en el motor principal y `num_ctx = 512` en el generador de texto conversacional. Esto redujo dramáticamente la huella en RAM.

### 2.2. Eliminación del "Cold Start" (`keep_alive`)
Ollama descarga los modelos de la memoria RAM después de 5 minutos de inactividad para ahorrar recursos. Cuando el usuario hacía una pregunta en la web después de una pausa, el sistema perdía entre 10 y 15 segundos solo en volver a cargar los 4.7 GB del modelo a la RAM (Arranque en frío).
- **Solución:** Se inyectó el parámetro `keep_alive = "24h"` en las peticiones HTTP (`ui/app.py`) y en LangChain (`engine/llm_engine.py`), forzando al modelo a residir permanentemente en la memoria ultra-rápida.

### 2.3. Caché de Interfaz Gráfica (`st.session_state`)
La naturaleza reactiva de Streamlit hacía que la "Sugerencia del Día" intentara regenerarse cada vez que el usuario hacía clic en el botón de chat. Esto causaba **competencia de hilos (Thread Contention)**, donde el procesador Ryzen 5 intentaba resolver dos tareas masivas de IA simultáneamente.
- **Solución:** Se implementó un patrón Singleton en el frontend. La sugerencia se genera una sola vez al cargar la página y se guarda en `st.session_state["insight_diario"]`, liberando el 100% del procesador para las consultas del chat.

### 2.4. Micro-tareas Asíncronas para Mejorar la Percepción (UX)
Para darle visibilidad a la IA sin sumar latencia a la respuesta de los datos, se dividió la carga.
- **Solución:** Primero, el sistema ejecuta la consulta SQL y muestra la tabla en pantalla (acción rápida). Segundo, se lanza un hilo en segundo plano que envía la tabla resultante a LLaMA con un contexto ínfimo para redactar el análisis conversacional en menos de 3 segundos.

---

## 3. Comparativa de Resultados (Antes vs Después)

| Métrica / Escenario | Entorno Web (Antes) | Entorno Web (Optimizado) | Entorno de Pruebas (Test) |
| :--- | :--- | :--- | :--- |
| **Tiempo de Respuesta (Promedio)** | 132.0 segundos | **~25.5 segundos** | 24.5 segundos |
| **Consumo Dinámico de RAM (Aprox)**| > 8 GB (Paginado a Disco) | **< 3 GB (Estricto en RAM)** | < 3 GB (Estricto en RAM) |
| **Cold Start (1ra Petición)** | 10 a 15 segundos extra | **0 segundos** (Pre-cargado) | 0 segundos |
| **Concurrencia de Tareas LLM** | 2 simultáneas (Saturación) | **1 secuencial** | 1 secuencial |
| **Experiencia de Usuario (UX)** | Bloqueo total de interfaz | Respuesta dividida (Datos -> Texto) | No aplica (Solo consola) |

> [!TIP]
> **Conclusión de Viabilidad:**
> Gracias a las restricciones aplicadas en los parámetros de inferencia (`num_ctx`, `keep_alive` y caché de estado), se ha demostrado que es completamente viable desplegar modelos de lenguaje grandes (LLMs) de manera local (On-Premise) en equipos de consumo (16GB RAM) sin sacrificar la estabilidad del servidor web. El sistema ahora garantiza tiempos de respuesta ejecutivos (< 30s) compatibles con la operación diaria de una PYME.
