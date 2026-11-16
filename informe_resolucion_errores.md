# Informe de Incidencias Técnicas y Resolución de Errores

Este documento recopila los principales desafíos técnicos, fallas y cuellos de botella encontrados durante el desarrollo del **Sistema de Inteligencia de Negocios con IA Generativa** para el Recreo Campestre Real Córdova, así como la ingeniería aplicada para resolverlos. Este reporte es ideal para el capítulo de *"Desafíos y Soluciones"* de la tesis.

---

## 1. Problema de Rutas de Importación en Streamlit (ModuleNotFoundError)
**Descripción del Error:**  
Al levantar la interfaz gráfica ejecutando `streamlit run .\ui\app.py`, el sistema arrojaba el error `ModuleNotFoundError: No module named 'database'`.  
**Causa Raíz:**  
Python y Streamlit aíslan el entorno de ejecución a la carpeta donde se encuentra el script (`/ui`). Por lo tanto, no podía encontrar los módulos hermanos (`database/` o `engine/`) ubicados en la raíz del proyecto.  
**Solución Aplicada:**  
Se inyectó código dinámico en las primeras líneas de `app.py` para forzar a Python a reconocer el directorio padre como parte del entorno del sistema:
```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

## 2. Alucinación de Tablas y Columnas Ambiguas (Text-to-SQL)
**Descripción del Error:**  
En las primeras pruebas, el modelo LLaMA 3.1 intentaba inventar tablas que no existían o generaba consultas con errores de ambigüedad como `column reference "reserva_id" is ambiguous` al unir las tablas `reservas` y `transacciones_financieras`.  
**Causa Raíz:**  
El modelo LLM no tenía un entendimiento profundo del negocio ni reglas explícitas de SQL para PostgreSQL (faltaba el uso estricto de alias `r.` o `t.`).  
**Solución Aplicada:**  
Se rediseñó el archivo `engine/schema_loader.py` implementando un **Glosario de Negocio**. Se le indicó explícitamente al LLM qué vistas usar (`v_resumen_diario`), el significado de términos de negocio (ej. "temporada alta = es_temporada_alta TRUE") y la obligación estricta de usar alias al hacer operaciones `JOIN`.

## 3. Caída del Parseo por Texto Conversacional (Error de Ejecución SQL)
**Descripción del Error:**  
Al ejecutar consultas, el sistema fallaba porque el LLM devolvía respuestas como: *"¡Claro! Aquí tienes la consulta: SELECT * FROM..."*. La base de datos intentaba ejecutar ese texto en español, provocando un error fatal (Syntax Error).  
**Causa Raíz:**  
A pesar de indicarle al modelo que solo devolviera SQL, los LLM generativos tienden a ser "amables" y añadir texto conversacional no deseado.  
**Solución Aplicada:**  
Se desarrolló una clase personalizada `SQLCleanOutputParser` utilizando expresiones regulares (Regex). Esta clase intercepta la respuesta de la IA antes de enviarla a la base de datos, extrae única y exclusivamente el bloque delimitado por ` ```sql ... ``` ` y desecha cualquier saludo o explicación extra.

## 4. Saturación de Memoria RAM (Latencia Extrema >130s)
**Descripción del Error:**  
Mientras el test aislado (K-Folds) demoraba ~24 segundos por consulta, en la web el tiempo de respuesta subió a 132 segundos.  
**Causa Raíz:**  
El componente web realizaba peticiones sin restricciones de memoria a Ollama (usando el máximo de 8192 tokens de contexto). Esto desbordaba los 16GB de RAM de la laptop ASUS Vivobook, forzando al sistema a realizar **Paginación (Swapping)** hacia el disco duro. Adicionalmente, se generaba una colisión de hilos de CPU al intentar renderizar la respuesta y el *Insight* al mismo tiempo.  
**Solución Aplicada:**  
1. Se limitó el contexto de la IA mediante el parámetro `num_ctx = 2048`.
2. Se implementó una caché en memoria usando `st.session_state` para que el *Insight Diario* se calculara solo una vez al abrir la página, dejando la CPU libre para las consultas del chat.

## 5. El Problema del "Arranque en Frío" (Cold Start)
**Descripción del Error:**  
Si el usuario dejaba la pestaña web abierta y hacía una pregunta después de 5 minutos, la IA demoraba 10 segundos extra antes de empezar a procesar.  
**Causa Raíz:**  
Por diseño, Ollama "suelta" los 4.7 GB del modelo de la memoria RAM tras 5 minutos de inactividad para ahorrar recursos. Volver a subirlos desde el SSD a la RAM toma un tiempo valioso.  
**Solución Aplicada:**  
Se inyectó el parámetro `keep_alive = "24h"` en todas las peticiones (tanto de LangChain como HTTP requests). Esto le da la orden a Ollama de fijar (lock) el modelo permanentemente en la memoria ultra-rápida durante todo el turno laboral, llevando el tiempo de "Arranque en Frío" a cero.
