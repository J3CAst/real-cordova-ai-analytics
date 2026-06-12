# Anexo: Justificación Técnica para el Cambio de Modelo de Lenguaje (LLM)

**Fecha:** 11 de junio de 2026  
**Proyecto:** Analítica Real Córdova (Tesis SENATI EEST)  
**Autor:** Acosta Cruz, Jean Steven  

## 1. Contexto Inicial y Definición en Metodología
En la propuesta inicial y metodología del proyecto, se estableció el uso del modelo **Meta LLaMA 3.1 8B (Q4_K_M)** ejecutado localmente mediante Ollama. El objetivo principal era maximizar el **Execution Accuracy (EX)** en la traducción de lenguaje natural a SQL para consultas de bases de datos. 

Las pruebas iniciales (Benchmark) demostraron que el modelo LLaMA 3.1 8B alcanzó un éxito excepcional, logrando un **96.67% de Execution Accuracy** en una validación cruzada K-Folds (K=5), superando con creces el umbral del 85% establecido en la Hipótesis (H₁).

## 2. El Problema Identificado: Cuello de Botella en Hardware (Latencia)
Durante la etapa de integración del modelo con la Interfaz de Usuario (Frontend en Streamlit) y el backend (FastAPI), se detectó un problema crítico que afectaba directamente los requisitos no funcionales del sistema (Experiencia de Usuario o UX).

* **Hardware de Ejecución:** El servidor local opera sobre un procesador **AMD Ryzen 5 5600H** utilizando inferencia puramente por CPU (sin tarjeta gráfica dedicada NVIDIA).
* **Peso del Modelo 8B:** El modelo de 8 billones de parámetros requiere trasladar aproximadamente 4.9 GB de pesos neuronales desde la memoria RAM hacia la CPU de manera constante.
* **Tiempos de Respuesta (Latencia):** Según los logs de auditoría extraídos del Benchmark, la latencia promedio por consulta fue de **49,029 milisegundos (49 segundos)**, con picos de hasta 64 segundos en consultas compuestas.

Esta alta latencia provocaba que:
1. El usuario percibiera el sistema como "congelado".
2. Se generaran errores de *Timeout* (tiempo de espera agotado) en el Frontend (cortes de conexión a los 60 segundos), lo cual causaba una tasa de fallo del 100% al intentar interactuar con el chat desde la interfaz web, interrumpiendo abruptamente el backend.

## 3. Resolución Técnica y Optimización
Dado que el tiempo de respuesta era inaceptable para un entorno interactivo de "Chat", se determinó que **el tiempo máximo aceptable de respuesta debía ser inferior a 10 segundos**. 

Debido a que el ancho de banda de memoria de la CPU es un límite físico inalterable, la única solución viable era reducir la cantidad de parámetros del modelo a evaluar por token generado, manteniendo una arquitectura moderna capaz de comprender razonamiento SQL (Chain-of-Thought).

**Acción Correctiva:**  
Se optó por sustituir el modelo LLaMA 3.1 de 8B por el modelo **Meta LLaMA 3.2 3B** (3 mil millones de parámetros). 

**Beneficios obtenidos:**
1. **Reducción drástica del tiempo de inferencia:** Al requerir mover menos de 2 GB de memoria, el procesador Ryzen 5 puede procesar y responder en una fracción del tiempo, cumpliendo la meta de < 10 segundos.
2. **Estabilidad del sistema:** Se eliminaron las caídas por *Timeout* entre la capa de interfaz y la API, permitiendo un uso fluido y continuo del chat conversacional.
3. **Eficiencia en el consumo de recursos:** Menor carga térmica y menor uso de RAM, lo cual es ideal para el despliegue en hardware estándar o *edge devices*.

## 4. Conclusión para el Informe de Tesis
Se documenta este ajuste iterativo propio del desarrollo de software. Queda establecido que se priorizó el equilibrio entre el nivel de precisión técnica (Accuracy) y la viabilidad operativa para el usuario final (Tiempo de Respuesta), justificando plenamente el cambio de la versión 8B a la versión 3B del modelo LLaMA en la arquitectura final del proyecto.
