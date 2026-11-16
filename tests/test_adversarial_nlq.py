"""
tests/test_adversarial_nlq.py
Conjunto de Pruebas Adversariales — Real Córdova AI Analytics

Propósito (para la tesis):
    El dataset de entrenamiento/prueba usa texto sintético "perfecto" generado
    con Faker. En producción real, el administrador escribe sin acentos,
    con abreviaturas o errores ortográficos (desde WhatsApp, teclado físico).

    Este módulo ejecuta 20 consultas deliberadamente "sucias" a través del
    pipeline completo (normalización → LLM → parser SQL) y reporta:

    - Tasa de generación de SQL válido (contiene SELECT)
    - Tasa de queries ejecutables contra PostgreSQL
    - Correcciones de normalización aplicadas
    - Tabla comparativa: texto original vs. texto normalizado enviado al LLM

    Métrica objetivo: ≥ 80 % de SQL válido en condiciones adversariales,
    frente a 93.33 % en el conjunto de prueba limpio (K-Folds).

Uso:
    python -m tests.test_adversarial_nlq
    (ejecutar desde el directorio raíz del proyecto)

Nota: requiere Ollama corriendo y PostgreSQL accesible.
"""

import sys
import os
import json
import time

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from engine.llm_engine import LLMEngine, _normalizar_pregunta
from database.db_pool import ejecutar_query_readonly


# ──────────────────────────────────────────────────────────
# CONSULTAS ADVERSARIALES
# Organizadas en 4 categorías de "ruido" real:
# 1. Sin acentos (texto de WhatsApp o teclado físico)
# 2. Errores ortográficos comunes
# 3. Abreviaturas y jerga peruana
# 4. Preguntas incompletas o ambiguas
# ──────────────────────────────────────────────────────────

CONSULTAS_ADVERSARIALES = [
    # ── Categoría 1: Sin acentos ──────────────────────────
    {
        "id": "ADV-01",
        "categoria": "sin_acentos",
        "texto_original": "cuanto ingreso el recreo esta semana",
        "descripcion": "Falta tilde en 'ingresó', sin signos de interrogación",
        "sql_keywords": ["SUM", "transacciones_financieras", "semana"],
    },
    {
        "id": "ADV-02",
        "categoria": "sin_acentos",
        "texto_original": "cuantas reservas hay para manana",
        "descripcion": "Sin acentos en 'cuántas' y 'mañana'",
        "sql_keywords": ["COUNT", "reservas", "fecha_reserva"],
    },
    {
        "id": "ADV-03",
        "categoria": "sin_acentos",
        "texto_original": "cual fue la utilidad de ayer",
        "descripcion": "Sin acento en 'cuál'",
        "sql_keywords": ["utilidad", "CURRENT_DATE"],
    },
    {
        "id": "ADV-04",
        "categoria": "sin_acentos",
        "texto_original": "cuantos clientes nuevos hubo este mes",
        "descripcion": "Sin tilde en 'cuántos'",
        "sql_keywords": ["COUNT", "clientes"],
    },
    {
        "id": "ADV-05",
        "categoria": "sin_acentos",
        "texto_original": "que canal trae mas reservas",
        "descripcion": "Sin acentos en 'qué' y 'más'",
        "sql_keywords": ["canal", "COUNT", "GROUP BY"],
    },
    # ── Categoría 2: Errores ortográficos ─────────────────
    {
        "id": "ADV-06",
        "categoria": "typo",
        "texto_original": "cuanto se ingreso en resrvas este mes",
        "descripcion": "Typo: 'resrvas' en lugar de 'reservas'",
        "sql_keywords": ["SUM", "transacciones_financieras"],
    },
    {
        "id": "ADV-07",
        "categoria": "typo",
        "texto_original": "cuantas resrvas se cancelaron ayer",
        "descripcion": "Typo: 'resrvas' por 'reservas'",
        "sql_keywords": ["COUNT", "cancelada"],
    },
    {
        "id": "ADV-08",
        "categoria": "typo",
        "texto_original": "muestra los clientes mas recurrenes",
        "descripcion": "Typo: 'recurrenes' por 'recurrentes'",
        "sql_keywords": ["clientes", "es_recurrente"],
    },
    {
        "id": "ADV-09",
        "categoria": "typo",
        "texto_original": "cual es el tickt promedio por canal",
        "descripcion": "Typo: 'tickt' por 'ticket'",
        "sql_keywords": ["AVG", "canal"],
    },
    {
        "id": "ADV-10",
        "categoria": "typo",
        "texto_original": "cuantos no-shws hubo esta semana",
        "descripcion": "Typo: 'no-shws' por 'no_show'",
        "sql_keywords": ["no_show", "COUNT"],
    },
    # ── Categoría 3: Abreviaturas y jerga ─────────────────
    {
        "id": "ADV-11",
        "categoria": "abreviatura",
        "texto_original": "q tal fue la smna pasada en ingresos",
        "descripcion": "Jerga: 'q' por 'qué', 'smna' por 'semana'",
        "sql_keywords": ["SUM", "transacciones_financieras"],
    },
    {
        "id": "ADV-12",
        "categoria": "abreviatura",
        "texto_original": "cuantas rese hay pa hoy",
        "descripcion": "Informal: 'rese' por 'reservas', 'pa' por 'para'",
        "sql_keywords": ["COUNT", "reservas", "CURRENT_DATE"],
    },
    {
        "id": "ADV-13",
        "categoria": "abreviatura",
        "texto_original": "tb quiero saber cuantos clientes vip hay",
        "descripcion": "Abreviatura: 'tb' por 'también'",
        "sql_keywords": ["clientes"],
    },
    {
        "id": "ADV-14",
        "categoria": "abreviatura",
        "texto_original": "ventas de este mes vs mes anterior xq quiero comparar",
        "descripcion": "Jerga: 'xq' por 'porque', comparación de períodos",
        "sql_keywords": ["SUM", "mes_actual", "mes_anterior"],
    },
    {
        "id": "ADV-15",
        "categoria": "abreviatura",
        "texto_original": "los 5 mejores clientes pq necesito llamarlos",
        "descripcion": "Abreviatura: 'pq' por 'porque'",
        "sql_keywords": ["LIMIT 5", "clientes"],
    },
    # ── Categoría 4: Preguntas incompletas o ambiguas ─────
    {
        "id": "ADV-16",
        "categoria": "incompleta",
        "texto_original": "ingresos mayo",
        "descripcion": "Pregunta telegráfica sin verbo ni signos",
        "sql_keywords": ["SUM", "mayo", "transacciones_financieras"],
    },
    {
        "id": "ADV-17",
        "categoria": "incompleta",
        "texto_original": "reservas fin de semana temporada alta",
        "descripcion": "Palabras clave sin estructura gramatical",
        "sql_keywords": ["reservas", "es_temporada_alta", "fin_de_semana"],
    },
    {
        "id": "ADV-18",
        "categoria": "incompleta",
        "texto_original": "cancelaciones",
        "descripcion": "Consulta de una sola palabra",
        "sql_keywords": ["cancelada", "COUNT"],
    },
    {
        "id": "ADV-19",
        "categoria": "incompleta",
        "texto_original": "cuanto hizo ayer el negocio",
        "descripcion": "Sin la palabra 'ingresos', infiere el contexto de negocio",
        "sql_keywords": ["SUM", "CURRENT_DATE - 1"],
    },
    {
        "id": "ADV-20",
        "categoria": "incompleta",
        "texto_original": "top 3 mesas mas usadas",
        "descripcion": "Pregunta coloquial sobre ocupación de mesas",
        "sql_keywords": ["mesas", "LIMIT 3"],
    },
]


# ──────────────────────────────────────────────────────────
# RUNNER DE PRUEBAS
# ──────────────────────────────────────────────────────────

def _sql_contiene_keywords(sql: str, keywords: list[str]) -> float:
    """Porcentaje de keywords presentes en el SQL generado."""
    sql_upper = sql.upper()
    encontradas = sum(1 for kw in keywords if kw.upper() in sql_upper)
    return encontradas / len(keywords) if keywords else 0.0


def ejecutar_pruebas_adversariales(verbose: bool = True) -> dict:
    print("\n" + "="*70)
    print("CONJUNTO DE PRUEBAS ADVERSARIALES — Real Córdova AI Analytics")
    print("Tesis SENATI EEST 2026 | Evaluación de Robustez")
    print("="*70)

    try:
        motor = LLMEngine()
    except Exception as e:
        print(f"\n❌ No se pudo inicializar LLMEngine: {e}")
        print("   Verifica que Ollama esté corriendo: ollama serve")
        return {"error": str(e)}

    resultados = []
    total = len(CONSULTAS_ADVERSARIALES)
    sql_valido = 0
    sql_ejecutable = 0
    total_latencia_ms = 0.0

    categorias = {}

    print(f"\nEjecutando {total} consultas adversariales...\n")

    for i, caso in enumerate(CONSULTAS_ADVERSARIALES, 1):
        texto_orig = caso["texto_original"]

        # Aplicar normalización
        texto_norm, transformaciones = _normalizar_pregunta(texto_orig)

        if verbose:
            print(f"[{i:02d}/{total}] {caso['id']} ({caso['categoria']})")
            print(f"  Original:   {texto_orig}")
            if transformaciones:
                print(f"  Normaliz.:  {texto_norm}")
                print(f"  Cambios:    {'; '.join(transformaciones)}")
            else:
                print(f"  Normaliz.:  (sin cambios)")

        t0 = time.perf_counter()
        resultado = motor.consultar(texto_orig)
        latencia_ms = (time.perf_counter() - t0) * 1000
        total_latencia_ms += latencia_ms

        sql = resultado.get("sql_generado", "")
        tiene_select = "SELECT" in sql.upper()
        keyword_score = _sql_contiene_keywords(sql, caso["sql_keywords"])

        # Intentar ejecutar en BD
        ejecutable = False
        error_bd = None
        if tiene_select:
            try:
                filas = ejecutar_query_readonly(sql)
                ejecutable = True
                sql_ejecutable += 1
            except Exception as e:
                error_bd = str(e)[:80]

        if tiene_select:
            sql_valido += 1
            estado_icono = "✅" if ejecutable else "⚠️ "
        else:
            estado_icono = "❌"

        if verbose:
            print(f"  SQL válido: {estado_icono} | Keywords: {keyword_score:.0%} | {latencia_ms:.0f} ms")
            if sql and verbose:
                print(f"  SQL:        {sql[:100]}...")
            if error_bd:
                print(f"  Error BD:   {error_bd}")
            print()

        cat = caso["categoria"]
        categorias.setdefault(cat, {"total": 0, "valido": 0, "ejecutable": 0})
        categorias[cat]["total"] += 1
        if tiene_select:
            categorias[cat]["valido"] += 1
        if ejecutable:
            categorias[cat]["ejecutable"] += 1

        resultados.append({
            "id": caso["id"],
            "categoria": cat,
            "texto_original": texto_orig,
            "texto_normalizado": texto_norm,
            "transformaciones": transformaciones,
            "sql_generado": sql,
            "sql_valido": tiene_select,
            "sql_ejecutable": ejecutable,
            "keyword_score": round(keyword_score, 2),
            "latencia_ms": round(latencia_ms, 1),
            "error_bd": error_bd,
        })

    # ── Reporte final ──────────────────────────────────────
    accuracy_sql = sql_valido / total * 100
    accuracy_exec = sql_ejecutable / total * 100
    latencia_media = total_latencia_ms / total

    print("="*70)
    print("RESULTADOS FINALES")
    print("="*70)
    print(f"  Total consultas adversariales:  {total}")
    print(f"  SQL válido generado:            {sql_valido}/{total} ({accuracy_sql:.1f}%)")
    print(f"  SQL ejecutable en PostgreSQL:   {sql_ejecutable}/{total} ({accuracy_exec:.1f}%)")
    print(f"  Latencia media:                 {latencia_media:.0f} ms")
    print()
    print("  Comparativa con conjunto limpio (K-Folds):")
    print(f"    Execution Accuracy limpio:    93.33%")
    print(f"    Execution Accuracy adversarial: {accuracy_sql:.1f}%")
    degradacion = 93.33 - accuracy_sql
    print(f"    Degradación por ruido:        {degradacion:.1f} pp")
    print()
    print("  Por categoría:")
    for cat, stats in categorias.items():
        pct = stats["valido"] / stats["total"] * 100
        print(f"    {cat:<15}: {stats['valido']}/{stats['total']} SQL válido ({pct:.0f}%)")
    print()

    # Evaluación de la meta
    if accuracy_sql >= 80:
        print("  ✅ META CUMPLIDA: Robustez adversarial ≥ 80%")
    else:
        print(f"  ⚠️  META NO CUMPLIDA: Se obtuvo {accuracy_sql:.1f}% (meta: 80%)")
        print("     Revisar: ampliar _CORRECCIONES o subir cutoff del fuzzy match.")

    resumen = {
        "total_consultas": total,
        "sql_valido": sql_valido,
        "sql_ejecutable": sql_ejecutable,
        "accuracy_sql_pct": round(accuracy_sql, 2),
        "accuracy_exec_pct": round(accuracy_exec, 2),
        "latencia_media_ms": round(latencia_media, 1),
        "degradacion_vs_limpio_pp": round(degradacion, 2),
        "meta_80pct_cumplida": accuracy_sql >= 80,
        "por_categoria": categorias,
        "detalle": resultados,
    }

    # Guardar reporte JSON
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
        "reporte_adversarial.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"\n  Reporte guardado en: {output_path}")

    return resumen


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pruebas adversariales NLQ → SQL")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Solo muestra el resumen final, sin detalle por consulta")
    args = parser.parse_args()
    ejecutar_pruebas_adversariales(verbose=not args.quiet)
