"""
tests/run_benchmark.py
Runner no-interactivo del benchmark K-Folds para ejecucion automatizada.
"""
import sys
import io
import time
from pathlib import Path

# Forzar UTF-8 en stdout/stderr para evitar UnicodeEncodeError en Windows (cp1252)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_sql_accuracy import (
    DATASET_EVALUACION,
    K_FOLDS,
    ejecutar_kfolds,
    generar_reporte,
    exportar_resultados,
)

if __name__ == "__main__":
    print("=" * 65)
    print("  BENCHMARK DE VALIDACIÓN — SISTEMA REAL CÓRDOVA")
    print("  Tesis SENATI EEST | Acosta Cruz, Jean Steven | 2026")
    print("=" * 65)
    print(f"\n  Dataset   : {len(DATASET_EVALUACION)} consultas (sin data leakage)")
    print(f"  K-Folds   : K={K_FOLDS}")
    print(f"  Métricas  : EX clásica + SA semántica (Answer Accuracy)")
    print(f"  Objetivo  : EX > 85%\n")

    t_inicio = time.perf_counter()
    resultado = ejecutar_kfolds(DATASET_EVALUACION, K_FOLDS)
    elapsed = time.perf_counter() - t_inicio

    reporte = generar_reporte(resultado)
    print("\n" + reporte)
    exportar_resultados(resultado, reporte)

    print(f"\n  Tiempo total: {elapsed/60:.1f} minutos")
    print(f"  EX Media    : {resultado['ex_media']*100:.2f}%")
    if resultado.get('sa_media') is not None:
        print(f"  SA Media    : {resultado['sa_media']*100:.2f}%")
    print(f"  Latencia    : {resultado['latencia_media_ms']:.0f} ms promedio")
    h = resultado['hipotesis_aceptada']
    print(f"  Hipótesis H1: {'✅ ACEPTADA' if h else '❌ NO ACEPTADA'} (umbral EX > 85%)")
