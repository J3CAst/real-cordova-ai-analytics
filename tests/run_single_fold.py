from sklearn.model_selection import KFold
import statistics
import csv
import time
from pathlib import Path

from test_sql_accuracy import DATASET_EVALUACION, evaluar_consulta


def run_single_fold(fold_index: int = 1, k: int = 5, seed: int = 42):
    indices = list(range(len(DATASET_EVALUACION)))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(kf.split(indices))
    if fold_index < 1 or fold_index > k:
        raise ValueError("fold_index out of range")

    _, test_idx = folds[fold_index - 1]
    subset = [DATASET_EVALUACION[i] for i in test_idx]

    print(f"Running fold {fold_index}/{k} — {len(subset)} queries (seed={seed})")

    results = []
    for i, item in enumerate(subset, start=1):
        print(f"[{i:02d}/{len(subset)}] {item['id']} — {item['pregunta']}")
        r = evaluar_consulta(item, max_retries=3, delay_seconds=3.0)
        results.append(r)
        estado = "✓" if r["exito"] else "✗"
        print(f"  {estado} lat={r.get('latencia_ms',0):.0f}ms error={r.get('error','')}")

    ex_mean = statistics.mean(r["execution_accuracy"] for r in results)
    lat_list = [r["latencia_ms"] for r in results if r.get("latencia_ms", 0) > 0]
    lat_mean = statistics.mean(lat_list) if lat_list else 0.0

    print(f"\nFold {fold_index} summary: EX={ex_mean*100:.1f}% | Lat mean={lat_mean:.0f}ms")

    outdir = Path(__file__).parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"single_fold_{fold_index}_results.csv"
    campos = [
        "id", "categoria", "dificultad", "pregunta",
        "sql_generado", "exito", "execution_accuracy",
        "latencia_ms", "n_filas", "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    run_single_fold(fold_index=1, k=5, seed=42)
