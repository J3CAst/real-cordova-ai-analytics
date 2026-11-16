"""
engine/prediccion.py
Motor de Predicción de Demanda — Recreo Real Córdova

Usa RandomForestRegressor de scikit-learn entrenado con los registros
históricos de reservas (15,000 registros, 2024-2026) para predecir
el volumen de reservas para los próximos 7 días.

Features de entrada: dia_semana, mes, semana_anio, es_temporada_alta,
                     es_fin_de_semana
Target: COUNT(reservas) agrupado por fecha (excluyendo cancelaciones)
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sqlalchemy import text

load_dotenv()
logger = logging.getLogger("prediccion")

CAPACIDAD_MESAS = 30
MESES_TEMPORADA_ALTA = {12, 1, 2, 3}   # Verano peruano (dic–mar)
DIAS_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

MODEL_PATH = Path(__file__).parent.parent / "models" / "predictor_demanda.joblib"
FEATURES = ["dia_semana", "mes", "semana_anio", "es_temporada_alta", "es_fin_de_semana"]


class PredictorDemanda:
    """
    Modelo ML para predecir demanda de reservas por día.
    Se entrena al iniciar la API (via lifespan) y expone predecir_semana().
    """

    def __init__(self):
        self.modelo = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        self._entrenado = False
        self.mae: float | None = None
        self.rmse: float | None = None
        self.n_muestras: int = 0

    # ──────────────────────────────────────────────────────
    # PERSISTENCIA EN DISCO
    # ──────────────────────────────────────────────────────

    def guardar_modelo(self) -> None:
        """Serializa el modelo y sus métricas en disco (joblib)."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "modelo": self.modelo,
            "mae": self.mae,
            "rmse": self.rmse,
            "n_muestras": self.n_muestras,
        }
        joblib.dump(payload, MODEL_PATH)
        logger.info("Modelo serializado en %s", MODEL_PATH)

    @classmethod
    def cargar_modelo(cls) -> "PredictorDemanda | None":
        """
        Deserializa el modelo desde disco si existe.
        Retorna None si no hay modelo guardado o si falla la carga.
        Evita re-entrenar en cada reinicio del servidor cuando los datos no cambiaron.
        """
        if not MODEL_PATH.exists():
            return None
        try:
            payload = joblib.load(MODEL_PATH)
            instancia = cls.__new__(cls)
            instancia.modelo = payload["modelo"]
            instancia.mae = payload["mae"]
            instancia.rmse = payload["rmse"]
            instancia.n_muestras = payload["n_muestras"]
            instancia._entrenado = True
            logger.info(
                "Modelo cargado desde disco. MAE=%.2f reservas, RMSE=%.2f, muestras=%d",
                instancia.mae, instancia.rmse, instancia.n_muestras,
            )
            return instancia
        except Exception as e:
            logger.warning("Fallo al cargar modelo desde disco: %s — se reentrenará.", e)
            return None

    # ──────────────────────────────────────────────────────
    # ENTRENAMIENTO
    # ──────────────────────────────────────────────────────

    def _cargar_datos(self) -> pd.DataFrame:
        from database.db_pool import pool_engine

        query = text("""
            SELECT
                fecha_reserva,
                COUNT(*)::INT                               AS total_reservas,
                EXTRACT(DOW  FROM fecha_reserva)::INT       AS dia_semana,
                EXTRACT(MONTH FROM fecha_reserva)::INT      AS mes,
                EXTRACT(WEEK  FROM fecha_reserva)::INT      AS semana_anio,
                (BOOL_OR(es_temporada_alta))::INT           AS es_temporada_alta,
                CASE WHEN EXTRACT(DOW FROM fecha_reserva) IN (0, 6)
                     THEN 1 ELSE 0 END::INT                 AS es_fin_de_semana
            FROM reservas
            WHERE estado NOT IN ('cancelada', 'no_show')
              AND fecha_reserva < CURRENT_DATE
            GROUP BY fecha_reserva
            ORDER BY fecha_reserva
        """)
        with pool_engine.connect() as conn:
            return pd.read_sql(query, conn)

    def entrenar(self) -> bool:
        """
        Entrena el modelo con datos históricos de PostgreSQL.
        Retorna True si el entrenamiento fue exitoso y guarda el modelo en disco.
        """
        try:
            df = self._cargar_datos()
            if len(df) < 30:
                logger.warning("Datos insuficientes para entrenar el predictor (<30 días).")
                return False

            X = df[FEATURES].values
            y = df["total_reservas"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.modelo.fit(X_train, y_train)

            y_pred = self.modelo.predict(X_test)
            self.mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
            self.rmse = round(float(root_mean_squared_error(y_test, y_pred)), 2)
            self.n_muestras = len(df)
            self._entrenado = True

            logger.info(
                "Predictor entrenado. MAE=%.2f reservas, RMSE=%.2f, muestras=%d",
                self.mae, self.rmse, self.n_muestras,
            )
            self.guardar_modelo()
            return True

        except Exception as e:
            logger.error("Error entrenando predictor: %s", e)
            return False

    # ──────────────────────────────────────────────────────
    # PREDICCIÓN
    # ──────────────────────────────────────────────────────

    def predecir_semana(self) -> List[Dict[str, Any]]:
        """
        Predice la demanda para los próximos 7 días.
        Retorna lista de dicts ordenada por fecha.
        """
        if not self._entrenado:
            raise RuntimeError(
                "El predictor no está entrenado. Llama a entrenar() primero."
            )

        resultados = []
        hoy = date.today()

        for i in range(1, 8):
            fecha = hoy + timedelta(days=i)

            # PostgreSQL DOW: 0=Dom, 1=Lun, ..., 6=Sáb
            dia_pg = fecha.isoweekday() % 7  # isoweekday: 1=Lun..7=Dom → 0=Dom
            mes = fecha.month
            semana = fecha.isocalendar()[1]
            temporada = 1 if mes in MESES_TEMPORADA_ALTA else 0
            es_fds = 1 if fecha.weekday() >= 5 else 0  # weekday: 5=Sáb, 6=Dom

            X_pred = np.array([[dia_pg, mes, semana, temporada, es_fds]])
            pred_central = max(0, round(float(self.modelo.predict(X_pred)[0])))

            # Intervalo de confianza usando la dispersión entre árboles individuales
            preds_arboles = np.array([
                t.predict(X_pred)[0] for t in self.modelo.estimators_
            ])
            std = float(np.std(preds_arboles))
            rango_min = max(0, round(pred_central - std))
            rango_max = round(pred_central + std)

            # Tasa de ocupación (~1.8 mesas promedio por reserva en datos históricos)
            mesas_est = pred_central * 1.8
            ocupacion_pct = min(round((mesas_est / CAPACIDAD_MESAS) * 100), 100)

            if ocupacion_pct >= 80:
                nivel = "alta"
                recomendacion = (
                    "Demanda alta — reforzar personal, preparar insumos adicionales "
                    "y confirmar disponibilidad de mesas con antelación."
                )
            elif ocupacion_pct >= 50:
                nivel = "media"
                recomendacion = "Demanda moderada — operación estándar recomendada."
            else:
                nivel = "baja"
                recomendacion = (
                    "Demanda baja — oportunidad para mantenimiento preventivo "
                    "o capacitación de personal."
                )

            resultados.append({
                "fecha": fecha.isoformat(),
                "dia_nombre": DIAS_ES[fecha.weekday()],  # weekday(): 0=Lun
                "es_fin_de_semana": fecha.weekday() >= 5,
                "es_temporada_alta": bool(temporada),
                "reservas_predichas": pred_central,
                "rango_min": rango_min,
                "rango_max": rango_max,
                "ocupacion_estimada_pct": ocupacion_pct,
                "nivel_demanda": nivel,
                "recomendacion": recomendacion,
            })

        return resultados

    def get_metricas(self) -> Dict[str, Any]:
        """Retorna métricas de calidad del modelo entrenado."""
        return {
            "entrenado": self._entrenado,
            "mae_reservas": self.mae,
            "rmse_reservas": self.rmse,
            "muestras_entrenamiento": self.n_muestras,
            "features": FEATURES,
            "n_estimadores": self.modelo.n_estimators if self._entrenado else None,
        }
