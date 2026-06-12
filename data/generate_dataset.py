import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL — Parámetros calibrados con la
# bitácora física del Recreo Real Córdova
# ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# Faker does not ship with es_PE in this version, so use generic Spanish.
fake = Faker("es")
Faker.seed(SEED)

# Totales objetivo del documento de tesis
N_CLIENTES = 1_200
N_RESERVAS = 15_000         # Tamaño muestral declarado en metodología

# Período de simulación (ciclo académico 2026-I ampliado
# hacia 2024 para simular histórico)
FECHA_INICIO = date(2024, 1, 1)
FECHA_FIN = date(2026, 6, 30)

# Meses de temporada alta (verano peruano costero)
MESES_VERANO = {12, 1, 2, 3}     # 60% de reservas
MESES_INVIERNO = {6, 7, 8, 9}      # 40% de reservas

# ──────────────────────────────────────────────────────────
# PARÁMETROS DE DISTRIBUCIONES
# Calibrados con datos diagnósticos del establecimiento
# ──────────────────────────────────────────────────────────

# Distribución Poisson: λ = tasa de llegada diaria de clientes
LAMBDA_VERANO_FDS = 42    # fin de semana, temporada alta
LAMBDA_VERANO_SEM = 18    # entre semana, temporada alta
LAMBDA_INVIERNO_FDS = 24   # fin de semana, temporada baja
LAMBDA_INVIERNO_SEM = 9    # entre semana, temporada baja

# Distribución Normal: montos de consumo (S/. por persona)
CONSUMO_ENTRADA_MU = 25.0
CONSUMO_ENTRADA_SD = 5.0
CONSUMO_GASTRO_MU = 45.0
CONSUMO_GASTRO_SD = 15.0
CONSUMO_BEBIDA_MU = 18.0
CONSUMO_BEBIDA_SD = 6.0
CONSUMO_EXTRA_MU = 30.0
CONSUMO_EXTRA_SD = 10.0

# Precios fijos de servicios
PRECIO_ENTRADA_ADULTO = 20.0
PRECIO_ENTRADA_NINO = 12.0
PRECIO_PARRILLA = 35.0
PRECIO_PISCINA = 15.0

# Tasas de comportamiento (diagnosticadas en el establecimiento)
TASA_CANCELACION = 0.10   # 10% de reservas se cancelan
TASA_NO_SHOW = 0.05   # 5% de reservas no se presentan
TASA_RECURRENCIA = 0.35   # 35% de clientes son recurrentes

# Distribución de canales (observación directa diagnóstico)
CANALES = ["whatsapp", "llamada", "presencial", "web"]
PESO_CANALES = [0.65, 0.20, 0.10, 0.05]   # WhatsApp dominante

# Distribución de métodos de pago (contexto MYPE peruana)
METODOS_PAGO = ["efectivo", "yape", "plin", "transferencia", "tarjeta"]
PESO_PAGO = [0.40, 0.30, 0.15, 0.10, 0.05]

# Segmentos de clientes
SEGMENTOS = ["familiar", "corporativo", "pareja", "grupo"]
PESO_SEG = [0.55, 0.15, 0.20, 0.10]

# Distritos de procedencia (Lima Metropolitana)
DISTRITOS = [
    "Lurigancho-Chosica", "El Agustino", "Santa Anita",
    "San Juan de Lurigancho", "Ate", "La Molina",
    "Chaclacayo", "Cieneguilla", "San Borja", "Surco"
]
PESO_DISTRITOS = [0.25, 0.10, 0.10, 0.15, 0.10,
                  0.08, 0.07, 0.05, 0.05, 0.05]


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

def es_temporada_alta(d: date) -> bool:
    """Retorna True si la fecha pertenece a temporada alta (verano)."""
    return d.month in MESES_VERANO


def es_fin_de_semana(d: date) -> bool:
    """Retorna True si es sábado (5) o domingo (6)."""
    return d.weekday() >= 5


def get_lambda(d: date) -> int:
    """Retorna λ de Poisson según temporada y día de semana."""
    if es_temporada_alta(d) and es_fin_de_semana(d):
        return LAMBDA_VERANO_FDS
    elif es_temporada_alta(d):
        return LAMBDA_VERANO_SEM
    elif es_fin_de_semana(d):
        return LAMBDA_INVIERNO_FDS
    else:
        return LAMBDA_INVIERNO_SEM


def clip_positivo(val: float, minimo: float = 0.0) -> float:
    """Garantiza que valores monetarios no sean negativos."""
    return max(round(float(val), 2), minimo)


def generar_hora_llegada(es_fds: bool) -> str:
    """Genera hora de llegada realista según tipo de día."""
    if es_fds:
        hora = np.random.choice(
            range(9, 16), p=[0.05, 0.15, 0.25, 0.25, 0.20, 0.07, 0.03])
    else:
        hora = np.random.choice(range(10, 15), p=[0.20, 0.35, 0.30, 0.10, 0.05])
    minuto = random.randint(0, 59)
    return f"{hora:02d}:{minuto:02d}:00"


def generar_hora_salida(hora_llegada_str: str) -> str:
    """Genera hora de salida (2-6 horas después de llegada)."""
    h, m, _ = map(int, hora_llegada_str.split(":"))
    duracion = random.randint(2, 6)
    h_salida = min(h + duracion, 20)
    return f"{h_salida:02d}:{m:02d}:00"


def obtener_fechas_por_temporada() -> Tuple[List[date], List[date]]:
    """
    Retorna dos listas de fechas (verano e invierno) para forzar
    una estratificación estricta.
    """
    delta = (FECHA_FIN - FECHA_INICIO).days
    todas = [FECHA_INICIO + timedelta(days=i) for i in range(delta + 1)]
    
    verano = [d for d in todas if d.month in MESES_VERANO]
    invierno = [d for d in todas if d.month in MESES_INVIERNO]
    
    return verano, invierno


def get_pesos_dias(fechas: List[date]) -> List[float]:
    """Asigna mayor peso a los fines de semana."""
    pesos = [2.5 if es_fin_de_semana(d) else 1.0 for d in fechas]
    suma = sum(pesos)
    return [p / suma for p in pesos]


# ──────────────────────────────────────────────────────────
# GENERADORES PRINCIPALES
# ──────────────────────────────────────────────────────────

def generar_clientes(n: int) -> pd.DataFrame:
    """
    Genera N perfiles de clientes sintéticos.
    Privacidad por Diseño: nombres con Faker (locale es_PE).
    """
    print(f"\n[1/4] Generando {n} clientes sintéticos...")
    rows = []
    for i in tqdm(range(1, n + 1), unit="cliente"):
        segmento = np.random.choice(SEGMENTOS, p=PESO_SEG)
        rows.append({
            "cliente_id": i,
            "nombre_codigo": fake.name(),          # Nombre sintético
            "distrito": np.random.choice(DISTRITOS, p=PESO_DISTRITOS),
            "segmento": segmento,
            "es_recurrente": random.random() < TASA_RECURRENCIA,
            "fecha_registro": fake.date_between(
                start_date=FECHA_INICIO,
                end_date=FECHA_FIN
            ).isoformat(),
        })
    return pd.DataFrame(rows)


def generar_reservas(n: int, clientes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera N reservas con distribución de fechas estratificada
    (Poisson para tasa de llegada, Normal para personas/mesas).
    """
    print(f"\n[2/4] Generando {n} reservas transaccionales (60% Verano / 40% Invierno)...")
    fechas_verano, fechas_invierno = obtener_fechas_por_temporada()
    
    # Cuotas exactas
    n_verano = int(n * 0.60)
    n_invierno = n - n_verano
    
    # Muestreo estricto
    muestreo_verano = np.random.choice(
        fechas_verano, size=n_verano, p=get_pesos_dias(fechas_verano)
    )
    muestreo_invierno = np.random.choice(
        fechas_invierno, size=n_invierno, p=get_pesos_dias(fechas_invierno)
    )
    
    fechas = list(muestreo_verano) + list(muestreo_invierno)
    random.shuffle(fechas)

    ids_clientes = clientes_df["cliente_id"].tolist()
    rows = []

    for i, (reserva_id, fecha) in enumerate(
        tqdm(enumerate(fechas, start=1), total=n, unit="reserva"), start=0
    ):
        es_alta = es_temporada_alta(fecha)
        es_fds = es_fin_de_semana(fecha)
        lam = get_lambda(fecha)

        # Número de personas: Poisson truncado en [1, 200]
        n_personas = int(np.clip(np.random.poisson(lam / 4), 1, 200))

        # Mesas: ~1 mesa por cada 5 personas
        n_mesas = int(np.clip(
            np.random.normal(n_personas / 5, 1), 1, 40
        ))

        # Estado de la reserva
        rand = random.random()
        if rand < TASA_CANCELACION:
            estado = "cancelada"
            motivo = random.choice([
                "cliente canceló por lluvia",
                "cambio de fecha solicitado",
                "fuerza mayor",
                "sin respuesta del cliente",
                "duplicado accidental",
            ])
        elif rand < TASA_CANCELACION + TASA_NO_SHOW:
            estado = "no_show"
            motivo = None
        else:
            estado = random.choice(["completada", "completada", "confirmada"])
            motivo = None

        # Solicitud hecha entre 1-7 días antes de la visita
        dias_antelacion = random.randint(1, 7)
        fecha_solicitud = datetime.combine(
            fecha - timedelta(days=dias_antelacion),
            datetime.min.time()
        ).isoformat()

        hora_llegada = None
        hora_salida = None
        if estado in ("completada",):
            hora_llegada = generar_hora_llegada(es_fds)
            hora_salida = generar_hora_salida(hora_llegada)

        rows.append({
            "reserva_id": reserva_id,
            "cliente_id": random.choice(ids_clientes),
            "fecha_reserva": fecha.isoformat(),
            "fecha_solicitud": fecha_solicitud,
            "canal": np.random.choice(CANALES, p=PESO_CANALES),
            "numero_personas": n_personas,
            "numero_mesas": n_mesas,
            "estado": estado,
            "motivo_cancelacion": motivo,
            "hora_llegada": hora_llegada,
            "hora_salida": hora_salida,
            "es_temporada_alta": es_alta,
            "observaciones": None,
        })

    return pd.DataFrame(rows)


def generar_items_reserva(reservas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera ítems de consumo para reservas completadas.
    Montos modelados con distribución Normal μ,σ calibrados
    con el ticket promedio diagnóstico del establecimiento.
    """
    print("\n[3/4] Generando ítems de consumo por reserva...")
    completadas = reservas_df[reservas_df["estado"] == "completada"]
    rows = []
    item_id = 1

    for _, res in tqdm(completadas.iterrows(),
                       total=len(completadas), unit="reserva"):
        n_personas = int(res["numero_personas"])
        n_adultos = max(1, int(n_personas * 0.75))
        n_ninos = n_personas - n_adultos

        # ── ENTRADA (siempre presente) ──────────────────────
        rows.append({
            "item_id": item_id,
            "reserva_id": int(res["reserva_id"]),
            "descripcion": "Entrada adulto",
            "categoria": "entrada",
            "cantidad": n_adultos,
            "precio_unitario": clip_positivo(
                np.random.normal(PRECIO_ENTRADA_ADULTO, 2.0)
            ),
        })
        item_id += 1

        if n_ninos > 0:
            rows.append({
                "item_id": item_id,
                "reserva_id": int(res["reserva_id"]),
                "descripcion": "Entrada niño",
                "categoria": "entrada",
                "cantidad": n_ninos,
                "precio_unitario": clip_positivo(
                    np.random.normal(PRECIO_ENTRADA_NINO, 1.5)
                ),
            })
            item_id += 1

        # ── GASTRONOMÍA (70% de reservas) ────────────────────
        if random.random() < 0.70:
            platos = random.randint(1, max(1, n_personas // 3))
            rows.append({
                "item_id": item_id,
                "reserva_id": int(res["reserva_id"]),
                "descripcion": random.choice([
                    "Parrilla familiar", "Pollo a la brasa",
                    "Ceviche", "Arroz con leche", "Chicharrón"
                ]),
                "categoria": "gastronomia",
                "cantidad": platos,
                "precio_unitario": clip_positivo(
                    np.random.normal(CONSUMO_GASTRO_MU, CONSUMO_GASTRO_SD),
                    5.0
                ),
            })
            item_id += 1

        # ── BEBIDAS (85% de reservas) ─────────────────────────
        if random.random() < 0.85:
            bebidas = random.randint(1, max(1, n_personas // 2))
            rows.append({
                "item_id": item_id,
                "reserva_id": int(res["reserva_id"]),
                "descripcion": random.choice([
                    "Gaseosas", "Chicha morada", "Cerveza",
                    "Agua mineral", "Jugos naturales"
                ]),
                "categoria": "bebida",
                "cantidad": bebidas,
                "precio_unitario": clip_positivo(
                    np.random.normal(CONSUMO_BEBIDA_MU, CONSUMO_BEBIDA_SD),
                    3.0
                ),
            })
            item_id += 1

        # ── INSTALACIONES (50% de reservas) ──────────────────
        if random.random() < 0.50:
            rows.append({
                "item_id": item_id,
                "reserva_id": int(res["reserva_id"]),
                "descripcion": random.choice([
                    "Parrilla alquiler", "Piscina acceso extra",
                    "Cancha fútbol", "Zona infantil"
                ]),
                "categoria": "instalacion",
                "cantidad": 1,
                "precio_unitario": clip_positivo(
                    np.random.normal(PRECIO_PARRILLA, 8.0), 10.0
                ),
            })
            item_id += 1

        # ── EXTRAS (25% de reservas) ──────────────────────────
        if random.random() < 0.25:
            rows.append({
                "item_id": item_id,
                "reserva_id": int(res["reserva_id"]),
                "descripcion": random.choice([
                    "Decoración cumpleaños", "Torta personalizada",
                    "Música en vivo", "Servicio catering extra"
                ]),
                "categoria": "extra",
                "cantidad": 1,
                "precio_unitario": clip_positivo(
                    np.random.normal(CONSUMO_EXTRA_MU, CONSUMO_EXTRA_SD),
                    15.0
                ),
            })
            item_id += 1

    return pd.DataFrame(rows)


def generar_transacciones(
    reservas_df: pd.DataFrame,
    items_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Genera el libro de caja sintético.
    Incluye ingresos por reservas, gastos operativos y
    algunas devoluciones (simulando errores pre-sistema).
    """
    print("\n[4/4] Generando transacciones financieras (libro de caja)...")
    rows = []
    transac_id = 1

    # ── INGRESOS POR RESERVAS COMPLETADAS ───────────────────
    completadas = reservas_df[reservas_df["estado"] == "completada"]

    # Agrupar subtotales por reserva
    if not items_df.empty:
        items_df["subtotal"] = items_df["cantidad"] * \
            items_df["precio_unitario"]
        totales_por_reserva = (
            items_df.groupby("reserva_id")["subtotal"].sum().to_dict()
        )
    else:
        totales_por_reserva = {}

    for _, res in tqdm(completadas.iterrows(),
                       total=len(completadas), unit="trans"):
        monto_reserva = totales_por_reserva.get(int(res["reserva_id"]), 0.0)
        if monto_reserva <= 0:
            continue

        # Ingreso principal de la reserva
        rows.append({
            "transaccion_id": transac_id,
            "reserva_id": int(res["reserva_id"]),
            "fecha_transaccion": res["fecha_reserva"],
            "tipo": "ingreso_reserva",
            "concepto": f"Ingreso reserva #{int(res['reserva_id'])}",
            "monto": round(monto_reserva, 2),
            "metodo_pago": np.random.choice(METODOS_PAGO, p=PESO_PAGO),
            "es_conciliado": True,
            "numero_comprobante": f"B{transac_id:06d}",
            "notas": None,
        })
        transac_id += 1

    # ── GASTOS OPERATIVOS (distribuidos en el período) ───────
    # Simula compras de insumos, servicios básicos, etc.
    delta_dias = (FECHA_FIN - FECHA_INICIO).days
    n_gastos = int(delta_dias * 0.8)   # ~0.8 gastos por día en promedio

    print(f"    Generando {n_gastos} gastos operativos...")
    for _ in tqdm(range(n_gastos), unit="gasto"):
        dias_offset = random.randint(0, delta_dias)
        fecha_g = FECHA_INICIO + timedelta(days=dias_offset)
        monto_gasto = clip_positivo(
            np.random.normal(180, 60), 20.0
        )
        rows.append({
            "transaccion_id": transac_id,
            "reserva_id": None,
            "fecha_transaccion": fecha_g.isoformat(),
            "tipo": "gasto_operativo",
            "concepto": random.choice([
                "Compra insumos gastronomía",
                "Servicio agua potable",
                "Mantenimiento instalaciones",
                "Pago personal día",
                "Compra gas GLP",
                "Limpieza y desinfección",
                "Compra bebidas e insumos",
            ]),
            "monto": -round(monto_gasto, 2),   # negativo = egreso
            "metodo_pago": random.choice(["efectivo", "transferencia"]),
            "es_conciliado": True,
            "numero_comprobante": f"G{transac_id:06d}",
            "notas": None,
        })
        transac_id += 1

    # ── DEVOLUCIONES (cancelaciones confirmadas con adelanto) ─
    canceladas = reservas_df[reservas_df["estado"] == "cancelada"].sample(
        frac=0.15, random_state=SEED   # 15% de canceladas tenían adelanto
    )
    for _, res in canceladas.iterrows():
        monto_dev = round(random.uniform(20, 60), 2)
        rows.append({
            "transaccion_id": transac_id,
            "reserva_id": int(res["reserva_id"]),
            "fecha_transaccion": res["fecha_reserva"],
            "tipo": "devolucion",
            "concepto": "Devolución adelanto por cancelación",
            "monto": -monto_dev,
            "metodo_pago": "yape",
            "es_conciliado": True,
            "numero_comprobante": f"D{transac_id:06d}",
            "notas": res["motivo_cancelacion"],
        })
        transac_id += 1

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# VALIDACIONES DE INTEGRIDAD
# ──────────────────────────────────────────────────────────

def validar_dataset(
    clientes_df: pd.DataFrame,
    reservas_df: pd.DataFrame,
    items_df: pd.DataFrame,
    transacciones_df: pd.DataFrame
) -> None:
    """
    Ejecuta un conjunto de assertions de integridad para
    verificar que el dataset cumple las especificaciones
    del diseño metodológico.
    """
    print("\n── VALIDACIONES DE INTEGRIDAD ──────────────────────")

    # 1. Total de reservas
    n = len(reservas_df)
    print(f"  Total reservas generadas   : {n:,}")
    assert n == N_RESERVAS, f"Se esperaban {N_RESERVAS}, se generaron {n}"

    # 2. Estratificación temporal (60/40 verano/invierno)
    pct_alta = reservas_df["es_temporada_alta"].mean()
    print(f"  % temporada alta (verano)  : {pct_alta:.1%}  [objetivo: 60.0%]")
    assert abs(pct_alta - 0.60) < 0.001, f"Estratificación INEXACTA: {pct_alta:.1%} != 60.0%"

    # 3. Estados de reserva
    dist_estados = reservas_df["estado"].value_counts(normalize=True)
    print(f"  Distribución de estados:")
    for est, pct in dist_estados.items():
        print(f"    {est:<15}: {pct:.1%}")

    # 4. Integridad referencial: todos los items apuntan a reservas existentes
    ids_reservas = set(reservas_df["reserva_id"].tolist())
    ids_en_items = set(items_df["reserva_id"].tolist())
    huerfanos = ids_en_items - ids_reservas
    assert len(huerfanos) == 0, f"Items huérfanos: {huerfanos}"
    print(f"  Integridad referencial items: OK ✓")

    # 5. No hay montos negativos en items
    assert (items_df["precio_unitario"] >=
            0).all(), "Precios negativos detectados"
    print(f"  Precios unitarios ≥ 0       : OK ✓")

    # 6. Personas dentro de rango
    assert reservas_df["numero_personas"].between(1, 200).all()
    print(f"  Rango número_personas [1–200]: OK ✓")

    # 7. Resumen financiero
    ingresos = transacciones_df[transacciones_df["monto"] > 0]["monto"].sum()
    gastos = transacciones_df[transacciones_df["monto"] < 0]["monto"].sum()
    print(f"\n  Ingresos totales simulados : S/ {ingresos:,.2f}")
    print(f"  Egresos totales simulados  : S/ {abs(gastos):,.2f}")
    print(f"  Utilidad neta simulada     : S/ {(ingresos + gastos):,.2f}")
    print("\n  ✅ Todas las validaciones pasadas correctamente.")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main() -> None:
    """Pipeline completo de generación del dataset sintético."""
    print("=" * 60)
    print("  GENERADOR DE DATASET SINTÉTICO - REAL CÓRDOVA")
    print("  Tesis SENATI EEST | Acosta Cruz, Jean Steven")
    print("=" * 60)
    print(f"\n  Período     : {FECHA_INICIO} → {FECHA_FIN}")
    print(f"  Registros   : {N_RESERVAS:,} reservas objetivo")
    print(f"  Seed        : {SEED} (reproducibilidad garantizada)")
    print(f"  Privacidad  : Faker locale es_PE (Ley N°29733)\n")

    # ── Generación ────────────────────────────────────────
    clientes_df = generar_clientes(N_CLIENTES)
    reservas_df = generar_reservas(N_RESERVAS, clientes_df)
    items_df = generar_items_reserva(reservas_df)
    transacciones_df = generar_transacciones(reservas_df, items_df)

    # ── Validación ────────────────────────────────────────
    validar_dataset(clientes_df, reservas_df, items_df, transacciones_df)

    # ── Exportar CSV ─────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    archivos = {
        "clientes.csv": clientes_df,
        "reservas.csv": reservas_df,
        "items_reserva.csv": items_df,
        "transacciones_financieras.csv": transacciones_df,
    }

    print("\n── EXPORTANDO CSV ──────────────────────────────────")
    for nombre, df in archivos.items():
        ruta = os.path.join(output_dir, nombre)
        df.to_csv(ruta, index=False, encoding="utf-8")
        print(f"  ✓ {nombre:<40} {len(df):>7,} filas → {ruta}")

    print(f"\n  📁 Archivos guardados en: {output_dir}/")
    print("\n  Siguiente paso:")
    print("  → python data/load_to_postgres.py\n")


if __name__ == "__main__":
    main()
