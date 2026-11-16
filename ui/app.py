import os
import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime, timedelta
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────

API_BASE = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

st.set_page_config(
    page_title="Analítica Real Córdova",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# ESTILOS CSS
# ──────────────────────────────────────────────────────────

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}

    .header-box {
        background: linear-gradient(135deg, #1a5c2a 0%, #2d8a45 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        color: white;
    }
    .header-box h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .header-box p  { margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.85; }

    .chat-user {
        background: #e8f5e9;
        border-left: 4px solid #2d8a45;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background: #f0f4ff;
        border-left: 4px solid #3b5bdb;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .chat-ts { font-size: 0.72rem; color: #adb5bd; margin-top: 0.2rem; }
    .latencia-ok    { color: #2d8a45; font-weight: 600; font-size: 0.82rem; }
    .latencia-lenta { color: #e67700; font-weight: 600; font-size: 0.82rem; }

    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .metric-label { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px;}
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #212529; }

    .pred-alta   { background: #fff3cd; border-left: 4px solid #fd7e14; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.3rem 0; color: #111111 !important; }
    .pred-media  { background: #d1ecf1; border-left: 4px solid #17a2b8; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.3rem 0; color: #111111 !important; }
    .pred-baja   { background: #d4edda; border-left: 4px solid #28a745; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.3rem 0; color: #111111 !important; }
    .pred-alta *, .pred-media *, .pred-baja * { color: inherit !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# PATH para imports de módulos del proyecto
# ──────────────────────────────────────────────────────────

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_pool import pool_engine


# ──────────────────────────────────────────────────────────
# HELPERS — Comunicación con FastAPI
# ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_salud() -> dict:
    try:
        r = httpx.get(f"{API_BASE}/salud", timeout=5.0)
        return r.json()
    except Exception as e:
        return {"estado": "error", "ollama_activo": False, "postgres_activo": False,
                "modelo": "N/A", "mensaje": f"API no responde: {e}"}


@st.cache_data(ttl=10)
def fetch_estadisticas() -> dict:
    try:
        r = httpx.get(f"{API_BASE}/estadisticas", timeout=5.0)
        return r.json()
    except Exception:
        return {}


def fetch_historial(n: int = 30) -> list:
    try:
        r = httpx.get(f"{API_BASE}/historial", params={"n": n}, timeout=5.0)
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=120)
def fetch_predicciones() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}/prediccion/semana", timeout=30.0)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def enviar_consulta(pregunta: str, historial: list | None = None) -> dict:
    """Envía consulta al motor LLM, opcionalmente con contexto de conversación previa."""
    payload = {"pregunta": pregunta, "historial": historial or []}
    try:
        r = httpx.post(f"{API_BASE}/consulta", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        return {"exito": False,
                "error": "Tiempo de espera agotado. Reintenta en 10 segundos.",
                "respuesta_nl": "", "sql_generado": "", "datos": [], "latencia_ms": 0}
    except Exception as e:
        return {"exito": False, "error": f"Error de conexión: {e}",
                "respuesta_nl": "", "sql_generado": "", "datos": [], "latencia_ms": 0}


# ──────────────────────────────────────────────────────────
# HELPERS — LLaMA directo (insight + respuesta + cierre)
# ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def generar_sugerencia_dia(ingresos: float, reservas: int, ocupacion: float) -> str:
    if not ingresos and not reservas:
        return "No hay datos operativos suficientes hoy para generar un insight."
    prompt = (
        f"Eres un analista de negocios para un restaurante campestre. "
        f"Los datos de hoy son: Ingresos S/. {ingresos}, Reservas Activas: {reservas}, Ocupación: {ocupacion:.1f}%. "
        f"Genera UNA ÚNICA LÍNEA de insight o recomendación operativa directa para hoy. Sin saludos ni explicaciones largas."
    )
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "24h",
            "options": {"num_ctx": 1024, "num_thread": 6, "num_predict": 60, "temperature": 0.0}
        }, timeout=120.0)
        return r.json().get("response", "").strip()
    except Exception:
        return "Insight no disponible (LLaMA está ocupado)."


def generar_respuesta_natural(pregunta: str, datos: list) -> str:
    if not datos:
        return ""
    datos_resumen = str(datos[:3])
    if len(datos) > 3:
        datos_resumen += f" ... y {len(datos)-3} registros más."
    prompt = (
        f"El usuario preguntó: '{pregunta}'. "
        f"La base de datos devolvió: {datos_resumen}. "
        f"Responde amablemente en máximo 2 oraciones usando SOLO esos datos. Sin explicaciones técnicas."
    )
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "24h",
            "options": {"num_ctx": 512, "num_thread": 6, "num_predict": 100, "temperature": 0.0}
        }, timeout=60.0)
        return r.json().get("response", "").strip()
    except Exception:
        return ""


def generar_cierre_caja() -> dict:
    """
    Genera el reporte de cierre de caja del día usando datos reales + LLaMA narrativo.
    Retorna dict con reporte_texto, datos, tiempo_generacion_s.
    """
    t_inicio = datetime.now()

    try:
        with pool_engine.connect() as conn:
            # Ingresos por tipo
            q_ing = text("""
                SELECT tipo, COALESCE(SUM(monto), 0) AS total
                FROM transacciones_financieras
                WHERE DATE(fecha_transaccion) = CURRENT_DATE AND tipo LIKE 'ingreso%'
                GROUP BY tipo
            """)
            df_ing = pd.read_sql(q_ing, conn)

            # Egresos
            q_egr = text("""
                SELECT COALESCE(SUM(ABS(monto)), 0) AS total_egresos
                FROM transacciones_financieras
                WHERE DATE(fecha_transaccion) = CURRENT_DATE AND tipo = 'gasto_operativo'
            """)
            total_egresos = float(conn.execute(q_egr).scalar() or 0.0)

            # Reservas del día
            q_res = text("""
                SELECT estado, COUNT(*) AS total
                FROM reservas
                WHERE fecha_reserva = CURRENT_DATE
                GROUP BY estado
            """)
            df_res = pd.read_sql(q_res, conn)

            # Ítem más vendido
            q_item = text("""
                SELECT ir.descripcion, COUNT(*) AS veces
                FROM items_reserva ir
                JOIN reservas r ON r.reserva_id = ir.reserva_id
                WHERE r.fecha_reserva = CURRENT_DATE AND r.estado = 'completada'
                GROUP BY ir.descripcion
                ORDER BY veces DESC LIMIT 1
            """)
            row_item = conn.execute(q_item).fetchone()
            item_top = f"{row_item[0]} ({row_item[1]} veces)" if row_item else "Sin datos"

            # Promedio de ingresos últimos 30 días
            q_prom = text("""
                SELECT COALESCE(AVG(total_ingresos), 0)
                FROM v_resumen_diario
                WHERE fecha >= CURRENT_DATE - INTERVAL '30 days'
                  AND fecha < CURRENT_DATE
            """)
            prom_30d = float(conn.execute(q_prom).scalar() or 0.0)

        # Totales
        total_ingresos = float(df_ing["total"].sum()) if not df_ing.empty else 0.0
        utilidad_neta = total_ingresos - total_egresos

        ing_por_tipo = {row["tipo"]: float(row["total"]) for _, row in df_ing.iterrows()}
        res_por_estado = {row["estado"]: int(row["total"]) for _, row in df_res.iterrows()}

        completadas = res_por_estado.get("completada", 0)
        canceladas = res_por_estado.get("cancelada", 0)
        no_shows = res_por_estado.get("no_show", 0)

        vs_promedio = ((total_ingresos - prom_30d) / prom_30d * 100) if prom_30d > 0 else 0.0
        vs_texto = (
            f"{vs_promedio:+.1f}% respecto al promedio de los últimos 30 días"
            if prom_30d > 0 else "sin datos históricos suficientes para comparar"
        )

        # Prompt narrativo para LLaMA
        prompt = f"""Eres el contador del Recreo Campestre Real Córdova.
Genera un REPORTE DE CIERRE DE CAJA para el {date.today().strftime('%d/%m/%Y')} con los siguientes datos exactos:

INGRESOS DEL DÍA:
{chr(10).join(f'  - {k}: S/. {v:,.2f}' for k, v in ing_por_tipo.items()) if ing_por_tipo else '  (Sin ingresos registrados)'}
  TOTAL INGRESOS: S/. {total_ingresos:,.2f}

EGRESOS: S/. {total_egresos:,.2f}
UTILIDAD NETA: S/. {utilidad_neta:,.2f}

OPERACIONES:
  - Reservas completadas: {completadas}
  - Cancelaciones: {canceladas}
  - No-shows: {no_shows}
  - Ítem más vendido: {item_top}

COMPARATIVA: Los ingresos de hoy están {vs_texto}.

Redacta el reporte en lenguaje formal contable en exactamente 3 párrafos:
1. Resumen de ingresos y utilidad
2. Análisis de operaciones (reservas, cancelaciones)
3. Observación comparativa y recomendación para el próximo día"""

        r = httpx.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "24h",
            "options": {"num_ctx": 2048, "num_thread": 6, "num_predict": 400, "temperature": 0.1}
        }, timeout=180.0)
        reporte_texto = r.json().get("response", "").strip()

        tiempo_s = (datetime.now() - t_inicio).total_seconds()
        return {
            "ok": True,
            "reporte_texto": reporte_texto,
            "total_ingresos": total_ingresos,
            "total_egresos": total_egresos,
            "utilidad_neta": utilidad_neta,
            "completadas": completadas,
            "canceladas": canceladas,
            "no_shows": no_shows,
            "item_top": item_top,
            "tiempo_generacion_s": round(tiempo_s, 1),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────────────────────
# HELPERS — Dashboard y Anomalías
# ──────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def cargar_datos_dashboard(fecha_ini: date = None, fecha_fin: date = None):
    if fecha_ini is None:
        fecha_ini = date.today() - timedelta(days=6)
    if fecha_fin is None:
        fecha_fin = date.today()
    try:
        with pool_engine.connect() as conn:
            q_hoy = "SELECT COALESCE(SUM(monto), 0) FROM transacciones_financieras WHERE DATE(fecha_transaccion) = CURRENT_DATE AND tipo::text LIKE 'ingreso%'"
            ingresos_hoy = float(conn.execute(text(q_hoy)).scalar() or 0.0)

            q_ayer = "SELECT COALESCE(SUM(monto), 0) FROM transacciones_financieras WHERE DATE(fecha_transaccion) = CURRENT_DATE - 1 AND tipo::text LIKE 'ingreso%'"
            ingresos_ayer = float(conn.execute(text(q_ayer)).scalar() or 0.0)

            q_res = "SELECT COUNT(*) FROM reservas WHERE estado = 'confirmada' AND fecha_reserva >= CURRENT_DATE"
            reservas_activas = int(conn.execute(text(q_res)).scalar() or 0)

            q_res_ayer = "SELECT COUNT(*) FROM reservas WHERE estado IN ('confirmada','completada') AND fecha_reserva = CURRENT_DATE - 1"
            reservas_ayer = int(conn.execute(text(q_res_ayer)).scalar() or 0)

            q_ocup = "SELECT COALESCE(SUM(numero_mesas), 0) FROM reservas WHERE estado IN ('confirmada', 'completada') AND fecha_reserva = CURRENT_DATE"
            mesas_ocupadas = int(conn.execute(text(q_ocup)).scalar() or 0)
            tasa_ocupacion = min((mesas_ocupadas / 30.0) * 100, 100.0)

            q_flujo = text("""
                SELECT DATE(fecha_transaccion) as fecha, SUM(monto) as ingresos
                FROM transacciones_financieras
                WHERE tipo::text LIKE 'ingreso%'
                  AND fecha_transaccion >= :fi AND fecha_transaccion <= :ff
                GROUP BY DATE(fecha_transaccion)
                ORDER BY fecha ASC
            """)
            df_flujo = pd.read_sql(q_flujo, conn, params={"fi": fecha_ini, "ff": fecha_fin})

        return {
            "ingresos_hoy": ingresos_hoy, "ingresos_ayer": ingresos_ayer,
            "reservas_activas": reservas_activas, "reservas_ayer": reservas_ayer,
            "tasa_ocupacion": tasa_ocupacion, "df_flujo": df_flujo,
        }
    except Exception as e:
        st.error(f"Error cargando datos del Dashboard: {e}")
        return None


@st.cache_data(ttl=300)
def detectar_anomalia_ingreso(ingresos_hoy: float) -> dict | None:
    """
    Detecta si el ingreso del día es estadísticamente anormal
    comparado con los últimos 30 días (z-score).
    """
    try:
        q = text("""
            SELECT total_ingresos FROM v_resumen_diario
            WHERE fecha >= CURRENT_DATE - INTERVAL '30 days'
              AND fecha < CURRENT_DATE
        """)
        with pool_engine.connect() as conn:
            vals = [float(r[0]) for r in conn.execute(q).fetchall()]

        if len(vals) < 7:
            return None

        media = np.mean(vals)
        std = np.std(vals)
        if std == 0:
            return None

        z = (ingresos_hoy - media) / std

        if z > 2.0:
            return {
                "tipo": "positiva",
                "z": round(z, 2),
                "mensaje": f"¡Día excepcional! Los ingresos de hoy (S/. {ingresos_hoy:,.2f}) "
                           f"superan en {z:.1f}σ el promedio de los últimos 30 días (S/. {media:,.2f})."
            }
        elif z < -2.0:
            return {
                "tipo": "negativa",
                "z": round(z, 2),
                "mensaje": f"Alerta: Los ingresos de hoy (S/. {ingresos_hoy:,.2f}) "
                           f"están {abs(z):.1f}σ por debajo del promedio (S/. {media:,.2f}). Verifica operaciones."
            }
        return None
    except Exception:
        return None


def intentar_grafico(datos: list, pregunta: str) -> None:
    if not datos or len(datos) < 2:
        return
    df = pd.DataFrame(datos)
    cols_num = df.select_dtypes(include="number").columns.tolist()
    cols_str = df.select_dtypes(include="object").columns.tolist()
    if not cols_num:
        return
    try:
        if cols_str and cols_num:
            fig = px.bar(df, x=cols_str[0], y=cols_num[0],
                         title=f"Resultado: {pregunta[:60]}",
                         color=cols_num[0], color_continuous_scale="Greens",
                         template="plotly_white")
            fig.update_layout(showlegend=False, height=350,
                              margin=dict(l=20, r=20, t=50, b=20), title_font_size=13)
            st.plotly_chart(fig, use_container_width=True)
        elif len(cols_num) >= 2:
            fig = px.line(df, y=cols_num[0], title=f"Resultado: {pregunta[:60]}",
                          template="plotly_white")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────

def render_sidebar(dash_data):
    with st.sidebar:
        st.markdown("### 🌿 Real Córdova")
        st.markdown("*Sistema de Analítica Conversacional*")
        st.divider()

        st.markdown("#### Estado del Sistema")
        salud = fetch_salud()
        col1, col2 = st.columns(2)
        with col1:
            ic = "🟢" if salud.get("ollama_activo") else "🔴"
            st.markdown(f"{ic} **LLaMA**")
        with col2:
            ic = "🟢" if salud.get("postgres_activo") else "🔴"
            st.markdown(f"{ic} **PostgreSQL**")
        if salud.get("estado") != "ok":
            st.warning(salud.get("mensaje", ""))
        st.caption(f"Modelo: `{salud.get('modelo', 'N/A')}`")
        st.divider()

        # Alerta de anomalía financiera
        if dash_data:
            anomalia = detectar_anomalia_ingreso(dash_data["ingresos_hoy"])
            if anomalia:
                if anomalia["tipo"] == "positiva":
                    st.success(f"🌟 {anomalia['mensaje']}")
                else:
                    st.error(f"⚠️ {anomalia['mensaje']}")
                st.divider()

        st.markdown("#### Métricas de Rendimiento")
        stats = fetch_estadisticas()
        if stats:
            ea = stats.get("ex_accuracy") or 0.0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Execution Accuracy</div><div class="metric-value">{ea:.1f}%</div></div>', unsafe_allow_html=True)
            lat = stats.get("latencia_media_ms") or 0.0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Latencia Promedio</div><div class="metric-value">{lat:.0f} ms</div></div>', unsafe_allow_html=True)
            total = stats.get("total_consultas", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Consultas</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
        else:
            st.caption("Sin consultas registradas aún.")
        st.divider()

        st.caption("Tesis SENATI EEST 2026 | Acosta Cruz, Jean Steven")


# ──────────────────────────────────────────────────────────
# TAB 1: DASHBOARD OPERATIVO
# ──────────────────────────────────────────────────────────

def render_dashboard():
    st.markdown("### 📊 Dashboard Operativo")
    st.caption("Métricas en tiempo real obtenidas directamente de PostgreSQL.")

    col_f1, col_f2, _ = st.columns([2, 2, 4])
    with col_f1:
        fecha_ini = st.date_input("Desde", value=date.today() - timedelta(days=6), key="dash_fi")
    with col_f2:
        fecha_fin = st.date_input("Hasta", value=date.today(), key="dash_ff")
    if fecha_ini > fecha_fin:
        st.warning("La fecha de inicio debe ser anterior a la fecha fin.")
        return

    dash_data = cargar_datos_dashboard(fecha_ini, fecha_fin)
    if not dash_data:
        st.warning("No se pudieron cargar los datos operativos.")
        return

    # Métricas con delta comparativo
    c1, c2, c3 = st.columns(3)
    with c1:
        delta_ing = dash_data['ingresos_hoy'] - dash_data['ingresos_ayer']
        st.metric("💰 Ingresos del Día", f"S/. {dash_data['ingresos_hoy']:,.2f}",
                  delta=f"S/. {delta_ing:+,.2f} vs. ayer")
    with c2:
        delta_res = dash_data['reservas_activas'] - dash_data['reservas_ayer']
        st.metric("📅 Reservas Activas", str(dash_data['reservas_activas']),
                  delta=f"{delta_res:+d} vs. ayer")
    with c3:
        st.metric("🪑 Tasa de Ocupación", f"{dash_data['tasa_ocupacion']:.1f}%")

    st.markdown("---")

    df_flujo = dash_data['df_flujo']
    if not df_flujo.empty:
        st.markdown(f"#### Flujo de Ingresos ({fecha_ini.strftime('%d/%m/%Y')} — {fecha_fin.strftime('%d/%m/%Y')})")
        st.bar_chart(df_flujo.set_index('fecha')['ingresos'], color="#2d8a45")
    else:
        st.info("No hay transacciones de ingreso en el período seleccionado.")

    # ── Cierre de Caja Automático ──
    st.markdown("---")
    st.markdown("#### 🔐 Cierre de Caja con IA")
    st.caption("Genera en segundos el reporte narrativo de cierre de caja del día usando LLaMA.")

    if st.button("🔐 Generar Cierre de Caja del Día", type="primary"):
        with st.spinner("LLaMA está redactando el reporte de cierre... (~30-60 segundos)"):
            t_inicio_ui = datetime.now()
            resultado = generar_cierre_caja()
            tiempo_ui = (datetime.now() - t_inicio_ui).total_seconds()

        if resultado.get("ok"):
            st.success(f"✅ Reporte generado en **{resultado['tiempo_generacion_s']} segundos**"
                       f" (vs. proceso manual ~150 minutos)")

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Total Ingresos", f"S/. {resultado['total_ingresos']:,.2f}")
            col_r2.metric("Egresos", f"S/. {resultado['total_egresos']:,.2f}")
            col_r3.metric("Utilidad Neta", f"S/. {resultado['utilidad_neta']:,.2f}")

            reporte_editado = st.text_area(
                "Reporte narrativo (editable antes de descargar)",
                value=resultado["reporte_texto"],
                height=300,
            )
            txt_bytes = reporte_editado.encode("utf-8")
            st.download_button(
                "⬇ Descargar Reporte TXT",
                data=txt_bytes,
                file_name=f"cierre_caja_{date.today().isoformat()}.txt",
                mime="text/plain",
            )
            st.session_state["ultimo_cierre"] = resultado
        else:
            st.error(f"Error generando el cierre de caja: {resultado.get('error')}")

    elif "ultimo_cierre" in st.session_state:
        with st.expander("Ver último cierre de caja generado", expanded=False):
            c = st.session_state["ultimo_cierre"]
            st.text(c.get("reporte_texto", ""))


# ──────────────────────────────────────────────────────────
# TAB 2: CHAT CONVERSACIONAL (con memoria multi-turno)
# ──────────────────────────────────────────────────────────

def render_chat(dash_data):
    st.markdown(
        '<div class="header-box"><h1>🤖 Asistente Inteligente</h1>'
        '<p>Realiza preguntas sobre reservas y control financiero. '
        'El asistente recuerda el contexto de la conversación.</p></div>',
        unsafe_allow_html=True
    )

    if dash_data:
        with st.container(border=True):
            st.markdown("💡 **Sugerencia del Día (Generada por LLaMA)**")
            if "insight_diario" not in st.session_state:
                with st.spinner("Pensando..."):
                    st.session_state.insight_diario = generar_sugerencia_dia(
                        dash_data['ingresos_hoy'],
                        dash_data['reservas_activas'],
                        dash_data['tasa_ocupacion']
                    )
            st.info(st.session_state.insight_diario)

    # Inicialización con historial persistente del día
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
        registros_previos = fetch_historial(10)
        hoy = date.today().isoformat()
        for rec in registros_previos:
            ts_raw = rec.get("timestamp", "")
            if not ts_raw.startswith(hoy):
                continue
            ts_fmt = ts_raw[11:16] if len(ts_raw) >= 16 else ""
            st.session_state.mensajes.append({
                "rol": "usuario", "texto": rec.get("pregunta_nl", ""), "ts": ts_fmt
            })
            st.session_state.mensajes.append({
                "rol": "asistente",
                "texto": rec.get("sql_generado", "") if rec.get("resultado_ok") else f"❌ {rec.get('error_msg', '')}",
                "sql": rec.get("sql_generado", ""),
                "datos": [], "latencia_ms": rec.get("latencia_ms", 0),
                "pregunta": rec.get("pregunta_nl", ""), "ts": ts_fmt,
            })

    # Renderizado del historial
    for msg in st.session_state.mensajes:
        if msg["rol"] == "usuario":
            st.markdown(
                f'<div class="chat-user">👤 <strong>Tú:</strong> {msg["texto"]}'
                f'<div class="chat-ts">{msg.get("ts","")}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">🤖 <strong>Asistente:</strong> {msg["texto"]}'
                f'<div class="chat-ts">{msg.get("ts","")}</div></div>',
                unsafe_allow_html=True
            )
            if msg.get("datos"):
                df_res = pd.DataFrame(msg["datos"])
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Descargar CSV", csv_bytes, "resultado.csv", "text/csv",
                                   key=f"dl_{msg.get('ts','')}_{msg.get('pregunta','')[:15]}")
                intentar_grafico(msg["datos"], msg.get("pregunta", ""))
            if msg.get("sql"):
                with st.expander("🔍 Ver SQL generado", expanded=False):
                    st.code(msg["sql"], language="sql")
            if msg.get("latencia_ms"):
                lat = msg["latencia_ms"]
                cls = "latencia-ok" if lat < 3000 else "latencia-lenta"
                st.markdown(f'<span class="{cls}">⏱ {lat:.0f} ms</span>', unsafe_allow_html=True)

    st.markdown("---")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        pregunta = st.text_input("Consulta", placeholder="Ej: ¿Cuánto ingresó esta semana? → ¿Y la semana pasada?",
                                 label_visibility="collapsed", key="input_consulta", max_chars=500)
    with col_btn:
        enviar = st.button("Consultar", type="primary", use_container_width=True)

    if enviar and pregunta.strip():
        ts_ahora = datetime.now().strftime("%H:%M")
        st.session_state.mensajes.append({"rol": "usuario", "texto": pregunta.strip(), "ts": ts_ahora})

        # Construir contexto multi-turno: últimos 3 pares usuario→asistente
        msgs = st.session_state.mensajes
        ctx = []
        for i, m in enumerate(msgs[:-1]):
            if m["rol"] == "usuario" and i + 1 < len(msgs) and msgs[i+1]["rol"] == "asistente":
                sql_previo = msgs[i+1].get("sql", "")
                if sql_previo:
                    ctx.append({"pregunta": m["texto"], "sql": sql_previo})
        ctx = ctx[-3:]

        with st.spinner("🤖 Consultando a LLaMA..."):
            resultado = enviar_consulta(pregunta.strip(), historial=ctx)

        texto_respuesta = resultado.get("respuesta_nl") or ("✅ SQL generado." if resultado["exito"] else f"❌ {resultado['error']}")
        nuevo_mensaje = {
            "rol": "asistente", "texto": texto_respuesta,
            "datos": resultado.get("datos", []),
            "sql": resultado.get("sql_generado", ""),
            "latencia_ms": resultado.get("latencia_ms", 0),
            "pregunta": pregunta.strip(), "ts": ts_ahora,
        }
        st.session_state.mensajes.append(nuevo_mensaje)

        if resultado["exito"] and resultado.get("datos"):
            with st.spinner("🧠 Redactando análisis..."):
                texto_ia = generar_respuesta_natural(pregunta.strip(), resultado["datos"])
                if texto_ia:
                    st.session_state.mensajes[-1]["texto"] = texto_ia

        st.rerun()

    if st.session_state.mensajes:
        if st.button("🗑 Limpiar conversación"):
            st.session_state.mensajes = []
            st.rerun()


# ──────────────────────────────────────────────────────────
# TAB 3: PREDICCIONES ML
# ──────────────────────────────────────────────────────────

def render_predicciones():
    st.markdown("### 📈 Predicción de Demanda (Machine Learning)")
    st.caption(
        "Modelo RandomForestRegressor entrenado con 15,000 registros históricos (2024-2026). "
        "Predice el volumen de reservas para los próximos 7 días."
    )

    data = fetch_predicciones()

    if data is None:
        st.warning("El predictor ML no está disponible. Verifica que la API esté corriendo con suficientes datos.")
        return

    preds = data.get("predicciones", [])
    modelo_info = data.get("modelo", {})

    # Métricas del modelo
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("MAE del Modelo", f"{modelo_info.get('mae_reservas', '—')} reservas",
                  help="Error absoluto medio en reservas/día")
    col_m2.metric("RMSE del Modelo", f"{modelo_info.get('rmse_reservas', '—')} reservas")
    col_m3.metric("Muestras de Entrenamiento", f"{modelo_info.get('muestras_entrenamiento', '—')} días")

    st.markdown("---")

    if not preds:
        st.info("Sin predicciones disponibles.")
        return

    # Gráfico de barras con intervalo de confianza
    df_pred = pd.DataFrame(preds)
    # Formato "Lun 30-jun" — evita que Plotly interprete como fecha
    df_pred["etiqueta"] = df_pred.apply(
        lambda r: f"{r['dia_nombre']} {r['fecha'][8:]}/{r['fecha'][5:7]}", axis=1
    )

    colores = {
        "alta": "#fd7e14",
        "media": "#17a2b8",
        "baja": "#28a745",
    }
    df_pred["color"] = df_pred["nivel_demanda"].map(colores)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_pred["etiqueta"],
        y=df_pred["reservas_predichas"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=(df_pred["rango_max"] - df_pred["reservas_predichas"]).tolist(),
            arrayminus=(df_pred["reservas_predichas"] - df_pred["rango_min"]).tolist(),
        ),
        marker_color=df_pred["color"].tolist(),
        text=df_pred["reservas_predichas"],
        textposition="outside",
        name="Reservas previstas",
    ))
    fig.update_layout(
        title="Predicción de Reservas — Próximos 7 Días",
        xaxis=dict(type="category"),
        yaxis_title="Reservas estimadas",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tarjetas de recomendación por día
    st.markdown("#### Recomendaciones Operativas")
    for pred in preds:
        nivel = pred["nivel_demanda"]
        clase = f"pred-{nivel}"
        fds = "🔥 Fin de semana" if pred.get("es_fin_de_semana", False) else ""
        temporada = "☀️ Temporada alta" if pred["es_temporada_alta"] else ""
        badges = " · ".join(filter(None, [fds, temporada]))
        st.markdown(
            f'<div class="{clase}">'
            f'<strong>{pred["dia_nombre"]} {pred["fecha"][5:]}</strong>'
            f'{" · " + badges if badges else ""} — '
            f'<strong>{pred["reservas_predichas"]}</strong> reservas estimadas '
            f'(rango: {pred["rango_min"]}–{pred["rango_max"]}) · '
            f'Ocupación est.: {pred["ocupacion_estimada_pct"]}%<br>'
            f'<small>{pred["recomendacion"]}</small>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Exportar predicciones
    st.markdown("---")
    csv_pred = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Exportar Predicciones CSV", csv_pred,
                       f"prediccion_{date.today().isoformat()}.csv", "text/csv")


# ──────────────────────────────────────────────────────────
# TAB 4: AUDITORÍA
# ──────────────────────────────────────────────────────────

def render_auditoria():
    st.markdown("### 📋 Panel de Auditoría")

    col_ctrl1, col_ctrl2 = st.columns([2, 3])
    with col_ctrl1:
        n_registros = st.slider("Registros a mostrar", 10, 200, 50, 10)
    with col_ctrl2:
        filtro_estado = st.selectbox("Filtrar por estado", ["Todos", "✅ Exitosas", "❌ Con error"])

    historial = fetch_historial(n_registros)
    if not historial:
        st.info("No hay inferencias registradas.")
        return

    df_hist = pd.DataFrame(historial)
    if "resultado_ok" in df_hist.columns:
        df_hist["estado"] = df_hist["resultado_ok"].apply(lambda x: "✅ OK" if str(x) == "True" else "❌ Error")
        if filtro_estado == "✅ Exitosas":
            df_hist = df_hist[df_hist["resultado_ok"].astype(str) == "True"]
        elif filtro_estado == "❌ Con error":
            df_hist = df_hist[df_hist["resultado_ok"].astype(str) != "True"]

    columnas = [c for c in ["timestamp", "pregunta_nl", "estado", "latencia_ms", "ex_score"] if c in df_hist.columns]
    st.dataframe(df_hist[columnas], use_container_width=True, hide_index=True)

    csv_audit = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Exportar Auditoría CSV", csv_audit,
                       f"auditoria_{date.today().isoformat()}.csv", "text/csv")


# ──────────────────────────────────────────────────────────
# TAB 5: INGRESO DE DATOS
# ──────────────────────────────────────────────────────────

def render_ingreso_datos():
    st.markdown("### 📝 Ingreso Rápido de Servicios")
    st.caption("Registra consumos, entradas de piscina o alquileres en 10 segundos.")

    with st.form("form_nueva_transaccion"):
        col1, col2 = st.columns(2)
        with col1:
            nombre_cliente = st.text_input("👤 Nombre del Cliente", value="Cliente General")
            servicio = st.selectbox("🏢 Servicio Principal", [
                "Piscina (Entradas)", "Restaurante (Consumo)", "Cancha Sintética", "Evento Especial"
            ])
            n_personas = st.number_input("👥 N° de Personas", min_value=1, max_value=200, value=2, step=1)
        with col2:
            monto = st.number_input("💵 Monto Total (S/.)", min_value=0.01, value=30.00, step=1.0)
            metodo_pago = st.selectbox("💳 Método de Pago", ["efectivo", "yape", "plin", "tarjeta", "transferencia"])

        submitted = st.form_submit_button("Registrar Ingreso", type="primary")
        if submitted:
            try:
                map_servicio = {
                    "Piscina (Entradas)": "entrada",
                    "Restaurante (Consumo)": "gastronomia",
                    "Cancha Sintética": "instalacion",
                    "Evento Especial": "extra"
                }
                categoria = map_servicio.get(servicio, "extra")

                with pool_engine.begin() as conn:
                    res_c = conn.execute(text("""
                        INSERT INTO clientes (nombre_codigo, segmento, distrito, es_recurrente)
                        VALUES (:n, 'familiar', 'General', FALSE) RETURNING cliente_id
                    """), {"n": nombre_cliente}).fetchone()

                    res_r = conn.execute(text("""
                        INSERT INTO reservas
                            (cliente_id, fecha_reserva, fecha_solicitud, canal,
                             numero_personas, numero_mesas, estado, es_temporada_alta)
                        VALUES (:cid, CURRENT_DATE, CURRENT_TIMESTAMP, 'presencial',
                                :np, 1, 'completada', TRUE)
                        RETURNING reserva_id
                    """), {"cid": res_c[0], "np": n_personas}).fetchone()

                    conn.execute(text("""
                        INSERT INTO items_reserva (reserva_id, descripcion, categoria, cantidad, precio_unitario)
                        VALUES (:rid, :desc, :cat, 1, :m)
                    """), {"rid": res_r[0], "desc": f"Pago por {servicio}", "cat": categoria, "m": monto})

                    conn.execute(text("""
                        INSERT INTO transacciones_financieras
                            (reserva_id, fecha_transaccion, tipo, concepto, monto, metodo_pago, es_conciliado)
                        VALUES (:rid, CURRENT_DATE, 'ingreso_consumo', :conc, :m, :mp, FALSE)
                    """), {"rid": res_r[0], "conc": f"Pago por {servicio}", "m": monto, "mp": metodo_pago})

                st.success(f"✅ Ingreso de S/. {monto:.2f} por {servicio} registrado con éxito.")
                cargar_datos_dashboard.clear()
                generar_sugerencia_dia.clear()
                detectar_anomalia_ingreso.clear()
                if "insight_diario" in st.session_state:
                    del st.session_state["insight_diario"]

            except Exception as e:
                st.error(f"Error al registrar: {str(e)}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    dash_data = cargar_datos_dashboard()
    render_sidebar(dash_data)

    tab_dash, tab_chat, tab_pred, tab_audit, tab_ingreso = st.tabs([
        "📊 Dashboard",
        "💬 Asistente IA",
        "📈 Predicciones ML",
        "📋 Auditoría",
        "📝 Ingreso de Datos",
    ])

    with tab_dash:
        render_dashboard()
    with tab_chat:
        render_chat(dash_data)
    with tab_pred:
        render_predicciones()
    with tab_audit:
        render_auditoria()
    with tab_ingreso:
        render_ingreso_datos()


if __name__ == "__main__":
    main()
