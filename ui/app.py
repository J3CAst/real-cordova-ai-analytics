import os
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────

API_BASE = f"http://localhost:{os.getenv('API_PORT', '8000')}"

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
    /* Ocultar header por defecto de Streamlit */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}

    /* Cabecera personalizada */
    .header-box {
        background: linear-gradient(135deg, #1a5c2a 0%, #2d8a45 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        color: white;
    }
    .header-box h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .header-box p  { margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.85; }

    /* Burbuja de chat — usuario */
    .chat-user {
        background: #e8f5e9;
        border-left: 4px solid #2d8a45;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    /* Burbuja de chat — asistente */
    .chat-assistant {
        background: #f0f4ff;
        border-left: 4px solid #3b5bdb;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    /* Chip de latencia */
    .latencia-ok    { color: #2d8a45; font-weight: 600; font-size: 0.82rem; }
    .latencia-lenta { color: #e67700; font-weight: 600; font-size: 0.82rem; }

    /* Métrica en sidebar */
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .metric-label { font-size: 0.75rem; color: #6c757d; }
    .metric-value { font-size: 1.1rem; font-weight: 700; color: #212529; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# HELPERS — Comunicación con FastAPI
# ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_salud() -> dict:
    """Consulta el health check de la API (cache 30s)."""
    try:
        r = httpx.get(f"{API_BASE}/salud", timeout=5.0)
        return r.json()
    except Exception:
        return {
            "estado": "error",
            "ollama_activo": False,
            "postgres_activo": False,
            "modelo": "N/A",
            "mensaje": "No se puede conectar con la API. ¿Está corriendo FastAPI?",
        }


@st.cache_data(ttl=10)
def fetch_estadisticas() -> dict:
    """Consulta métricas del motor (cache 10s)."""
    try:
        r = httpx.get(f"{API_BASE}/estadisticas", timeout=5.0)
        return r.json()
    except Exception:
        return {}


def fetch_historial(n: int = 30) -> list:
    """Obtiene las últimas N inferencias auditadas."""
    try:
        r = httpx.get(f"{API_BASE}/historial", params={"n": n}, timeout=5.0)
        return r.json()
    except Exception:
        return []


def enviar_consulta(pregunta: str) -> dict:
    """Envía una consulta al motor LLM via FastAPI."""
    try:
        r = httpx.post(
            f"{API_BASE}/consulta",
            json={"pregunta": pregunta},
            timeout=60.0,    # LLM puede tardar hasta ~30s en inferencia
        )
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        return {
            "exito": False,
            "error": "Tiempo de espera agotado. El modelo LLaMA puede estar cargando. Reintenta en 10 segundos.",
            "respuesta_nl": "",
            "sql_generado": "",
            "datos": [],
            "latencia_ms": 0,
            "pregunta": pregunta,
        }
    except Exception as e:
        return {
            "exito": False,
            "error": f"Error de conexión con la API: {str(e)}",
            "respuesta_nl": "",
            "sql_generado": "",
            "datos": [],
            "latencia_ms": 0,
            "pregunta": pregunta,
        }


# ──────────────────────────────────────────────────────────
# HELPERS — Visualización automática de resultados
# ──────────────────────────────────────────────────────────

def intentar_grafico(datos: list, pregunta: str) -> None:
    """
    Intenta generar un gráfico automático según el tipo
    de datos retornados por la consulta.
    Solo grafica si hay más de 1 fila y columnas numéricas.
    """
    if not datos or len(datos) < 2:
        return

    df = pd.DataFrame(datos)
    cols_num = df.select_dtypes(include="number").columns.tolist()
    cols_str = df.select_dtypes(include="object").columns.tolist()

    if not cols_num:
        return

    try:
        # Gráfico de barras si hay categoría + métrica
        if cols_str and cols_num:
            x_col = cols_str[0]
            y_col = cols_num[0]
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=f"Resultado: {pregunta[:60]}",
                labels={x_col: x_col.replace("_", " ").title(),
                        y_col: y_col.replace("_", " ").title()},
                color=y_col,
                color_continuous_scale="Greens",
                template="plotly_white",
            )
            fig.update_layout(
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                title_font_size=13,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Gráfico de línea si solo hay numéricos (serie temporal)
        elif len(cols_num) >= 2:
            fig = px.line(
                df, y=cols_num[0],
                title=f"Resultado: {pregunta[:60]}",
                template="plotly_white",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    except Exception:
        pass    # Si el gráfico falla, simplemente no se muestra


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🌿 Real Córdova")
        st.markdown("*Sistema de Analítica Conversacional*")
        st.divider()

        # ── Estado del sistema ───────────────────────────
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

        # ── Métricas de rendimiento ──────────────────────
        st.markdown("#### Métricas de Rendimiento")
        stats = fetch_estadisticas()

        if stats:
            ea = stats.get("execution_accuracy", 0)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Execution Accuracy</div>'
                f'<div class="metric-value">{ea:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            lat = stats.get("latencia_promedio_ms", 0)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Latencia Promedio</div>'
                f'<div class="metric-value">{lat:.0f} ms</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Consultas Totales</div>'
                f'<div class="metric-value">{stats.get("total_consultas", 0):,}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.caption("Sin consultas registradas aún.")

        st.divider()

        # ── Sugerencias de consulta ──────────────────────
        st.markdown("#### 💡 Consultas Sugeridas")
        sugerencias = [
            "¿Cuántas reservas hay para hoy?",
            "¿Cuánto ingresó esta semana?",
            "¿Cuántas reservas por WhatsApp este mes?",
            "Dame el resumen financiero de hoy.",
            "¿Cuál es el ticket promedio este mes?",
            "¿Cuántas reservas se cancelaron este mes?",
            "¿Qué día tuvo más reservas esta semana?",
            "¿Cuánto se gastó en operaciones este mes?",
        ]

        if "sugerencia_seleccionada" not in st.session_state:
            st.session_state.sugerencia_seleccionada = ""

        for s in sugerencias:
            if st.button(s, key=f"sug_{s}", use_container_width=True):
                st.session_state.sugerencia_seleccionada = s
                st.rerun()

        st.divider()
        st.caption(
            "Tesis SENATI EEST 2026 | "
            "Acosta Cruz, Jean Steven | "
            "Stack: LLaMA · FastAPI · PostgreSQL · Streamlit"
        )


# ──────────────────────────────────────────────────────────
# TAB 1: CHAT CONVERSACIONAL
# ──────────────────────────────────────────────────────────

def render_chat():
    # Cabecera
    st.markdown(
        '<div class="header-box">'
        '<h1>🌿 Analítica Conversacional — Recreo Real Córdova</h1>'
        '<p>Realiza preguntas sobre reservas y control financiero en lenguaje natural</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Inicializar historial de chat en session_state
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    # ── Mostrar historial de conversación ────────────────
    for msg in st.session_state.mensajes:
        if msg["rol"] == "usuario":
            st.markdown(
                f'<div class="chat-user">👤 <strong>Administrador:</strong> {msg["texto"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">🤖 <strong>Asistente:</strong> {msg["texto"]}</div>',
                unsafe_allow_html=True
            )
            # Mostrar tabla de datos si hay resultados
            if msg.get("datos"):
                df = pd.DataFrame(msg["datos"])
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )
                intentar_grafico(msg["datos"], msg.get("pregunta", ""))

            # SQL expandible
            if msg.get("sql"):
                with st.expander("🔍 Ver SQL generado", expanded=False):
                    st.code(msg["sql"], language="sql")

            # Latencia
            if msg.get("latencia_ms"):
                lat = msg["latencia_ms"]
                cls = "latencia-ok" if lat < 3000 else "latencia-lenta"
                st.markdown(
                    f'<span class="{cls}">⏱ {lat:.0f} ms</span>',
                    unsafe_allow_html=True
                )

    # ── Input de consulta ────────────────────────────────
    st.markdown("---")
    col_input, col_btn = st.columns([5, 1])

    # Rellenar input si se seleccionó una sugerencia
    valor_inicial = st.session_state.pop("sugerencia_seleccionada", "")

    with col_input:
        pregunta = st.text_input(
            label="Escribe tu consulta",
            value=valor_inicial,
            placeholder="Ej: ¿Cuánto ingresó el recreo esta semana?",
            label_visibility="collapsed",
            key="input_consulta",
        )

    with col_btn:
        enviar = st.button("Consultar", type="primary",
                           use_container_width=True)

    # ── Procesar consulta ────────────────────────────────
    if enviar and pregunta.strip():
        # Guardar mensaje del usuario
        st.session_state.mensajes.append({
            "rol": "usuario",
            "texto": pregunta.strip(),
        })

        # Llamar a la API con spinner
        with st.spinner("🤖 Consultando a LLaMA..."):
            resultado = enviar_consulta(pregunta.strip())

        # Guardar respuesta del asistente
        if resultado["exito"]:
            texto_respuesta = resultado["respuesta_nl"] or "Consulta procesada correctamente."
        else:
            texto_respuesta = f"❌ {resultado['error']}"

        st.session_state.mensajes.append({
            "rol": "asistente",
            "texto": texto_respuesta,
            "datos": resultado.get("datos", []),
            "sql": resultado.get("sql_generado", ""),
            "latencia_ms": resultado.get("latencia_ms", 0),
            "pregunta": pregunta.strip(),
        })

        st.rerun()

    # ── Botón limpiar conversación ───────────────────────
    if st.session_state.mensajes:
        if st.button("🗑 Limpiar conversación", use_container_width=False):
            st.session_state.mensajes = []
            st.rerun()


# ──────────────────────────────────────────────────────────
# TAB 2: PANEL DE AUDITORÍA
# ──────────────────────────────────────────────────────────

def render_auditoria():
    st.markdown("### 📋 Panel de Auditoría de Inferencias")
    st.caption(
        "Registro completo de consultas procesadas por el motor LLaMA. "
        "Fuente de datos para el Capítulo IV de la tesis (métricas pre/post-test)."
    )

    historial = fetch_historial(50)

    if not historial:
        st.info(
            "No hay inferencias registradas aún. Realiza consultas en la pestaña Chat.")
        return

    df_hist = pd.DataFrame(historial)

    # ── KPIs de auditoría ────────────────────────────────
    stats = fetch_estadisticas()
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Execution Accuracy",
            f"{stats.get('execution_accuracy', 0):.1%}",
            help="% de queries SQL que ejecutaron sin error"
        )
        c2.metric(
            "Latencia Promedio",
            f"{stats.get('latencia_promedio_ms', 0):.0f} ms",
            help="Tiempo promedio de inferencia completa"
        )
        c3.metric(
            "Total Consultas",
            f"{stats.get('total_consultas', 0):,}",
        )
        c4.metric(
            "Consultas Fallidas",
            f"{stats.get('consultas_fallidas', 0):,}",
            delta=f"-{stats.get('consultas_fallidas', 0)} errores",
            delta_color="inverse",
        )

    st.divider()

    # ── Tabla de historial ───────────────────────────────
    st.markdown("#### Últimas 50 inferencias")

    if "sql_ejecutado_ok" in df_hist.columns:
        df_hist["estado"] = df_hist["sql_ejecutado_ok"].apply(
            lambda x: "✅ OK" if str(x) == "True" else "❌ Error"
        )

    columnas_mostrar = [
        c for c in [
            "timestamp", "consulta_nl", "estado",
            "latencia_ms", "n_filas_resultado", "execution_accuracy"
        ] if c in df_hist.columns
    ]

    st.dataframe(
        df_hist[columnas_mostrar],
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Timestamp", width=140),
            "consulta_nl": st.column_config.TextColumn("Consulta", width=300),
            "estado": st.column_config.TextColumn("Estado", width=80),
            "latencia_ms": st.column_config.NumberColumn("Latencia (ms)", format="%.0f"),
            "n_filas_resultado": st.column_config.NumberColumn("Filas", width=60),
            "execution_accuracy": st.column_config.NumberColumn("EX Acc.", format="%.1f"),
        }
    )

    # ── Gráfico de latencia ──────────────────────────────
    if "latencia_ms" in df_hist.columns and len(df_hist) > 1:
        st.markdown("#### Latencia por consulta (ms)")
        df_hist["latencia_ms"] = pd.to_numeric(
            df_hist["latencia_ms"], errors="coerce"
        )
        fig = px.line(
            df_hist.iloc[::-1],   # orden cronológico
            y="latencia_ms",
            labels={"latencia_ms": "Latencia (ms)", "index": "Consulta #"},
            template="plotly_white",
            color_discrete_sequence=["#2d8a45"],
        )
        fig.add_hline(
            y=3000,
            line_dash="dash",
            line_color="red",
            annotation_text="Objetivo: < 3,000 ms",
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────
# MAIN — Estructura de tabs
# ──────────────────────────────────────────────────────────

def main():
    render_sidebar()

    tab_chat, tab_auditoria = st.tabs([
        "💬 Chat Conversacional",
        "📋 Auditoría de Inferencias",
    ])

    with tab_chat:
        render_chat()

    with tab_auditoria:
        render_auditoria()


if __name__ == "__main__":
    main()
