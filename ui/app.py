import os
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────

API_BASE = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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

    /* Métrica en sidebar y dashboard */
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .metric-label { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px;}
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #212529; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# HELPERS — Comunicación con FastAPI y DB
# ──────────────────────────────────────────────────────────

# Importamos directamente db_pool para las consultas nativas del Dashboard
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_pool import pool_engine

@st.cache_data(ttl=30)
def fetch_salud() -> dict:
    """Consulta el health check de la API (cache 30s)."""
    try:
        r = httpx.get(f"{API_BASE}/salud", timeout=5.0)
        return r.json()
    except Exception as e:
        return {
            "estado": "error",
            "ollama_activo": False,
            "postgres_activo": False,
            "modelo": "N/A",
            "mensaje": f"No se puede conectar con la API. Error: {str(e)}",
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
            timeout=300.0,
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

@st.cache_data(ttl=300)
def generar_sugerencia_dia(ingresos: float, reservas: int, ocupacion: float) -> str:
    """Genera un insight de 1 línea usando LLaMA directamente según los datos del Dashboard."""
    if not ingresos and not reservas:
        return "No hay datos operativos suficientes hoy para generar un insight."
        
    prompt = (
        f"Eres un analista de negocios para un restaurante campestre. "
        f"Los datos de hoy son: Ingresos S/. {ingresos}, Reservas Activas: {reservas}, Ocupación: {ocupacion:.1f}%. "
        f"En base a esto, genera UNA ÚNICA LÍNEA de insight, consejo o recomendación operativa directa para hoy. No saludes ni des explicaciones largas."
    )
    
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "24h",
            "options": {
                "num_ctx": 2048,
                "num_thread": 6,
                "num_predict": 50,
                "temperature": 0.0
            }
        }, timeout=120.0)
        return r.json().get("response", "").strip()
    except Exception as e:
        return "Insight no disponible (LLaMA está ocupado)."

def generar_respuesta_natural(pregunta: str, datos: list) -> str:
    """Usa LLaMA para redactar un resumen conversacional de la data extraída."""
    if not datos:
        return ""
    
    # Acotar los datos para que no reviente el contexto (max 3 registros)
    datos_resumen = str(datos[:3])
    if len(datos) > 3:
        datos_resumen += f" ... y {len(datos)-3} registros más."
        
    prompt = f"""El usuario te hizo esta pregunta sobre el negocio: '{pregunta}'
La base de datos devolvió estos datos exactos: {datos_resumen}
Tu tarea: Responde amablemente la pregunta en lenguaje natural usando EXCLUSIVAMENTE los datos proveídos. No agregues explicaciones técnicas, no saludes, máximo 2 oraciones."""

    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "24h",
            "options": {
                "num_ctx": 512,
                "num_thread": 6,
                "num_predict": 100,
                "temperature": 0.0
            }
        }, timeout=60.0)
        return r.json().get("response", "").strip()
    except Exception as e:
        return ""

def cargar_datos_dashboard():
    """Ejecuta consultas nativas a PostgreSQL mediante Pandas para el Tab 1."""
    try:
        with pool_engine.connect() as conn:
            # 1. Ingresos del Día
            query_ing = "SELECT COALESCE(SUM(monto), 0) FROM transacciones_financieras WHERE DATE(fecha_transaccion) = CURRENT_DATE AND tipo::text LIKE 'ingreso%'"
            ingresos_hoy = float(conn.execute(text(query_ing)).scalar() or 0.0)

            # 2. Reservas Activas (Estado: Confirmada)
            query_res = "SELECT COUNT(*) FROM reservas WHERE estado = 'confirmada' AND fecha_reserva >= CURRENT_DATE"
            reservas_activas = int(conn.execute(text(query_res)).scalar() or 0)

            # 3. Tasa de Ocupación Hoy (Asumimos cap. 30 mesas como métrica ejemplo)
            query_ocup = "SELECT COALESCE(SUM(numero_mesas), 0) FROM reservas WHERE estado IN ('confirmada', 'completada') AND fecha_reserva = CURRENT_DATE"
            mesas_ocupadas = int(conn.execute(text(query_ocup)).scalar() or 0)
            tasa_ocupacion = min((mesas_ocupadas / 30.0) * 100, 100.0)

            # 4. Flujo de ingresos (últimos 7 días)
            query_flujo = """
                SELECT DATE(fecha_transaccion) as fecha, SUM(monto) as ingresos
                FROM transacciones_financieras
                WHERE tipo::text LIKE 'ingreso%' AND fecha_transaccion >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(fecha_transaccion)
                ORDER BY fecha ASC
            """
            df_flujo = pd.read_sql(text(query_flujo), conn)
            
        return {
            "ingresos_hoy": ingresos_hoy,
            "reservas_activas": reservas_activas,
            "tasa_ocupacion": tasa_ocupacion,
            "df_flujo": df_flujo
        }
    except Exception as e:
        st.error(f"Error cargando datos del Dashboard: {e}")
        return None

# ──────────────────────────────────────────────────────────
# HELPERS — Visualización automática
# ──────────────────────────────────────────────────────────

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
            x_col = cols_str[0]
            y_col = cols_num[0]
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=f"Resultado: {pregunta[:60]}",
                labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
                color=y_col, color_continuous_scale="Greens", template="plotly_white",
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=50, b=20), title_font_size=13)
            st.plotly_chart(fig, use_container_width=True)
        elif len(cols_num) >= 2:
            fig = px.line(df, y=cols_num[0], title=f"Resultado: {pregunta[:60]}", template="plotly_white")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────

def render_sidebar():
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

        st.markdown("#### Métricas de Rendimiento")
        stats = fetch_estadisticas()
        if stats:
            ea = stats.get("ex_accuracy") or 0.0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Execution Accuracy</div><div class="metric-value">{ea:.1f}%</div></div>', unsafe_allow_html=True)
            lat = stats.get("latencia_media_ms") or 0.0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Latencia Promedio</div><div class="metric-value">{lat:.0f} ms</div></div>', unsafe_allow_html=True)
        else:
            st.caption("Sin consultas registradas aún.")
        st.divider()

        st.caption("Tesis SENATI EEST 2026 | Acosta Cruz, Jean Steven")

# ──────────────────────────────────────────────────────────
# TAB 1: DASHBOARD OPERATIVO (BI TRADICIONAL)
# ──────────────────────────────────────────────────────────

def render_dashboard():
    st.markdown("### 📊 Dashboard Operativo")
    st.caption("Métricas en tiempo real obtenidas directamente de PostgreSQL.")
    
    dash_data = cargar_datos_dashboard()
    if not dash_data:
        st.warning("No se pudieron cargar los datos operativos.")
        return

    # ── Métricas Superiores ──
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #2d8a45;">
                <div class="metric-label">Ingresos del Día</div>
                <div class="metric-value">S/. {dash_data['ingresos_hoy']:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #3b5bdb;">
                <div class="metric-label">Reservas Activas</div>
                <div class="metric-value">{dash_data['reservas_activas']}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #e67700;">
                <div class="metric-label">Tasa de Ocupación</div>
                <div class="metric-value">{dash_data['tasa_ocupacion']:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Gráfico de Flujo Semanal ──
    df_flujo = dash_data['df_flujo']
    if not df_flujo.empty:
        st.markdown("#### Flujo de Ingresos (Últimos 7 días)")
        # Asegurarnos de que el índice sea fecha para usar st.bar_chart correctamente
        df_flujo.set_index('fecha', inplace=True)
        st.bar_chart(df_flujo['ingresos'], color="#2d8a45")
    else:
        st.info("No hay transacciones de ingreso registradas en los últimos 7 días.")

# ──────────────────────────────────────────────────────────
# TAB 2: CHAT CONVERSACIONAL (CAPA IA)
# ──────────────────────────────────────────────────────────

def render_chat(dash_data):
    # Cabecera
    st.markdown(
        '<div class="header-box">'
        '<h1>🤖 Asistente Inteligente</h1>'
        '<p>Realiza preguntas sobre reservas y control financiero usando Lenguaje Natural</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Sugerencias del Día (LLM + Dashboard) ──
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

    # ── Lógica del Chat ──
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for msg in st.session_state.mensajes:
        if msg["rol"] == "usuario":
            st.markdown(f'<div class="chat-user">👤 <strong>Tú:</strong> {msg["texto"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 <strong>Asistente:</strong> {msg["texto"]}</div>', unsafe_allow_html=True)
            if msg.get("datos"):
                st.dataframe(pd.DataFrame(msg["datos"]), use_container_width=True, hide_index=True)
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
        pregunta = st.text_input("Consulta", placeholder="Ej: ¿Cuánto ingresó esta semana?", label_visibility="collapsed", key="input_consulta")

    with col_btn:
        enviar = st.button("Consultar", type="primary", use_container_width=True)

    if enviar and pregunta.strip():
        st.session_state.mensajes.append({"rol": "usuario", "texto": pregunta.strip()})
        with st.spinner("🤖 Consultando a LLaMA..."):
            resultado = enviar_consulta(pregunta.strip())

        if resultado["exito"]:
            texto_respuesta = resultado["respuesta_nl"] or "Consulta procesada correctamente."
        else:
            texto_respuesta = f"❌ {resultado['error']}"

        nuevo_mensaje = {
            "rol": "asistente",
            "texto": texto_respuesta,
            "datos": resultado.get("datos", []),
            "sql": resultado.get("sql_generado", ""),
            "latencia_ms": resultado.get("latencia_ms", 0),
            "pregunta": pregunta.strip(),
        }
        st.session_state.mensajes.append(nuevo_mensaje)
        
        # Generar respuesta natural asíncrona si hubo datos
        if resultado["exito"] and resultado.get("datos"):
            with st.spinner("🧠 Redactando análisis..."):
                texto_ia = generar_respuesta_natural(pregunta.strip(), resultado["datos"])
                if texto_ia:
                    st.session_state.mensajes[-1]["texto"] = texto_ia

        st.rerun()

    if st.session_state.mensajes:
        if st.button("🗑 Limpiar", use_container_width=False):
            st.session_state.mensajes = []
            st.rerun()


# ──────────────────────────────────────────────────────────
# TAB 3: AUDITORÍA
# ──────────────────────────────────────────────────────────
def render_auditoria():
    st.markdown("### 📋 Panel de Auditoría")
    historial = fetch_historial(50)
    if not historial:
        st.info("No hay inferencias registradas.")
        return

    df_hist = pd.DataFrame(historial)
    if "resultado_ok" in df_hist.columns:
        df_hist["estado"] = df_hist["resultado_ok"].apply(lambda x: "✅ OK" if str(x) == "True" else "❌ Error")

    columnas_mostrar = [c for c in ["timestamp", "pregunta_nl", "estado", "latencia_ms", "ex_score"] if c in df_hist.columns]
    st.dataframe(df_hist[columnas_mostrar], use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────
# TAB 4: INGRESO DE DATOS
# ──────────────────────────────────────────────────────────
def render_ingreso_datos():
    st.markdown("### 📝 Ingreso Rápido de Servicios")
    st.caption("Registra consumos, entradas de piscina o alquileres en 10 segundos para alimentar al asistente IA.")
    
    with st.form("form_nueva_transaccion"):
        col1, col2 = st.columns(2)
        with col1:
            nombre_cliente = st.text_input("👤 Nombre del Cliente", placeholder="Ej. Familia Pérez", value="Cliente General")
            servicio = st.selectbox("🏢 Servicio Principal", [
                "Piscina (Entradas)", 
                "Restaurante (Consumo)", 
                "Cancha Sintética", 
                "Evento Especial"
            ])
            n_personas = st.number_input("👥 N° de Personas", min_value=1, value=2, step=1)
            
        with col2:
            monto = st.number_input("💵 Monto Total Pagado (S/.)", min_value=0.01, value=30.00, step=1.0)
            metodo_pago = st.selectbox("💳 Método de Pago", ["efectivo", "yape", "plin", "tarjeta", "transferencia"])
            
        submitted = st.form_submit_button("Registrar Ingreso", type="primary")
        
        if submitted:
            try:
                map_servicio = {
                    "Piscina (Entradas)": "piscina",
                    "Restaurante (Consumo)": "restaurante",
                    "Cancha Sintética": "cancha",
                    "Evento Especial": "evento"
                }
                categoria = map_servicio.get(servicio, "otros")
                
                # Ejecutamos las 4 inserciones en una sola transacción segura (ACID)
                with pool_engine.begin() as conn:
                    # 1. Crear perfil de cliente (ficticio rápido)
                    res_cliente = conn.execute(text("""
                        INSERT INTO clientes (nombre_codigo, segmento, distrito, es_recurrente)
                        VALUES (:nombre, 'familiar', 'General', FALSE)
                        RETURNING cliente_id
                    """), {"nombre": nombre_cliente}).fetchone()
                    cliente_id = res_cliente[0]
                    
                    # 2. Crear la reserva (como completada directamente)
                    res_reserva = conn.execute(text("""
                        INSERT INTO reservas (
                            cliente_id, fecha_reserva, fecha_solicitud, canal, 
                            numero_personas, numero_mesas, estado, es_temporada_alta
                        ) VALUES (
                            :cliente_id, CURRENT_DATE, CURRENT_TIMESTAMP, 'presencial',
                            :n_personas, 1, 'completada', TRUE
                        )
                        RETURNING reserva_id
                    """), {
                        "cliente_id": cliente_id,
                        "n_personas": n_personas
                    }).fetchone()
                    reserva_id = res_reserva[0]
                    
                    # 3. Registrar qué consumieron (el ítem)
                    conn.execute(text("""
                        INSERT INTO items_reserva (
                            reserva_id, descripcion, categoria, cantidad, precio_unitario
                        ) VALUES (
                            :reserva_id, :descripcion, :categoria, 1, :monto
                        )
                    """), {
                        "reserva_id": reserva_id,
                        "descripcion": f"Pago por {servicio}",
                        "categoria": categoria,
                        "monto": monto
                    })
                    
                    # 4. Registrar el ingreso a caja
                    conn.execute(text("""
                        INSERT INTO transacciones_financieras (
                            reserva_id, tipo, monto, metodo_pago, estado, es_conciliado
                        ) VALUES (
                            :reserva_id, 'ingreso_consumo', :monto, :metodo_pago, 'completado', FALSE
                        )
                    """), {
                        "reserva_id": reserva_id,
                        "monto": monto,
                        "metodo_pago": metodo_pago
                    })
                    
                st.success(f"✅ ¡Ingreso de S/. {monto:.2f} por {servicio} registrado con éxito!")
                
                # Refrescar métricas del Dashboard y borrar caché de IA
                cargar_datos_dashboard.clear()
                generar_sugerencia_dia.clear()
                if "insight_diario" in st.session_state:
                    del st.session_state["insight_diario"]
                    
            except Exception as e:
                st.error(f"Error al registrar en la base de datos: {str(e)}")

# ──────────────────────────────────────────────────────────
# MAIN 
# ──────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Pre-cargamos datos operativos para usarlos en Tab 1 y Tab 2
    dash_data = cargar_datos_dashboard()

    tab_dashboard, tab_chat, tab_auditoria, tab_ingreso = st.tabs([
        "📊 Dashboard Operativo",
        "💬 Asistente Inteligente",
        "📋 Auditoría",
        "📝 Ingreso de Datos"
    ])

    with tab_dashboard:
        render_dashboard()

    with tab_chat:
        # Pasamos los datos operativos para que el LLM genere la sugerencia
        render_chat(dash_data)

    with tab_auditoria:
        render_auditoria()
        
    with tab_ingreso:
        render_ingreso_datos()

if __name__ == "__main__":
    main()
