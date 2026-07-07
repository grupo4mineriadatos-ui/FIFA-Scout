"""
FIFA Scout — Web App de Scouting de Jugadores
================================================
TP4 · Licenciatura en Ciencias de Datos · Inteligencia Artificial y Aprendizaje Automático I

App de scouting que despliega el modelo ganador del TP2 (Gradient Boosting
Optimizado, Pipeline StandardScaler + GradientBoostingRegressor) para predecir
el potencial de crecimiento de jugadores de fútbol, con explicabilidad SHAP,
carga masiva por CSV y un espacio reservado para el clasificador de posición del TP3.
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# (el tema de colores nativo vive en .streamlit/config.toml)
# ============================================================
st.set_page_config(
    page_title="FIFA Scout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PALETA — dark sports-tech: navy profundo, verde césped, dorado élite
# Contraste verificado (WCAG): ver notas junto a cada color.
# (debe coincidir con .streamlit/config.toml)
# ============================================================
COLOR_BG = "#0B1220"             # fondo de página
COLOR_SURFACE = "#16213B"        # superficie de tarjetas
COLOR_SURFACE_2 = "#1B2A47"      # superficie elevada (hero)
COLOR_BORDER = "#2C3B57"         # borde sutil de tarjetas
COLOR_TEXT = "#F3F4F6"           # texto principal — 17:1 sobre COLOR_BG
COLOR_GRAY = "#94A3B8"           # texto secundario — 7.3:1 sobre COLOR_BG
COLOR_ACCENT = "#16A34A"         # verde césped — botones/badges (con texto oscuro encima)
COLOR_ACCENT_LIGHT = "#22C55E"   # verde claro — íconos y gráficos decorativos
COLOR_GOLD = "#D4AF37"           # dorado — tier élite, 8.9:1 sobre COLOR_BG
COLOR_WARN = "#F59E0B"           # ámbar — alertas suaves
COLOR_CHART_NEUTRAL = "#5B8DEF"  # azul — gráficos de conteo/ranking neutrales
COLOR_GRID = "#22304A"           # líneas de grilla, sutiles sobre fondo oscuro
COLOR_ON_ACCENT = "#08170D"      # texto oscuro sobre fondo verde — 5.7:1, AA
COLOR_DARK = COLOR_TEXT          # alias retro-compatible: "texto fuerte"

# ============================================================
# ÍCONOS — set propio en SVG (sin emojis), trazo consistente 1.9px
# ============================================================
ICONS = {
    "search": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
              'stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/>'
              '<line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "bolt": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
            'stroke-linecap="round" stroke-linejoin="round">'
            '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "folder": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
              'stroke-linecap="round" stroke-linejoin="round">'
              '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
    "target": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
              'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>'
              '<circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none"/></svg>',
    "bar-chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
                 'stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/>'
                 '<line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "soccer": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
              'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>'
              '<polygon points="12,8.8 15.04,10.99 13.88,14.59 10.12,14.59 8.96,10.99" '
              'fill="currentColor" stroke="none"/></svg>',
}


def svg_icon(name: str, size: int = 22, color: str = "currentColor") -> str:
    """Devuelve el SVG inline de un ícono, listo para insertar en HTML propio."""
    markup = ICONS.get(name, ICONS["search"])
    return markup.replace("<svg ", f'<svg style="width:{size}px;height:{size}px;display:block;color:{color};" ', 1)


# ============================================================
# CSS — sistema de diseño (hero headers, cards, tipografía)
# ============================================================
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

        html {{ color-scheme: dark; }}
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }}
        footer {{ visibility: hidden; }}

        h1, h2, h3 {{ color: {COLOR_TEXT}; font-weight: 700; font-family: 'Sora', sans-serif; }}
        h1 {{ font-size: 1.7rem; }}
        h2 {{ font-size: 1.25rem; }}
        h3 {{ font-size: 1.05rem; }}

        /* Foco visible — accesibilidad, nunca se remueve */
        :focus-visible {{ outline: 2px solid {COLOR_ACCENT_LIGHT} !important; outline-offset: 2px; }}

        /* Hero header por página */
        .fifa-hero {{
            display: flex; align-items: center; gap: 16px;
            padding: 18px 24px 18px 20px; margin-bottom: 24px;
            background: {COLOR_SURFACE_2};
            border: 1px solid {COLOR_BORDER}; border-left: 4px solid {COLOR_ACCENT_LIGHT};
            border-radius: 14px;
        }}
        .fifa-hero-icon {{ line-height: 1; flex-shrink: 0; }}
        .fifa-hero-title {{
            font-family: 'Sora', sans-serif; font-size: 1.4rem; font-weight: 700;
            color: {COLOR_TEXT}; margin: 0; line-height: 1.2;
        }}
        .fifa-hero-subtitle {{ font-size: 0.92rem; color: {COLOR_GRAY}; margin-top: 3px; }}

        /* Tarjetas KPI */
        .fifa-kpi {{
            background: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; border-radius: 12px;
            padding: 14px 16px;
        }}
        .fifa-kpi-label {{
            font-size: 0.74rem; color: {COLOR_GRAY}; text-transform: uppercase;
            letter-spacing: .04em; font-weight: 600;
        }}
        .fifa-kpi-value {{
            font-family: 'Sora', sans-serif; font-size: 1.6rem; color: {COLOR_TEXT};
            font-weight: 700; margin-top: 2px;
        }}

        .fifa-badge {{
            display: inline-block; padding: 4px 14px; border-radius: 20px;
            background: {COLOR_ACCENT}; color: {COLOR_ON_ACCENT}; font-size: 0.8rem; font-weight: 700;
        }}

        /* Etiquetas de sección (reemplazan bold suelto) */
        .fifa-section-label {{
            font-size: 0.76rem; font-weight: 700; color: {COLOR_GRAY};
            text-transform: uppercase; letter-spacing: .05em;
            border-bottom: 2px solid {COLOR_ACCENT_LIGHT}; padding-bottom: 6px; margin-bottom: 10px;
        }}

        /* Sidebar */
        .fifa-sidebar-brand {{ display: flex; align-items: center; gap: 10px; padding: 2px 0 10px 0; }}
        .fifa-sidebar-icon {{ line-height: 1; flex-shrink: 0; }}
        .fifa-sidebar-title {{
            font-family: 'Sora', sans-serif; font-size: 1.1rem; font-weight: 700;
            color: {COLOR_TEXT}; line-height: 1.1;
        }}
        .fifa-sidebar-subtitle {{ font-size: 0.72rem; color: {COLOR_GRAY}; }}

        .fifa-kpi-mini {{
            background: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; border-radius: 10px;
            padding: 10px 12px; text-align: center;
        }}
        .fifa-kpi-mini-value {{
            font-family: 'Sora', sans-serif; font-size: 1.35rem; font-weight: 700; color: {COLOR_ACCENT_LIGHT};
        }}
        .fifa-kpi-mini-label {{ font-size: 0.7rem; color: {COLOR_GRAY}; }}

        /* Dataframes con esquinas redondeadas */
        [data-testid="stDataFrame"] {{ border: 1px solid {COLOR_BORDER}; border-radius: 10px; overflow: hidden; }}

        button[kind="primary"] {{
            background-color: {COLOR_ACCENT} !important; color: {COLOR_ON_ACCENT} !important;
            border: none !important; border-radius: 8px; font-weight: 700;
        }}
        button[kind="primary"]:hover {{
            background-color: {COLOR_ACCENT_LIGHT} !important; color: {COLOR_ON_ACCENT} !important;
        }}

        /* Números tabulares para alinear cifras en tarjetas y tablas */
        .fifa-kpi-value, .fifa-kpi-mini-value, [data-testid="stDataFrame"] {{
            font-variant-numeric: tabular-nums;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(icon_key: str, titulo: str, subtitulo: str):
    """Header consistente tipo 'hero' para el tope de cada página."""
    icon_html = svg_icon(icon_key, size=26, color=COLOR_ACCENT_LIGHT)
    st.markdown(
        f"""
        <div class="fifa-hero">
            <div class="fifa-hero-icon">{icon_html}</div>
            <div>
                <div class="fifa-hero-title">{titulo}</div>
                <div class="fifa-hero-subtitle">{subtitulo}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(col, label: str, value: str):
    """Tarjeta KPI dentro de una columna de st.columns()."""
    with col:
        st.markdown(
            f"""
            <div class="fifa-kpi">
                <div class="fifa-kpi-label">{label}</div>
                <div class="fifa-kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_label(texto: str):
    st.markdown(f'<div class="fifa-section-label">{texto}</div>', unsafe_allow_html=True)


def estilizar_fig(fig, height=420, showlegend=False):
    """Aplica un template visual único a todos los gráficos de Plotly de la app."""
    fig.update_layout(
        height=height,
        font_family="Inter, sans-serif",
        font_color=COLOR_GRAY,
        title_font_size=16,
        title_font_color=COLOR_TEXT,
        plot_bgcolor=COLOR_SURFACE,
        paper_bgcolor=COLOR_SURFACE,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=showlegend,
        hoverlabel=dict(bgcolor=COLOR_SURFACE_2, font_size=12, font_family="Inter, sans-serif",
                         font_color=COLOR_TEXT, bordercolor=COLOR_BORDER),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=COLOR_GRID, zeroline=False)
    return fig


# ============================================================
# CONSTANTES DEL MODELO — deben coincidir EXACTO con el TP2
# ============================================================
MODEL_PATH = "modelo_gb_pipeline.joblib"
DATA_PATH = "fifa_players_model_ready.csv"
TP3_MODEL_PATH = "modelo_clasificador_tp3.joblib"

FEATURES_BASE = [
    "age", "overall_rating",
    "vision", "agility", "standing_tackle", "strength",
    "international_reputation(1-5)", "weak_foot(1-5)", "skill_moves(1-5)",
]
FEATURES_TP1 = [
    "attack_score", "defense_score", "playmaking_score", "physical_score",
    "has_release_clause",
]
FEATURES_NUEVAS = [
    "preferred_foot_enc",   # 0 = Izquierdo, 1 = Derecho
    "nationality_freq",     # frecuencia de aparición de la nacionalidad en el dataset
]
POSITION_COLS = [
    "pos_Arquero", "pos_Defensor Central", "pos_Delantero Centro",
    "pos_Extremo", "pos_Lateral", "pos_Mediocampista Defensivo",
    "pos_Mediocampista Ofensivo",
]
ALL_FEATURES = FEATURES_BASE + FEATURES_TP1 + FEATURES_NUEVAS + POSITION_COLS
TARGET = "potential"

POSICIONES = [c.replace("pos_", "") for c in POSITION_COLS]

# Métricas reales del TP2 (extraídas del notebook, tabla comparativa final)
COMPARATIVA_MODELOS = pd.DataFrame([
    {"Modelo": "Regresión Lineal",              "R2_test": 0.8328, "MAE": 1.9558, "RMSE": 2.4794, "Gap_%": 0.05},
    {"Modelo": "Árbol de Decisión",              "R2_test": 0.9200, "MAE": 1.1234, "RMSE": 1.7151, "Gap_%": 0.98},
    {"Modelo": "KNN (7 features, K=9)",          "R2_test": 0.9029, "MAE": 1.3427, "RMSE": 1.8898, "Gap_%": 2.14},
    {"Modelo": "Random Forest (base)",           "R2_test": 0.9301, "MAE": 1.0205, "RMSE": 1.6033, "Gap_%": 5.99},
    {"Modelo": "GB Optimizado (GridSearchCV) — elegido", "R2_test": 0.9355, "MAE": 0.9973, "RMSE": 1.5402, "Gap_%": 2.23},
])

HIPERPARAMETROS_GB = {
    "learning_rate": 0.05,
    "max_depth": 5,
    "n_estimators": 300,
    "subsample": 0.8,
}


# ============================================================
# CARGA CON CACHÉ
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Carga el pipeline GB Optimizado serializado en el TP2."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_tp3_model():
    """Carga el clasificador de posición del TP3 si ya está disponible."""
    if not os.path.exists(TP3_MODEL_PATH):
        return None
    return joblib.load(TP3_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    """Carga el dataset del TP1 y agrega la predicción de potencial para todo el dataset."""
    df = pd.read_csv(DATA_PATH)
    modelo = load_model()
    if modelo is not None:
        df["potential_predicho"] = np.round(modelo.predict(df[ALL_FEATURES]), 2)
    else:
        df["potential_predicho"] = np.nan
    return df


def resolve_display_columns(df: pd.DataFrame):
    """
    Resuelve de forma robusta los nombres de columnas de presentación
    (nombre corto, nombre largo, club, posición legible), ya que pueden
    variar levemente según la versión del dataset procesado en el TP1.
    """
    short_col = next((c for c in ["short_name", "name"] if c in df.columns), df.columns[0])
    long_col = next((c for c in ["long_name", "full_name"] if c in df.columns), short_col)
    club_col = next((c for c in ["club_name", "club"] if c in df.columns), None)
    position_col = next((c for c in ["specific_position", "player_positions", "positions"] if c in df.columns), None)
    return short_col, long_col, club_col, position_col


def formatear_euros(valor: float) -> str:
    if pd.isna(valor):
        return "—"
    if valor >= 1_000_000:
        return f"€{valor / 1_000_000:.2f}M"
    if valor >= 1_000:
        return f"€{valor / 1_000:.0f}K"
    return f"€{valor:.0f}"


def clasificar_potencial(p: float) -> str:
    if p < 65:
        return "Jugador de relleno"
    if p < 75:
        return "Suplente confiable"
    if p < 82:
        return "Titular sólido"
    if p < 88:
        return "Jugador de alto nivel"
    return "Élite mundial"


# ============================================================
# ESTADO DE SESIÓN — monitoreo básico (actividad de extensión)
# ============================================================
if "n_predicciones" not in st.session_state:
    st.session_state.n_predicciones = 0
if "n_predicciones_lote" not in st.session_state:
    st.session_state.n_predicciones_lote = 0

# ============================================================
# CARGA INICIAL
# ============================================================
modelo = load_model()
modelo_disponible = modelo is not None

if modelo_disponible:
    df = load_data()
    SHORT_COL, LONG_COL, CLUB_COL, POSITION_COL = resolve_display_columns(df)
else:
    df = None
    SHORT_COL = LONG_COL = CLUB_COL = POSITION_COL = None

# ============================================================
# SIDEBAR — navegación + contador de sesión
# ============================================================
st.sidebar.markdown(
    f"""
    <div class="fifa-sidebar-brand">
        <div class="fifa-sidebar-icon">{svg_icon('soccer', size=28, color=COLOR_ACCENT_LIGHT)}</div>
        <div>
            <div class="fifa-sidebar-title">FIFA Scout</div>
            <div class="fifa-sidebar-subtitle">Scouting con Machine Learning</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegación",
    [
        ":material/search: Explorador de Joyas",
        ":material/bolt: Predictor Individual",
        ":material/folder_open: Análisis en Lote",
        ":material/adjust: Clasificador de Posición",
        ":material/bar_chart: Sobre el Modelo",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
total_predicciones = st.session_state.n_predicciones + st.session_state.n_predicciones_lote
st.sidebar.markdown(
    f"""
    <div class="fifa-kpi-mini">
        <div class="fifa-kpi-mini-value">{total_predicciones}</div>
        <div class="fifa-kpi-mini-label">Predicciones en esta sesión</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Monitoreo básico de uso (session_state)")

if not modelo_disponible:
    st.sidebar.error(
        f"No se encontró '{MODEL_PATH}'. Ejecutá el notebook de entrenamiento "
        f"y colocá el archivo serializado en la carpeta de la app."
    )

# ============================================================
# PÁGINA 1 — EXPLORADOR DE JOYAS
# ============================================================
if pagina == ":material/search: Explorador de Joyas":
    hero("search", "Explorador de Joyas", "Identificá jugadores jóvenes con alto potencial predicho y bajo valor de mercado.")

    if not modelo_disponible:
        st.error(
            "Esta página requiere el modelo entrenado. Colocá "
            f"`{MODEL_PATH}` en la carpeta de la app y recargá."
        )
        st.stop()

    # ---------------- FILTROS (sidebar) ----------------
    with st.sidebar.expander(":material/tune: Filtros", expanded=True):
        posiciones_sel = st.multiselect(
            "Posición", options=POSICIONES, default=[],
            help="Sin selección = todas las posiciones",
        )

        edad_min, edad_max = st.slider("Rango de edad", 15, 45, (15, 45))

        valor_max_dataset = float(df["value_euro"].max()) / 1_000_000
        valor_min_sel, valor_max_sel = st.slider(
            "Valor de mercado (millones €)", 0.0, round(valor_max_dataset, 1),
            (0.0, round(valor_max_dataset, 1)), step=0.5,
        )

        nacionalidades_sel = st.multiselect(
            "Nacionalidad", options=sorted(df["nationality"].unique()), default=[],
            help="Sin selección = todas las nacionalidades",
        )

        overall_min = st.slider("Overall rating mínimo", 50, 99, 50)

    # ---------------- APLICAR FILTROS ----------------
    df_f = df.copy()
    if POSITION_COL and posiciones_sel:
        df_f = df_f[df_f[POSITION_COL].isin(posiciones_sel)]
    df_f = df_f[(df_f["age"] >= edad_min) & (df_f["age"] <= edad_max)]
    df_f = df_f[
        (df_f["value_euro"] >= valor_min_sel * 1_000_000)
        & (df_f["value_euro"] <= valor_max_sel * 1_000_000)
    ]
    if nacionalidades_sel:
        df_f = df_f[df_f["nationality"].isin(nacionalidades_sel)]
    df_f = df_f[df_f["overall_rating"] >= overall_min]

    k1, k2, k3 = st.columns(3)
    kpi_card(k1, "Jugadores filtrados", f"{len(df_f):,}")
    kpi_card(k2, "Total en el dataset", f"{len(df):,}")
    kpi_card(k3, "Potencial promedio (filtro)", f"{df_f['potential_predicho'].mean():.1f}" if not df_f.empty else "—")
    st.write("")

    # ---------------- SCATTER PRINCIPAL ----------------
    if df_f.empty:
        st.warning("Ningún jugador cumple con los filtros seleccionados. Ajustá los criterios.")
    else:
        hover_cols = {
            SHORT_COL: True,
            "nationality": True,
            "age": True,
            "overall_rating": True,
            "potential_predicho": True,
            "value_euro": ":,.0f",
        }
        if CLUB_COL:
            hover_cols[CLUB_COL] = True

        fig = px.scatter(
            df_f,
            x=df_f["value_euro"] / 1_000_000,
            y="potential_predicho",
            color="age",
            size="overall_rating",
            size_max=18,
            color_continuous_scale=[COLOR_ACCENT_LIGHT, COLOR_GOLD, COLOR_WARN],
            hover_name=SHORT_COL,
            hover_data=hover_cols,
            labels={
                "x": "Valor de mercado (millones €)",
                "potential_predicho": "Potencial predicho",
                "age": "Edad",
            },
            title="Valor de Mercado vs. Potencial Predicho",
            opacity=0.72,
        )
        fig = estilizar_fig(fig, height=560)
        fig.update_xaxes(title="Valor de mercado (millones €)")
        fig.update_yaxes(title="Potencial predicho")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------- TABLA JOYAS OCULTAS ----------------
    st.subheader(":material/diamond: Joyas Ocultas")
    st.caption("Jóvenes (≤23 años) con potencial predicho ≥ 80 y valor de mercado en el 25% más bajo del subconjunto filtrado.")

    def calcular_joyas(data, edad_tope, percentil, potencial_min):
        if data.empty:
            return data
        umbral_valor = data["value_euro"].quantile(percentil)
        return data[
            (data["potential_predicho"] >= potencial_min)
            & (data["value_euro"] <= umbral_valor)
            & (data["age"] <= edad_tope)
        ]

    criterios_relajados = []
    joyas = calcular_joyas(df_f, 23, 0.25, 80)

    if joyas.empty:
        joyas = calcular_joyas(df_f, 25, 0.25, 78)
        criterios_relajados.append("edad ≤ 25 y potencial ≥ 78")
    if joyas.empty:
        joyas = calcular_joyas(df_f, 25, 0.40, 75)
        criterios_relajados.append("percentil de valor 40% y potencial ≥ 75")
    if joyas.empty and not df_f.empty:
        joyas = df_f.sort_values("potential_predicho", ascending=False).head(20)
        criterios_relajados.append("sin restricción de edad/valor — se muestra el top 20 por potencial predicho")

    if criterios_relajados:
        st.info(
            "No se encontraron jugadores con los criterios estrictos de 'joya oculta'. "
            f"Se relajaron los filtros automáticamente ({criterios_relajados[-1]})."
        )

    joyas = joyas.sort_values("potential_predicho", ascending=False).head(20)

    if joyas.empty:
        st.warning("No hay jugadores para mostrar con los filtros actuales.")
    else:
        tabla = pd.DataFrame({
            "Nombre": joyas[SHORT_COL].values,
            "Nacionalidad": joyas["nationality"].values,
            "Club": joyas[CLUB_COL].values if CLUB_COL else "No disponible",
            "Edad": joyas["age"].values,
            "Overall": joyas["overall_rating"].values,
            "Potencial Predicho": joyas["potential_predicho"].values,
            "Valor (€)": [formatear_euros(v) for v in joyas["value_euro"].values],
        })
        st.dataframe(
            tabla, use_container_width=True, hide_index=True,
            column_config={
                "Overall": st.column_config.ProgressColumn("Overall", min_value=40, max_value=99, format="%d"),
                "Potencial Predicho": st.column_config.ProgressColumn(
                    "Potencial Predicho", min_value=40, max_value=99, format="%.1f"
                ),
            },
        )

# ============================================================
# PÁGINA 2 — PREDICTOR INDIVIDUAL
# ============================================================
elif pagina == ":material/bolt: Predictor Individual":
    hero("bolt", "Predictor Individual", "Ingresá los atributos de un jugador y obtené su potencial de crecimiento predicho.")

    if not modelo_disponible:
        st.error(
            "Esta página requiere el modelo entrenado. Colocá "
            f"`{MODEL_PATH}` en la carpeta de la app y recargá."
        )
        st.stop()

    rangos = df[FEATURES_BASE + FEATURES_TP1[:4]].agg(["min", "max"]).to_dict()

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            section_label("Atributos base")
            age = st.slider("Edad", int(rangos["age"]["min"]), int(rangos["age"]["max"]), 22)
            overall_rating = st.slider(
                "Overall rating", int(rangos["overall_rating"]["min"]), int(rangos["overall_rating"]["max"]), 70
            )
            vision = st.slider("Visión", int(rangos["vision"]["min"]), int(rangos["vision"]["max"]), 55)
            agility = st.slider("Agilidad", int(rangos["agility"]["min"]), int(rangos["agility"]["max"]), 65)
            standing_tackle = st.slider(
                "Entrada de pie (standing tackle)", int(rangos["standing_tackle"]["min"]),
                int(rangos["standing_tackle"]["max"]), 48
            )
            strength = st.slider("Fuerza", int(rangos["strength"]["min"]), int(rangos["strength"]["max"]), 65)

    with col2:
        with st.container(border=True):
            section_label("Reputación y scores de juego")
            international_reputation = st.slider("Reputación internacional (1-5)", 1, 5, 1)
            weak_foot = st.slider("Pie malo (1-5)", 1, 5, 3)
            skill_moves = st.slider("Habilidad de gambeta (1-5)", 1, 5, 2)
            attack_score = st.slider(
                "Índice de ataque", float(rangos["attack_score"]["min"]), float(rangos["attack_score"]["max"]), 48.0
            )
            defense_score = st.slider(
                "Índice de defensa", float(rangos["defense_score"]["min"]), float(rangos["defense_score"]["max"]), 48.0
            )

    with col3:
        with st.container(border=True):
            section_label("Contexto y posición")
            playmaking_score = st.slider(
                "Índice de juego asociado", float(df["playmaking_score"].min()), float(df["playmaking_score"].max()), 55.0
            )
            physical_score = st.slider(
                "Índice físico", float(df["physical_score"].min()), float(df["physical_score"].max()), 65.0
            )
            has_release_clause = st.selectbox("¿Tiene cláusula de rescisión?", ["Sí", "No"]) == "Sí"
            preferred_foot = st.selectbox("Pie preferido", ["Derecho", "Izquierdo"])
            nacionalidad_sel = st.selectbox("Nacionalidad", sorted(df["nationality"].unique()))
            posicion_sel = st.selectbox("Posición específica", POSICIONES)

    st.caption(
        "Nacionalidad y posición se ingresan por nombre: la app las codifica automáticamente "
        "en segundo plano para evitar combinaciones inválidas que el modelo nunca vio en entrenamiento."
    )
    with st.expander("Detalle técnico de la codificación"):
        st.caption(
            "`nationality_freq` = frecuencia real de esa nacionalidad en el dataset de entrenamiento. "
            "`pos_*` = codificación one-hot exclusiva de la posición seleccionada."
        )

    predecir = st.button("Predecir Potencial", type="primary", use_container_width=True)

    if predecir:
        nationality_freq = int(df["nationality"].value_counts().get(nacionalidad_sel, 1))
        fila = {
            "age": age, "overall_rating": overall_rating,
            "vision": vision, "agility": agility,
            "standing_tackle": standing_tackle, "strength": strength,
            "international_reputation(1-5)": international_reputation,
            "weak_foot(1-5)": weak_foot, "skill_moves(1-5)": skill_moves,
            "attack_score": attack_score, "defense_score": defense_score,
            "playmaking_score": playmaking_score, "physical_score": physical_score,
            "has_release_clause": int(has_release_clause),
            "preferred_foot_enc": 1 if preferred_foot == "Derecho" else 0,
            "nationality_freq": nationality_freq,
        }
        for pos in POSICIONES:
            fila[f"pos_{pos}"] = 1 if pos == posicion_sel else 0

        X_input = pd.DataFrame([fila])[ALL_FEATURES]

        try:
            potencial_pred = float(modelo.predict(X_input)[0])
        except Exception as e:
            st.error(f"No se pudo calcular la predicción: {e}")
            st.stop()

        st.session_state.n_predicciones += 1

        st.divider()
        with st.container(border=True):
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown(
                    f"<div style='color:{COLOR_GRAY}; font-size:0.85rem; font-weight:600; "
                    f"text-transform:uppercase; letter-spacing:.05em;'>Potencial predicho</div>"
                    f"<h1 style='color:{COLOR_TEXT}; font-family:Sora,sans-serif; font-size:4rem; margin:0;'>{potencial_pred:.1f}</h1>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span class='fifa-badge'>{clasificar_potencial(potencial_pred)}</span>",
                    unsafe_allow_html=True,
                )

            with res_col2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=potencial_pred,
                    domain={"x": [0, 1], "y": [0, 1]},
                    number={"font": {"color": COLOR_TEXT, "size": 40}},
                    gauge={
                        "axis": {"range": [50, 99], "tickcolor": COLOR_GRAY},
                        "bar": {"color": COLOR_ACCENT_LIGHT},
                        "bgcolor": COLOR_SURFACE,
                        "borderwidth": 0,
                        "steps": [
                            {"range": [50, 65], "color": COLOR_BORDER},
                            {"range": [65, 75], "color": "#3B4F73"},
                            {"range": [75, 82], "color": COLOR_CHART_NEUTRAL},
                            {"range": [82, 88], "color": COLOR_ACCENT},
                            {"range": [88, 99], "color": COLOR_GOLD},
                        ],
                    },
                ))
                gauge.update_layout(
                    height=260, margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor=COLOR_SURFACE, font_family="Inter, sans-serif",
                    font_color=COLOR_TEXT,
                )
                st.plotly_chart(gauge, use_container_width=True)

        # ---------------- SHAP — explicabilidad ----------------
        st.divider()
        st.subheader("¿Por qué el modelo predijo esto?")

        with st.spinner("Calculando explicabilidad…"):
            try:
                import shap

                scaler = modelo.named_steps["scaler"]
                arbol = modelo.named_steps["model"]
                X_scaled = scaler.transform(X_input)

                explainer = shap.TreeExplainer(arbol)
                shap_values = explainer.shap_values(X_scaled, check_additivity=False)

                explicacion = shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_input.values[0],
                    feature_names=ALL_FEATURES,
                )

                import matplotlib.pyplot as plt

                with st.container(border=True):
                    fig_shap, ax = plt.subplots(figsize=(9, 5))
                    fig_shap.patch.set_facecolor("white")
                    ax.set_facecolor("white")
                    shap.plots.waterfall(explicacion, max_display=10, show=False)
                    st.pyplot(fig_shap, use_container_width=True)
                    plt.close(fig_shap)

                top_feats = pd.Series(np.abs(shap_values[0]), index=ALL_FEATURES).sort_values(ascending=False)
                top3 = ", ".join(top_feats.head(3).index.tolist())
                st.markdown(f"Las variables que más influyeron en esta predicción son: **{top3}**.")
            except ImportError:
                st.warning("La librería `shap` no está instalada. Agregala a requirements.txt para ver esta sección.")
            except Exception as e:
                st.error(f"No se pudo calcular la explicabilidad SHAP: {e}")

# ============================================================
# PÁGINA 3 — ANÁLISIS EN LOTE
# ============================================================
elif pagina == ":material/folder_open: Análisis en Lote":
    hero("folder", "Análisis en Lote", "Subí un CSV con múltiples jugadores y obtené el potencial predicho para todos a la vez.")

    if not modelo_disponible:
        st.error(
            "Esta página requiere el modelo entrenado. Colocá "
            f"`{MODEL_PATH}` en la carpeta de la app y recargá."
        )
        st.stop()

    with st.expander(":material/checklist: Columnas requeridas en el CSV", expanded=False):
        st.write(f"El archivo debe contener exactamente estas **{len(ALL_FEATURES)} columnas**:")
        st.code(", ".join(ALL_FEATURES))

    plantilla = pd.DataFrame(columns=ALL_FEATURES)
    st.download_button(
        ":material/download: Descargar plantilla CSV",
        data=plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_fifa_scout.csv",
        mime="text/csv",
    )

    archivo = st.file_uploader("Subí tu CSV", type=["csv"])
    st.caption("Formato CSV · tamaño máximo 200 MB.")

    if archivo is not None:
        try:
            df_lote = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()

        faltantes = [c for c in ALL_FEATURES if c not in df_lote.columns]
        if faltantes:
            st.error(
                f"Al CSV le faltan {len(faltantes)} columna(s) requerida(s): "
                f"{', '.join(faltantes)}"
            )
        else:
            with st.spinner("Aplicando el modelo al lote…"):
                df_pred = df_lote.copy()
                for col in ALL_FEATURES:
                    df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

                filas_invalidas = df_pred[ALL_FEATURES].isnull().any(axis=1).sum()
                if filas_invalidas > 0:
                    st.warning(
                        f"{filas_invalidas} fila(s) tienen valores no numéricos o vacíos en alguna "
                        f"columna requerida y fueron excluidas del cálculo."
                    )
                df_validas = df_pred.dropna(subset=ALL_FEATURES).copy()

                if df_validas.empty:
                    st.error("Ninguna fila del CSV tiene datos válidos para predecir.")
                else:
                    df_validas["potential_predicho"] = np.round(
                        modelo.predict(df_validas[ALL_FEATURES]), 2
                    )
                    st.session_state.n_predicciones_lote += len(df_validas)

                    st.success(f"Predicciones calculadas para {len(df_validas):,} jugadores.", icon=":material/check_circle:")

                    m1, m2, m3, m4 = st.columns(4)
                    kpi_card(m1, "Jugadores procesados", f"{len(df_validas):,}")
                    kpi_card(m2, "Potencial promedio", f"{df_validas['potential_predicho'].mean():.1f}")
                    kpi_card(m3, "Potencial máximo", f"{df_validas['potential_predicho'].max():.1f}")
                    kpi_card(m4, "Potencial mínimo", f"{df_validas['potential_predicho'].min():.1f}")
                    st.write("")

                    st.dataframe(
                        df_validas, use_container_width=True, hide_index=True,
                        column_config={
                            "potential_predicho": st.column_config.ProgressColumn(
                                "Potencial Predicho", min_value=40, max_value=99, format="%.1f"
                            ),
                        },
                    )

                    st.download_button(
                        ":material/download: Descargar resultados",
                        data=df_validas.to_csv(index=False).encode("utf-8"),
                        file_name="predicciones_fifa_scout.csv",
                        mime="text/csv",
                    )

# ============================================================
# PÁGINA 4 — CLASIFICADOR DE POSICIÓN (placeholder TP3)
# ============================================================
elif pagina == ":material/adjust: Clasificador de Posición":
    hero("target", "Clasificador de Posición", "Predicción de la posición más adecuada para un jugador a partir de sus atributos.")

    modelo_tp3 = load_tp3_model()

    if modelo_tp3 is None:
        st.info("Esta funcionalidad está en desarrollo y se habilitará próximamente.", icon=":material/autorenew:")
        with st.expander("Detalle técnico (equipo de desarrollo)"):
            st.caption(
                f"Cuando el archivo `{TP3_MODEL_PATH}` esté disponible, colocalo en la carpeta "
                "de la app: la interfaz de clasificación (sliders → posición predicha) se activa "
                "automáticamente, sin modificar el código."
            )

        if modelo_disponible and POSITION_COL:
            st.subheader("Distribución de posiciones en el dataset")
            conteo = df[POSITION_COL].value_counts().reset_index()
            conteo.columns = ["Posición", "Cantidad"]
            fig = px.bar(
                conteo, x="Cantidad", y="Posición", orientation="h",
                color_discrete_sequence=[COLOR_CHART_NEUTRAL],
                title="Cantidad de jugadores por posición",
            )
            fig = estilizar_fig(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # El modelo TP3 ya está disponible: se arma la interfaz dinámicamente
        # a partir de las features que el propio modelo espera.
        st.success("Modelo de clasificación cargado correctamente.", icon=":material/check_circle:")

        if hasattr(modelo_tp3, "feature_names_in_"):
            features_tp3 = list(modelo_tp3.feature_names_in_)
        else:
            st.warning(
                "No se pudo determinar automáticamente el nombre de las variables del modelo; "
                "se usa el mismo conjunto de variables que el predictor de potencial como aproximación."
            )
            features_tp3 = ALL_FEATURES

        st.markdown("**Ingresá los atributos del jugador:**")
        cols = st.columns(3)
        entrada_tp3 = {}
        for i, feat in enumerate(features_tp3):
            with cols[i % 3]:
                if modelo_disponible and feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                    lo, hi = float(df[feat].min()), float(df[feat].max())
                    default = float(df[feat].median())
                    entrada_tp3[feat] = st.slider(feat, lo, hi, default)
                else:
                    entrada_tp3[feat] = st.slider(feat, 0.0, 100.0, 50.0)

        if st.button("Predecir Posición", type="primary"):
            X_tp3 = pd.DataFrame([entrada_tp3])[features_tp3]
            try:
                pred_pos = modelo_tp3.predict(X_tp3)[0]
                st.session_state.n_predicciones += 1
                st.markdown(f"### Posición predicha: **{pred_pos}**")

                if hasattr(modelo_tp3, "predict_proba"):
                    proba = modelo_tp3.predict_proba(X_tp3)[0]
                    clases = modelo_tp3.classes_
                    df_proba = pd.DataFrame({"Posición": clases, "Probabilidad": proba}).sort_values(
                        "Probabilidad", ascending=False
                    )
                    fig = px.bar(
                        df_proba, x="Probabilidad", y="Posición", orientation="h",
                        color_discrete_sequence=[COLOR_ACCENT_LIGHT],
                    )
                    fig = estilizar_fig(fig, height=350)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"No se pudo calcular la predicción: {e}")

# ============================================================
# PÁGINA 5 — SOBRE EL MODELO
# ============================================================
elif pagina == ":material/bar_chart: Sobre el Modelo":
    hero("bar-chart", "Sobre el Modelo", "Transparencia y metodología del modelo de predicción de potencial.")

    st.subheader("Comparación de modelos evaluados")
    tabla_modelos = COMPARATIVA_MODELOS.rename(columns={
        "R2_test": "R² Test", "MAE": "MAE", "RMSE": "RMSE", "Gap_%": "Gap train-test (%)",
    })
    st.dataframe(
        tabla_modelos, use_container_width=True, hide_index=True,
        column_config={
            "R² Test": st.column_config.ProgressColumn("R² Test", min_value=0.8, max_value=1.0, format="%.4f"),
        },
    )

    fig_comp = px.bar(
        COMPARATIVA_MODELOS.sort_values("R2_test"),
        x="R2_test", y="Modelo", orientation="h",
        color="R2_test", color_continuous_scale=[COLOR_GRAY, COLOR_ACCENT_LIGHT],
        title="R² en test por modelo evaluado",
        labels={"R2_test": "R² Test"},
    )
    fig_comp = estilizar_fig(fig_comp, height=380)
    fig_comp.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(
        f"""
        **¿Por qué Gradient Boosting Optimizado?** Con hiperparámetros
        `learning_rate={HIPERPARAMETROS_GB['learning_rate']}`,
        `max_depth={HIPERPARAMETROS_GB['max_depth']}`,
        `n_estimators={HIPERPARAMETROS_GB['n_estimators']}`,
        `subsample={HIPERPARAMETROS_GB['subsample']}` (hallados con `GridSearchCV`), el GB
        Optimizado obtuvo el mejor R² test (0.9355), el menor error (MAE 0.9973, RMSE 1.5402)
        y el segundo menor gap train-test (2.23%) de todos los modelos comparados — muy por
        debajo del Random Forest base (5.99%), lo que indica mejor capacidad de generalización
        sin sacrificar precisión.
        """
    )

    if modelo_disponible:
        st.divider()
        st.subheader("Importancia de variables (modelo en producción)")
        try:
            gb_step = modelo.named_steps["model"]
            importancias = pd.DataFrame({
                "Variable": ALL_FEATURES,
                "Importancia": gb_step.feature_importances_,
            }).sort_values("Importancia", ascending=False)

            fig_imp = px.bar(
                importancias, x="Importancia", y="Variable", orientation="h",
                color_discrete_sequence=[COLOR_CHART_NEUTRAL],
                title="Importancia de las 23 variables del modelo (Gini importance)",
            )
            fig_imp = estilizar_fig(fig_imp, height=650)
            fig_imp.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo calcular la importancia de variables: {e}")

        st.divider()
        st.subheader("Distribución de residuos")
        st.caption(
            "Calculada sobre el dataset completo (no sólo el holdout de test) porque el "
            "split de entrenamiento no se persiste junto al modelo serializado; se muestra "
            "a fines ilustrativos de la forma del error, no como métrica de evaluación "
            "(las métricas oficiales de test están en la tabla comparativa de arriba)."
        )
        residuos = df[TARGET] - df["potential_predicho"]
        fig_res = px.histogram(
            residuos, nbins=60, color_discrete_sequence=[COLOR_ACCENT_LIGHT],
            title="Distribución de residuos (Real − Predicho)",
            labels={"value": "Residuo"},
        )
        fig_res = estilizar_fig(fig_res, height=380)
        st.plotly_chart(fig_res, use_container_width=True)

        st.divider()
        st.subheader("Sobre los datos")
        d1, d2, d3 = st.columns(3)
        kpi_card(d1, "Total de jugadores", f"{len(df):,}")
        kpi_card(d2, "Rango de edad", f"{int(df['age'].min())}–{int(df['age'].max())} años")
        kpi_card(d3, "Nacionalidades distintas", f"{df['nationality'].nunique()}")
        st.write("")

        if POSITION_COL:
            conteo_pos = df[POSITION_COL].value_counts().reset_index()
            conteo_pos.columns = ["Posición", "Cantidad"]
            fig_pos = px.bar(
                conteo_pos, x="Cantidad", y="Posición", orientation="h",
                color_discrete_sequence=[COLOR_CHART_NEUTRAL],
                title="Distribución de posiciones en el dataset",
            )
            fig_pos = estilizar_fig(fig_pos, height=380)
            st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.warning(f"Cargá `{MODEL_PATH}` para ver importancia de variables y residuos en vivo.")
