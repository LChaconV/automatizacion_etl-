
import streamlit as st
from streamlit_folium import st_folium

from analytics import (
    calcular_correlacion_lags,
    calcular_kpis,
    correlacion_todos_municipios,
    detectar_anomalias,
    lag_optimo,
    proyectar_riesgo_prospectivo,
)
from charts import (
    anomalias_tabla,
    mapa_correlacion,
    mapa_precipitacion,
    riesgo_prospectivo,
    serie_temporal,
    serie_temporal_anomalia,
)
from data_loader import (
    agregar_nacional,
    cargar_catalogo_municipios,
    cargar_climatologia_referencia,
    cargar_oni,
    cargar_oni_prediccion,
    cargar_precipitacion,
    combinar_datos,
    nombre_a_codigo,
    obtener_nombres_municipios,
    obtener_rango_anios,
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ENSO · Precipitación Colombia",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
.kpi-card {
    background: var(--background-color);
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px 20px 12px;
    text-align: center;
}
.kpi-label {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 600;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}
.block-container { padding-top: 1.2rem; }
.stTabs [data-baseweb="tab"] { font-size: 13px; padding: 6px 16px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DE DATOS (cacheados)
# ─────────────────────────────────────────────

df_precip        = cargar_precipitacion()
df_oni           = cargar_oni()
df_combinado     = combinar_datos(df_precip, df_oni)
df_nacional      = agregar_nacional(df_precip)
df_nac_combinado = combinar_datos(df_nacional, df_oni)
df_clima_ref     = cargar_climatologia_referencia()
catalogo         = cargar_catalogo_municipios()
try:
    df_oni_pred = cargar_oni_prediccion()
except FileNotFoundError:
    df_oni_pred = None

nombres_municipios = obtener_nombres_municipios(df_precip, catalogo)
anio_min, anio_max = obtener_rango_anios(df_precip)

# ── Rango manual de años ──────────────────────────────────────────────────
# Ajusta estos valores si el dataset cambia de cobertura temporal.
ANIO_MIN_FIJO = 2004
ANIO_MAX_FIJO = 2023
anio_min = ANIO_MIN_FIJO
anio_max = ANIO_MAX_FIJO
# ─────────────────────────────────────────────────────────────────────────

NOMBRES_MESES_ES = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]
FASES_ONI = ["El Niño", "La Niña", "Neutral"]

# ─────────────────────────────────────────────
# SIDEBAR — FILTROS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌊 Filtros")
    st.divider()

    nombre_seleccionado = st.selectbox(
        "Municipio",
        options=nombres_municipios,
        index=0,
        help="Afecta todos los paneles.",
    )
    muni_seleccionado = nombre_a_codigo(nombre_seleccionado, catalogo) or "NACIONAL"
    es_nacional  = nombre_seleccionado == "Nacional"
    titulo_vista = "Colombia (Nacional)" if es_nacional else nombre_seleccionado

    st.divider()

    anio_inicio, anio_fin = st.slider(
        "Rango de años",
        min_value=anio_min,
        max_value=anio_max,
        value=(anio_min, anio_max),
        step=1,
        help="Afecta ambos mapas, serie temporal y tabla.",
    )

    st.divider()

    mes_nombre = st.selectbox(
        "Mes",
        options=NOMBRES_MESES_ES,
        index=3,
        help="Afecta mapa de precipitación y tabla de anomalías.",
    )
    mes_num = NOMBRES_MESES_ES.index(mes_nombre) + 1

    anio_mapa = st.selectbox(
        "Año (mapa precipitación)",
        options=list(range(anio_min, anio_max + 1)),
        index=len(range(anio_min, anio_max + 1)) - 1,
        help="Afecta solo el mapa de precipitación.",
    )

    st.divider()

    fases_sel = st.multiselect(
        "Fases ONI",
        options=FASES_ONI,
        default=FASES_ONI,
        help="Filtra serie temporal y tabla de anomalías.",
    )
    if not fases_sel:
        fases_sel = FASES_ONI

    st.divider()

    variable_mapa = st.selectbox(
        "Variable mapa correlación",
        options=[
            "Correlación ONI (lag óptimo)",
            "Correlación ONI (lag 0)",
            "Correlación ONI (lag 1)",
            "Correlación ONI (lag 2)",
        ],
        help="Qué variable se visualiza en el mapa de correlación ONI.",
    )

    st.divider()
    st.caption(
        "🛰️ Precipitación: CHIRPS\n"
        "🌊 ONI: NOAA\n"
        "🗺️ Geometría: DIVIPOLA · DANE\n"
        "📐 Anomalías: mediana (P50) ref. OMM 1991–2020\n"
        "Pipeline ETL · Laura A. Chacón V. · 2025"
    )

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────

st.markdown("## 🌧️ Dashboard ENSO · Precipitación Colombia")
st.markdown(
    f"Período **{anio_inicio}–{anio_fin}** &nbsp;·&nbsp; "
    f"Municipio: **{titulo_vista}** &nbsp;·&nbsp; "
    f"Fases: **{', '.join(fases_sel)}**",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────
# PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────

if es_nacional:
    df_base = df_nac_combinado.copy()
else:
    df_base = df_combinado[df_combinado["muni_code"] == muni_seleccionado].copy()

# Filtro rango de años
df_base_rango = df_base[
    (df_base["anio"] >= anio_inicio) & (df_base["anio"] <= anio_fin)
]

# Filtro fases ONI
df_base_fases = df_base_rango[df_base_rango["fase_oni"].isin(fases_sel)]

# Filtro mes para tabla
df_base_mes = df_base_fases[df_base_fases["mes"] == mes_num]

df_anomalias = detectar_anomalias(df_base_rango, df_clima_ref)
df_lags      = calcular_correlacion_lags(df_anomalias)
df_anomalias_fases = detectar_anomalias(df_base_fases, df_clima_ref) if not df_base_fases.empty else df_anomalias
df_anomalias_mes   = detectar_anomalias(df_base_mes,   df_clima_ref) if not df_base_mes.empty   else df_anomalias_fases
kpis      = calcular_kpis(df_base_rango, df_lags, df_anomalias, df_oni)
mejor_lag = lag_optimo(df_lags)


df_combinado_rango = df_combinado[
    (df_combinado["anio"] >= anio_inicio) & (df_combinado["anio"] <= anio_fin)
]
with st.spinner("Calculando correlaciones..."):
    df_corr_mapa = correlacion_todos_municipios(df_combinado_rango, df_clima_ref)

# Ajustar lag del mapa según variable seleccionada
lag_mapa = None  # None = lag óptimo automático
if variable_mapa == "Correlación ONI (lag 0)":
    lag_mapa = 0
elif variable_mapa == "Correlación ONI (lag 1)":
    lag_mapa = 1
elif variable_mapa == "Correlación ONI (lag 2)":
    lag_mapa = 2

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

PHASE_COLORS = {"El Niño": "#E24B4A", "La Niña": "#185FA5", "Neutral": "#6B9E6B"}
PHASE_ICONS  = {"El Niño": "🔴", "La Niña": "🔵", "Neutral": "🟢"}

fase_actual  = kpis["fase_oni_actual"]
fase_color   = PHASE_COLORS.get(fase_actual, "#888")
fase_icon    = PHASE_ICONS.get(fase_actual, "⚪")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>☁️ Precipitación promedio</div>"
        f"<div class='kpi-value'>{kpis['precip_promedio_mm']} mm</div>"
        f"<div class='kpi-sub'>Mensual · {anio_inicio}–{anio_fin}</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k2:
    oni_date = df_oni.sort_values("date").iloc[-1]["date"].strftime("%b %Y")
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>🌊 Índice ONI · {oni_date}</div>"
        f"<div class='kpi-value' style='color:{fase_color}'>{kpis['oni_actual']:+.2f}</div>"
        f"<div class='kpi-sub'>{fase_icon} {fase_actual}</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k3:
    if kpis["correlacion_optima"] is not None:
        lag_txt = f"Lag {kpis['lag_optimo_meses']} mes{'es' if kpis['lag_optimo_meses'] != 1 else ''}"
        corr_color = "#E24B4A" if kpis["correlacion_optima"] < 0 else "#185FA5"
        sig_txt = "p &lt; 0.05*" if mejor_lag.get("significativo") else "no significativo*"
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>↔️ Correlación ONI óptima</div>"
            f"<div class='kpi-value' style='color:{corr_color}'>"
            f"{kpis['correlacion_optima']:.3f}</div>"
            f"<div class='kpi-sub'>{lag_txt} · {sig_txt}</div>"
            f"</div>", unsafe_allow_html=True,
        )
        st.caption("*sobre anomalía deseasonalizada · p-valor corregido por autocorrelación")
    else:
        st.markdown(
            "<div class='kpi-card'>"
            "<div class='kpi-label'>↔️ Correlación ONI óptima</div>"
            "<div class='kpi-value'>—</div>"
            "<div class='kpi-sub'>Sin datos suficientes</div>"
            "</div>", unsafe_allow_html=True,
        )

with k4:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>⚠️ Anomalías detectadas</div>"
        f"<div class='kpi-value'>{kpis['n_anomalias']}</div>"
        f"<div class='kpi-sub'>Mediana (P50) ref. OMM 1991–2020 · P5/P95</div>"
        f"</div>", unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab_mapas, tab_serie, tab_tabla, tab_prospectivo = st.tabs([
    "🗺️  Mapas",
    "📈  Serie temporal",
    "📋  Anomalías",
    "🔮  Riesgo prospectivo",
])

# ════════════════════════════════════════════
# TAB 1 — MAPAS
# ════════════════════════════════════════════

with tab_mapas:

    col_mapa_prec, col_mapa_corr = st.columns(2)

    with col_mapa_prec:
        st.markdown(
            f"**Precipitación absoluta** "
            f"<span style='background:#FAECE7;color:#993C1D;padding:1px 8px;"
            f"border-radius:6px;font-size:10px'>Mes + Año</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"{mes_nombre} {anio_mapa} · mm")
        try:
            st_folium(
                mapa_precipitacion(
                    df_combinado=df_combinado,
                    mes=mes_num,
                    anio=anio_mapa,
                    muni_seleccionado=None if es_nacional else muni_seleccionado,
                ),
                width="100%", height=480, key="mapa_precip",
            )
        except ValueError as e:
            st.warning(str(e))

    with col_mapa_corr:
        st.markdown(
            f"**{variable_mapa}** "
            f"<span style='background:#E6F1FB;color:#185FA5;padding:1px 8px;"
            f"border-radius:6px;font-size:10px'>Rango años</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{anio_inicio}–{anio_fin} · todos los meses · "
            f"{'lag óptimo automático' if lag_mapa is None else f'lag {lag_mapa} fijo'} · "
            "atenuado = no significativo (FDR)"
        )

        # Si se seleccionó un lag fijo, recalcular con ese lag
        if lag_mapa is not None:
            df_corr_lag = correlacion_todos_municipios(
                df_combinado_rango, df_clima_ref, max_lag=lag_mapa
            )
        else:
            df_corr_lag = df_corr_mapa

        resultado_mapa = st_folium(
            mapa_correlacion(
                df_corr=df_corr_lag,
                muni_seleccionado=None if es_nacional else muni_seleccionado,
            ),
            width="100%", height=480, key="mapa_corr",
        )

        # Clic en mapa → sugerencia de municipio
        if resultado_mapa and resultado_mapa.get("last_active_drawing"):
            props = resultado_mapa["last_active_drawing"].get("properties", {})
            id_mun_clic = props.get("id_mun")
            if id_mun_clic:
                st.session_state["muni_desde_mapa"] = id_mun_clic
                st.rerun()

    if "muni_desde_mapa" in st.session_state:
        muni_desde_mapa = st.session_state.pop("muni_desde_mapa")
        if muni_desde_mapa != muni_seleccionado:
            st.info(
                f"Municipio seleccionado desde el mapa: **{muni_desde_mapa}**. "
                "Confirma en el selector del sidebar."
            )

    # Estadísticas rápidas debajo de los mapas
    st.divider()
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Municipios mapeados",  f"{df_corr_lag['muni_code'].nunique():,}")
    s2.metric("Corr. máxima (|r|)",   f"{df_corr_lag['correlacion'].abs().max():.3f}")
    s3.metric("Corr. mínima (|r|)",   f"{df_corr_lag['correlacion'].abs().min():.3f}")
    s4.metric("Corr. media (|r|)",    f"{df_corr_lag['correlacion'].abs().mean():.3f}")
    pct_significativo = df_corr_lag["significativo_fdr"].mean() * 100
    s5.metric(
        "Significativos (FDR)",
        f"{pct_significativo:.0f}%",
        help="% de municipios con relación ONI-precipitación significativa "
             "tras corregir por comparaciones múltiples (Benjamini-Hochberg).",
    )
    pct_concuerda = df_corr_lag["concuerda_spearman"].mean() * 100
    s6.metric(
        "Concuerda con Spearman",
        f"{pct_concuerda:.0f}%",
        help="% de municipios donde Pearson y Spearman coinciden en signo y "
             "significancia (FDR). Spearman no asume normalidad ni relación "
             "lineal — es un chequeo de robustez frente a la asimetría de la "
             "precipitación. Un % bajo indicaría que la conclusión depende "
             "del método elegido, no solo de los datos.",
    )

# ════════════════════════════════════════════
# TAB 2 — SERIE TEMPORAL
# ════════════════════════════════════════════

with tab_serie:

    st.caption(
        "Solo responde al municipio y al slider propio. "
        "No responde al filtro de mes."
    )

    # ── Rango manual del slider de serie temporal ──────────────────────
    # Ajusta estos valores si el dataset cambia de cobertura temporal.
    TS_ANIO_MIN = 1981
    TS_ANIO_MAX = 2026
    # ────────────────────────────────────────────────────────────────────

    ts_col1, ts_col2 = st.columns([4, 1])
    with ts_col1:
        ts_rango = st.slider(
            "Período",
            min_value=TS_ANIO_MIN,
            max_value=TS_ANIO_MAX,
            value=(TS_ANIO_MIN, TS_ANIO_MAX),
            step=1,
            label_visibility="collapsed",
        )
    with ts_col2:
        st.caption(f"{ts_rango[0]} – {ts_rango[1]}")

    df_ts = df_base[
        (df_base["anio"] >= ts_rango[0]) & (df_base["anio"] <= ts_rango[1])
    ]
    df_anomalias_ts = detectar_anomalias(df_ts, df_clima_ref)

    # Gráfico 1 — precipitación absoluta + ONI
    st.plotly_chart(
        serie_temporal(df_anomalias_ts, titulo=titulo_vista),
        use_container_width=True,
    )

    st.divider()

    # Gráfico 2 — anomalía de precipitación + ONI
    st.plotly_chart(
        serie_temporal_anomalia(df_anomalias_ts, titulo=titulo_vista),
        use_container_width=True,
    )

    # Interpretación automática del lag
    if mejor_lag:
        direccion = "negativa" if mejor_lag["correlacion"] < 0 else "positiva"
        efecto = (
            "cuando el ONI sube, la precipitación tiende a **bajar**"
            if mejor_lag["correlacion"] < 0
            else "cuando el ONI sube, la precipitación tiende a **subir**"
        )
        sig_txt = "significativa (p < 0.05)" if mejor_lag["significativo"] else "no significativa (p ≥ 0.05)"
        st.info(
            f"Correlación {direccion} en lag **{mejor_lag['lag']} meses** "
            f"({sig_txt}): {efecto}."
        )

# ════════════════════════════════════════════
# TAB 3 — TABLA DE ANOMALÍAS
# ════════════════════════════════════════════

with tab_tabla:

    st.markdown(
        f"**Anomalías históricas · {mes_nombre} · {anio_inicio}–{anio_fin} · {titulo_vista}**"
    )
    st.caption("Mediana (P50) ref. OMM 1991–2020 · P5/P95 por mes del año")

    if df_anomalias_mes.empty:
        st.warning(
            f"No hay datos para {mes_nombre} en el período {anio_inicio}–{anio_fin} "
            f"con las fases seleccionadas."
        )
    else:
        st.plotly_chart(
            anomalias_tabla(df_anomalias_mes),
            use_container_width=True,
        )

        # Resumen rápido
        n_def = (df_anomalias_mes["anomalia"] == "déficit").sum()
        n_exc = (df_anomalias_mes["anomalia"] == "exceso").sum()
        n_nor = (df_anomalias_mes["anomalia"] == "normal").sum()

        r1, r2, r3 = st.columns(3)
        r1.metric("Meses en déficit",  n_def,
                  help="Precipitación < P5 de referencia OMM 1991–2020")
        r2.metric("Meses en exceso",   n_exc,
                  help="Precipitación > P95 de referencia OMM 1991–2020")
        r3.metric("Meses normales",    n_nor,
                  help="Precipitación dentro del rango P5–P95")

# ════════════════════════════════════════════
# TAB 4 — RIESGO PROSPECTIVO (ONI PREDICHO)
# ════════════════════════════════════════════

with tab_prospectivo:

    st.markdown(f"**Riesgo prospectivo · {titulo_vista}**")
    st.caption(
        "Pronóstico ONI (promedio del ensamble CPC/IRI) aplicado hacia adelante "
        "usando la sensibilidad histórica ya validada del municipio (lag óptimo "
        "deseasonalizado). No responde a los filtros de mes, año ni fases ONI — "
        "solo al municipio seleccionado."
    )

    if df_oni_pred is None or df_oni_pred.empty:
        st.warning(
            "No se encontró el pronóstico ONI en "
            "`data/processed/noaa_prediction/`. Ejecuta el pipeline de "
            "extracción de pronóstico (extract_noaa_prediction.py) primero."
        )
    else:
        fecha_referencia = df_oni["date"].max()
        df_riesgo = proyectar_riesgo_prospectivo(df_oni_pred, mejor_lag, fecha_referencia)

        if df_riesgo.empty:
            st.info("No hay datos de pronóstico ONI disponibles para proyectar.")
        else:
            if df_riesgo["pronostico_vencido"].iloc[0]:
                st.error(
                    f"⚠️ **Pronóstico vencido**: el último pronóstico ONI disponible "
                    f"no cubre ningún período posterior al último ONI observado "
                    f"({fecha_referencia.strftime('%b %Y')}). Se muestra de todas formas "
                    "la última vigencia cargada, pero **no representa el riesgo actual** — "
                    "hay que volver a correr el ETL de pronóstico "
                    "(`extract_noaa_prediction.py`) para refrescarlo."
                )

            st.plotly_chart(
                riesgo_prospectivo(df_riesgo, titulo=titulo_vista),
                use_container_width=True,
            )

            st.caption(
                f"Último ONI observado: **{fecha_referencia.strftime('%b %Y')}** "
                "(ancla usada para estimar el horizonte de cada pronóstico). "
                "⚠️ El ETL de pronóstico no conserva la fecha real de emisión de "
                "cada predicción, por lo que el horizonte (lead time) y, por tanto, "
                "el nivel de confianza mostrado son una **aproximación**, no un dato exacto. "
                "La confianza por horizonte es una categorización cualitativa basada en "
                "la habilidad de pronóstico de ENOS reportada por CPC/IRI (alta 1–3 meses, "
                "moderada 4–6, baja más allá de 6 meses), no el spread real del ensamble "
                "de modelos."
            )

            if not mejor_lag.get("significativo", False):
                st.warning(
                    "La relación histórica ONI ↔ precipitación de este municipio "
                    "**no es significativa** tras la corrección por autocorrelación. "
                    "La anomalía proyectada en el panel inferior debe tratarse con "
                    "baja confiabilidad."
                )
