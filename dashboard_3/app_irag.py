"""
app_irag.py
-----------
Dashboard Streamlit: Análisis Espacial IRAG Inusitada
Clusters LISA e Índice de Moran Global

Componentes:
    Sidebar  → variable (ONI / precipitación) y nivel α
    Fila 1   → KPIs: Moran I, p-valor, z-score, clusters significativos
    Fila 2   → Mapa LISA (Folium)
    Fila 3   → Tabla top municipios HH y LL
    Fila 4   → Interpretación automática

Ejecución:
    streamlit run app_irag.py
"""

import geopandas as gpd
import libpysal
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
from streamlit_folium import st_folium
import folium

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="IRAG · Análisis Espacial",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

DIVIPOLA_PATH    = (
    r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon"
    r"\ETL_code\dashboard_2\divipola\divipola.parquet"
)
CORR_ONI_PATH    = (
    r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon"
    r"\ETL_code\dashboard_3\processed\join\corr_oni_irag.parquet"
)
CORR_PRECIP_PATH = (
    r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon"
    r"\ETL_code\dashboard_3\processed\join\corr_precip_irag.parquet"
)

COLORES_LISA = {
    "HH": "#E24B4A",
    "LL": "#185FA5",
    "HL": "#F4A460",
    "LH": "#87CEEB",
    "NS": "#DDDDDD",
}

ETIQUETAS_LISA = {
    "HH": "Alto-Alto (hotspot)",
    "LL": "Bajo-Bajo (coldspot)",
    "HL": "Alto-Bajo (outlier)",
    "LH": "Bajo-Alto (outlier)",
    "NS": "No significativo",
}

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
    font-size: 11px; color: #888;
    text-transform: uppercase;
    letter-spacing: .05em; margin-bottom: 4px;
}
.kpi-value  { font-size: 26px; font-weight: 600; line-height: 1.1; }
.kpi-sub    { font-size: 12px; color: #666; margin-top: 4px; }
.insight-box {
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14px;
}
.block-container { padding-top: 1.2rem; }
.stTabs [data-baseweb="tab"] { font-size: 13px; padding: 6px 16px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Cargando geometrías...")
def cargar_geodataframe() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(DIVIPOLA_PATH)
    gdf["id_mun"] = gdf["id_mun"].astype(str).str.zfill(5)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


@st.cache_data(show_spinner="Cargando correlaciones...")
def cargar_correlaciones() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_oni    = pd.read_parquet(CORR_ONI_PATH)
    df_precip = pd.read_parquet(CORR_PRECIP_PATH)
    df_oni["muni_code"]    = df_oni["muni_code"].astype(str).str.zfill(5)
    df_precip["muni_code"] = df_precip["muni_code"].astype(str).str.zfill(5)
    return df_oni, df_precip


@st.cache_data(show_spinner="Calculando Moran y LISA (primera carga)...")
def calcular_lisa(
    _gdf: gpd.GeoDataFrame,
    col: str,
    alpha: float,
) -> tuple[gpd.GeoDataFrame, Moran]:
    """
    Calcula índice de Moran global y LISA local.
    Retorna GeoDataFrame con clusters y objeto Moran.
    """
    gdf_v = _gdf[_gdf[col].notna()].copy().reset_index(drop=True)

    w = Queen.from_dataframe(gdf_v, silence_warnings=True, use_index=False)
    w.transform = "r"

    moran      = Moran(gdf_v[col], w)
    moran_local = Moran_Local(gdf_v[col], w, seed=42)

    media   = gdf_v[col].mean()
    lag_esp = libpysal.weights.lag_spatial(w, gdf_v[col].values)

    clusters = []
    for i in range(len(gdf_v)):
        if moran_local.p_sim[i] >= alpha:
            clusters.append("NS")
        elif gdf_v[col].iloc[i] >= media and lag_esp[i] >= media:
            clusters.append("HH")
        elif gdf_v[col].iloc[i] < media and lag_esp[i] < media:
            clusters.append("LL")
        elif gdf_v[col].iloc[i] >= media and lag_esp[i] < media:
            clusters.append("HL")
        else:
            clusters.append("LH")

    gdf_v["lisa_cluster"] = clusters
    gdf_v["lisa_p"]       = moran_local.p_sim
    gdf_v["lisa_I"]       = moran_local.Is

    return gdf_v, moran


# ─────────────────────────────────────────────
# MAPA FOLIUM
# ─────────────────────────────────────────────

def construir_mapa_lisa(
    gdf_base: gpd.GeoDataFrame,
    gdf_lisa: gpd.GeoDataFrame,
    col_id: str = "id_mun",
) -> folium.Map:
    """Construye mapa Folium con clusters LISA coloreados."""

    # Unir clusters al GDF base
    gdf_mapa = gdf_base.merge(
        gdf_lisa[[col_id, "lisa_cluster", "lisa_p", "lisa_I", "corr"]],
        on=col_id, how="left"
    )
    gdf_mapa["lisa_cluster"] = gdf_mapa["lisa_cluster"].fillna("NS")
    gdf_mapa["color"]        = gdf_mapa["lisa_cluster"].map(COLORES_LISA)

    geojson_str = gdf_mapa.to_json()

    mapa = folium.Map(
        location=[4.5, -74.0],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    folium.GeoJson(
        geojson_str,
        style_function=lambda feature: {
            "fillColor":   feature["properties"].get("color") or "#DDDDDD",
            "color":       "white",
            "weight":      0.3,
            "fillOpacity": 0.8,
        },
        highlight_function=lambda feature: {
            "color": "#333333",
            "weight": 1.5,
            "fillOpacity": 0.95,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["name_mun", "lisa_cluster", "corr", "lisa_p"],
            aliases=["Municipio:", "Cluster:", "Correlación:", "p-valor LISA:"],
            localize=True,
        ),
    ).add_to(mapa)

    # Leyenda CSS
    leyenda_html = """
    <div style='position:fixed;bottom:50px;right:10px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;line-height:1.8'>
        <b>Clusters LISA</b><br>
    """
    for cat, color in COLORES_LISA.items():
        etiqueta = ETIQUETAS_LISA[cat]
        leyenda_html += (
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{color};border-radius:2px;margin-right:6px'></span>"
            f"{etiqueta}<br>"
        )
    leyenda_html += "</div>"
    mapa.get_root().html.add_child(folium.Element(leyenda_html))

    return mapa


# ─────────────────────────────────────────────
# INTERPRETACIÓN AUTOMÁTICA
# ─────────────────────────────────────────────

def generar_interpretacion(
    moran: Moran,
    gdf_lisa: gpd.GeoDataFrame,
    variable: str,
    alpha: float,
) -> list[dict]:
    """Genera textos de interpretación automática basados en los resultados."""

    insights = []
    conteo = gdf_lisa["lisa_cluster"].value_counts()
    n_hh   = conteo.get("HH", 0)
    n_ll   = conteo.get("LL", 0)
    n_hl   = conteo.get("HL", 0)
    n_lh   = conteo.get("LH", 0)
    n_sig  = n_hh + n_ll + n_hl + n_lh

    var_label = "el ONI" if variable == "ONI" else "la precipitación"

    # Moran global
    sig_txt = "significativa" if moran.p_sim < alpha else "no significativa"
    insights.append({
        "color": "#6B3A9E",
        "titulo": "Autocorrelación espacial global",
        "texto": (
            f"El índice de Moran I = {moran.I:.4f} indica autocorrelación espacial "
            f"positiva {sig_txt} (p = {moran.p_sim:.4f}, z = {moran.z_sim:.2f}). "
            f"Los municipios vecinos tienden a compartir patrones similares en la "
            f"relación entre {var_label} y los casos de IRAG inusitada."
        ),
    })

    # Hotspots
    if n_hh > 0:
        insights.append({
            "color": "#E24B4A",
            "titulo": f"Hotspots (Alto-Alto): {n_hh} municipios",
            "texto": (
                f"Hay {n_hh} municipios con correlación positiva alta entre {var_label} "
                f"y los casos de IRAG, rodeados de vecinos con el mismo patrón. "
                f"Estas zonas son las más sensibles al efecto de La Niña sobre las "
                f"enfermedades respiratorias graves."
            ),
        })

    # Coldspots
    if n_ll > 0:
        insights.append({
            "color": "#185FA5",
            "titulo": f"Coldspots (Bajo-Bajo): {n_ll} municipios",
            "texto": (
                f"Hay {n_ll} municipios con correlación baja entre {var_label} y los "
                f"casos de IRAG, rodeados de vecinos igualmente insensibles al clima. "
                f"En estas zonas otros factores no climáticos dominan la dinámica "
                f"de la enfermedad."
            ),
        })

    # Outliers
    if n_hl + n_lh > 0:
        insights.append({
            "color": "#BA7517",
            "titulo": f"Outliers espaciales: {n_hl + n_lh} municipios",
            "texto": (
                f"Hay {n_hl} municipios Alto-Bajo y {n_lh} municipios Bajo-Alto. "
                f"Estos municipios tienen un comportamiento distinto al de sus vecinos, "
                f"lo que puede deberse a características locales como altitud, densidad "
                f"poblacional o calidad del reporte al SIVIGILA."
            ),
        })

    # Resumen
    insights.append({
        "color": "#2E7D32",
        "titulo": "Resumen del análisis",
        "texto": (
            f"De los {len(gdf_lisa):,} municipios analizados, {n_sig} ({n_sig/len(gdf_lisa)*100:.1f}%) "
            f"muestran clustering espacial significativo en la relación {var_label} → IRAG. "
            f"El patrón no es aleatorio en el territorio colombiano (Moran p = {moran.p_sim:.4f}), "
            f"lo que sugiere que factores climáticos regionales median la distribución "
            f"espacial de la IRAG inusitada."
        ),
    })

    return insights


# ─────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────

# Carga
gdf          = cargar_geodataframe()
df_oni, df_precip = cargar_correlaciones()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 IRAG · Análisis Espacial")
    st.divider()

    variable = st.selectbox(
        "Variable climática",
        options=["ONI", "Precipitación"],
        help="Variable cuya correlación con IRAG se analiza espacialmente.",
    )

    alpha = st.selectbox(
        "Nivel de significancia (α)",
        options=[0.05, 0.01, 0.10],
        index=0,
        help="Umbral para clasificar clusters LISA como significativos.",
    )

    st.divider()
    st.caption(
        "Fuentes: SIVIGILA · CHIRPS · NOAA ONI\n"
        "Geometría: DIVIPOLA · DANE\n"
        "Método: Moran Local (LISA) · Queen contiguity\n"
        "Período: 2018–2024"
    )

# Preparar datos según variable seleccionada
if variable == "ONI":
    df_corr  = df_oni.copy()
    col_corr = "corr_oni"
    col_label = "ONI → IRAG"
else:
    df_corr  = df_precip.copy()
    col_corr = "corr_precip"
    col_label = "Precipitación → IRAG"

# Unir correlaciones a geometría
df_corr_rename = df_corr.rename(columns={"corr": col_corr})
gdf_joined = gdf.merge(
    df_corr_rename[["muni_code", col_corr, "lag", "p_valor", "sig"]],
    left_on="id_mun", right_on="muni_code", how="left"
)
gdf_joined["corr"] = gdf_joined[col_corr]

# Calcular LISA
with st.spinner("Calculando LISA..."):
    gdf_lisa, moran = calcular_lisa(gdf_joined, col_corr, alpha)

# ── Encabezado ────────────────────────────────────────────────────────────
st.markdown(f"## 🫁 IRAG Inusitada · Análisis Espacial LISA")
st.markdown(
    f"Variable: **{col_label}** &nbsp;·&nbsp; "
    f"α = **{alpha}** &nbsp;·&nbsp; "
    f"Período: **2018–2024** &nbsp;·&nbsp; "
    f"Municipios: **{len(gdf_lisa):,}**"
)
st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────
conteo = gdf_lisa["lisa_cluster"].value_counts()
n_hh   = conteo.get("HH", 0)
n_ll   = conteo.get("LL", 0)
n_sig  = conteo.drop("NS", errors="ignore").sum()

sig_color = "#2E7D32" if moran.p_sim < alpha else "#888"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>📐 Índice de Moran I</div>"
        f"<div class='kpi-value' style='color:{sig_color}'>{moran.I:.4f}</div>"
        f"<div class='kpi-sub'>{'✓ Significativo' if moran.p_sim < alpha else '✗ No significativo'}</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>📊 p-valor · z-score</div>"
        f"<div class='kpi-value'>{moran.p_sim:.4f}</div>"
        f"<div class='kpi-sub'>z = {moran.z_sim:.2f}</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>🔴 Hotspots (HH)</div>"
        f"<div class='kpi-value' style='color:#E24B4A'>{n_hh}</div>"
        f"<div class='kpi-sub'>municipios · α={alpha}</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>🔵 Coldspots (LL)</div>"
        f"<div class='kpi-value' style='color:#185FA5'>{n_ll}</div>"
        f"<div class='kpi-sub'>municipios · α={alpha}</div>"
        f"</div>", unsafe_allow_html=True,
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_mapa, tab_tabla, tab_interpretacion = st.tabs([
    "🗺️  Mapa LISA",
    "📋  Top municipios",
    "💡  Interpretación",
])

# ── Tab 1: Mapa LISA ──────────────────────────────────────────────────────
with tab_mapa:
    st.caption(
        f"Clusters LISA · {col_label} · α={alpha} · "
        f"Contigüidad Queen · Row-standardized weights"
    )
    mapa = construir_mapa_lisa(gdf, gdf_lisa)
    st_folium(mapa, width="100%", height=560, returned_objects=[])

# ── Tab 2: Tabla top municipios ───────────────────────────────────────────
with tab_tabla:

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 🔴 Top 10 Hotspots (HH)")
        st.caption("Municipios con correlación positiva alta y vecinos similares")
        top_hh = (
            gdf_lisa[gdf_lisa["lisa_cluster"] == "HH"]
            [["id_mun", "name_mun", "corr", "lag", "p_valor", "lisa_p"]]
            .sort_values("corr", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_hh.columns = ["Código", "Municipio", "Correlación", "Lag", "p corr.", "p LISA"]
        top_hh.index += 1
        st.dataframe(top_hh, use_container_width=True)

    with col_t2:
        st.markdown("#### 🔵 Top 10 Coldspots (LL)")
        st.caption("Municipios con correlación baja y vecinos igualmente insensibles")
        top_ll = (
            gdf_lisa[gdf_lisa["lisa_cluster"] == "LL"]
            [["id_mun", "name_mun", "corr", "lag", "p_valor", "lisa_p"]]
            .sort_values("corr")
            .head(10)
            .reset_index(drop=True)
        )
        top_ll.columns = ["Código", "Municipio", "Correlación", "Lag", "p corr.", "p LISA"]
        top_ll.index += 1
        st.dataframe(top_ll, use_container_width=True)

    st.divider()
    st.markdown("#### Distribución completa de clusters")
    dist = (
        gdf_lisa["lisa_cluster"]
        .value_counts()
        .reset_index()
        .rename(columns={"lisa_cluster": "Cluster", "count": "Municipios"})
    )
    dist["Descripción"] = dist["Cluster"].map(ETIQUETAS_LISA)
    dist["% del total"] = (dist["Municipios"] / len(gdf_lisa) * 100).round(1)
    st.dataframe(dist, use_container_width=True, hide_index=True)

# ── Tab 3: Interpretación ─────────────────────────────────────────────────
with tab_interpretacion:

    st.markdown(
        f"### Análisis de resultados · {col_label}\n"
        f"*Generado automáticamente a partir de los resultados del análisis espacial*"
    )
    st.divider()

    insights = generar_interpretacion(moran, gdf_lisa, variable, alpha)

    for ins in insights:
        _c = ins["color"]
        st.markdown(
            f"<div class='insight-box' style='border-color:{_c};background:#fafafa'>"
            f"<span style='font-weight:600;color:{_c}'>{ins['titulo']}</span><br>"
            f"{ins['texto']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "Nota metodológica: el análisis usa correlación de Spearman con lag óptimo 0–6 meses "
        "calculada sobre datos mensuales 2018–2024. Los clusters LISA usan contigüidad Queen "
        "con pesos estandarizados por fila (row-standardized). "
        "La significancia se evalúa con 999 permutaciones aleatorias."
    )
