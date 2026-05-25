# ============================================================
#  app.py  ·  Dashboard Ejecutivo ENSO y Precipitación · Colombia
#  Laura Andrea Chacón Velásquez · Tesis Maestría Ciencia de Datos
#  Escuela Colombiana de Ingeniería Julio Garavito · 2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
import json
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from plotly.subplots import make_subplots


from data_loader import build_dataset, CHIRPS_YEARS_AVAILABLE

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ENSO · Colombia",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paleta y constantes ──────────────────────────────────────────────────────
PHASE_COLORS  = {"El Niño": "#E24B4A", "La Niña": "#185FA5", "Neutro": "#6B9E6B"}
PHASE_ICONS   = {"El Niño": "🔴", "La Niña": "🔵", "Neutro": "🟢"}
RISK_CFG = {
    "Alto — Sequía":     {"color": "#E24B4A", "bg": "#FCEBEB", "icon": "🔴"},
    "Alto — Exceso":     {"color": "#185FA5", "bg": "#E6F1FB", "icon": "🔵"},
    "Medio":             {"color": "#BA7517", "bg": "#FAEEDA", "icon": "🟡"},
    "Bajo":              {"color": "#2E7D32", "bg": "#E8F5E9", "icon": "🟢"},
}

REGIONES = {
    "Caribe":     ["08","13","20","23","44","47","70"],
    "Andina":     ["05","15","17","19","25","41","52","63","66","68","73","76","11"],
    "Pacífico":   ["27"],
    "Orinoquía":  ["81","85","50","99"],
    "Amazonia":   ["18","91","94","95","86","97"],
}
MUNI_TO_REGION = {
    muni[:2]: region
    for region, depts in REGIONES.items()
    for muni in depts
}

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background: var(--background-color);
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px 20px 12px;
    text-align: center;
}
.kpi-label  { font-size: 11px; color: #888; text-transform: uppercase;
              letter-spacing: .05em; margin-bottom: 4px; }
.kpi-value  { font-size: 28px; font-weight: 600; line-height: 1.1; }
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


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 Filtros ENSO")
    st.divider()

    yr_min = min(CHIRPS_YEARS_AVAILABLE)
    yr_max = max(CHIRPS_YEARS_AVAILABLE)
    year_range = st.slider("Período de análisis", yr_min, yr_max,
                           (max(yr_min, 2010), yr_max))

    fase_sel = st.multiselect(
        "Fases ONI",
        ["El Niño", "La Niña", "Neutro"],
        default=["El Niño", "La Niña", "Neutro"],
    )

    region_sel = st.multiselect(
        "Regiones",
        list(REGIONES.keys()),
        default=list(REGIONES.keys()),
    )

    variable_mapa = st.selectbox(
        "Variable en el mapa",
        options=[
            "Correlación ONI",
            "Anomalía media (mm)",
            "Anomalía en El Niño (mm)",
            "Anomalía en La Niña (mm)",
        ],
    )

    st.divider()
    st.caption("🛰️ Precipitación: CHIRPS")
    st.caption("🌊 ONI histórico: NOAA")
    st.caption("📡 Predicción: IRI")
    st.caption("🗺️ Geometría: DIVIPOLA · DANE")
    st.divider()
    st.caption("Pipeline ETL · Laura A. Chacón V. · 2025")


# ════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════
data = build_dataset(year_range[0], year_range[1])

chirps   : pd.DataFrame     = data["chirps"]
oni_hist : pd.DataFrame     = data["oni_hist"]
oni_pred : pd.DataFrame     = data["oni_pred"]
divipola                    = data["divipola"]
corr_muni: pd.DataFrame     = data["corr_muni"]

# Agregar región
if "id_dept" not in chirps.columns:
    chirps["id_dept"] = chirps["muni_code"].astype(str).str.zfill(5).str[:2]
chirps["region"] = chirps["id_dept"].map(MUNI_TO_REGION).fillna("Otra")

# Filtros activos
depts_sel = [d for r in region_sel for d in REGIONES.get(r, [])]
mask = (
    chirps["oni_phase"].isin(fase_sel) &
    chirps["id_dept"].isin(depts_sel)
)
cf = chirps[mask].copy()    # chirps filtrado

# ONI más reciente disponible
oni_actual_row = oni_hist.sort_values("date").iloc[-1]
oni_val        = oni_actual_row["value_oni"]
oni_fase       = oni_actual_row["oni_phase"]
oni_date       = oni_actual_row["date"].strftime("%b %Y")


# ════════════════════════════════════════════════════════════════════════════
#  ENCABEZADO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 🌧️ Dashboard Ejecutivo ENSO · Colombia")
st.markdown(
    f"Período **{year_range[0]}–{year_range[1]}** &nbsp;·&nbsp; "
    f"Fases: **{', '.join(fase_sel)}** &nbsp;·&nbsp; "
    f"Regiones: **{', '.join(region_sel) if region_sel else 'Ninguna'}**",
    unsafe_allow_html=True,
)
st.divider()


# ════════════════════════════════════════════════════════════════════════════
#  KPIs
# ════════════════════════════════════════════════════════════════════════════
last_month_data = chirps[chirps["date"] == chirps["date"].max()]
n_deficit   = (last_month_data["anomaly_mm"] < -10).sum()
n_exceso    = (last_month_data["anomaly_mm"] >  10).sum()
n_total_mun = last_month_data["muni_code"].nunique()
pct_deficit = n_deficit / n_total_mun * 100 if n_total_mun > 0 else 0
pct_exceso  = n_exceso  / n_total_mun * 100 if n_total_mun > 0 else 0

# Riesgo dominante
if oni_fase == "El Niño" and pct_deficit > 40:
    riesgo_dom = "Alto — Sequía"
elif oni_fase == "La Niña" and pct_exceso > 40:
    riesgo_dom = "Alto — Exceso"
elif abs(oni_val) >= 0.5:
    riesgo_dom = "Medio"
else:
    riesgo_dom = "Bajo"

risk_cfg = RISK_CFG[riesgo_dom]

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    fase_color = PHASE_COLORS[oni_fase]
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>Índice ONI · {oni_date}</div>"
        f"<div class='kpi-value' style='color:{fase_color}'>{oni_val:+.2f}</div>"
        f"<div class='kpi-sub'>{PHASE_ICONS[oni_fase]} {oni_fase}</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>Municipios en déficit</div>"
        f"<div class='kpi-value' style='color:#E24B4A'>{n_deficit:,}</div>"
        f"<div class='kpi-sub'>{pct_deficit:.0f}% del total · &lt; −10 mm</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>Municipios en exceso</div>"
        f"<div class='kpi-value' style='color:#185FA5'>{n_exceso:,}</div>"
        f"<div class='kpi-sub'>{pct_exceso:.0f}% del total · &gt; +10 mm</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k4:
    anom_media = cf["anomaly_mm"].mean()
    anom_color = "#E24B4A" if anom_media < 0 else "#185FA5"
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>Anomalía media (período)</div>"
        f"<div class='kpi-value' style='color:{anom_color}'>{anom_media:+.1f} mm</div>"
        f"<div class='kpi-sub'>{'Déficit' if anom_media < 0 else 'Exceso'} hídrico</div>"
        f"</div>", unsafe_allow_html=True,
    )

with k5:
    _rc = risk_cfg["color"]
    _ri = risk_cfg["icon"]
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>Riesgo climático</div>"
        f"<div class='kpi-value' style='color:{_rc}'>"
        f"{_ri}</div>"
        f"<div class='kpi-sub' style='color:{_rc};font-weight:600'>"
        f"{riesgo_dom}</div>"
        f"</div>", unsafe_allow_html=True,
    )

st.divider()


# ════════════════════════════════════════════════════════════════════════════
#  TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════════════════
tab_mapa, tab_serie, tab_semaforo, tab_pred, tab_insights, tab_prec = st.tabs([
    "🌧️  Precipitación",
    "🗺️  Mapa ENSO",
    "📈  Serie temporal",
    "🚦  Semáforo de riesgo",
    "🔮  Proyección y alertas",
    "💡  Insights climáticos",
    
])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — MAPA COROPLÉTICO
# ════════════════════════════════════════════════════════════════════════════
with tab_mapa:

    # ── Calcular variable seleccionada por municipio ──────────────────────
    if variable_mapa == "Correlación ONI":
        muni_vals = corr_muni[["muni_code", "corr_oni"]].copy()
        muni_vals.columns = ["muni_code", "valor"]
        color_scale  = "RdBu"          # rojo = correlación negativa (El Niño = sequía)
        color_midpoint = 0
        legend_label = "Correlación ONI"
        range_color  = [-1, 1]
        nota = (
            "🔴 **Rojo**: el municipio tiende a tener **menos lluvia** cuando sube el ONI (sensible a El Niño).  "
            "🔵 **Azul**: el municipio tiende a tener **más lluvia** cuando sube el ONI."
        )

    elif variable_mapa == "Anomalía media (mm)":
        muni_vals = (
            cf.groupby("muni_code")["anomaly_mm"]
            .mean().round(1).reset_index()
        )
        muni_vals.columns = ["muni_code", "valor"]
        color_scale  = "RdBu"
        color_midpoint = 0
        legend_label = "Anomalía (mm)"
        lim = max(abs(muni_vals["valor"].quantile(0.05)),
                  abs(muni_vals["valor"].quantile(0.95)))
        range_color = [-lim, lim]
        nota = (
            "🔴 **Rojo**: déficit de lluvia en el período seleccionado.  "
            "🔵 **Azul**: exceso de lluvia."
        )

    elif variable_mapa == "Anomalía en El Niño (mm)":
        muni_vals = (
            chirps[chirps["oni_phase"] == "El Niño"]
            .groupby("muni_code")["anomaly_mm"]
            .mean().round(1).reset_index()
        )
        muni_vals.columns = ["muni_code", "valor"]
        color_scale  = "RdBu"
        color_midpoint = 0
        legend_label = "Anomalía El Niño (mm)"
        lim = max(abs(muni_vals["valor"].quantile(0.05)),
                  abs(muni_vals["valor"].quantile(0.95)))
        range_color = [-lim, lim]
        nota = (
            "Anomalía media de precipitación **durante meses de El Niño** (ONI ≥ +0.5).  "
            "🔴 Rojo = déficit hídrico típico en El Niño."
        )

    else:  # Anomalía en La Niña
        muni_vals = (
            chirps[chirps["oni_phase"] == "La Niña"]
            .groupby("muni_code")["anomaly_mm"]
            .mean().round(1).reset_index()
        )
        muni_vals.columns = ["muni_code", "valor"]
        color_scale  = "RdBu"
        color_midpoint = 0
        legend_label = "Anomalía La Niña (mm)"
        lim = max(abs(muni_vals["valor"].quantile(0.05)),
                  abs(muni_vals["valor"].quantile(0.95)))
        range_color = [-lim, lim]
        nota = (
            "Anomalía media de precipitación **durante meses de La Niña** (ONI ≤ −0.5).  "
            "🔵 Azul = exceso hídrico típico en La Niña."
        )

    # ── Unir con geometría ────────────────────────────────────────────────
    geo_plot = divipola.merge(
        muni_vals, left_on="id_mun", right_on="muni_code", how="left"
    )

    # ── Mapa ─────────────────────────────────────────────────────────────
    st.info(nota)

    # ── Construir mapa Folium ─────────────────────────────────────────────

    m = folium.Map(
        location=[4.5, -74.0],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # Colormap divergente centrado en 0
    vals = geo_plot["valor"].dropna()
    vmin = float(vals.quantile(0.05))
    vmax = float(vals.quantile(0.95))
    vlim = max(abs(vmin), abs(vmax))

    colormap = LinearColormap(
        colors=["#E24B4A", "#f7f7f7", "#185FA5"],
        vmin=-vlim, vmax=vlim,
        caption=legend_label,
    )

    def style_fn(feature):
            try:
                val = float(feature["properties"].get("valor") or 0)
            except (TypeError, ValueError):
                val = 0.0
            return {
                "fillColor": colormap(np.clip(val, -vlim, vlim)),
                "color":     "#555555",
                "weight":    0.3,
                "fillOpacity": 0.8,
            }

    def highlight_fn(feature):
        return {"color": "#333333", "weight": 1.5, "fillOpacity": 0.95}
    geo_json_data = json.loads(geo_plot.to_json())
    for f in geo_json_data["features"]:
        v = f["properties"].get("valor")
        f["properties"]["valor"] = float(v) if v is not None else 0.0

    folium.GeoJson(
        geo_json_data,
        style_function    = style_fn,
        highlight_function= highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields   =["name_mun", "id_dept", "valor"],
            aliases  =["Municipio", "Dpto.", legend_label],
            localize =True,
        ),
    ).add_to(m)

    colormap.add_to(m)

    st_folium(m, width="stretch", height=580, returned_objects=[])
    # ── Estadística rápida debajo del mapa ───────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municipios mapeados", f"{geo_plot['valor'].notna().sum():,}")
    c2.metric("Valor máximo", f"{geo_plot['valor'].max():.2f}")
    c3.metric("Valor mínimo", f"{geo_plot['valor'].min():.2f}")
    c4.metric("Promedio nacional", f"{geo_plot['valor'].mean():.2f}")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SERIE TEMPORAL DUAL
# ════════════════════════════════════════════════════════════════════════════
with tab_serie:

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        munis_disponibles = sorted(
            chirps[["muni_code", "name_mun"]]
            .dropna()
            .drop_duplicates()
            .apply(lambda r: f"{r['name_mun']} ({r['muni_code']})", axis=1)
            .tolist()
        )
        muni_label = st.selectbox(
            "Municipio",
            ["Nacional (promedio)"] + munis_disponibles,
        )
    with col_ctrl2:
        agg_nivel = st.radio("Agregación", ["Mensual", "Trimestral", "Anual"],
                             horizontal=True)

    # ── Preparar serie ────────────────────────────────────────────────────
    if muni_label == "Nacional (promedio)":
        serie = (
            chirps.groupby("date")
            .agg(
                precip=("precip_mean_mm", "mean"),
                anom  =("anomaly_mm",     "mean"),
            ).reset_index()
        )
        titulo_serie = "Precipitación nacional promedio"
    else:
        cod = muni_label.split("(")[-1].replace(")", "").strip()
        serie = (
            chirps[chirps["muni_code"] == cod]
            .groupby("date")
            .agg(
                precip=("precip_mean_mm", "mean"),
                anom  =("anomaly_mm",     "mean"),
            ).reset_index()
        )
        titulo_serie = f"Precipitación · {muni_label}"

    # Agregar ONI
    oni_slim = oni_hist[["date", "value_oni", "oni_phase"]].copy()
    serie = serie.merge(oni_slim, on="date", how="left")
    serie["oni_phase"] = serie["oni_phase"].fillna("Neutro")

    # Resampling si no es mensual
    if agg_nivel == "Trimestral":
        serie = serie.set_index("date").resample("QS").agg(
            precip=("precip", "mean"),
            anom  =("anom",   "mean"),
            value_oni=("value_oni", "mean"),
        ).reset_index()
        serie["oni_phase"] = serie["value_oni"].apply(
            lambda v: "El Niño" if v>=0.5 else "La Niña" if v<=-0.5 else "Neutro"
        )
    elif agg_nivel == "Anual":
        serie = serie.set_index("date").resample("YS").agg(
            precip=("precip", "mean"),
            anom  =("anom",   "mean"),
            value_oni=("value_oni", "mean"),
        ).reset_index()
        serie["oni_phase"] = serie["value_oni"].apply(
            lambda v: "El Niño" if v>=0.5 else "La Niña" if v<=-0.5 else "Neutro"
        )

    # ── Gráfico dual ─────────────────────────────────────────────────────
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    # Barras de anomalía (eje izquierdo)
    bar_cols = ["#E24B4A" if v < 0 else "#185FA5" for v in serie["anom"]]
    fig_ts.add_trace(
        go.Bar(
            x=serie["date"], y=serie["anom"],
            name="Anomalía precipitación (mm)",
            marker_color=bar_cols, opacity=0.75,
            hovertemplate="%{x}<br>Anomalía: %{y:.1f} mm<extra></extra>",
         ),
        secondary_y=False,
    )

    # Línea ONI (eje derecho)
    fig_ts.add_trace(
        go.Scatter(
            x=serie["date"], y=serie["value_oni"],
            name="Índice ONI",
            mode="lines",
            line=dict(color="#333333", width=2),
            hovertemplate="ONI: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Banda ±0.5 ONI
    fig_ts.add_hrect(y0=0.5, y1=3.5,  fillcolor="rgba(226,75,74,0.06)",
                     line_width=0, secondary_y=True)
    fig_ts.add_hrect(y0=-3.5, y1=-0.5, fillcolor="rgba(24,95,165,0.06)",
                     line_width=0, secondary_y=True)

    # Línea cero anomalía
    fig_ts.add_hline(y=0, line_dash="dot", line_color="#cccccc",
                     line_width=1, secondary_y=False)

    fig_ts.update_layout(
        title      = titulo_serie,
        height     = 420,
        hovermode  = "x unified",
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        legend     = dict(orientation="h", y=1.10, x=0),
        margin     = dict(t=65, b=40, l=10, r=10),
    )
    fig_ts.update_yaxes(
        title_text="Anomalía de precipitación (mm)",
        secondary_y=False,
        zeroline=True, zerolinecolor="#eeeeee",
    )
    fig_ts.update_yaxes(
        title_text="Índice ONI (°C)",
        secondary_y=True,
        range=[-3.5, 3.5],
        showgrid=False,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Heatmap debajo: municipio × mes del año ───────────────────────────
    with st.expander("🔥 Heatmap — Anomalía mensual por año"):
        if muni_label == "Nacional (promedio)":
            heat_data = (
                chirps.groupby(["year", "month"])["anomaly_mm"]
                .mean().round(1).reset_index()
            )
        else:
            heat_data = (
                chirps[chirps["muni_code"] == cod]
                .groupby(["year", "month"])["anomaly_mm"]
                .mean().round(1).reset_index()
            )

        pivot = heat_data.pivot(index="year", columns="month", values="anomaly_mm")
        pivot.columns = ["Ene","Feb","Mar","Abr","May","Jun",
                         "Jul","Ago","Sep","Oct","Nov","Dic"]

        fig_heat = px.imshow(
            pivot,
            color_continuous_scale  = "RdBu",
            color_continuous_midpoint = 0,
            aspect     = "auto",
            title      = "Anomalía de precipitación (mm) · año × mes",
            labels     = dict(color="Anomalía (mm)", x="Mes", y="Año"),
        )
        fig_heat.update_layout(
            height = max(300, len(pivot) * 22),
            margin = dict(t=50, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SEMÁFORO DE RIESGO
# ════════════════════════════════════════════════════════════════════════════
with tab_semaforo:

    st.markdown("### Clasificación de riesgo climático por región")
    st.caption(
        "Basado en: fase ONI actual · anomalía media del período · "
        "porcentaje de municipios en déficit o exceso"
    )

    # ── Calcular riesgo por región ────────────────────────────────────────
    if "id_dept" not in chirps.columns:
        chirps["id_dept"] = chirps["muni_code"].astype(str).str.zfill(5).str[:2]
    chirps["region"] = chirps["id_dept"].map(MUNI_TO_REGION).fillna("Otra")

    reg_stats = (
        chirps.groupby("region")
        .agg(
            anom_media   = ("anomaly_mm",  "mean"),
            pct_deficit  = ("anomaly_mm",  lambda x: (x < -10).mean() * 100),
            pct_exceso   = ("anomaly_mm",  lambda x: (x >  10).mean() * 100),
            n_municipios = ("muni_code",   "nunique"),
            oni_media    = ("value_oni",   "mean"),
        ).round(1).reset_index()
    )

    def asignar_riesgo(row):
        if row["oni_media"] >= 0.5 and row["pct_deficit"] > 35:
            return "Alto — Sequía"
        if row["oni_media"] <= -0.5 and row["pct_exceso"] > 35:
            return "Alto — Exceso"
        if abs(row["oni_media"]) >= 0.5 or row["pct_deficit"] > 25 or row["pct_exceso"] > 25:
            return "Medio"
        return "Bajo"

    reg_stats["riesgo"] = reg_stats.apply(asignar_riesgo, axis=1)

    # ── Tarjetas semáforo ─────────────────────────────────────────────────
    cols = st.columns(len(reg_stats))
    for i, (_, row) in enumerate(reg_stats.sort_values("region").iterrows()):
        cfg  = RISK_CFG[row["riesgo"]]
        anom = row["anom_media"]
        with cols[i]:
            _cbg = cfg["bg"]
            _cc  = cfg["color"]
            _ci  = cfg["icon"]
            st.markdown(
                f"<div style='background:{_cbg};border:2px solid {_cc};"
                f"border-radius:12px;padding:14px 10px;text-align:center;'>"
                f"<div style='font-size:28px;margin-bottom:4px'>{_ci}</div>"
                f"<div style='font-size:14px;font-weight:600;color:{_cc}'>"
                f"{row['region']}</div>"
                f"<div style='font-size:12px;font-weight:600;color:{_cc};"
                f"margin:4px 0'>{row['riesgo']}</div>"
                f"<div style='font-size:11px;color:#555;margin-top:6px'>"
                f"Anomalía: <b>{anom:+.1f} mm</b><br>"
                f"Déficit: <b>{row['pct_deficit']:.0f}%</b> · "
                f"Exceso: <b>{row['pct_exceso']:.0f}%</b><br>"
                f"ONI medio: <b>{row['oni_media']:+.2f}</b><br>"
                f"{row['n_municipios']} municipios"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Gráfico de barras agrupadas ───────────────────────────────────────
    fig_sem = go.Figure()
    fig_sem.add_trace(go.Bar(
        name="% Municipios en déficit",
        x=reg_stats["region"],
        y=reg_stats["pct_deficit"],
        marker_color="#E24B4A", opacity=0.80,
        text=reg_stats["pct_deficit"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
    ))
    fig_sem.add_trace(go.Bar(
        name="% Municipios en exceso",
        x=reg_stats["region"],
        y=reg_stats["pct_exceso"],
        marker_color="#185FA5", opacity=0.80,
        text=reg_stats["pct_exceso"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
    ))
    fig_sem.update_layout(
        barmode       = "group",
        title         = "Porcentaje de municipios en déficit o exceso hídrico por región",
        height        = 350,
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        legend        = dict(orientation="h", y=1.08),
        yaxis         = dict(title="% de municipios", range=[0, 100]),
    )
    st.plotly_chart(fig_sem, use_container_width=True)

    # ── Tabla detalle ─────────────────────────────────────────────────────
    with st.expander("📋 Tabla detallada por región"):
        tbl = reg_stats.copy()
        tbl.columns = ["Región", "Anomalía media (mm)", "% Déficit",
                       "% Exceso", "N° municipios", "ONI medio", "Riesgo"]
        st.dataframe(tbl.sort_values("Anomalía media (mm)"),
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PROYECCIÓN Y ALERTAS
# ════════════════════════════════════════════════════════════════════════════
with tab_pred:

    st.markdown("### Proyección de precipitación · Próximos 3 meses")
    st.caption("Basada en predicciones ONI del IRI × correlación histórica CHIRPS por municipio")

    if oni_pred.empty:
        st.warning("No se encontraron datos de predicción IRI.")
    else:
        # Últimas 3 predicciones
        pred3 = oni_pred.sort_values("date").tail(3).reset_index(drop=True)

        # ── Cards de predicción ONI ───────────────────────────────────────
        pc1, pc2, pc3 = st.columns(3)
        for i, col in enumerate([pc1, pc2, pc3]):
            if i >= len(pred3): break
            row   = pred3.iloc[i]
            phase = row["oni_phase"]
            pcfg  = RISK_CFG.get(
                "Alto — Sequía" if phase == "El Niño" else
                "Alto — Exceso" if phase == "La Niña" else "Bajo"
            )
            _pbg = pcfg["bg"]
            _pc  = pcfg["color"]
            col.markdown(
                f"<div style='background:{_pbg};border:1.5px solid "
                f"{_pc}33;border-radius:12px;padding:14px;text-align:center;'>"
                f"<div style='font-size:12px;font-weight:600;color:#555;margin-bottom:6px'>"
                f"{row['prediction_period']} · {row['date'].strftime('%b %Y')}</div>"
                f"<div style='font-size:32px;font-weight:700;color:{_pc}'>"
                f"{row['prediction_oni']:+.2f}</div>"
                f"<div style='font-size:13px;color:{_pc};margin-top:4px;font-weight:600'>"
                f"{PHASE_ICONS[phase]} {phase}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Proyección de anomalía por región ────────────────────────────
        st.markdown("#### Anomalía de precipitación proyectada por región")

        # Correlación media ONI × anomalía por región
        corr_region = (
            chirps.groupby("region")
            .apply(lambda g: g["value_oni"].corr(g["anomaly_mm"]))
            .reset_index()
        )
        corr_region.columns = ["region", "corr_oni"]

        anom_media_region = (
            chirps.groupby("region")["anomaly_mm"].mean().reset_index()
        )
        anom_media_region.columns = ["region", "anom_media"]
        corr_region = corr_region.merge(anom_media_region, on="region")

        next_oni = pred3.iloc[0]["prediction_oni"]
        next_phase = pred3.iloc[0]["oni_phase"]

        corr_region["anom_proj"] = (
            corr_region["corr_oni"] * next_oni * 30
        ).round(1)
        corr_region["riesgo_proj"] = corr_region["anom_proj"].apply(
            lambda v: "Alto — Sequía" if v < -15
            else "Alto — Exceso" if v > 15
            else "Medio" if abs(v) > 7
            else "Bajo"
        )

        fig_proj = go.Figure()
        for _, row in corr_region.iterrows():
            cfg = RISK_CFG[row["riesgo_proj"]]
            fig_proj.add_trace(go.Bar(
                x=[row["region"]], y=[row["anom_proj"]],
                name=row["region"],
                marker_color=cfg["color"], opacity=0.85,
                text=f"{row['anom_proj']:+.1f} mm",
                textposition="outside",
                showlegend=False,
            ))

        fig_proj.add_hline(y=0, line_dash="dot", line_color="#aaaaaa")
        fig_proj.update_layout(
            title  = f"Anomalía proyectada · ONI {next_oni:+.2f} ({next_phase})",
            height = 340,
            yaxis  = dict(title="Anomalía proyectada (mm)"),
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
        )
        st.plotly_chart(fig_proj, use_container_width=True)

        # ── Alertas tempranas ────────────────────────────────────────────
        st.markdown("#### 🚨 Alertas tempranas")

        alertas = []
        for _, row in corr_region.iterrows():
            if row["anom_proj"] < -15:
                alertas.append({
                    "tipo": "Déficit hídrico",
                    "region": row["region"],
                    "valor": row["anom_proj"],
                    "color": "#E24B4A",
                    "msg": (
                        f"La región **{row['region']}** podría registrar "
                        f"**{row['anom_proj']:+.1f} mm** de anomalía. "
                        f"Riesgo de sequía en cultivos y abastecimiento hídrico."
                    ),
                })
            elif row["anom_proj"] > 15:
                alertas.append({
                    "tipo": "Exceso hídrico",
                    "region": row["region"],
                    "valor": row["anom_proj"],
                    "color": "#185FA5",
                    "msg": (
                        f"La región **{row['region']}** podría registrar "
                        f"**{row['anom_proj']:+.1f} mm** de anomalía. "
                        f"Riesgo de inundaciones y deslizamientos."
                    ),
                })

        if alertas:
            for a in alertas:
                _ac  = a["color"]
                _key = "Alto — Sequía" if a["valor"] < 0 else "Alto — Exceso"
                _abg = RISK_CFG[_key]["bg"]
                st.markdown(
                    f"<div class='insight-box' style='border-color:{_ac};background:{_abg}'>"
                    f"⚠️ <b>{a['tipo']} · {a['region']}</b><br>{a['msg']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success(
                f"✅ No se proyectan anomalías extremas para el próximo trimestre "
                f"con ONI {next_oni:+.2f} ({next_phase})."
            )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — INSIGHTS CLIMÁTICOS AUTOMÁTICOS
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:

    st.markdown("### 💡 Hallazgos clave del análisis climático")
    st.caption(
        f"Generados automáticamente · Período {year_range[0]}–{year_range[1]} · "
        f"Regiones: {', '.join(region_sel)}"
    )

    insights = []

    # ── Insight 1: sensibilidad regional al ENSO ─────────────────────────
    nino_months = chirps[chirps["oni_phase"] == "El Niño"]
    nina_months = chirps[chirps["oni_phase"] == "La Niña"]

    if not nino_months.empty:
        pct_deficit_nino = (nino_months["anomaly_mm"] < -10).mean() * 100
        insights.append({
            "color": "#E24B4A",
            "titulo": "Sensibilidad a El Niño",
            "texto": (
                f"Durante meses de El Niño, el **{pct_deficit_nino:.0f}%** de los "
                f"registros municipales presentan déficit de precipitación mayor a 10 mm. "
                f"Esto confirma la relación inversa entre el ONI positivo y la lluvia en Colombia."
            ),
        })

    if not nina_months.empty:
        pct_exceso_nina = (nina_months["anomaly_mm"] > 10).mean() * 100
        insights.append({
            "color": "#185FA5",
            "titulo": "Sensibilidad a La Niña",
            "texto": (
                f"Durante meses de La Niña, el **{pct_exceso_nina:.0f}%** de los "
                f"registros municipales presentan exceso de precipitación mayor a 10 mm."
            ),
        })

    # ── Insight 2: región más y menos sensible ───────────────────────────
    sens = (
        chirps.groupby("region")
        .apply(lambda g: abs(g["value_oni"].corr(g["anomaly_mm"])))
        .reset_index()
    )
    sens.columns = ["region", "sensibilidad"]
    mas_sens  = sens.loc[sens["sensibilidad"].idxmax()]
    menos_sens = sens.loc[sens["sensibilidad"].idxmin()]

    insights.append({
        "color": "#6B3A9E",
        "titulo": "Región más sensible al ENSO",
        "texto": (
            f"La región **{mas_sens['region']}** presenta la mayor correlación "
            f"entre ONI y precipitación (|r| = {mas_sens['sensibilidad']:.2f}), "
            f"lo que indica alta vulnerabilidad a los ciclos ENSO."
        ),
    })
    insights.append({
        "color": "#2E7D32",
        "titulo": "Región menos sensible al ENSO",
        "texto": (
            f"La región **{menos_sens['region']}** presenta la menor correlación "
            f"(|r| = {menos_sens['sensibilidad']:.2f}), lo que sugiere que su "
            f"precipitación está controlada principalmente por factores locales o marítimos."
        ),
    })

    # ── Insight 3: mes del año con mayor impacto ENSO ────────────────────
    mes_impact = (
        nino_months.groupby("month")["anomaly_mm"].mean()
        if not nino_months.empty
        else pd.Series(dtype=float)
    )
    if not mes_impact.empty:
        mes_max = mes_impact.idxmin()  # mes con mayor déficit en El Niño
        meses_nombres = ["Ene","Feb","Mar","Abr","May","Jun",
                         "Jul","Ago","Sep","Oct","Nov","Dic"]
        insights.append({
            "color": "#E24B4A",
            "titulo": "Mes más crítico en El Niño",
            "texto": (
                f"El mes de **{meses_nombres[mes_max-1]}** registra la mayor "
                f"reducción de precipitación durante El Niño "
                f"(anomalía media: {mes_impact[mes_max]:+.1f} mm). "
                f"Este período concentra el mayor riesgo de déficit hídrico nacional."
            ),
        })

    # ── Insight 4: municipios con correlación más fuerte ─────────────────
    top_corr_neg = corr_muni.nsmallest(5, "corr_oni")[["muni_code", "corr_oni"]]
    top_corr_neg = top_corr_neg.merge(
        chirps[["muni_code","name_mun"]].drop_duplicates(),
        on="muni_code", how="left"
    )
    nombres_top = ", ".join(
        top_corr_neg["name_mun"].fillna(top_corr_neg["muni_code"]).tolist()
    )
    insights.append({
        "color": "#E24B4A",
        "titulo": "Municipios más vulnerables a El Niño",
        "texto": (
            f"Los municipios con mayor correlación negativa ONI–precipitación son: "
            f"**{nombres_top}**. "
            f"Estos son los más susceptibles a experimentar sequía durante eventos El Niño."
        ),
    })

    # ── Insight 5: tendencia reciente ────────────────────────────────────
    reciente = chirps[chirps["year"] >= year_range[1] - 3]
    anom_reciente = reciente["anomaly_mm"].mean()
    anom_total    = chirps["anomaly_mm"].mean()
    diferencia    = anom_reciente - anom_total
    tendencia_txt = (
        f"**incremento** de {diferencia:+.1f} mm" if diferencia > 2
        else f"**reducción** de {diferencia:+.1f} mm" if diferencia < -2
        else "**estabilidad**"
    )
    insights.append({
        "color": "#BA7517",
        "titulo": f"Tendencia reciente ({year_range[1]-3}–{year_range[1]})",
        "texto": (
            f"Los últimos 3 años muestran una {tendencia_txt} en la anomalía "
            f"de precipitación respecto al promedio del período completo "
            f"({anom_total:+.1f} mm → {anom_reciente:+.1f} mm)."
        ),
    })

    # ── Renderizar todos los insights ────────────────────────────────────
    for ins in insights:
        _ic = ins["color"]
        st.markdown(
            f"<div class='insight-box' style='border-color:{_ic};background:#fafafa'>"
            f"<span style='font-weight:600;color:{_ic}'>{ins['titulo']}</span><br>"
            f"{ins['texto']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — PRECIPITACIÓN ABSOLUTA
# ════════════════════════════════════════════════════════════════════════════
with tab_prec:

    st.markdown("### 🌧️ Precipitación absoluta por municipio")
    st.caption("Precipitación media mensual en mm · Fuente: CHIRPS")

    # ── Controles ────────────────────────────────────────────────────────
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        mes_sel = st.selectbox(
            "Mes a visualizar en el mapa",
            options=list(range(1, 13)),
            format_func=lambda m: ["Enero","Febrero","Marzo","Abril","Mayo",
                                   "Junio","Julio","Agosto","Septiembre",
                                   "Octubre","Noviembre","Diciembre"][m-1],
        )
    with col_c2:
        anio_sel = st.selectbox(
            "Año",
            options=sorted(chirps["year"].unique(), reverse=True),
        )

    # ── Datos del mes/año seleccionado ───────────────────────────────────
    prec_mes = (
        chirps[(chirps["year"] == anio_sel) & (chirps["month"] == mes_sel)]
        .groupby("muni_code")["precip_mean_mm"]
        .mean().round(1).reset_index()
    )
    prec_mes.columns = ["muni_code", "precip"]

    geo_prec = divipola.merge(
        prec_mes, left_on="id_mun", right_on="muni_code", how="left"
    )

    # Forzar float nativo
    geo_json_prec = json.loads(geo_prec.to_json())
    for feat in geo_json_prec["features"]:
        v = feat["properties"].get("precip")
        feat["properties"]["precip"] = float(v) if v is not None else 0.0

    # ── KPIs rápidos del mes ─────────────────────────────────────────────
    p1, p2, p3, p4 = st.columns(4)
    if prec_mes.empty:
        st.warning("Sin datos para el mes y año seleccionados.")
        st.stop()

    p1.metric("Precipitación máxima",  f"{prec_mes['precip'].max():.0f} mm")
    p2.metric("Precipitación mínima",  f"{prec_mes['precip'].min():.0f} mm")
    p3.metric("Promedio nacional",     f"{prec_mes['precip'].mean():.0f} mm")
    p4.metric("Municipios con datos",  f"{prec_mes['precip'].notna().sum():,}")

    st.divider()

    # ── Mapa Folium ──────────────────────────────────────────────────────
    col_mapa, col_serie = st.columns([3, 2])

    with col_mapa:
        mes_nombre = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                      "Julio","Agosto","Septiembre","Octubre","Noviembre",
                      "Diciembre"][mes_sel - 1]
        st.markdown(f"**Mapa · {mes_nombre} {anio_sel}**")

        vmax_prec = float(prec_mes["precip"].quantile(0.95))

        colormap_prec = LinearColormap(
            colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
            vmin=0, vmax=vmax_prec,
            caption="Precipitación (mm)",
        )

        m_prec = folium.Map(
            location=[4.5, -74.0],
            zoom_start=5,
            tiles="CartoDB positron",
        )

        def style_prec(feature):
            try:
                val = float(feature["properties"].get("precip") or 0)
            except (TypeError, ValueError):
                val = 0.0
            return {
                "fillColor":   colormap_prec(min(val, vmax_prec)),
                "color":       "#555555",
                "weight":      0.3,
                "fillOpacity": 0.85,
            }

        def highlight_prec(feature):
            return {"color": "#222222", "weight": 1.5, "fillOpacity": 0.95}

        folium.GeoJson(
            geo_json_prec,
            style_function     = style_prec,
            highlight_function = highlight_prec,
            tooltip=folium.GeoJsonTooltip(
                fields  =["name_mun", "id_dept", "precip"],
                aliases =["Municipio", "Dpto.", "Precipitación (mm)"],
                localize=True,
            ),
        ).add_to(m_prec)

        colormap_prec.add_to(m_prec)
        st_folium(m_prec, width="stretch", height=500, returned_objects=[])

    # ── Serie temporal por municipio ─────────────────────────────────────
    with col_serie:
        st.markdown("**Serie temporal**")

        munis_prec = sorted(
            chirps[["muni_code", "name_mun"]]
            .dropna()
            .drop_duplicates()
            .apply(lambda r: f"{r['name_mun']} ({r['muni_code']})", axis=1)
            .tolist()
        )
        muni_prec_sel = st.selectbox(
            "Municipio",
            ["Nacional (promedio)"] + munis_prec,
            key="muni_prec",
        )

        if muni_prec_sel == "Nacional (promedio)":
            serie_prec = (
                chirps.groupby("date")["precip_mean_mm"]
                .mean().round(1).reset_index()
            )
            titulo_prec = "Precipitación nacional promedio"
        else:
            cod_prec = muni_prec_sel.split("(")[-1].replace(")", "").strip()
            serie_prec = (
                chirps[chirps["muni_code"] == cod_prec]
                .groupby("date")["precip_mean_mm"]
                .mean().round(1).reset_index()
            )
            titulo_prec = muni_prec_sel

        # Unir ONI para contexto
        serie_prec = serie_prec.merge(
            oni_hist[["date", "value_oni", "oni_phase"]], on="date", how="left"
        )

        # Colorear barras por fase ONI
        bar_cols_prec = serie_prec["oni_phase"].map(
            {"El Niño": "#E24B4A", "La Niña": "#185FA5", "Neutro": "#6B9E6B"}
        ).fillna("#6B9E6B")

        fig_prec = go.Figure()
        fig_prec.add_trace(go.Bar(
            x=serie_prec["date"],
            y=serie_prec["precip_mean_mm"],
            marker_color=bar_cols_prec,
            opacity=0.80,
            name="Precipitación (mm)",
            hovertemplate="%{x}<br>%{y:.1f} mm<extra></extra>",
        ))

        # Promedio histórico como línea de referencia
        prom_hist = serie_prec["precip_mean_mm"].mean()
        fig_prec.add_hline(
            y=prom_hist,
            line_dash="dot", line_color="#333333", line_width=1.5,
            annotation_text=f"Promedio {prom_hist:.0f} mm",
            annotation_position="top right",
            annotation_font_size=10,
        )

        fig_prec.update_layout(
            title         = titulo_prec,
            height        = 460,
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            yaxis         = dict(title="mm"),
            xaxis         = dict(title=""),
            margin        = dict(t=50, b=30, l=10, r=10),
            showlegend    = False,
        )
        st.plotly_chart(fig_prec, width="stretch")

        # Leyenda de fases
        st.markdown(
            "<div style='font-size:11px;color:#666;margin-top:-10px'>"
            "<span style='color:#E24B4A'>■</span> El Niño &nbsp;"
            "<span style='color:#185FA5'>■</span> La Niña &nbsp;"
            "<span style='color:#6B9E6B'>■</span> Neutro"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Resumen ejecutivo descargable ─────────────────────────────────────
    with st.expander("📄 Resumen ejecutivo en texto"):
        resumen = f"""RESUMEN EJECUTIVO · Dashboard ENSO Colombia
Período analizado: {year_range[0]}–{year_range[1]}
Generado automáticamente por el pipeline ETL · Laura A. Chacón V. · 2025
{'='*55}

ESTADO CLIMÁTICO ACTUAL
  Valor ONI más reciente : {oni_val:+.2f} ({oni_date})
  Fase ENSO              : {oni_fase}
  Riesgo dominante       : {riesgo_dom}

IMPACTO EN PRECIPITACIÓN
"""
        for ins in insights:
            resumen += f"\n{ins['titulo'].upper()}\n{ins['texto']}\n"

        st.code(resumen, language=None)
        st.download_button(
            "⬇️ Descargar resumen (.txt)",
            data=resumen,
            file_name=f"resumen_enso_{year_range[0]}_{year_range[1]}.txt",
            mime="text/plain",
        )