

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from folium.plugins import Fullscreen
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────

COLORES = {
    "El Niño":  "#D85A30",
    "La Niña":  "#378ADD",
    "Neutral":  "#888780",
    "déficit":  "#D85A30",
    "exceso":   "#378ADD",
    "normal":   "rgba(200,200,200,0.15)",
    "oni_line": "#D85A30",
    "precip":   "#378ADD",
}

NOMBRES_MESES = ["Ene","Feb","Mar","Abr","May","Jun",
                 "Jul","Ago","Sep","Oct","Nov","Dic"]

# ─────────────────────────────────────────────
# 1. SERIE TEMPORAL
# ─────────────────────────────────────────────

def serie_temporal(df_anomalias: pd.DataFrame, titulo: str = "") -> go.Figure:

    df = df_anomalias.sort_values("date").copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Barras por tipo de anomalía
    for tipo, color in [("déficit", COLORES["déficit"]),
                        ("exceso",  COLORES["exceso"]),
                        ("normal",  "rgba(55,138,221,0.45)")]:
        mask = df["anomalia"] == tipo
        label = {"déficit": "Déficit (< P5)",
                 "exceso":  "Exceso (> P95)",
                 "normal":  "Normal"}[tipo]
        fig.add_trace(
            go.Bar(
                x=df.loc[mask, "date"],
                y=df.loc[mask, "precip_mm"],
                name=label,
                marker_color=color if tipo == "normal" else color,
                marker_opacity=0.85 if tipo != "normal" else 0.5,
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Precipitación: %{y:.1f} mm<br>"
                    f"Tipo: {label}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    # Línea ONI
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["oni"],
            name="ONI",
            line=dict(color=COLORES["oni_line"], width=1.8),
            mode="lines",
            hovertemplate="<b>%{x|%b %Y}</b><br>ONI: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Banda neutral ONI (±0.5)
    fig.add_hrect(
        y0=-0.5, y1=0.5,
        fillcolor="rgba(136,135,128,0.10)",
        line_width=0,
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=f"Precipitación mensual y ONI — {titulo}", font_size=13),
        barmode="stack",
        legend=dict(orientation="h", y=-0.18, font_size=11),
        margin=dict(t=50, b=60, l=50, r=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Precipitación (mm)", secondary_y=False,
                     gridcolor="rgba(200,200,200,0.4)")
    fig.update_yaxes(title_text="ONI", secondary_y=True,
                     range=[-3, 3], zeroline=True,
                     zerolinecolor="rgba(200,200,200,0.6)")
    fig.update_xaxes(showgrid=False)

    return fig


# ─────────────────────────────────────────────
# 2. DISPERSIÓN ONI vs. PRECIPITACIÓN
# ─────────────────────────────────────────────

def dispersion_oni(df: pd.DataFrame, titulo: str = "") -> go.Figure:
    """
    Scatter precipitación vs. ONI coloreado por fase ENOS.
    Incluye línea de tendencia por fase.
    """
    fig = go.Figure()

    simbolos = {"El Niño": "circle", "La Niña": "triangle-up", "Neutral": "square"}

    for fase in ["El Niño", "La Niña", "Neutral"]:
        mask = df["fase_oni"] == fase
        sub  = df[mask]
        if sub.empty:
            continue

        fig.add_trace(go.Scatter(
            x=sub["oni"],
            y=sub["precip_mm"],
            mode="markers",
            name=fase,
            marker=dict(color=COLORES[fase], size=6,
                        symbol=simbolos[fase], opacity=0.65),
            hovertemplate=(
                f"<b>{fase}</b><br>"
                "ONI: %{x:.2f}<br>"
                "Precipitación: %{y:.1f} mm<extra></extra>"
            ),
        ))

        # Línea de tendencia lineal
        if len(sub) >= 5:
            m, b = np.polyfit(sub["oni"].values, sub["precip_mm"].values, 1)
            x_line = np.linspace(sub["oni"].min(), sub["oni"].max(), 50)
            fig.add_trace(go.Scatter(
                x=x_line, y=m * x_line + b,
                mode="lines",
                line=dict(color=COLORES[fase], width=1.2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.add_vline(x=0.5,  line_dash="dot", line_color="rgba(216,90,48,0.4)",
                  annotation_text="El Niño", annotation_font_size=9)
    fig.add_vline(x=-0.5, line_dash="dot", line_color="rgba(55,138,221,0.4)",
                  annotation_text="La Niña", annotation_font_size=9,
                  annotation_position="bottom right")

    fig.update_layout(
        title=dict(text=f"Precipitación vs. ONI — {titulo}", font_size=13),
        xaxis_title="ONI",
        yaxis_title="Precipitación (mm)",
        legend=dict(orientation="h", y=-0.18, font_size=11),
        margin=dict(t=50, b=60, l=50, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="rgba(200,200,200,0.4)", zeroline=True,
                     zerolinecolor="rgba(200,200,200,0.6)")
    fig.update_yaxes(gridcolor="rgba(200,200,200,0.4)")

    return fig


# ─────────────────────────────────────────────
# 3. BARRAS DE REZAGO
# ─────────────────────────────────────────────

def lags_barras(df_lags: pd.DataFrame) -> go.Figure:
    """
    Barras horizontales de correlación por lag.
    El lag óptimo se resalta con borde más grueso.
    Barras sin significancia estadística (p > 0.05) se muestran atenuadas.
    """
    if df_lags.empty:
        return go.Figure()

    df = df_lags.copy().sort_values("lag")
    idx_optimo = df["abs_corr"].idxmax()

    colores, opacidades, bordes = [], [], []
    for i, row in df.iterrows():
        significativo = row["p_valor"] < 0.05
        es_optimo     = i == idx_optimo
        color = COLORES["precip"] if row["correlacion"] < 0 else COLORES["déficit"]
        colores.append(color)
        opacidades.append(0.85 if significativo else 0.30)
        bordes.append(3 if es_optimo else 0)

    fig = go.Figure(go.Bar(
        x=df["correlacion"],
        y=[f"Lag {l}m" for l in df["lag"]],
        orientation="h",
        marker=dict(
            color=colores,
            opacity=opacidades,
            line=dict(color="rgba(0,0,0,0.5)", width=bordes),
        ),
        customdata=np.stack([df["p_valor"], df["lag"]], axis=-1),
        hovertemplate=(
            "<b>Lag %{customdata[1]} meses</b><br>"
            "Correlación: %{x:.3f}<br>"
            "p-valor: %{customdata[0]:.4f}<extra></extra>"
        ),
    ))

    # Línea vertical en 0
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.2)", line_width=1)

    # Anotación lag óptimo
    lag_opt = df.loc[idx_optimo]
    fig.add_annotation(
        x=lag_opt["correlacion"],
        y=f"Lag {int(lag_opt['lag'])}m",
        text="★ óptimo",
        showarrow=False,
        xshift=40 if lag_opt["correlacion"] < 0 else -40,
        font=dict(size=10, color="rgba(0,0,0,0.6)"),
    )

    fig.update_layout(
        title=dict(text="Correlación cruzada ONI ↔ precipitación por rezago", font_size=13),
        xaxis=dict(title="Correlación de Pearson", range=[-1, 1],
                   gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=40, l=70, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        annotations=[
            dict(
                text="Barras atenuadas: p > 0.05 (no significativo)",
                xref="paper", yref="paper",
                x=0, y=-0.12, showarrow=False,
                font=dict(size=9, color="gray"),
            )
        ],
    )

    return fig


# ─────────────────────────────────────────────
# 4. CLIMATOLOGÍA POR FASE ONI
# ─────────────────────────────────────────────

def climatologia_fases(df_clima: pd.DataFrame, titulo: str = "") -> go.Figure:
    """
    Ciclo anual de precipitación (ene-dic) por fase ENOS.
    Incluye banda de dispersión P25-P75 por fase.
    """
    fig = go.Figure()

    for fase in ["El Niño", "La Niña", "Neutral"]:
        sub = df_clima[df_clima["fase_oni"] == fase].sort_values("mes")
        if sub.empty:
            continue

        color = COLORES[fase]

        # Banda P25-P75 — convertir hex a rgba para soportar transparencia
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.15)"

        fig.add_trace(go.Scatter(
            x=list(sub["mes"]) + list(sub["mes"])[::-1],
            y=list(sub["precip_p75"]) + list(sub["precip_p25"])[::-1],
            fill="toself",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Línea media
        fig.add_trace(go.Scatter(
            x=sub["mes"],
            y=sub["precip_media"],
            name=fase,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=(
                f"<b>{fase}</b> — %{{x}}<br>"
                "Media: %{y:.1f} mm<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=f"Climatología mensual por fase ONI — {titulo}", font_size=13),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=NOMBRES_MESES,
            gridcolor="rgba(200,200,200,0.4)",
        ),
        yaxis=dict(title="Precipitación media (mm)",
                   gridcolor="rgba(200,200,200,0.4)"),
        legend=dict(orientation="h", y=-0.18, font_size=11),
        margin=dict(t=50, b=60, l=50, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )

    return fig


# ─────────────────────────────────────────────
# 5. TABLA DE ANOMALÍAS HISTÓRICAS
# ─────────────────────────────────────────────

def anomalias_tabla(df_anomalias: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Tabla interactiva de los eventos extremos más significativos
    ordenados por magnitud de desvío respecto al P50 del mes.
    """
    extremos = (
        df_anomalias[df_anomalias["anomalia"] != "normal"]
        .assign(abs_desvio=lambda d: d["desvio_pct"].abs())
        .sort_values("abs_desvio", ascending=False)
        .head(top_n)
        .copy()
    )

    if extremos.empty:
        return go.Figure()

    extremos["Fecha"]     = extremos["date"].dt.strftime("%b %Y")
    extremos["Tipo"]      = extremos["anomalia"].str.capitalize()
    extremos["Precip mm"] = extremos["precip_mm"].round(1)
    extremos["P50 mm"]    = extremos["p50"].round(1)
    extremos["Desvío %"]  = extremos["desvio_pct"].apply(
        lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
    )
    extremos["ONI"]       = extremos["oni"].round(2)
    extremos["Fase ONI"]  = extremos["fase_oni"]

    def _hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    colores_filas_alpha = [
        _hex_rgba(COLORES["exceso"]) if t == "Exceso" else _hex_rgba(COLORES["déficit"])
        for t in extremos["Tipo"]
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Fecha</b>", "<b>Tipo</b>", "<b>Precip mm</b>",
                    "<b>P50 mm</b>", "<b>Desvío %</b>", "<b>ONI</b>", "<b>Fase ONI</b>"],
            fill_color="#F0F2F6",
            font=dict(size=11),
            align="left",
            line_color="white",
        ),
        cells=dict(
            values=[
                extremos["Fecha"], extremos["Tipo"], extremos["Precip mm"],
                extremos["P50 mm"], extremos["Desvío %"],
                extremos["ONI"], extremos["Fase ONI"],
            ],
            fill_color=[colores_filas_alpha],
            font=dict(size=11),
            align="left",
            line_color="white",
            height=26,
        ),
    ))

    fig.update_layout(
        title=dict(text=f"Top {top_n} anomalías históricas por magnitud", font_size=13),
        margin=dict(t=50, b=10, l=10, r=10),
    )

    return fig


# ─────────────────────────────────────────────
# 6. MAPA COROPLÉTICO
# ─────────────────────────────────────────────

DIVIPOLA_PATH = (
    r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon"
    r"\ETL_code\dashboard_2\divipola\divipola.parquet"
)


@st.cache_data(show_spinner="Cargando geometrías de municipios...")
def _cargar_geodataframe():

    import geopandas as gpd

    gdf = gpd.read_parquet(DIVIPOLA_PATH)
    gdf["id_mun"] = gdf["id_mun"].astype(str).str.zfill(5)

    # Asegurar CRS geográfico para Folium (requiere EPSG:4326)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def mapa_correlacion(
    df_corr: pd.DataFrame,
    muni_seleccionado: str | None = None,
) -> folium.Map:

    import geopandas as gpd

    gdf = _cargar_geodataframe()

    # Unir correlaciones a las geometrías por id_mun
    df_corr = df_corr.copy()
    df_corr["muni_code"] = df_corr["muni_code"].astype(str).str.zfill(5)
    if "significativo_fdr" not in df_corr.columns:
        df_corr["significativo_fdr"] = True  # compatibilidad si no viene calculado
    if "p_valor_fdr" not in df_corr.columns:
        df_corr["p_valor_fdr"] = df_corr.get("p_valor")
    if "concuerda_spearman" not in df_corr.columns:
        df_corr["concuerda_spearman"] = None  # compatibilidad si no viene calculado
    gdf = gdf.merge(df_corr, left_on="id_mun", right_on="muni_code", how="left")

    geojson_str = gdf.to_json()

    mapa = folium.Map(
        location=[4.5, -74.0],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    folium.Choropleth(
        geo_data=geojson_str,
        data=df_corr,
        columns=["muni_code", "correlacion"],
        key_on="feature.properties.id_mun",
        fill_color="RdBu",       # rojo = correlación negativa, azul = positiva
        fill_opacity=0.75,
        line_opacity=0.15,
        line_weight=0.3,
        nan_fill_color="rgba(200,200,200,0.3)",
        legend_name="Correlación ONI ↔ anomalía de precipitación (lag óptimo)",
        bins=[-1, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 1],
    ).add_to(mapa)

    # Capa semitransparente que atenúa los municipios NO significativos
    # (FDR) por encima del coroplético, simulando opacidad reducida.
    folium.GeoJson(
        geojson_str,
        style_function=lambda feature: {
            "fillOpacity": 0 if feature["properties"].get("significativo_fdr") else 0.55,
            "fillColor": "#FFFFFF",
            "weight": 0,
            "color": "transparent",
        },
    ).add_to(mapa)

    # Capa de tooltip + resaltado del municipio seleccionado
    folium.GeoJson(
        geojson_str,
        style_function=lambda feature: {
            "fillOpacity": 0,
            "weight": 2.5 if feature["properties"].get("id_mun") == muni_seleccionado
                      else 0.2,
            "color": "#000000" if feature["properties"].get("id_mun") == muni_seleccionado
                     else "transparent",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["id_mun", "name_mun", "correlacion", "lag", "p_valor_fdr",
                    "significativo_fdr", "spearman", "concuerda_spearman"],
            aliases=["Código:", "Municipio:", "Correlación (Pearson):", "Lag (meses):",
                     "p-valor (FDR):", "Significativo (FDR):", "Correlación (Spearman):",
                     "¿Concuerda Pearson/Spearman?:"],
            localize=True,
        ),
    ).add_to(mapa)

    Fullscreen().add_to(mapa)

    css = """
    <style>
    .legend.leaflet-control {
        bottom: 10px !important;
        right: 20px !important;
        left: auto !important;
        top: auto !important;
        font-size: 11px !important;
        max-width: 220px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    </style>
    """
    mapa.get_root().html.add_child(folium.Element(css))

    return mapa


# ─────────────────────────────────────────────
# 7. MAPA COROPLÉTICO DE PRECIPITACIÓN
# ─────────────────────────────────────────────

def mapa_precipitacion(
    df_combinado: pd.DataFrame,
    mes: int,
    anio: int,
    muni_seleccionado: str | None = None,
) -> folium.Map:

    import geopandas as gpd

    gdf = _cargar_geodataframe()

    # Filtrar solo el mes y año seleccionados — un valor por municipio
    df_mes = (
        df_combinado[(df_combinado["mes"] == mes) & (df_combinado["anio"] == anio)]
        [["muni_code", "precip_mm"]]
        .copy()
    )
    df_mes["muni_code"] = df_mes["muni_code"].astype(str).str.zfill(5)

    if df_mes.empty:
        raise ValueError(
            f"No hay datos de precipitación para mes={mes}, año={anio}."
        )

    # Unir a geometrías
    gdf = gdf.merge(df_mes, left_on="id_mun", right_on="muni_code", how="left")
    geojson_str = gdf.to_json()

    nombre_mes = NOMBRES_MESES[mes - 1]

    mapa = folium.Map(
        location=[4.5, -74.0],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    folium.Choropleth(
        geo_data=geojson_str,
        data=df_mes,
        columns=["muni_code", "precip_mm"],
        key_on="feature.properties.id_mun",
        fill_color="Blues",      # escala de azules: más oscuro = más lluvia
        fill_opacity=0.75,
        line_opacity=0.15,
        line_weight=0.3,
        nan_fill_color="rgba(200,200,200,0.3)",
        legend_name=f"Precipitación (mm) — {nombre_mes} {anio}",
    ).add_to(mapa)

    # Capa de tooltip + resaltado del municipio seleccionado
    folium.GeoJson(
        geojson_str,
        style_function=lambda feature: {
            "fillOpacity": 0,
            "weight": 2.5 if feature["properties"].get("id_mun") == muni_seleccionado
                      else 0.2,
            "color": "#000000" if feature["properties"].get("id_mun") == muni_seleccionado
                     else "transparent",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["id_mun", "name_mun", "precip_mm"],
            aliases=["Código:", "Municipio:", "Precipitación (mm):"],
            localize=True,
        ),
    ).add_to(mapa)

    Fullscreen().add_to(mapa)

    # Mover leyenda del Choropleth a esquina inferior derecha para evitar
    # que quede sobre el contenido del mapa.
    css = """
    <style>
    .legend.leaflet-control {
        bottom: 10px !important;
        right: 20px !important;
        left: auto !important;
        top: auto !important;
        font-size: 11px !important;
        max-width: 220px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    </style>
    """
    mapa.get_root().html.add_child(folium.Element(css))

    return mapa


# ─────────────────────────────────────────────
# 8. SERIE TEMPORAL DE ANOMALÍAS
# ─────────────────────────────────────────────

def serie_temporal_anomalia(df_anomalias: pd.DataFrame, titulo: str = "") -> go.Figure:
    """
    Barras de anomalía de precipitación en mm (precip_mm - p50),
    coloreadas en rojo para déficit y azul para exceso,
    con línea ONI en eje secundario y bandas de fondo por fase ENOS.

    Parámetros
    ----------
    df_anomalias : salida de analytics.detectar_anomalias()
                   debe tener: date, precip_mm, p50, oni, fase_oni, anomalia
    titulo       : nombre del municipio o 'Nacional'
    """
    df = df_anomalias.sort_values("date").copy()
    df["anomalia_mm"] = df["precip_mm"] - df["p50"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bandas de fondo por fase ONI
    fig.add_hrect(
        y0=0.5, y1=4,
        fillcolor="rgba(226,75,74,0.06)",
        line_width=0,
        secondary_y=True,
    )
    fig.add_hrect(
        y0=-4, y1=-0.5,
        fillcolor="rgba(24,95,165,0.06)",
        line_width=0,
        secondary_y=True,
    )

    # Barras de anomalía coloreadas por tipo
    for tipo, color in [("déficit", "#E24B4A"), ("exceso", "#185FA5"), ("normal", "#6baed6")]:
        mask = df["anomalia"] == tipo
        opacidad = 0.85 if tipo != "normal" else 0.45
        label = {"déficit": "Déficit (< P5 OMM)",
                 "exceso":  "Exceso (> P95 OMM)",
                 "normal":  "Normal"}[tipo]
        fig.add_trace(
            go.Bar(
                x=df.loc[mask, "date"],
                y=df.loc[mask, "anomalia_mm"],
                name=label,
                marker_color=color,
                marker_opacity=opacidad,
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Anomalía: %{y:.1f} mm<br>"
                    f"Tipo: {label}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    # Línea cero
    fig.add_hline(
        y=0, line_dash="dot",
        line_color="rgba(100,100,100,0.4)",
        line_width=1,
        secondary_y=False,
    )

    # Línea ONI
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["oni"],
            name="Índice ONI",
            mode="lines",
            line=dict(color="#333333", width=2),
            hovertemplate="<b>%{x|%b %Y}</b><br>ONI: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=f"Anomalía de precipitación y ONI — {titulo}", font_size=13),
        barmode="relative",
        legend=dict(orientation="h", y=-0.18, font_size=11),
        margin=dict(t=50, b=60, l=60, r=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_yaxes(
        title_text="Anomalía de precipitación (mm)",
        secondary_y=False,
        zeroline=True,
        zerolinecolor="rgba(150,150,150,0.4)",
        gridcolor="rgba(200,200,200,0.4)",
    )
    fig.update_yaxes(
        title_text="Índice ONI (°C)",
        secondary_y=True,
        range=[-3.5, 3.5],
        showgrid=False,
    )
    fig.update_xaxes(showgrid=False)

    return fig


# ─────────────────────────────────────────────
# 9. RIESGO PROSPECTIVO (ONI PREDICHO)
# ─────────────────────────────────────────────

CONFIANZA_OPACIDAD = {"Alta": 1.0, "Media": 0.55, "Baja": 0.28, "Vencido": 0.35}
CONFIANZA_SIMBOLO  = {"Alta": "circle", "Media": "diamond", "Baja": "x", "Vencido": "triangle-down"}


def riesgo_prospectivo(df_riesgo: pd.DataFrame, titulo: str = "") -> go.Figure:
    """
    Panel de riesgo prospectivo: ONI pronosticado (promedio del ensamble
    CPC/IRI) con su nivel de confianza por horizonte de pronóstico, y la
    anomalía de precipitación esperada para el municipio activo, proyectada
    aplicando hacia adelante la sensibilidad histórica ya validada
    (lag óptimo deseasonalizado, con p-valor corregido).

    La confianza por horizonte es una categorización cualitativa basada en
    la habilidad de pronóstico de ENOS reportada en verificaciones CPC/IRI
    (alta 1-3 meses, moderada 4-6, baja >6 meses), no una incertidumbre
    calculada del spread real del ensamble de modelos (no disponible en
    este insumo). El horizonte mismo es una aproximación, ya que el ETL no
    conserva la fecha de emisión real de cada pronóstico — ver
    analytics.proyectar_riesgo_prospectivo().

    Parámetros
    ----------
    df_riesgo : salida de analytics.proyectar_riesgo_prospectivo()
    titulo    : nombre del municipio o 'Nacional'
    """
    if df_riesgo.empty:
        return go.Figure()

    df = df_riesgo.copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.58],
        vertical_spacing=0.10,
        subplot_titles=(
            "ONI pronosticado (promedio del ensamble CPC/IRI)",
            "Anomalía de precipitación esperada (proyectada desde el lag histórico)",
        ),
    )

    for nivel in ["Alta", "Media", "Baja", "Vencido"]:
        sub = df[df["confianza_pronostico"] == nivel]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["date"], y=sub["oni_predicho"],
                mode="markers",
                name=f"Confianza {nivel}",
                marker=dict(size=11, symbol=CONFIANZA_SIMBOLO[nivel],
                            opacity=CONFIANZA_OPACIDAD[nivel], color="#D85A30",
                            line=dict(width=1, color="#333333")),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>ONI predicho: %{y:.2f}<br>"
                    f"Confianza: {nivel}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["oni_predicho"],
            mode="lines", line=dict(color="#333333", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="rgba(136,135,128,0.10)",
                  line_width=0, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.4)", row=1, col=1)

    if df["anomalia_esperada_mm"].notna().any():
        colores = ["#185FA5" if v >= 0 else "#E24B4A" for v in df["anomalia_esperada_mm"]]
        opacidades = [CONFIANZA_OPACIDAD[c] for c in df["confianza_pronostico"]]
        fig.add_trace(
            go.Bar(
                x=df["mes_impacto_esperado"], y=df["anomalia_esperada_mm"],
                name="Anomalía esperada (mm)",
                marker=dict(color=colores, opacity=opacidades),
                hovertemplate=(
                    "<b>Impacto esperado: %{x|%b %Y}</b><br>"
                    "Anomalía proyectada: %{y:.1f} mm<extra></extra>"
                ),
            ),
            row=2, col=1,
        )
        es_significativo = bool(df["relacion_significativa"].iloc[0])
        nota_sig = (
            "Relación histórica del municipio significativa tras corrección FDR."
            if es_significativo else
            "⚠ Relación histórica del municipio NO significativa tras corrección FDR "
            "— proyección de baja confiabilidad."
        )
    else:
        nota_sig = "Sin lag óptimo disponible para este municipio: no se proyecta anomalía esperada."

    es_vencido = bool(df["pronostico_vencido"].iloc[0])
    titulo_fig = (
        f"Riesgo prospectivo ONI ↔ precipitación — {titulo} "
        + ("⚠ PRONÓSTICO VENCIDO" if es_vencido else "")
    )

    fig.update_layout(
        title=dict(text=titulo_fig,
                    font=dict(size=13, color="#B33A3A" if es_vencido else "#1f1f1f")),
        legend=dict(orientation="h", y=-0.20, font_size=10),
        margin=dict(t=60, b=90, l=60, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.add_annotation(
        text=nota_sig, xref="paper", yref="paper", x=0, y=-0.26,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    fig.update_yaxes(title_text="ONI", range=[-3, 3], row=1, col=1)
    fig.update_yaxes(title_text="Anomalía (mm)", zeroline=True,
                      zerolinecolor="rgba(150,150,150,0.4)", row=2, col=1)
    fig.update_xaxes(showgrid=False)

    return fig
