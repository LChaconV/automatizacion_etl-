# ============================================================
#  data_loader.py
#  Módulo de carga y procesamiento base
#  Dashboard "Impacto ENSO en Colombia"
#  Laura Andrea Chacón Velásquez · 2025
# ============================================================

import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import streamlit as st
from pathlib import Path

# ─── RUTAS ───────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code")

CHIRPS_DIR  = BASE / "data/processed/chirps_municipal"
NOAA_HIST   = BASE / "data/processed/noaa_historical"
NOAA_PRED   = BASE / "data/processed/noaa_prediction"
DIVIPOLA    = BASE / "dashboard/divipola.parquet"

# ─── AÑOS DISPONIBLES ────────────────────────────────────────────────────────
CHIRPS_YEARS_AVAILABLE = sorted([
    int(p.name.split("=")[1])
    for p in Path(CHIRPS_DIR).glob("year=*")
    if p.is_dir() and "=" in p.name
])
NOAA_HIST_YEARS_AVAILABLE = sorted([
    int(p.name.split("=")[1])
    for p in Path(NOAA_HIST).glob("year=*")
    if p.is_dir() and "=" in p.name
])

# ─── CLASIFICACIÓN ONI ────────────────────────────────────────────────────────
def classify_oni(v: float) -> str:
    """Clasifica el valor ONI en fase según umbral oficial NOAA (±0.5 °C)."""
    if pd.isna(v):   return "Neutro"
    if v >=  0.5:    return "El Niño"
    if v <= -0.5:    return "La Niña"
    return "Neutro"

# ─── CARGA DIVIPOLA ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando geometría DIVIPOLA…")
def load_divipola() -> gpd.GeoDataFrame:
    """
    Carga la geometría municipal.
    Garantiza:
      - id_mun como string de 5 dígitos con ceros a la izquierda
      - id_dept como string de 2 dígitos
      - CRS en WGS84 (EPSG:4326) para compatibilidad con Plotly
    """
    gdf = gpd.read_parquet(DIVIPOLA)
    gdf["id_mun"]  = gdf["id_mun"].astype(str).str.zfill(5)
    gdf["id_dept"] = gdf["id_mun"].str[:2]

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:9377")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Simplificar geometría para mejor rendimiento en el mapa
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.005, preserve_topology=True)

    return gdf[["id_mun", "name_mun", "id_dept", "geometry"]].copy()


# ─── CARGA NOAA HISTÓRICO ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando índice ONI (NOAA)…")
def load_oni_historico() -> pd.DataFrame:
    """
    Carga todos los años del ONI histórico.
    Columnas resultantes:
      date (datetime), value_oni (float), oni_phase (str)
    """
    frames = []
    for y in NOAA_HIST_YEARS_AVAILABLE:
        p = NOAA_HIST / f"year={y}/noaa_oni_{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))

    if not frames:
        st.error("No se encontraron datos ONI históricos.")
        st.stop()

    df = pd.concat(frames, ignore_index=True)
    df["date"]      = pd.to_datetime(df["date"])
    df["value_oni"] = pd.to_numeric(df["value_oni"], errors="coerce")
    df["oni_phase"] = df["value_oni"].apply(classify_oni)

    return df.sort_values("date").reset_index(drop=True)


# ─── CARGA NOAA PREDICCIÓN ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando predicciones IRI…")
def load_oni_prediccion() -> pd.DataFrame:
    """
    Carga todos los años de predicción ONI del IRI.
    Columnas resultantes:
      date (datetime), prediction_period (str),
      prediction_oni (float), oni_phase (str)
    """
    frames = []
    for p in sorted(Path(NOAA_PRED).glob("year=*/noaa_oni_pred_*.parquet")):
        frames.append(pd.read_parquet(p))

    if not frames:
        return pd.DataFrame(columns=["date", "prediction_period",
                                     "prediction_oni", "oni_phase"])

    df = pd.concat(frames, ignore_index=True)
    df["date"]           = pd.to_datetime(df["date"])
    df["prediction_oni"] = pd.to_numeric(df["prediction_oni"], errors="coerce")
    df["oni_phase"]      = df["prediction_oni"].apply(classify_oni)

    return df.sort_values("date").reset_index(drop=True)


# ─── CARGA CHIRPS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando precipitación CHIRPS…")
def load_chirps(year_start: int, year_end: int) -> pd.DataFrame:
    """
    Carga los parquets CHIRPS para el rango de años indicado.
    Columnas resultantes:
      date, year, month, muni_code, id_dept,
      precip_mean_mm, precip_min, precip_max, std_dev, n_pixels
    """
    years = [y for y in CHIRPS_YEARS_AVAILABLE
             if year_start <= y <= year_end]

    if not years:
        st.error(f"Sin datos CHIRPS para {year_start}–{year_end}.")
        st.stop()

    frames = []
    for y in years:
        p = CHIRPS_DIR / f"year={y}/fact_chirps_muni_{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))

    df = pd.concat(frames, ignore_index=True)
    df["date"]      = pd.to_datetime(df["date"])
    df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
    df["id_dept"]   = df["muni_code"].str[:2]
    df["year"]      = df["date"].dt.year
    df["month"]     = df["date"].dt.month

    # Columnas numéricas
    for col in ["precip_mean_mm", "precip_min", "precip_max", "std_dev"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["muni_code", "date"]).reset_index(drop=True)


# ─── CÁLCULO DE ANOMALÍAS ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Calculando anomalías de precipitación…")
def compute_anomalies(chirps: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la anomalía de precipitación respecto al promedio histórico
    del mismo municipio y mes (baseline = todo el período disponible).

    Nuevas columnas:
      hist_mean   : promedio histórico del municipio en ese mes
      anomaly_mm  : desviación en mm (precip - hist_mean)
      anomaly_pct : desviación porcentual
      z_score     : número de desviaciones estándar respecto al histórico
    """
    # Baseline: media histórica por municipio × mes
    baseline = (
        chirps.groupby(["muni_code", "month"])["precip_mean_mm"]
        .agg(hist_mean="mean", hist_std="std")
        .reset_index()
    )

    df = chirps.merge(baseline, on=["muni_code", "month"], how="left")

    df["anomaly_mm"]  = (df["precip_mean_mm"] - df["hist_mean"]).round(2)
    df["anomaly_pct"] = (
        (df["anomaly_mm"] / df["hist_mean"].replace(0, np.nan)) * 100
    ).round(1)
    df["z_score"] = (
        df["anomaly_mm"] / df["hist_std"].replace(0, np.nan)
    ).round(3)

    return df


# ─── UNIÓN CON ONI ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Uniendo precipitación con fases ONI…")
def merge_oni(chirps: pd.DataFrame, oni: pd.DataFrame) -> pd.DataFrame:
    """
    Une cada registro CHIRPS con el valor ONI del mismo mes.
    Nuevas columnas:
      value_oni (float), oni_phase (str)
    """
    # Normalizar a primer día del mes para el join
    oni_slim = oni[["date", "value_oni", "oni_phase"]].copy()
    oni_slim["date"] = oni_slim["date"].dt.to_period("M").dt.to_timestamp()

    chirps_m = chirps.copy()
    chirps_m["date_m"] = chirps_m["date"].dt.to_period("M").dt.to_timestamp()

    merged = chirps_m.merge(
        oni_slim.rename(columns={"date": "date_m"}),
        on="date_m", how="left"
    ).drop(columns=["date_m"])

    merged["value_oni"] = merged["value_oni"].fillna(0)
    merged["oni_phase"] = merged["oni_phase"].fillna("Neutro")

    return merged


# ─── CORRELACIÓN ONI × PRECIPITACIÓN ────────────────────────────────────────
@st.cache_data(show_spinner="Calculando correlación ONI por municipio…")
def compute_oni_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la correlación de Pearson entre ONI y anomalía de precipitación
    para cada municipio, usando toda la serie histórica disponible.

    Resultado: un DataFrame con una fila por municipio.
    Columnas:
      muni_code, corr_oni (float -1 a 1), n_meses (int),
      mean_anomaly_nino, mean_anomaly_nina, mean_anomaly_neutro
    """
    # Correlación global ONI × anomalía por municipio
    corr = (
        df.groupby("muni_code")
        .apply(
            lambda g: pd.Series({
                "corr_oni": (
                    g[["value_oni", "anomaly_mm"]]
                    .dropna()
                    .pipe(lambda x: x["value_oni"].corr(x["anomaly_mm"]))
                    if len(g.dropna(subset=["value_oni", "anomaly_mm"])) >= 12
                    else np.nan
                ),
                "n_meses": len(g),
            })
        )
        .reset_index()
    )

    # Anomalía media por fase ONI
    phase_means = (
        df.groupby(["muni_code", "oni_phase"])["anomaly_mm"]
        .mean()
        .unstack(fill_value=0)
        .rename(columns={
            "El Niño": "mean_anom_nino",
            "La Niña": "mean_anom_nina",
            "Neutro":  "mean_anom_neutro",
        })
        .reset_index()
    )

    result = corr.merge(phase_means, on="muni_code", how="left")
    result["corr_oni"] = result["corr_oni"].round(3)

    return result


# ─── FUNCIÓN PRINCIPAL: PIPELINE COMPLETO ────────────────────────────────────
def build_dataset(year_start: int, year_end: int):
    """
    Ejecuta el pipeline completo de carga y procesamiento.
    Retorna todos los objetos necesarios para las visualizaciones.
    """
    divipola  = load_divipola()
    oni_hist  = load_oni_historico()
    oni_pred  = load_oni_prediccion()
    chirps    = load_chirps(year_start, year_end)
    chirps    = compute_anomalies(chirps)
    chirps    = merge_oni(chirps, oni_hist)
    corr_muni = compute_oni_correlation(chirps)

    # Unir nombres de municipios y departamentos desde DIVIPOLA
    names = divipola[["id_mun", "name_mun", "id_dept"]].copy()
    chirps = chirps.merge(
        names.rename(columns={"id_mun": "muni_code"}),
        on="muni_code", how="left"
    )

    return {
        "chirps":    chirps,
        "oni_hist":  oni_hist,
        "oni_pred":  oni_pred,
        "divipola":  divipola,
        "corr_muni": corr_muni,
    }


# ─── RESUMEN DE VALIDACIÓN ────────────────────────────────────────────────────
def print_summary(data: dict) -> None:
    """Muestra un resumen en consola para verificar que los datos cargaron bien."""
    c  = data["chirps"]
    o  = data["oni_hist"]
    cr = data["corr_muni"]
    d  = data["divipola"]

    print("\n" + "="*55)
    print("  RESUMEN DE DATOS · ENSO Dashboard")
    print("="*55)
    print(f"  CHIRPS")
    print(f"    Filas totales   : {len(c):,}")
    print(f"    Municipios      : {c['muni_code'].nunique():,}")
    print(f"    Período         : {c['date'].min().date()} → {c['date'].max().date()}")
    print(f"    Anomalía media  : {c['anomaly_mm'].mean():.2f} mm")
    print(f"    Rango anomalía  : [{c['anomaly_mm'].min():.1f}, {c['anomaly_mm'].max():.1f}] mm")
    print(f"\n  ONI histórico")
    print(f"    Meses totales   : {len(o):,}")
    print(f"    Período         : {o['date'].min().date()} → {o['date'].max().date()}")
    print(f"    Fase El Niño    : {(o['oni_phase']=='El Niño').sum()} meses")
    print(f"    Fase La Niña    : {(o['oni_phase']=='La Niña').sum()} meses")
    print(f"    Fase Neutro     : {(o['oni_phase']=='Neutro').sum()} meses")
    print(f"\n  Correlación ONI")
    print(f"    Municipios con correlación: {cr['corr_oni'].notna().sum():,}")
    print(f"    Correlación media     : {cr['corr_oni'].mean():.3f}")
    print(f"    Correlación mín/máx   : {cr['corr_oni'].min():.3f} / {cr['corr_oni'].max():.3f}")
    print(f"\n  DIVIPOLA")
    print(f"    Municipios geometría  : {len(d):,}")
    print(f"    CRS                   : {d.crs}")
    print("="*55 + "\n")


# ─── VERIFICACIÓN DIRECTA ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Ejecutando pipeline de datos (2004–2025)…")
    data = build_dataset(2004, 2025)
    print_summary(data)
    print("✅ Pipeline completado sin errores.")