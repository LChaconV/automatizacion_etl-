"""
data_loader.py
--------------
Carga y cacheo de datos de precipitación mensual (CHIRPS) y ONI (NOAA).
Todas las funciones usan @st.cache_data para evitar re-lecturas en cada
interacción del dashboard.
"""

import os
import glob
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# RUTAS  (ajusta si cambian)
# ─────────────────────────────────────────────

CHIRPS_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\chirps_municipal"
ONI_DIR    = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\noaa_historical"
CLIMA_REF  = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_2\referencia\climatologia_referencia_1991_2020.parquet"


# ─────────────────────────────────────────────
# CARGA DE PRECIPITACIÓN
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Cargando datos de precipitación...")
def cargar_precipitacion() -> pd.DataFrame:
    """
    Lee todos los parquet anuales de CHIRPS y retorna un DataFrame limpio.

    Columnas resultantes:
        date          datetime64  primer día del mes
        muni_code     str (5 dígitos, zero-padded)
        dep_code      str (2 dígitos)
        precip_mm     float       precipitación media mensual en mm
        mes           int         1-12
        anio          int
    """
    archivos = glob.glob(os.path.join(CHIRPS_DIR, "year=*", "*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos parquet en {CHIRPS_DIR}")

    df = pd.concat(
        [pd.read_parquet(f, columns=["date", "muni_code", "precip_mean_mm", "n_pixels"])
         for f in archivos],
        ignore_index=True,
    )

    df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
    df["dep_code"]  = df["muni_code"].str[:2]
    df["date"]      = pd.to_datetime(df["date"])
    df["mes"]       = df["date"].dt.month
    df["anio"]      = df["date"].dt.year
    df = df.rename(columns={"precip_mean_mm": "precip_mm"})
    df = df.dropna(subset=["precip_mm"])
    df = df.sort_values(["muni_code", "date"]).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# CARGA DE ONI
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Cargando datos ONI...")
def cargar_oni() -> pd.DataFrame:
    """
    Lee todos los parquet anuales de ONI y retorna un DataFrame limpio.

    Columnas resultantes:
        date      datetime64  primer día del mes
        oni       float       valor ONI mensual
        fase_oni  str         'El Niño' | 'La Niña' | 'Neutral'
        mes       int         1-12
        anio      int
    """
    archivos = glob.glob(os.path.join(ONI_DIR, "year=*", "*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos parquet en {ONI_DIR}")

    # Carga flexible: toma todas las columnas y detecta la del valor ONI
    dfs = [pd.read_parquet(f) for f in archivos]
    df  = pd.concat(dfs, ignore_index=True)

    # Normalizar nombre de columna ONI (acepta variantes comunes)
    columnas_oni = [c for c in df.columns if "oni" in c.lower()]
    if not columnas_oni:
        raise ValueError(f"No se encontró columna ONI en los parquet. Columnas disponibles: {df.columns.tolist()}")
    df = df.rename(columns={columnas_oni[0]: "oni"})

    # Normalizar fecha
    columnas_fecha = [c for c in df.columns if "date" in c.lower() or "fecha" in c.lower()]
    if columnas_fecha:
        df = df.rename(columns={columnas_fecha[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])

    df["mes"]  = df["date"].dt.month
    df["anio"] = df["date"].dt.year
    df["fase_oni"] = df["oni"].apply(_clasificar_fase)
    df = df[["date", "oni", "fase_oni", "mes", "anio"]].dropna(subset=["oni"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def _clasificar_fase(valor: float) -> str:
    """Clasifica el valor ONI en fase ENOS."""
    if valor >= 0.5:
        return "El Niño"
    elif valor <= -0.5:
        return "La Niña"
    else:
        return "Neutral"


# ─────────────────────────────────────────────
# COMBINACIÓN PRECIPITACIÓN + ONI
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Combinando series...")
def combinar_datos(df_precip: pd.DataFrame, df_oni: pd.DataFrame) -> pd.DataFrame:
    """
    Une precipitación y ONI por fecha (año-mes).
    El ONI es un valor único para todo el país, se replica para cada municipio.

    Retorna DataFrame con todas las columnas de precipitación + oni + fase_oni.
    """
    df_oni_slim = df_oni[["date", "oni", "fase_oni"]].copy()
    df = df_precip.merge(df_oni_slim, on="date", how="left")
    return df


# ─────────────────────────────────────────────
# AGREGACIÓN NACIONAL (ponderada por n_pixels)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Calculando serie nacional...")
def agregar_nacional(df_precip: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega precipitación a nivel nacional usando promedio ponderado por n_pixels.
    Retorna una serie mensual con la misma estructura que un municipio individual,
    con muni_code = 'NACIONAL'.
    """
    grp = (
        df_precip
        .groupby("date")
        .apply(
            lambda g: pd.Series({
                "precip_mm": (g["precip_mm"] * g["n_pixels"]).sum() / g["n_pixels"].sum(),
                "n_pixels":  g["n_pixels"].sum(),
            })
        )
        .reset_index()
    )
    grp["muni_code"] = "NACIONAL"
    grp["dep_code"]  = "NACIONAL"
    grp["mes"]       = grp["date"].dt.month
    grp["anio"]      = grp["date"].dt.year
    return grp


# ─────────────────────────────────────────────
# LISTAS PARA FILTROS DEL DASHBOARD
# ─────────────────────────────────────────────

DIVIPOLA_PATH = (
    r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon"
    r"\ETL_code\dashboard_2\divipola\divipola.parquet"
)


@st.cache_data(show_spinner="Cargando nombres de municipios...")
def cargar_catalogo_municipios() -> pd.DataFrame:
    """
    Lee el parquet DIVIPOLA y retorna un catálogo con id_mun y name_mun.
    Usado para mostrar nombres legibles en el selector del dashboard y
    para traducir nombre → muni_code al filtrar los datos.

    Columnas resultantes:
        muni_code   str (5 dígitos, zero-padded)  código DIVIPOLA
        nombre      str                            nombre del municipio
    """
    # Leer solo las columnas necesarias con pandas — no requiere geopandas
    # ya que no necesitamos la geometría para el catálogo de nombres.
    df = pd.read_parquet(DIVIPOLA_PATH, columns=["id_mun", "name_mun"])
    df["muni_code"] = df["id_mun"].astype(str).str.zfill(5)
    df = df.rename(columns={"name_mun": "nombre"})
    df = df[["muni_code", "nombre"]].drop_duplicates("muni_code")
    df = df.sort_values("nombre").reset_index(drop=True)
    return df


@st.cache_data
def obtener_nombres_municipios(
    df_precip: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> list[str]:
    """
    Retorna lista de nombres de municipios disponibles en los datos,
    ordenada alfabéticamente. Incluye 'Nacional' como primera opción.

    Parámetros
    ----------
    df_precip : DataFrame de precipitación (para saber qué muni_codes existen)
    catalogo  : salida de cargar_catalogo_municipios()
    """
    codes_disponibles = df_precip["muni_code"].unique()
    nombres = (
        catalogo[catalogo["muni_code"].isin(codes_disponibles)]["nombre"]
        .sort_values()
        .tolist()
    )
    return ["Nacional"] + nombres


@st.cache_data
def nombre_a_codigo(nombre: str, catalogo: pd.DataFrame) -> str | None:
    """
    Traduce el nombre seleccionado en el selector al muni_code correspondiente.
    Retorna None si el nombre es 'Nacional'.
    """
    if nombre == "Nacional":
        return None
    fila = catalogo[catalogo["nombre"] == nombre]
    if fila.empty:
        return None
    return fila.iloc[0]["muni_code"]


@st.cache_data
def obtener_rango_anios(df: pd.DataFrame) -> tuple[int, int]:
    """Año mínimo y máximo disponibles en los datos."""
    return int(df["anio"].min()), int(df["anio"].max())


# ─────────────────────────────────────────────
# CARGA DE CLIMATOLOGÍA DE REFERENCIA OMM 1991–2020
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Cargando climatología de referencia...")
def cargar_climatologia_referencia() -> pd.DataFrame:
    """
    Lee el parquet de climatología de referencia OMM 1991–2020.
    Generado por construir_climatologia_referencia.py.

    Columnas: muni_code, mes, p5, p50, p95, media, std, n_anios,
              periodo_ref_inicio, periodo_ref_fin

    Este archivo define los umbrales P5/P95 fijos para detección de
    anomalías, independientemente del filtro de período del dashboard.
    """
    if not os.path.exists(CLIMA_REF):
        raise FileNotFoundError(
            f"No se encontró el archivo de climatología de referencia en:\n{CLIMA_REF}\n"
            "Ejecuta construir_climatologia_referencia.py primero."
        )

    df = pd.read_parquet(CLIMA_REF)
    df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
    return df
