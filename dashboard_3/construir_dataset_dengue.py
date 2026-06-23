"""
construir_dataset_dengue.py
---------------------------
Construye un dataset de análisis integrando:
    - Casos de dengue (SIVIGILA) por municipio y mes
    - Precipitación mensual (CHIRPS) por municipio
    - Índice ONI histórico (NOAA)

Lógica del join:
    1. CHIRPS + Dengue  → join por date + muni_code (left join desde CHIRPS)
       Si no hay casos ese mes → se rellena con 0
    2. Resultado + ONI  → join por date
       Si no hay ONI ese mes  → se rellena con NaN

Salida:
    date            datetime64
    muni_code       str (5 dígitos)
    anio            int
    mes             int
    precip_mean_mm  float
    precip_min      float
    precip_max      float
    std_dev         float
    n_pixels        int
    cod_eve         int
    casos           int
    value_oni       float
"""

import glob
import os

import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

CHIRPS_DIR  = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\chirps_municipal"
ONI_DIR     = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\noaa_historical"
DENGUE_PATH = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\processed\dengue.parquet"
OUTPUT_DIR  = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\processed"

# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def cargar_dengue(ruta_parquet: str, muni_code: str) -> pd.DataFrame:
    """
    Lee el parquet de dengue y filtra por municipio.

    Parámetros
    ----------
    ruta_parquet : ruta al parquet generado por etl_enfermedades.py
    muni_code    : código DIVIPOLA de 5 dígitos (ej. '81001')

    Retorna
    -------
    DataFrame con columnas: date, muni_code, anio, mes, cod_eve, casos
    """
    print("Cargando datos de dengue...")

    if not os.path.exists(ruta_parquet):
        raise FileNotFoundError(f"No se encontró el parquet de dengue: {ruta_parquet}")

    df = pd.read_parquet(ruta_parquet)
    df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
    df = df[df["muni_code"] == muni_code].copy()

    if df.empty:
        raise ValueError(f"No hay datos de dengue para muni_code={muni_code}")

    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "muni_code", "anio", "mes", "cod_eve", "casos"]]
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  → {len(df)} registros · {df['anio'].min()}–{df['anio'].max()} · "
          f"{df['casos'].sum():,} casos totales")
    return df


def cargar_chirps(data_dir: str, muni_code: str,
                  anio_inicio: int, anio_fin: int) -> pd.DataFrame:
    """
    Lee los parquet anuales de CHIRPS para el rango de años y municipio indicados.

    Parámetros
    ----------
    data_dir    : carpeta raíz con subcarpetas year=XXXX
    muni_code   : código DIVIPOLA de 5 dígitos
    anio_inicio : primer año del rango (inclusive)
    anio_fin    : último año del rango (inclusive)

    Retorna
    -------
    DataFrame con todas las columnas del parquet CHIRPS + date normalizada
    """
    print(f"Cargando CHIRPS {anio_inicio}–{anio_fin} para municipio {muni_code}...")

    archivos = [
        os.path.join(data_dir, f"year={anio}", f"fact_chirps_muni_{anio}.parquet")
        for anio in range(anio_inicio, anio_fin + 1)
    ]
    archivos_existentes = [f for f in archivos if os.path.exists(f)]

    if not archivos_existentes:
        raise FileNotFoundError(
            f"No se encontraron parquet CHIRPS en {data_dir} "
            f"para el rango {anio_inicio}–{anio_fin}"
        )

    dfs = []
    for f in archivos_existentes:
        df = pd.read_parquet(f)
        df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
        df = df[df["muni_code"] == muni_code]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No hay datos CHIRPS para muni_code={muni_code}")

    df_chirps = pd.concat(dfs, ignore_index=True)
    df_chirps["date"] = pd.to_datetime(df_chirps["date"])
    df_chirps = df_chirps.sort_values("date").reset_index(drop=True)

    # Renombrar year → anio si existe
    if "year" in df_chirps.columns:
        df_chirps = df_chirps.rename(columns={"year": "anio"})

    print(f"  → {len(df_chirps)} registros · "
          f"{df_chirps['date'].dt.year.min()}–{df_chirps['date'].dt.year.max()}")
    return df_chirps


def cargar_oni(data_dir: str, anio_inicio: int, anio_fin: int) -> pd.DataFrame:
    """
    Lee los parquet anuales de ONI para el rango de años indicado.
    ONI es el mismo para todos los municipios — no requiere filtro de municipio.

    Parámetros
    ----------
    data_dir    : carpeta raíz con subcarpetas year=XXXX
    anio_inicio : primer año del rango (inclusive)
    anio_fin    : último año del rango (inclusive)

    Retorna
    -------
    DataFrame con columnas: date, value_oni
    """
    print(f"Cargando ONI {anio_inicio}–{anio_fin}...")

    archivos = [
        os.path.join(data_dir, f"year={anio}", f"noaa_oni_{anio}.parquet")
        for anio in range(anio_inicio, anio_fin + 1)
    ]
    archivos_existentes = [f for f in archivos if os.path.exists(f)]

    if not archivos_existentes:
        raise FileNotFoundError(
            f"No se encontraron parquet ONI en {data_dir} "
            f"para el rango {anio_inicio}–{anio_fin}"
        )

    df_oni = pd.concat(
        [pd.read_parquet(f) for f in archivos_existentes],
        ignore_index=True,
    )
    df_oni["date"] = pd.to_datetime(df_oni["date"])
    df_oni = df_oni[["date", "value_oni"]].sort_values("date").reset_index(drop=True)

    print(f"  → {len(df_oni)} registros · "
          f"{df_oni['date'].dt.year.min()}–{df_oni['date'].dt.year.max()}")
    return df_oni


def construir_dataset(
    df_dengue: pd.DataFrame,
    df_chirps: pd.DataFrame,
    df_oni:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Une las tres fuentes en un único dataset de análisis.

    Join en dos pasos:
        1. CHIRPS + Dengue  → left join por date + muni_code
           Meses sin casos  → casos = 0, cod_eve = 210
        2. Resultado + ONI  → left join por date
           Meses sin ONI    → value_oni = NaN

    Parámetros
    ----------
    df_dengue : salida de cargar_dengue()
    df_chirps : salida de cargar_chirps()
    df_oni    : salida de cargar_oni()

    Retorna
    -------
    DataFrame con columnas de las tres fuentes ordenado por date
    """
    print("Construyendo dataset integrado...")

    # Paso 1: CHIRPS + Dengue (left join — base es CHIRPS)
    df = df_chirps.merge(
        df_dengue[["date", "muni_code", "cod_eve", "casos"]],
        on=["date", "muni_code"],
        how="left",
    )


    cod_eve_valor = int(df_dengue["cod_eve"].iloc[0])
    df["casos"]   = df["casos"].fillna(0).astype(int)
    df["cod_eve"] = df["cod_eve"].fillna(cod_eve_valor).astype(int)

    # Paso 2: Resultado + ONI (left join por date)
    df = df.merge(df_oni, on="date", how="left")
    # value_oni faltante → NaN (ya queda así por defecto en left join)

    # Ordenar columnas
    cols_orden = [
        "date", "muni_code", "anio", "mes",
        "precip_mean_mm", "precip_min", "precip_max", "std_dev", "n_pixels",
        "cod_eve", "casos",
        "value_oni",
    ]
    # Solo incluir columnas que existen
    cols_orden = [c for c in cols_orden if c in df.columns]
    df = df[cols_orden].sort_values("date").reset_index(drop=True)

    print(f"  → {len(df)} registros · "
          f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    print(f"  → Meses con casos > 0: {(df['casos'] > 0).sum()}")
    print(f"  → Meses sin ONI (NaN): {df['value_oni'].isna().sum()}")
    print(f"  → Meses sin precipitación (NaN): {df['precip_mean_mm'].isna().sum()}")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Parámetros de análisis ────────────────────────────────────────────
    MUNI_CODE   = "52001"
    ANIO_INICIO = 2013
    ANIO_FIN    = 2023

    # ── Carga ─────────────────────────────────────────────────────────────
    df_dengue = cargar_dengue(DENGUE_PATH, MUNI_CODE)
    df_chirps = cargar_chirps(CHIRPS_DIR, MUNI_CODE, ANIO_INICIO, ANIO_FIN)
    df_oni    = cargar_oni(ONI_DIR, ANIO_INICIO, ANIO_FIN)

    # ── Integración ───────────────────────────────────────────────────────
    df_final = construir_dataset(df_dengue, df_chirps, df_oni)

    # ── Guardar ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nombre_archivo = f"dataset_{MUNI_CODE}_{ANIO_INICIO}_{ANIO_FIN}.parquet"
    ruta_salida    = os.path.join(OUTPUT_DIR, nombre_archivo)
    df_final.to_parquet(ruta_salida, index=False)

    print(f"\n✓ Dataset guardado: {ruta_salida}")
    print(f"\nMuestra:")
    print(df_final.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
