"""
construir_dataset_irag.py
--------------------------
Construye el dataset de análisis integrando:
    - Casos de IRAG inusitada (SIVIGILA) por municipio y mes
    - Precipitación mensual (CHIRPS) por municipio
    - Índice ONI histórico (NOAA)

Todos los municipios disponibles en el parquet de IRAG.

Lógica del join:
    Base : CHIRPS (todos los municipios en el rango de años)
    + IRAG  → left join por date + muni_code
              Meses sin casos → casos = 0, cod_eve = 348
    + ONI   → left join por date
              Meses sin ONI   → value_oni = NaN

Salida:
    dashboard_3/processed/join/dataset_irag_2018_2023.parquet

Columnas:
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

import os
import glob
import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

CHIRPS_DIR  = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\chirps_municipal"
ONI_DIR     = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\noaa_historical"
IRAG_PATH   = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\processed\irag_inusitada.parquet"
OUTPUT_DIR  = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\processed\join"
ANIO_INICIO = 2018
ANIO_FIN    = 2024
OUTPUT_FILE = f"dataset_irag_{ANIO_INICIO}_{ANIO_FIN}.parquet"



# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def cargar_irag(ruta_parquet: str) -> pd.DataFrame:
    """
    Lee el parquet de IRAG inusitada.
    Retorna todos los municipios disponibles.
    """
    print("Cargando datos de IRAG...")

    if not os.path.exists(ruta_parquet):
        raise FileNotFoundError(f"No se encontró el parquet de IRAG: {ruta_parquet}")

    df = pd.read_parquet(ruta_parquet)
    df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
    df["date"]      = pd.to_datetime(df["date"])
    df = df[["date", "muni_code", "cod_eve", "casos"]]
    df = df.sort_values(["muni_code", "date"]).reset_index(drop=True)

    print(f"  → {len(df):,} registros · "
          f"{df['muni_code'].nunique()} municipios · "
          f"{df['casos'].sum():,} casos totales · "
          f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    return df


def cargar_chirps(data_dir: str, anio_inicio: int,
                  anio_fin: int) -> pd.DataFrame:
    """
    Lee los parquet anuales de CHIRPS para el rango de años indicado.
    Carga todos los municipios.
    """
    print(f"Cargando CHIRPS {anio_inicio}–{anio_fin}...")

    archivos = [
        os.path.join(data_dir, f"year={a}", f"fact_chirps_muni_{a}.parquet")
        for a in range(anio_inicio, anio_fin + 1)
        if os.path.exists(
            os.path.join(data_dir, f"year={a}", f"fact_chirps_muni_{a}.parquet")
        )
    ]

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron parquet CHIRPS en {data_dir} "
            f"para el rango {anio_inicio}–{anio_fin}"
        )

    dfs = []
    for f in archivos:
        df = pd.read_parquet(f)
        df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
        df["date"]      = pd.to_datetime(df["date"])
        dfs.append(df)

    df_chirps = pd.concat(dfs, ignore_index=True)

    # Renombrar year → anio si existe
    if "year" in df_chirps.columns:
        df_chirps = df_chirps.rename(columns={"year": "anio"})

    df_chirps = df_chirps.sort_values(["muni_code", "date"]).reset_index(drop=True)

    print(f"  → {len(df_chirps):,} registros · "
          f"{df_chirps['muni_code'].nunique()} municipios · "
          f"{df_chirps['date'].dt.year.min()}–{df_chirps['date'].dt.year.max()}")
    return df_chirps


def cargar_oni(data_dir: str, anio_inicio: int,
               anio_fin: int) -> pd.DataFrame:
    """
    Lee los parquet anuales de ONI para el rango de años indicado.
    ONI es el mismo para todos los municipios.
    """
    print(f"Cargando ONI {anio_inicio}–{anio_fin}...")

    archivos = [
        os.path.join(data_dir, f"year={a}", f"noaa_oni_{a}.parquet")
        for a in range(anio_inicio, anio_fin + 1)
        if os.path.exists(
            os.path.join(data_dir, f"year={a}", f"noaa_oni_{a}.parquet")
        )
    ]

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron parquet ONI en {data_dir} "
            f"para el rango {anio_inicio}–{anio_fin}"
        )

    df_oni = pd.concat(
        [pd.read_parquet(f) for f in archivos],
        ignore_index=True,
    )
    df_oni["date"] = pd.to_datetime(df_oni["date"])
    df_oni = df_oni[["date", "value_oni"]].sort_values("date").reset_index(drop=True)

    print(f"  → {len(df_oni)} registros · "
          f"{df_oni['date'].dt.year.min()}–{df_oni['date'].dt.year.max()}")
    return df_oni


def construir_dataset(
    df_irag:   pd.DataFrame,
    df_chirps: pd.DataFrame,
    df_oni:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Une las tres fuentes en un único dataset de análisis.

    Join en dos pasos:
        1. CHIRPS + IRAG  → left join por date + muni_code
           Meses sin casos → casos = 0, cod_eve = 348
        2. Resultado + ONI → left join por date
           Meses sin ONI   → value_oni = NaN
    """
    print("Construyendo dataset integrado...")

    # Paso 1: CHIRPS + IRAG
    df = df_chirps.merge(
        df_irag[["date", "muni_code", "cod_eve", "casos"]],
        on=["date", "muni_code"],
        how="left",
    )

    # Meses sin casos → 0; cod_eve faltante → 348 (IRAG)
    cod_eve_valor = int(df_irag["cod_eve"].iloc[0])
    df["casos"]   = df["casos"].fillna(0).astype(int)
    df["cod_eve"] = df["cod_eve"].fillna(cod_eve_valor).astype(int)

    # Paso 2: Resultado + ONI
    df = df.merge(df_oni, on="date", how="left")

    # Agregar mes y anio si no existen
    if "mes" not in df.columns:
        df["mes"] = df["date"].dt.month
    if "anio" not in df.columns:
        df["anio"] = df["date"].dt.year

    # Ordenar columnas
    cols_orden = [
        "date", "muni_code", "anio", "mes",
        "precip_mean_mm", "precip_min", "precip_max", "std_dev", "n_pixels",
        "cod_eve", "casos", "value_oni",
    ]
    cols_orden = [c for c in cols_orden if c in df.columns]
    df = df[cols_orden].sort_values(["muni_code", "date"]).reset_index(drop=True)

    print(f"  → {len(df):,} registros · "
          f"{df['muni_code'].nunique()} municipios · "
          f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    print(f"  → Meses con casos > 0    : {(df['casos'] > 0).sum():,}")
    print(f"  → Meses sin ONI (NaN)    : {df['value_oni'].isna().sum():,}")
    print(f"  → Meses sin precip (NaN) : {df['precip_mean_mm'].isna().sum():,}")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    # ── Carga ─────────────────────────────────────────────────────────────
    df_irag   = cargar_irag(IRAG_PATH)
    df_chirps = cargar_chirps(CHIRPS_DIR, ANIO_INICIO, ANIO_FIN)
    df_oni    = cargar_oni(ONI_DIR, ANIO_INICIO, ANIO_FIN)

    # ── Integración ───────────────────────────────────────────────────────
    df_final = construir_dataset(df_irag, df_chirps, df_oni)

    # ── Guardar ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta_salida = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df_final.to_parquet(ruta_salida, index=False)

    print(f"\n✓ Dataset guardado: {ruta_salida}")
    print(f"\nMuestra:")
    print(df_final.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
