"""
etl_enfermedades.py
-------------------
ETL para datos de enfermedades del SIVIGILA (2018–2024).

Estructura de carpetas esperada:
    INPUT_BASE/
        dengue/
            Datos_2018_210.xlsx
            Datos_2019_210.csv
            ...
        leptospirosis/
            ...

Uso:
    Definir en el main() las enfermedades a procesar con su ruta y COD_EVE.
    Se puede ejecutar una o varias enfermedades.

Salida por enfermedad:
    date        datetime64  primer día del mes
    muni_code   str (5 dígitos)
    anio        int
    mes         int
    cod_eve     int
    casos       int
"""

import gc
import glob
import os

import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

INPUT_BASE = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\enfermedades"
OUTPUT_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\processed"

ANIO_MIN = 2010
ANIO_MAX = 2024

COLS_REQUERIDAS = ["COD_EVE", "ANO", "FEC_NOT", "COD_DPTO_O", "COD_MUN_O", "CONFIRMADOS"]

# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def leer_archivo(ruta: str) -> pd.DataFrame | None:
    """
    Lee un archivo CSV, XLSX o XLS.
    Normaliza nombres de columnas a mayúsculas.
    Retorna solo las columnas requeridas o None si falla.
    """
    ext = os.path.splitext(ruta)[-1].lower()
    nombre = os.path.basename(ruta)

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(ruta, dtype=str, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(ruta, dtype=str, low_memory=False, encoding="latin-1")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(ruta, dtype=str)
            print(f"    DEBUG columnas raw: {df.columns.tolist()[:10]}")
        else:
            print(f"    ⚠ {nombre}: formato no soportado")
            return None

        # Normalizar columnas
        df.columns = df.columns.str.strip().str.upper()

        # Verificar columnas requeridas
        faltantes = [c for c in COLS_REQUERIDAS if c not in df.columns]
        if faltantes:
            print(f"    ⚠ {nombre}: faltan columnas {faltantes} — omitido")
            return None

        return df[COLS_REQUERIDAS].copy()

    except Exception as e:
        print(f"    ✗ {nombre}: error al leer — {e}")
        return None


def limpiar(df: pd.DataFrame, cod_eve: int) -> pd.DataFrame:
    """
    Limpia el DataFrame:
      - Filtra por COD_EVE
      - Convierte tipos
      - Reconstruye muni_code de 5 dígitos
      - Parsea FEC_NOT como fecha
      - Filtra rango de años
    """
    # Filtrar por enfermedad
    df["COD_EVE"] = pd.to_numeric(df["COD_EVE"], errors="coerce")
    df = df[df["COD_EVE"] == cod_eve].copy()

    if df.empty:
        return pd.DataFrame()

    # Reconstruir muni_code
    df["COD_DPTO_O"] = df["COD_DPTO_O"].astype(str).str.strip().str.zfill(2)
    df["COD_MUN_O"]  = df["COD_MUN_O"].astype(str).str.strip().str.zfill(3)
    df["muni_code"]  = df["COD_DPTO_O"] + df["COD_MUN_O"]

    # Parsear fecha
    df["FEC_NOT"] = pd.to_datetime(df["FEC_NOT"], errors="coerce", dayfirst=True)

    # Año y mes
    df["anio"] = df["FEC_NOT"].dt.year
    df["mes"]  = df["FEC_NOT"].dt.month
    df["date"] = df["FEC_NOT"].values.astype("datetime64[M]").astype("datetime64[ns]")

    # Filtrar rango
    df = df[(df["anio"] >= ANIO_MIN) & (df["anio"] <= ANIO_MAX)]

    debug = df[(df["muni_code"] == "81001") & (df["anio"] == 2013) & (df["mes"] == 1)]

    return df


def agregar_mensual(df: pd.DataFrame, cod_eve: int) -> pd.DataFrame:
    """
    Agrega casos a nivel mensual por municipio.
    Incluye cod_eve en el resultado.
    """
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(["date", "muni_code", "anio", "mes"])["CONFIRMADOS"]
        .sum()
        .reset_index()
        .rename(columns={"CONFIRMADOS": "casos"})
    )
    agg["cod_eve"] = cod_eve
    agg["casos"]   = agg["casos"].astype(int)
    agg = agg[["date", "muni_code", "anio", "mes", "cod_eve", "casos"]]
    return agg


def procesar_enfermedad(nombre: str, carpeta: str, cod_eve: int, 
                        muni_code: str = None, 
                        archivo_prueba: str = None) -> None:
    """
    Procesa todos los archivos de una enfermedad:
      1. Lee cada archivo de la carpeta
      2. Limpia y filtra por COD_EVE
      3. Agrega mensualmente
      4. Guarda parquet final

    Parámetros
    ----------
    nombre  : nombre de la enfermedad (usado para el archivo de salida)
    carpeta : ruta a la carpeta con los archivos de la enfermedad
    cod_eve : código SIVIGILA de la enfermedad
    muni_code : código de municipio a filtrar (opcional)
    """
    print(f"\n{'='*60}")
    print(f"▶ {nombre.upper()} (COD_EVE={cod_eve})")
    print(f"  Carpeta: {carpeta}")
    print(f"{'='*60}")

    if not os.path.exists(carpeta):
        print(f"  ✗ Carpeta no encontrada: {carpeta}")
        return

    archivos = (
        glob.glob(os.path.join(carpeta, "*.csv"))  +
        glob.glob(os.path.join(carpeta, "*.xlsx")) +
        glob.glob(os.path.join(carpeta, "*.xls"))
    )

    if not archivos:
        print(f"  ⚠ Sin archivos en la carpeta")
        return

    print(f"  Archivos encontrados: {len(archivos)}")

    # Si se especifica un archivo de prueba, procesar solo ese
    if archivo_prueba:
        archivos = [archivo_prueba]
        print(f"  Modo prueba: {os.path.basename(archivo_prueba)}")

    acumulado = []

    for ruta in sorted(archivos):
        nombre_archivo = os.path.basename(ruta)
        df = leer_archivo(ruta)
        if df is None:
            continue

        df_limpio = limpiar(df, cod_eve)
        if df_limpio.empty:
            print(f"    — {nombre_archivo}: sin registros para COD_EVE={cod_eve}")
            del df, df_limpio
            gc.collect()
            continue
        if muni_code:
            df_limpio = df_limpio[df_limpio["muni_code"] == str(muni_code).zfill(5)]
            if df_limpio.empty:
                print(f"    — {nombre_archivo}: sin registros para muni_code={muni_code}")
                del df, df_limpio
                gc.collect()
                continue

        agg = agregar_mensual(df_limpio, cod_eve)
        if not agg.empty:
            acumulado.append(agg)
            print(f"    ✓ {nombre_archivo}: "
                  f"{agg['casos'].sum():,} casos · "
                  f"{agg['muni_code'].nunique()} municipios")

        del df, df_limpio, agg
        gc.collect()

    if not acumulado:
        print(f"  ⚠ Sin datos procesados para {nombre}")
        return

    # Consolidar — re-agregar por si hay solapamiento entre archivos
    df_final = (
        pd.concat(acumulado, ignore_index=True)
        .groupby(["date", "muni_code", "anio", "mes", "cod_eve"])["casos"]
        .sum()
        .reset_index()
        .sort_values(["muni_code", "date"])
        .reset_index(drop=True)
    )
    df_final = df_final[["date", "muni_code", "anio", "mes", "cod_eve", "casos"]]

    # Guardar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta_salida = os.path.join(OUTPUT_DIR, f"{nombre}.parquet")
    df_final.to_parquet(ruta_salida, index=False)

    print(f"\n  ✓ Guardado: {ruta_salida}")
    print(f"  → {len(df_final):,} registros")
    print(f"  → {df_final['muni_code'].nunique()} municipios")
    print(f"  → {df_final['casos'].sum():,} casos totales")
    print(f"  → Período: {df_final['anio'].min()}–{df_final['anio'].max()}")

    del df_final, acumulado
    gc.collect()


# ─────────────────────────────────────────────
# MAIN — definir aquí las enfermedades a procesar
# ─────────────────────────────────────────────

def main():
    """
    Define las enfermedades a procesar.
    Comenta o descomenta cada línea para ejecutar una o varias.
    """
    # procesar_enfermedad(
    # nombre  = "dengue_prueba",
    # carpeta = os.path.join(INPUT_BASE, "dengue"),
    # cod_eve = 210,
    # archivo_prueba = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\enfermedades\dengue\Datos_2013_210.xlsx",
    # muni_code = "81001",
    # )
    # ── Dengue ────────────────────────────────────────────────────────────
    # procesar_enfermedad(
    #     nombre  = "dengue",
    #     carpeta = os.path.join(INPUT_BASE, "dengue"),
    #     cod_eve = 210,
    #     muni_code = "81001",
    # )
    # procesar_enfermedad(
    #     nombre  = "dengue",
    #     carpeta = os.path.join(INPUT_BASE, "dengue"),
    #     cod_eve = 210
    # )
    # ── Leptospirosis ─────────────────────────────────────────────────────
    # procesar_enfermedad(
    #     nombre  = "leptospirosis",
    #     carpeta = os.path.join(INPUT_BASE, "leptospirosis"),
    #     cod_eve = 455,
    # )

    #── IRAG Inusitada ────────────────────────────────────────────────────
    procesar_enfermedad(
        nombre  = "irag_inusitada",
        carpeta = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_3\enfermedades\irag",
        cod_eve = 348,
    )

    # ── Malaria Vivax ─────────────────────────────────────────────────────
    # procesar_enfermedad(
    #     nombre  = "malaria_vivax",
    #     carpeta = os.path.join(INPUT_BASE, "malaria_vivax"),
    #     cod_eve = 490,
    # )

    # ── Malaria Falciparum ────────────────────────────────────────────────
    # procesar_enfermedad(
    #     nombre  = "malaria_falciparum",
    #     carpeta = os.path.join(INPUT_BASE, "malaria_falciparum"),
    #     cod_eve = 470,
    # )


if __name__ == "__main__":
    main()
