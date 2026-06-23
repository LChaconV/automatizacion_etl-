"""
construir_climatologia_referencia.py
--------------------------------------
Construye el archivo de climatología de referencia climática OMM 1991–2020
(ANIO_INICIO=1991 abajo; el nombre del archivo de salida quedó como
"..._1991_2020.parquet" de una versión anterior y no se renombró para no
romper la ruta ya referenciada en data_loader.py — el período real cubierto
es 1991–2020).

Para cada combinación municipio × mes del año calcula:
    p5, p50, p95   percentiles de precipitación mensual (p50 = MEDIANA,
                   no la media — ver nota en data_loader.cargar_climatologia_referencia())
    media, std     estadísticos complementarios (media = "normal" en
                   sentido estricto de la OMM, disponible pero no usada
                   actualmente por el dashboard)
    n_anios        cantidad de años con dato (máximo 30)

Salida:
    climatologia_referencia_1991_2020.parquet

Este archivo se usa en el dashboard para detectar anomalías con umbrales
fijos e independientes del filtro de período que el usuario seleccione.

Referencia: OMM No. 1203 — Normales Climatológicas (metodología; el
período efectivo aquí es 1991–2020, no 1991–2020).
"""

import glob
import os

import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

DATA_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\chirps_municipal"
OUTPUT_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_2\referencia"

ANIO_INICIO = 1991
ANIO_FIN    = 2020

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CARGA DE DATOS 1991–2020
# ─────────────────────────────────────────────

print(f"Cargando datos {ANIO_INICIO}–{ANIO_FIN}...")

archivos = glob.glob(os.path.join(DATA_DIR, "year=*", "*.parquet"))

# Filtrar solo los años del período de referencia
archivos_ref = [
    f for f in archivos
    if ANIO_INICIO <= int(os.path.basename(os.path.dirname(f)).replace("year=", "")) <= ANIO_FIN
]

if not archivos_ref:
    raise FileNotFoundError(
        f"No se encontraron archivos parquet para {ANIO_INICIO}–{ANIO_FIN} en {DATA_DIR}"
    )

print(f"  → {len(archivos_ref)} archivos encontrados ({ANIO_INICIO}–{ANIO_FIN})")

df = pd.concat(
    [pd.read_parquet(f, columns=["date", "muni_code", "precip_mean_mm"])
     for f in archivos_ref],
    ignore_index=True,
)

df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
df["date"]      = pd.to_datetime(df["date"])
df["mes"]       = df["date"].dt.month
df["anio"]      = df["date"].dt.year
df = df.dropna(subset=["precip_mean_mm"])
df = df.rename(columns={"precip_mean_mm": "precip_mm"})

print(f"  → {len(df):,} registros · {df['muni_code'].nunique()} municipios")

# ─────────────────────────────────────────────
# CÁLCULO DE PERCENTILES POR MUNICIPIO × MES
# ─────────────────────────────────────────────

print("Calculando climatología de referencia...")

def percentiles_mes(grupo):
    return pd.Series({
        "p5":    grupo["precip_mm"].quantile(0.05),
        "p50":   grupo["precip_mm"].quantile(0.50),
        "p95":   grupo["precip_mm"].quantile(0.95),
        "media": grupo["precip_mm"].mean(),
        "std":   grupo["precip_mm"].std(),
        "n_anios": grupo["anio"].nunique(),
    })

clima = (
    df.groupby(["muni_code", "mes"])
    .apply(percentiles_mes)
    .reset_index()
)

# ─────────────────────────────────────────────
# VALIDACIÓN
# ─────────────────────────────────────────────

n_esperado = df["muni_code"].nunique() * 12
n_obtenido = len(clima)
municipios_incompletos = clima[clima["n_anios"] < 25]["muni_code"].nunique()

print(f"\n{'='*55}")
print("VALIDACIÓN")
print(f"{'='*55}")
print(f"Combinaciones municipio×mes esperadas : {n_esperado:,}")
print(f"Combinaciones municipio×mes obtenidas : {n_obtenido:,}")
print(f"Municipios con < 25 años de datos     : {municipios_incompletos}")
print(f"n_anios promedio                      : {clima['n_anios'].mean():.1f}")
print(f"{'='*55}")

if municipios_incompletos > 0:
    print(
        f"\n⚠ {municipios_incompletos} municipios tienen menos de 25 años de datos.\n"
        "  Sus percentiles son menos robustos pero se incluyen igualmente.\n"
        "  Considera documentar esta limitación en la tesis."
    )

# ─────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────

# Agregar metadatos del período de referencia como columnas
clima["periodo_ref_inicio"] = ANIO_INICIO
clima["periodo_ref_fin"]    = ANIO_FIN

ruta_salida = os.path.join(OUTPUT_DIR, "climatologia_referencia_1991_2020.parquet")
clima.to_parquet(ruta_salida, index=False)

print(f"\n✓ Archivo guardado en:\n  {ruta_salida}")
print(f"\nColumnas: {clima.columns.tolist()}")
print(f"Muestra:\n{clima.head(12).to_string(index=False)}")
