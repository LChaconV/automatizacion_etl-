"""
Exploración de distribuciones de precipitación mensual por municipio
--------------------------------------------------------------------
Objetivo: determinar si la precipitación sigue distribución normal o sesgada,
para decidir el umbral de anomalías (Z-score vs. percentiles).

Salidas:
  - Figura con histogramas + curva normal teórica por municipio de muestra
  - Tabla resumen con estadísticos y resultado Shapiro-Wilk
  - Recomendación automática de método de detección de anomalías
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro, skew, kurtosis

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

DATA_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\chirps_municipal"
OUTPUT_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\dashboard_2\results\exploracion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Departamentos representativos (uno por región natural de Colombia)
# Primeros 2 dígitos del muni_code = código DIVIPOLA del departamento
DEPARTAMENTOS_MUESTRA = {
    "Atlántico (Caribe)":       "08",
    "Antioquia (Andina)":       "05",
    "Chocó (Pacífico)":         "27",
    "Casanare (Orinoquía)":     "85",
    "Amazonas (Amazonía)":      "91",
    "Cundinamarca (Andina)":    "25",
    "Nariño (Andina/Pacífico)": "52",
}

ALPHA_SHAPIRO = 0.05   # nivel de significancia para la prueba de normalidad
MIN_MESES = 60         # mínimo de observaciones mensuales para incluir un municipio

# ─────────────────────────────────────────────
# 2. CARGA DE DATOS
# ─────────────────────────────────────────────

print("Cargando archivos parquet...")
archivos = glob.glob(os.path.join(DATA_DIR, "year=*", "*.parquet"))
print(f"  → {len(archivos)} archivos encontrados")

df = pd.concat(
    [pd.read_parquet(f, columns=["date", "muni_code", "precip_mean_mm"]) for f in archivos],
    ignore_index=True
)
df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["precip_mean_mm"])
print(f"  → {len(df):,} registros cargados · {df['muni_code'].nunique()} municipios")

# ─────────────────────────────────────────────
# 3. SELECCIÓN DE MUNICIPIOS DE MUESTRA
# ─────────────────────────────────────────────

municipios_muestra = {}
for region, dep_code in DEPARTAMENTOS_MUESTRA.items():
    candidatos = df[df["muni_code"].str.startswith(dep_code)]["muni_code"].unique()
    for muni in candidatos:
        serie = df[df["muni_code"] == muni]["precip_mean_mm"]
        if len(serie) >= MIN_MESES:
            municipios_muestra[region] = muni
            break
    if region not in municipios_muestra:
        print(f"  ⚠ Sin municipio válido para {region} (mínimo {MIN_MESES} meses)")

print(f"\nMunicipios seleccionados:")
for region, muni in municipios_muestra.items():
    print(f"  {region}: {muni}")

# ─────────────────────────────────────────────
# 4. ANÁLISIS DE DISTRIBUCIÓN
# ─────────────────────────────────────────────

resultados = []

n_municipios = len(municipios_muestra)
n_cols = 2
n_rows = (n_municipios + 1) // n_cols

fig = plt.figure(figsize=(14, n_rows * 4))
fig.suptitle("Distribución de precipitación mensual por municipio de muestra\n(para definir método de detección de anomalías)",
             fontsize=13, y=1.01)

for idx, (region, muni_code) in enumerate(municipios_muestra.items()):
    serie = df[df["muni_code"] == muni_code]["precip_mean_mm"].dropna().values

    # Estadísticos
    media = np.mean(serie)
    desv  = np.std(serie)
    asim  = skew(serie)
    kurt  = kurtosis(serie)
    stat_sw, p_sw = shapiro(serie[:5000])  # Shapiro-Wilk tiene límite de 5000 obs.
    es_normal = p_sw > ALPHA_SHAPIRO

    resultados.append({
        "Región":         region,
        "muni_code":      muni_code,
        "n_meses":        len(serie),
        "media_mm":       round(media, 1),
        "desv_std":       round(desv, 1),
        "asimetría":      round(asim, 3),
        "curtosis":       round(kurt, 3),
        "shapiro_p":      round(p_sw, 4),
        "normal (p>0.05)": "Sí" if es_normal else "No",
        "método_anomalía": "Z-score ±2σ" if es_normal else "Percentiles 5-95"
    })

    # Gráfico
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    ax.hist(serie, bins=30, density=True, color="#378ADD", alpha=0.55,
            edgecolor="white", linewidth=0.4, label="Distribución observada")

    # Curva normal teórica superpuesta
    x = np.linspace(serie.min(), serie.max(), 300)
    ax.plot(x, stats.norm.pdf(x, media, desv), color="#D85A30",
            linewidth=1.5, linestyle="--", label="Normal teórica")

    # Líneas de percentiles 5 y 95
    p5, p95 = np.percentile(serie, [5, 95])
    ax.axvline(p5,  color="#888780", linewidth=1, linestyle=":", label=f"P5={p5:.0f}mm")
    ax.axvline(p95, color="#888780", linewidth=1, linestyle=":", label=f"P95={p95:.0f}mm")

    resultado_texto = "Normal (p>{:.3f})".format(ALPHA_SHAPIRO) if es_normal else f"No normal (p={p_sw:.4f})"
    ax.set_title(f"{region}\n{muni_code} · {resultado_texto}", fontsize=9)
    ax.set_xlabel("Precipitación mensual (mm)", fontsize=8)
    ax.set_ylabel("Densidad", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

plt.tight_layout()
ruta_figura = os.path.join(OUTPUT_DIR, "distribuciones_precipitacion.png")
fig.savefig(ruta_figura, dpi=150, bbox_inches="tight")
print(f"\nFigura guardada en: {ruta_figura}")

# ─────────────────────────────────────────────
# 5. TABLA RESUMEN Y RECOMENDACIÓN
# ─────────────────────────────────────────────

df_resultados = pd.DataFrame(resultados)
ruta_tabla = os.path.join(OUTPUT_DIR, "resumen_distribuciones.csv")
df_resultados.to_csv(ruta_tabla, index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print("RESUMEN DE DISTRIBUCIONES")
print("="*60)
print(df_resultados[["Región", "n_meses", "asimetría", "shapiro_p", "normal (p>0.05)", "método_anomalía"]].to_string(index=False))

# Recomendación global
n_normales = df_resultados["normal (p>0.05)"].eq("Sí").sum()
n_total    = len(df_resultados)

print("\n" + "="*60)
print("RECOMENDACIÓN")
print("="*60)
if n_normales == n_total:
    print("✓ Todos los municipios muestran distribución normal.")
    print("  → Usar Z-score con umbral ±2σ para detección de anomalías.")
elif n_normales == 0:
    print("✗ Ningún municipio muestra distribución normal.")
    print("  → Usar percentiles 5 y 95 para detección de anomalías.")
else:
    print(f"~ {n_normales}/{n_total} municipios con distribución normal.")
    print("  → Distribución mixta. Se recomienda usar percentiles 5 y 95")
    print("    como método robusto aplicable a todos los municipios.")
print("="*60)
print(f"\nTabla resumen guardada en: {ruta_tabla}")
