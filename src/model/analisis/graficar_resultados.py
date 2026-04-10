import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Definición de Rutas
PATH_DIM_MUNI = r"C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/dim_municipality.parquet"
PATH_PREDICTIONS = r"C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/model_outputs_demo/prediction_interval.parquet"
PATH_REAL_2025 = r"C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/chirps_municipal/year=2025/fact_chirps_muni_2025.parquet"

# 2. Carga de Datos
gdf_dim = gpd.read_parquet(PATH_DIM_MUNI)
df_preds = pd.read_parquet(PATH_PREDICTIONS)
df_real = pd.read_parquet(PATH_REAL_2025)

# 3. Procesamiento y Filtro
# Asegurar tipos de datos para el Join
gdf_dim['muni_code'] = gdf_dim['muni_code'].astype(str)
df_preds['muni_code'] = df_preds['muni_code'].astype(str)
df_real['muni_code'] = df_real['muni_code'].astype(str)

# Filtrar Enero 2025 en los datos reales
df_real['date'] = pd.to_datetime(df_real['date'])
df_enero_real = df_real[df_real['date'] == '2025-01-01'].copy()

# 4. Unión con Cartografía
# Unimos predicción y real con la geometría
gdf_preds_map = gdf_dim.merge(df_preds, on="muni_code", how="inner")
gdf_real_map = gdf_dim.merge(df_enero_real, on="muni_code", how="inner")

# 5. Estandarizar Escala de Colores (Fundamental para la comparación)
# Buscamos el mínimo y máximo global entre ambos datasets
vmin = min(gdf_preds_map['avg_precip_mm'].min(), gdf_real_map['precip_mean_mm'].min())
vmax = max(gdf_preds_map['avg_precip_mm'].max(), gdf_real_map['precip_mean_mm'].max())

# 6. Crear Visualización Lado a Lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Mapa 1: Predicción (Promedio Histórico)
gdf_preds_map.plot(
    column='avg_precip_mm', 
    cmap='Blues', 
    vmin=vmin, vmax=vmax, # Aplicar misma escala
    ax=ax1, 
    edgecolor='black', linewidth=0.1
)
ax1.set_title('Predicción: Promedio Histórico Ponderado', fontsize=14)
ax1.axis('off')

# Mapa 2: Realidad (Enero 2025)
gdf_real_map.plot(
    column='precip_mean_mm', 
    cmap='Blues', 
    vmin=vmin, vmax=vmax, # Aplicar misma escala
    ax=ax2, 
    edgecolor='black', linewidth=0.1
)
ax2.set_title('Realidad: Observación CHIRPS (Enero 2025)', fontsize=14)
ax2.axis('off')

# Añadir una sola barra de color para ambos mapas
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Precipitación (mm)')

plt.suptitle('Comparación: Predicción del Modelo vs Realidad Observada', fontsize=18, y=0.95)
plt.show()