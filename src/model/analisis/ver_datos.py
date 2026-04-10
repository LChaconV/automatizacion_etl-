import pandas as pd

# Leer el archivo completo
df_1 = pd.read_parquet("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/model_outputs_demo/prediction_interval.parquet")

# Ver las primeras filas
print(df_1.head())

# Leer el archivo completo
df = pd.read_parquet("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/dim_municipality.parquet")

# Ver las primeras filas
print(df.head())

