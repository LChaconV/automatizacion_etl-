import pandas as pd

path = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\model_outputs_demo\prediction_interval.parquet"
path= r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\reference\dim_municipality.parquet"
df = pd.read_parquet(path)

print(df.describe())