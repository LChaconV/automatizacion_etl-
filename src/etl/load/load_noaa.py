from __future__ import annotations
from pathlib import Path
import pandas as pd



def write_oni_historical(df: pd.DataFrame, base_dir: str) -> list[str]:

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    written: list[str] = []
    groups = df.groupby(df["date"].map(lambda d: d.year), sort=True)

    for year, g in groups:
        part_dir = Path(base_dir) / f"year={year}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / f"noaa_oni_{year}.parquet"
        g.to_parquet(out_path, index=False)
        written.append(str(out_path))

    return written

if __name__ == "__main__":

    data = pd.read_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/df_noaa_oni_transform.csv")
    df_example = pd.DataFrame(data)
    output_dir = "data/processed/noaa_oni"

    written_files = write_oni_historical(df_example, output_dir)
    print("Written files:")
    for f in written_files:
        print(f)
    parquet=pd.read_parquet('C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/noaa_oni/year=2025/noaa_oni_2025.parquet')
    print(parquet)