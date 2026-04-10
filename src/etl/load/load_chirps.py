from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def write_chirps_municipal_parquet(df: pd.DataFrame, config: Dict[str, Any]) -> List[Path]:

    if df.empty:
        logger.warning("CHIRPS load: received empty DataFrame. Nothing to save.")
        return []

    if "date" not in df.columns or "muni_code" not in df.columns:
        raise ValueError("CHIRPS DataFrame must contain 'date' and 'muni_code' columns.")

    # Asegurar tipos correctos
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["muni_code"] = df["muni_code"].astype(str)

    cols = list(df.columns)
    cols.insert(1, cols.pop(cols.index("year")))
    df = df[cols]


    # Rutas de salida desde config.yaml
    project_root = Path(config["project_root"])
    out_dir = project_root / config["processing"]["outputs"]["chirps_municipal_parquet_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    written_paths: List[Path] = []

   
    for year, df_year in df.groupby("year"):
        year_dir = out_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        out_path = year_dir / f"fact_chirps_muni_{year}.parquet"

        if out_path.exists():
           
            df_old = pd.read_parquet(out_path)
            df_old["date"] = pd.to_datetime(df_old["date"])
            logger.info(f"CHIRPS load: existing file found for {year}, {len(df_old)} rows.")

            # Eliminar duplicados de meses ya procesados
            existing_dates = df_old["date"].unique()
            df_new = df_year[~df_year["date"].isin(existing_dates)].copy()

            if df_new.empty:
                logger.info(f"CHIRPS load: no new months to add for {year}. Skipping write.")
                continue

            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            logger.info(f"CHIRPS load: appended {len(df_new)} new rows for {year}.")
        else:
            # Año nuevo
            df_combined = df_year.copy()
            logger.info(f"CHIRPS load: creating new Parquet for {year} ({len(df_year)} rows).")

        # Ordenar por fecha y municipio antes de guardar
        df_combined = df_combined.sort_values(["date", "muni_code"])
        df_combined.to_parquet(out_path, index=False)
        written_paths.append(out_path)

        logger.info(f"CHIRPS load: written Parquet → {out_path}")

    if not written_paths:
        logger.info("CHIRPS load: no Parquet files were updated or created.")

    return written_paths

if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    from src.utils.config_loader import load_config

    print("This module is intended to be imported and used within the ETL pipeline.")
    df_test=pd.read_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/df_precipitation_chirps_transform.csv")

    project_root = Path(__file__).parent.parent.parent.parent
    print("este es project root: ", project_root)
    
    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)
    config["project_root"] = str(project_root)

    written_files = write_chirps_municipal_parquet(df_test, config)

    if written_files:
        print("CHIRPS load completed. Files written:")
        for f in written_files:
            print(f" - {f}")
    else:
        print("No files were written.")

    df = pd.read_parquet("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/chirps/precip_raster_colombia/year=2025/fact_chirps_muni_2025.parquet")
    print(df)