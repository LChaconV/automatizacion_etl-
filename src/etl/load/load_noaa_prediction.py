from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def _resolve_pred_dir(config: Dict[str, Any]) -> Path:
    base = (
        config.get("processing", {})
              .get("outputs", {})
              .get("noaa_prediction_parquet_dir", "data/processed/noaa_prediction")
    )
    p = Path(base)
    if not p.is_absolute():
        p = Path(config["project_root"]) / p
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_noaa_pred_parquet_by_year(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> List[str]:

    out_base = _resolve_pred_dir(config)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    written: List[str] = []

    year = df["date"].iloc[0].year
    part = out_base / f"year={year}"
    part.mkdir(parents=True, exist_ok=True)

    out = part / f"noaa_oni_pred_{year}.parquet"

    if out.exists():
        df_old = pd.read_parquet(out)
        df = pd.concat([df_old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date")

    df.to_parquet(out, index=False)
    written.append(str(out))

    return written

def load_oni_prediction(
    df_prediction: pd.DataFrame,
    config: Dict[str, Any]
) -> Optional[List[str]]:

    
    logger.info("Starting ONI Prediction Load")


    if df_prediction is None or df_prediction.empty:
        logger.warning("Prediction DataFrame is empty or None. Nothing to load.")
        return None

    required_cols = {"date", "prediction_period", "prediction_oni"}
    if not required_cols.issubset(df_prediction.columns):
        logger.error("Prediction DataFrame must contain columns %s", required_cols)
        return None


    df = df_prediction.copy()


    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "prediction_oni"])

    if df.empty:
        logger.warning("No valid prediction records after cleaning.")
        return None

  
    written_files = write_noaa_pred_parquet_by_year(df, config)

    logger.info("ONI prediction load completed. Files written: %s", len(written_files))
    return written_files

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from src.utils.config_loader import load_config

  
    config_path = Path(__file__).resolve().parents[3] / "config" / "config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    if "project_root" not in config:
        config["project_root"] = str(config_path.parent.parent)

  
    df_test = pd.read_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/df_noaa_oni_prediction_transform.csv")
    print(df_test)

    
    written = load_oni_prediction(df_test, config)

   
    if written:
        print("\nParquet files written:")
        for p in written:
            print(p)

        print("\nContent of written Parquet:")
        df_check = pd.read_parquet(written[0])
        print(df_check)
    else:
        print("No Parquet files written.")
