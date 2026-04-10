import logging
import pandas as pd
from typing import Optional

import logging

logger = logging.getLogger(__name__)

def transform_oni_prediction_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:


    if df is None or df.empty:
        logger.warning("NOAA_PRED | Transform received empty DataFrame.")
        return None


    df_out = df.copy()

  
    df_out["year"] = df_out["year"].astype(int)
    df_out["month"] = df_out["month"].astype(int)
    df_out["prediction_oni"] = pd.to_numeric(df_out["prediction_oni"], errors="coerce")

    df_out["date"] = pd.to_datetime(
        dict(year=df_out["year"], month=df_out["month"], day=1),
        errors="coerce"
    )


    df_out = df_out[["date", "prediction_period", "prediction_oni"]].reset_index(drop=True)

    logger.info("NOAA_PRED | Transform completed.")
    logger.info("=" * 80)

    return df_out


if __name__ == "__main__":
    
    df_test = pd.read_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/raw/oni_prediction_extract.csv")

    df_transformed = transform_oni_prediction_df(df_test)

    print("Input DataFrame:")
    print(df_test)

    print("\nTransformed DataFrame:")
    print(df_transformed)
    df_transformed.to_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/df_noaa_oni_prediction_transform.csv", index=False)
