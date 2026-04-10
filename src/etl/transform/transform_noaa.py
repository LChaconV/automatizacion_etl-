import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from src.utils.config_loader import load_config
except ImportError:
    def load_config(path):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)

def transform_oni_historical(config: Dict[str, Any]) -> Optional[pd.DataFrame]:

    print("\n--- Starting ONI Data Transformation ---")

    try:
        project_root = Path(config['project_root'])

        raw_oni_input_dir = (
            project_root /
            config['data']['raw_base_dir'] /
            config['processing']['oni_processing']['input_raw_oni_sub_dir']
        )
        raw_oni_input_path = raw_oni_input_dir / config['processing']['oni_processing']['input_raw_oni_filename']

        processed_oni_output_dir = (
            project_root /
            config['data']['processed_base_dir'] /
            config['processing']['oni_processing']['output_processed_oni_sub_dir']
        )
        processed_oni_output_path = processed_oni_output_dir / config['processing']['oni_processing']['output_processed_oni_filename']

        processed_oni_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Looking for raw ONI data at: {raw_oni_input_path}")
        print(f"Saving processed ONI index to: {processed_oni_output_path}")

        if not raw_oni_input_path.exists():
            print(f"ERROR: Raw ONI input file not found: {raw_oni_input_path}")
            return None

        df_raw_oni = pd.read_csv(raw_oni_input_path)

        required_cols = ["year", "month", "anom"]
        if not all(col in df_raw_oni.columns for col in required_cols):
            print(f"ERROR: Raw ONI file does not contain all required columns ({required_cols}).")
            return None

        df_raw_oni["year"] = df_raw_oni["year"].astype(int)
        df_raw_oni["month"] = df_raw_oni["month"].astype(int)
        df_raw_oni = df_raw_oni.sort_values(by=["year", "month"]).reset_index(drop=True)

        print(f"Raw ONI data loaded. Total records: {len(df_raw_oni)}")
  
        df_raw_oni['oni_index'] = df_raw_oni['anom']
        
        df_oni_index = df_raw_oni[['year', 'month', 'oni_index']].copy()
        df_oni_index.dropna(subset=['oni_index'], inplace=True)

        
        df_oni_index["date"] = pd.to_datetime(dict(year=df_oni_index["year"],
                                                   month=df_oni_index["month"], day=1))
        df_oni_index.rename(columns={"oni_index": "value_oni"}, inplace=True)

        df_oni_index = df_oni_index[["date", "value_oni"]].sort_values("date").reset_index(drop=True)

       
        df_oni_index.to_csv(processed_oni_output_path, index=False)
        print(f"ONI transformation completed. CSV saved to: {processed_oni_output_path}")

      
        return df_oni_index

    except Exception as e:
        print(f"FATAL ERROR during ONI data transformation: {e}")
        return None

if __name__ == "__main__":

    config_path = Path(__file__).parent.parent.parent.parent / "config"/ "config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    config["project_root"] = str(config_path.parent.parent)


    df_transformed_oni = transform_oni_historical(config)
    if df_transformed_oni is not None:
        print("Transformed ONI DataFrame:")
        print(df_transformed_oni.head())
        df_transformed_oni.to_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/df_noaa_oni_transform.csv", index=False)
    else:
        print("ONI data transformation failed.")