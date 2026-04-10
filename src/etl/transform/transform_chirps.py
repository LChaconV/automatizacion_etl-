from pathlib import Path
import re
import datetime
from typing import Dict, Any, Optional, List

from datetime import  timezone
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

from src.utils.config_loader import load_config

import logging
logger = logging.getLogger(__name__)

def parse_chirps_date_from_name(tif_path: Path) -> datetime.date:
    
    name = tif_path.name
    #  YYYY.MM : chirps-v2.0.2005.01.tif
    m = re.search(r"(\d{4})\.(\d{2})", name)
    if not m:
        raise ValueError(f"Cannot infer CHIRPS date from filename: {name}")

    year = int(m.group(1))
    month = int(m.group(2))
    return datetime.date(year, month, 1)
def load_municipalities_gdf(config: Dict[str, Any],project_root: Optional[Path] = None) -> gpd.GeoDataFrame:

    project_root = project_root or Path(config.get("project_root", Path(__file__).parent.parent.parent.parent))
    base_cfg = config["data"]["base_dir"]
    ref_cfg = config["data"]["reference"]
    parquet_rel = ref_cfg["dim_municipality_geoparquet"]
    parquet_path = project_root /base_cfg/ parquet_rel


    if not parquet_path.exists():
        raise FileNotFoundError(f"Municipality reference Parquet not found at: {parquet_path}")

    logger.info(f"Loading municipality reference from Parquet: {parquet_path}")
    gdf = gpd.read_parquet(parquet_path)

    if "muni_code" not in gdf.columns:
        raise ValueError(
            f"'muni_code' column not found in municipality GeoJSON. "
            f"Available columns: {list(gdf.columns)}"
        )
    if gdf.geometry is None:
        raise ValueError("No geometry column found in municipality Parquet.")
    
    gdf["muni_code"] = gdf["muni_code"].astype(str)

    return gdf
def transform_chirps_to_municipal_table(
    config: Dict[str, Any],
    tif_path: Path,
) -> pd.DataFrame:

    date_obj = parse_chirps_date_from_name(tif_path)
    logger.info(f"CHIRPS transform for file: {tif_path.name} | date={date_obj}")

    # -----------------------------
    # Error Event Schema 
    # -----------------------------
    
    error_events: List[Dict[str, Any]] = []

    def add_error_event(
        event_type: str,
        muni_code: Any,
        message: str,
        severity: str = "WARNING",
    ) -> None:
        error_events.append(
            {
                "event_time_utc": datetime.now(timezone.utc).isoformat(),
                "process": "chirps_transform",
                "event_type": event_type,   
                "severity": severity,    
                "tif_name": tif_path.name,
                "date": str(date_obj),
                "muni_code": str(muni_code) if muni_code is not None else None,
                "message": message,
            }
        )

  
    municipalities_gdf = load_municipalities_gdf(config)

  
    if not tif_path.exists():
        raise FileNotFoundError(f"CHIRPS TIF not found at: {tif_path}")

    logger.info(f"Opening CHIRPS raster: {tif_path}")
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

     
        raster_cfg = (
            config
            .get("processing", {})
            .get("raster_transform", {})
        )
        config_nodata = raster_cfg.get("input_nodata_value", None)

        if nodata is None and config_nodata is not None:
            nodata = config_nodata

        if municipalities_gdf.crs != raster_crs:
            logger.info("Reprojecting municipalities to match raster CRS.")
            municipalities_gdf = municipalities_gdf.to_crs(raster_crs)

        rows = []


        for idx, muni_row in municipalities_gdf.iterrows():
            muni_code = muni_row["muni_code"]
            geom_obj = muni_row.geometry

            if geom_obj is None or getattr(geom_obj, "is_empty", True):
                add_error_event(
                    event_type="empty_geometry",
                    muni_code=muni_code,
                    message="Municipality geometry is empty or None. Skipping municipality.",
                    severity="WARNING",
                )
                continue

            geom = [geom_obj.__geo_interface__]

            try:
                # mask() lee solo los píxeles dentro del polígono
                data, _ = mask(
                    src,
                    geom,
                    crop=True,
                    filled=True,
                    nodata=nodata,
                    all_touched=True,
                )
          
                band = data[0]

                # Flatten y filtrar nodata / NaN
                flat = band.astype("float32").ravel()

                if nodata is not None:
                    valid = flat[flat != nodata]
                else:
                    valid = flat

                # Eliminar NaN
                valid = valid[~np.isnan(valid)]

                if valid.size == 0:
                    # No hay pixeles válidos para este municipio
                    add_error_event(
                        event_type="no_valid_pixels",
                        muni_code=muni_code,
                        message="No valid pixels after nodata/NaN filtering. Writing NaN stats.",
                        severity="WARNING",
                    )
                    n_pixels = 0
                    precip_mean = np.nan
                    precip_min = np.nan
                    precip_max = np.nan
                    std_dev = np.nan
                else:
                    n_pixels = int(valid.size)
                    precip_mean = float(valid.mean())
                    precip_min = float(valid.min())
                    precip_max = float(valid.max())
                    std_dev = float(valid.std(ddof=0))

                rows.append(
                    {
                        "date": date_obj,
                        "muni_code": muni_code,
                        "precip_mean_mm": precip_mean,
                        "n_pixels": n_pixels,
                        "precip_min": precip_min,
                        "precip_max": precip_max,
                        "std_dev": std_dev,
                    }
                )

            except Exception as e:
               
                add_error_event(
                    event_type="mask_error",
                    muni_code=muni_code,
                    message=str(e),
                    severity="ERROR",
                )
                logger.warning(
                    f"Error computing stats for muni_code={muni_code}: {e}",
                    exc_info=False,
                )
               
    df = pd.DataFrame(rows)


    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["muni_code"] = df["muni_code"].astype(str)

    
        dup_mask = df.duplicated(subset=["date", "muni_code"], keep=False)
        if dup_mask.any():
            n_dup = int(dup_mask.sum())
            add_error_event(
                event_type="deduplication_applied",
                muni_code="*",
                message=f"Found {n_dup} duplicate rows on (date, muni_code). Keeping first occurrence.",
                severity="WARNING",
            )
            df = df.drop_duplicates(subset=["date", "muni_code"], keep="first")

  
    if error_events:
        n_err = sum(1 for e in error_events if e.get("severity") == "ERROR")
        n_warn = sum(1 for e in error_events if e.get("severity") == "WARNING")
        logger.info(
            f"CHIRPS Error Events | total={len(error_events)} | errors={n_err} | warnings={n_warn}"
        )

    logger.info(
        f"CHIRPS municipal table created. Rows: {len(df)} "
        f"for date={date_obj}"
    )

   

    for event in error_events:
        print(event)
    return df


    


if __name__ == "__main__":

    try:
        project_root = Path(__file__).parent.parent.parent.parent
        print(f"Project root: {project_root}")
        config_file_path = project_root / "config" / "config.yaml"
        
        print(f"Loading configuration from: {config_file_path}")
        app_config = load_config(config_file_path)
        municipalities_gdf = load_municipalities_gdf(app_config)
        tif=Path("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/raw/chirps_raster_raw/chirps-v2.0.2025.01.tif")
        municipalities_transform=transform_chirps_to_municipal_table(
        app_config,
        tif)
        print("CHIRPS Transformation process completed successfully.")
        print(municipalities_transform.head())
        municipalities_transform_sorted = municipalities_transform.sort_values(
            by="n_pixels", ascending=True
        )

        print(municipalities_transform_sorted.info())

       
        print(municipalities_transform_sorted)
        municipalities_transform_sorted.to_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/processed/reference/df_precipitation_chirps_transform.csv", index=False)
                
                    
    except Exception as e:
        print(f"FATAL ERROR: While running the CHIRPS transformation script independently: {e}")


