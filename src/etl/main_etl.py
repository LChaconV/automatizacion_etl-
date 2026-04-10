
import datetime
from pathlib import Path
import sys
from typing import Optional, List
import pandas as pd
import logging
from typing import Dict, Any, Iterator

from dateutil.relativedelta import relativedelta
from src.utils.config_loader import load_config

from src.utils.status_manager import get_last_processed_date, update_last_processed_date, set_current_execution_date

# CHIRPS ETL functions
from src.etl.extract.extract_chirps import extract_chirps_data_range,run_extract_chirps
from src.etl.transform.transform_chirps import transform_chirps_to_municipal_table
from src.etl.load.load_chirps import write_chirps_municipal_parquet


# NOAA Historical
from src.etl.extract.extract_noaa import run_extract_noaa_data
from src.etl.transform.transform_noaa import transform_oni_historical
from src.etl.load.load_noaa import write_oni_historical

# NOAA Prediction
from src.etl.extract.extract_noaa_prediction import extract_oni_prediction_df
from src.etl.transform.transform_noaa_prediction import transform_oni_prediction_df
from src.etl.load.load_noaa_prediction import load_oni_prediction



project_root_for_logging_setup = Path(__file__).parent.parent.parent
log_dir = project_root_for_logging_setup / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_filename = f"pipeline_run_etl_{datetime.date.today().strftime('%Y-%m-%d')}.log"
log_file_path = log_dir / log_filename

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger()



def run_chirps_etl(
    config: Dict[str, Any],
    force_chirps_start_date: Optional[datetime.date] = None,
) -> Optional[Dict[str, Any]]:
   
    logger.info("\n" + "=" * 80)
    logger.info("--- Starting CHIRPS ETL ---")
    logger.info("=" * 80 + "\n")

    project_root = Path(config["project_root"])
    source_key = "chirps"
    today = datetime.date.today()

    # -------------------------------------------------------------------------
    # 0) Determine start date (incremental or forced)
    # -------------------------------------------------------------------------
    if force_chirps_start_date:
        start_date = force_chirps_start_date
        logger.info(f"CHIRPS | Forced start date: {start_date}")
    else:
        last_processed = get_last_processed_date(project_root, config, source_key)
        if last_processed:
       
            if last_processed.month == 12:
                start_date = datetime.date(last_processed.year + 1, 1, 1)
            else:
                start_date = datetime.date(last_processed.year, last_processed.month + 1, 1)

            logger.info(
                "CHIRPS | Resuming from last processed date: "
                f"{last_processed} -> new start date: {start_date}"
            )
        else:
            start_date = datetime.date(2005, 1, 1)
            logger.info(f"CHIRPS | No previous status. Starting from default: {start_date}")

    if start_date > today:
        logger.info(f"CHIRPS | Already up-to-date through {start_date}. Nothing to do.")
        return None

    # -------------------------------------------------------------------------
    # 1) EXTRACT (range)
    # -------------------------------------------------------------------------
    downloaded_files, latest_date_in_run = extract_chirps_data_range(config, start_date, today)

    if not downloaded_files:
        logger.info("CHIRPS | No new files downloaded/verified. Nothing to process.")
        return None

    logger.info(f"CHIRPS | {len(downloaded_files)} file(s) downloaded/verified:")
    for p in downloaded_files:
        logger.info(f"  - {p.name}")

    # -------------------------------------------------------------------------
    # 2) TRANSFORM (per tif -> municipal table rows)
    # -------------------------------------------------------------------------
    transformed_frames: List[pd.DataFrame] = []
    failures = 0

    for tif_path in downloaded_files:
        try:
            logger.info(f"CHIRPS | Transforming to municipal table: {tif_path.name}")
            df_muni = transform_chirps_to_municipal_table(config, tif_path)

            if df_muni is None or df_muni.empty:
                logger.warning(f"CHIRPS | Transform produced empty DF for {tif_path.name}. Skipping.")
                failures += 1
                continue

            transformed_frames.append(df_muni)

        except Exception as e:
            failures += 1
            logger.error(f"CHIRPS | Transform failed for {tif_path.name}: {e}", exc_info=True)

    if not transformed_frames:
        logger.warning("CHIRPS | No transformed tables produced. Nothing to load.")
        return {
            "source": "CHIRPS",
            "status": "failed_transform_all",
            "downloaded_files": len(downloaded_files),
            "success_transforms": 0,
            "failures": failures,
            "written_paths": [],
        }

    df_all = pd.concat(transformed_frames, ignore_index=True)
    logger.info(f"CHIRPS | Transform completed. Rows total: {len(df_all)}")

    # -------------------------------------------------------------------------
    # 3) LOAD (Parquet)
    # -------------------------------------------------------------------------
    written_paths: List[Path] = []
    try:
        logger.info("CHIRPS | Loading municipal table to Parquet...")
        written_paths = write_chirps_municipal_parquet(df_all, config)

        if not written_paths:
            logger.warning("CHIRPS | Load wrote no files (empty output). Status will NOT be updated.")
        else:
            logger.info(f"CHIRPS | Load wrote {len(written_paths)} parquet file(s).")

    except Exception as e:
        logger.error(f"CHIRPS | Load failed: {e}", exc_info=True)
        return {
            "source": "CHIRPS",
            "status": "failed_load",
            "downloaded_files": len(downloaded_files),
            "success_transforms": len(transformed_frames),
            "failures": failures,
            "written_paths": [],
        }

    # -------------------------------------------------------------------------
    # 4) UPDATE STATUS (only if load succeeded)
    # -------------------------------------------------------------------------
    if written_paths and latest_date_in_run:
        try:
            update_last_processed_date(project_root, config, source_key, latest_date_in_run)
            logger.info(f"CHIRPS | Updated last processed date to {latest_date_in_run}.")
        except Exception as e:
            logger.warning(f"CHIRPS | Could not update status file: {e}")

    logger.info(
        f"CHIRPS | Completed. Downloads={len(downloaded_files)} | "
        f"Transforms={len(transformed_frames)} | Failures={failures} | "
        f"ParquetFiles={len(written_paths)} | LastRunDate={latest_date_in_run}"
    )
    logger.info("--- END: CHIRPS ETL ---")

    return {
        "source": "CHIRPS",
        "status": "success" if written_paths else "success_no_write",
        "downloaded_files": len(downloaded_files),
        "success_transforms": len(transformed_frames),
        "failures": failures,
        "rows_out": int(len(df_all)),
        "written_paths": [str(p) for p in written_paths],
        "latest_date_in_run": str(latest_date_in_run) if latest_date_in_run else None,
    }

# -----------------------------------------------------------------------------
# NOAA PREDICTION ETL
# -----------------------------------------------------------------------------

def run_noaa_prediction_etl(
    config: Dict[str, Any],
    force_noaa_prediction_start_date: Optional[datetime.date] = None,
) -> Optional[Dict[str, Any]]:
    
    project_root = Path(config["project_root"])
    source_key = "noaa_oni_prediction"
    today = datetime.date.today()
    end_date = datetime.date(today.year, today.month, 1)

    summary: Dict[str, Any] = {
        "source": "NOAA_ONI_PREDICTION",
        "status": "started",
        "start_date": None,
        "end_date": str(end_date),
        "months_attempted": 0,
        "months_success": 0,
        "rows_extracted_total": 0,
        "rows_transformed_total": 0,
        "written_files_total": 0,
        "latest_success_date": None,
        "failed_month": None,
        "messages": [],
        "written_files": [],
    }

    logger.info("=" * 80)
    logger.info("--- Starting NOAA ONI Prediction ETL ---")
 

    # -------------------------------------------------------------------------
    # 0) Determine start date (incremental or forced)
    # -------------------------------------------------------------------------
    if force_noaa_prediction_start_date:
        start_date = force_noaa_prediction_start_date
        logger.info(f"ONI prediction | Forced start date: {start_date}")
    else:
        last_processed = get_last_processed_date(project_root, config, source_key)
        if last_processed:
            if last_processed.month == 12:
                start_date = datetime.date(last_processed.year + 1, 1, 1)
            else:
                start_date = datetime.date(last_processed.year, last_processed.month + 1, 1)

            logger.info(
                "ONI prediction | Resuming from last processed date: "
                f"{last_processed} -> new start date: {start_date}"
            )
        else:
            start_date = datetime.date(2005, 1, 1)
            logger.info(f"ONI prediction | No previous status. Starting from default: {start_date}")

    summary["start_date"] = str(start_date)

    if start_date > end_date:
        logger.info(f"ONI prediction | Already up-to-date through {start_date}. Nothing to do.")
        summary["status"] = "nothing_to_do"
        summary["messages"].append("Already up-to-date. No new months to process.")
        return summary

    # -------------------------------------------------------------------------
    # 1) Process month-by-month
    # -------------------------------------------------------------------------
    current = datetime.date(start_date.year, start_date.month, 1)
    latest_date_in_run: Optional[datetime.date] = None

    while current <= end_date:
        summary["months_attempted"] += 1
        y, m = current.year, current.month
        logger.info(f"ONI prediction | Processing month {y}-{m:02d}...")

        try:
           
            df_raw = extract_oni_prediction_df(config, y, m)
            if df_raw is None or df_raw.empty:
                raise RuntimeError(f"Extract returned empty DF for {y}-{m:02d}")

            summary["rows_extracted_total"] += int(len(df_raw))

        
            df_tr = transform_oni_prediction_df(df_raw)
            if df_tr is None or df_tr.empty:
                raise RuntimeError(f"Transform returned empty DF for {y}-{m:02d}")

            summary["rows_transformed_total"] += int(len(df_tr))

        
            written = load_oni_prediction(df_tr, config)  
            if not written:
                raise RuntimeError(f"Load produced no output for {y}-{m:02d}")

     
            for w in written:
                summary["written_files"].append(str(w))
            summary["written_files_total"] += len(written)

            
            latest_date_in_run = current
            summary["months_success"] += 1
            summary["latest_success_date"] = str(latest_date_in_run)

            logger.info(f"ONI prediction | Month {y}-{m:02d} completed successfully.")

        except Exception as e:
          
            summary["status"] = "failed"
            summary["failed_month"] = str(current)
            summary["messages"].append(f"Failed at {current}: {e}")
            logger.error(f"ONI prediction | Failed at {y}-{m:02d}: {e}", exc_info=True)

            if latest_date_in_run:
                try:
                    update_last_processed_date(project_root, config, source_key, latest_date_in_run)
                    logger.info(
                        f"ONI prediction | Status updated to latest successful month: {latest_date_in_run}"
                    )
                except Exception as ue:
                    logger.warning(f"ONI prediction | Could not update status after failure: {ue}")
            else:
                logger.warning(
                    "ONI prediction | No successful month in this run; status will not be updated."
                )

            return summary

        
        if current.month == 12:
            current = datetime.date(current.year + 1, 1, 1)
        else:
            current = datetime.date(current.year, current.month + 1, 1)

    # -------------------------------------------------------------------------
    # 2) All months processed successfully -> update status to latest month
    # -------------------------------------------------------------------------
    if latest_date_in_run:
        try:
            update_last_processed_date(project_root, config, source_key, latest_date_in_run)
            logger.info(f"ONI prediction | Status updated to {latest_date_in_run}")
        except Exception as e:
            summary["status"] = "success_status_not_updated"
            summary["messages"].append(f"Load succeeded but status update failed: {e}")
            logger.warning(f"ONI prediction | Status update failed: {e}", exc_info=True)
            return summary

    summary["status"] = "success"
    summary["messages"].append("NOAA ONI prediction ETL completed successfully.")
    logger.info("--- END: NOAA ONI Prediction ETL ---")
    return summary


# -----------------------------------------------------------------------------
# NOAA HISTORICAL ETL
# -----------------------------------------------------------------------------
def run_noaa_historical_etl(
    config: dict,
    force_noaa_historical_start_date: Optional[datetime.date] = None,
) -> Optional[dict]:


    project_root = Path(config["project_root"])
    source_key = "noaa_oni_historical"
    today = datetime.date.today()
    end_date = datetime.date(today.year, today.month, 1)

    summary = {
        "source": "NOAA_ONI_HISTORICAL",
        "status": "started",
        "start_date": None,
        "end_date": str(end_date),
        "latest_extracted_date": None,
        "written_files": [],
        "messages": [],
    }

    logger.info("\n" + "=" * 80)
    logger.info("--- Starting NOAA ONI Historical ETL ---")
    logger.info("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 0) Determine start date
    # ------------------------------------------------------------------
    if force_noaa_historical_start_date:
        start_date = force_noaa_historical_start_date
        logger.info(f"ONI historical | Forced start date: {start_date}")
    else:
        last_processed = get_last_processed_date(project_root, config, source_key)
        if last_processed:
            start_date = (
                datetime.date(last_processed.year, last_processed.month, 1)
                + relativedelta(months=1)
            )
            
        else:
            start_date = datetime.date(1950, 1, 1)
            logger.info(
                f"ONI historical | No previous status found. Starting from {start_date}"
            )

    summary["start_date"] = str(start_date)

    if start_date > end_date:
        logger.info("ONI historical | Already up-to-date. Nothing to process.")
        summary["status"] = "nothing_to_do"
        summary["messages"].append("Already up-to-date.")
        return summary

    # ------------------------------------------------------------------
    # 1) EXTRACT (range-based)
    # ------------------------------------------------------------------

    latest_extracted_date = run_extract_noaa_data(
        config=config,
        start_date=start_date,
        end_date=end_date
    )

    if not latest_extracted_date:
        summary["status"] = "extract_failed"
        summary["messages"].append("Extraction did not return a latest date.")
        logger.warning("ONI historical | Extraction returned no new data.")
        return summary

    summary["latest_extracted_date"] = str(latest_extracted_date)

    # ------------------------------------------------------------------
    # 2) TRANSFORM
    # ------------------------------------------------------------------
    logger.info("--- Transforming ONI historical data ---")

    df_transformed = transform_oni_historical(config)
    if df_transformed is None or df_transformed.empty:
        summary["status"] = "transform_failed"
        summary["messages"].append("Transform returned empty DataFrame.")
        logger.error("ONI historical | Transform returned empty DataFrame.")
        return summary

    # ------------------------------------------------------------------
    # 3) LOAD
    # ------------------------------------------------------------------
    logger.info("--- Writing ONI historical data ---")

    base_dir = config["processing"]["outputs"]["noaa_parquet_partitioned_dir"]
    written_files = write_oni_historical(df_transformed, base_dir)

    if not written_files:
        summary["status"] = "load_failed"
        summary["messages"].append("Load did not produce output files.")
        logger.error("ONI historical | Load step produced no files.")
        return summary

    summary["written_files"] = written_files

    # ------------------------------------------------------------------
    # 4) Update status (YAML)
    # ------------------------------------------------------------------
    try:
        update_last_processed_date(
            project_root,
            config,
            source_key,
            latest_extracted_date
        )
        logger.info(
            f"ONI historical | Status updated to {latest_extracted_date}"
        )
    except Exception as e:
        summary["status"] = "success_status_not_updated"
        summary["messages"].append(f"Status update failed: {e}")
        logger.warning(
            f"ONI historical | Failed to update status: {e}",
            exc_info=True
        )
        return summary

    # ------------------------------------------------------------------
    # 5) Success
    # ------------------------------------------------------------------
    summary["status"] = "success"
    summary["messages"].append(
        "NOAA ONI historical ETL completed successfully."
    )

    logger.info("--- END: NOAA ONI Historical ETL ---")
    return summary



def run_etl_pipeline(
    force_chirps_start_date: Optional[datetime.date] = None,
    force_noaa_historical_start_date: Optional[datetime.date] = None,
    noaa_pred_year: Optional[int] = None,
    noaa_pred_month: Optional[int] = None,


):
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)
    config["project_root"] = str(project_root)

    current_execution_date = datetime.date.today()
    set_current_execution_date(current_execution_date)
    logger.info("--- Starting Full ETL Pipeline ---")
    

    # CHIRPS
    chirps_summary = run_chirps_etl(config, force_chirps_start_date)
   
    # -------------------------
    # 2) NOAA Historical (ONI)
    # -------------------------

    noaa_hist_summary = run_noaa_historical_etl(
        config=config,
        
    )
    
    # -------------------------
    # 3) NOAA Prediction (ONI)
    # -------------------------

    
    noaa_pred_summary = run_noaa_prediction_etl(
        config=config,
        force_noaa_prediction_start_date=None,
    )
    
    
    logger.info("--- Full ETL Pipeline Completed ---")
    logger.info("=" * 80)

    return {
        "chirps": chirps_summary,
        "noaa_historical": noaa_hist_summary,
        "noaa_prediction": noaa_pred_summary,
    }

    
if __name__ == "__main__":
    logger.info("--- Script execution started from main block ---")
    run_etl_pipeline()
