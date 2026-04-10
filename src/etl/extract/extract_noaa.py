import pandas as pd
import requests
from pathlib import Path
import datetime
from typing import Optional, Tuple 
import logging
logger = logging.getLogger(__name__)


try:
    from src.utils.config_loader import load_config
except ImportError:
    def load_config(path):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)


def update_oni_data(
    config: dict,
    start_date: datetime.date,
    end_date: datetime.date
) -> Optional[datetime.date]:

    logger.info(
    "Starting ONI data update for range %s to %s",
    start_date,
    end_date
)
    latest_record_date_in_file = None

    try:
        project_root = Path(config["project_root"])

        # ---------------------------------------------------------------------
        # Configuration file paths
        # ---------------------------------------------------------------------
        oni_raw_data_sub_dir = config["extraction_params"]["oni_extraction"]["raw_sub_dir"]
        oni_raw_data_dir = (
            project_root
            / config["data"]["raw_base_dir"]
            / oni_raw_data_sub_dir
        )
        oni_raw_data_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # NOAA raw data filename
        # ---------------------------------------------------------------------
        raw_csv_filename = (
            config["extraction_params"]["oni_extraction"]["oni_data_raw_path"]
            .split("/")[-1]
            .replace(".csv", "_raw.csv")
        )
        raw_csv_path = oni_raw_data_dir / raw_csv_filename
        oni_remote_url = config["extraction_params"]["oni_extraction"]["remote_url"]

        logger.info("Local raw ONI CSV path: %s", raw_csv_path)
        logger.info("Remote ONI URL: %s", oni_remote_url)


        # ---------------------------------------------------------------------
        # Last record in the local CSV
        # ---------------------------------------------------------------------
        last_year = 0
        last_month = 0
        df_local_raw = pd.DataFrame(
            columns=["year", "month", "total", "climadjust", "anom"]
        )

        if raw_csv_path.exists() and raw_csv_path.stat().st_size > 0:
            try:
                temp_df = pd.read_csv(raw_csv_path)
                expected_cols = ["year", "month", "total", "climadjust", "anom"]

                if all(col in temp_df.columns for col in expected_cols):
                    df_local_raw = temp_df.dropna(subset=expected_cols)
                    df_local_raw["year"] = df_local_raw["year"].astype(int)
                    df_local_raw["month"] = df_local_raw["month"].astype(int)

                    df_local_raw = (
                        df_local_raw
                        .sort_values(by=["year", "month"])
                        .reset_index(drop=True)
                    )

                    if not df_local_raw.empty:
                        last_record = df_local_raw.iloc[-1]
                        last_year = int(last_record["year"])
                        last_month = int(last_record["month"])
                        latest_record_date_in_file = datetime.date(
                            last_year, last_month, 1
                        )
                        logger.info("Last local raw record: %s-%02d", last_year, last_month)
                    else:
                        logger.info(
                            "INFO: Local raw ONI CSV '%s' "
                            "is empty after removing NaNs. All data will be downloaded.",
                            raw_csv_path.name
                        )
                else:
                    logger.warning(
                        "WARNING: Local raw ONI CSV '%s' "
                        "has unexpected columns. It will be recreated from remote data.",
                        raw_csv_path.name
                    )

            except pd.errors.EmptyDataError:
                logger.info(
                    "Local raw ONI CSV '%s' is empty. It will be filled with remote data.",
                    raw_csv_path.name
                )
            except Exception as e:
                logger.error(
                    "Could not read local raw ONI CSV '%s': %s. Trying to download all remote data.",
                    raw_csv_path.name,
                    e
                )

        else:
            print(
                f"INFO: Local raw ONI CSV '{raw_csv_path.name}' "
                "does not exist or is empty. It will be filled with remote data."
            )

        # ---------------------------------------------------------------------
        # Download remote ONI data
        # ---------------------------------------------------------------------
        try:
            response = requests.get(oni_remote_url, timeout=120)
            response.raise_for_status()
            lines = response.text.strip().split("\n")
            logger.info(
                "Successfully downloaded ONI data from %s.",
                oni_remote_url
            )

        except requests.exceptions.RequestException as e:
            logger.error(
                "Could not download ONI data from %s: %s",
                oni_remote_url,
                e
            )

            return None

        # ---------------------------------------------------------------------
        # Parse and append new records
        # ---------------------------------------------------------------------
        SEAS_TO_MONTH = {
            "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
            "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
            "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
        }

        new_records = []

        for line in lines:
            parts = line.strip().split()
            try:
                seas = parts[0]
                year = int(parts[1])
                total = float(parts[2])
                anom = float(parts[3])
                month = SEAS_TO_MONTH.get(seas)

                if month is None:
                    continue

            except (ValueError, IndexError):
                continue

            if (year > last_year) or (year == last_year and month > last_month):
                new_records.append(
                    {
                        "year": year,
                        "month": month,
                        "total": total,
                        "anom": anom,
                    }
                )

        if new_records:
            new_df = pd.DataFrame(new_records)

            write_header = (
                not raw_csv_path.exists()
                or raw_csv_path.stat().st_size == 0
                or (last_year == 0 and last_month == 0)
            )

            mode = "w" if write_header else "a"
            header = write_header

            new_df.to_csv(raw_csv_path, mode=mode, index=False, header=header)

            logger.info(
                "%s new records %s raw ONI CSV file.",
                len(new_records),
                "written to a new" if write_header else "appended to the"
            )


            latest_record_date_in_file = datetime.date(
                int(new_df["year"].iloc[-1]),
                int(new_df["month"].iloc[-1]),
                1,
            )
        else:
            logger.info(
                "No new records found in the remote data for the raw ONI CSV."
            )


        return latest_record_date_in_file

    except KeyError as ke:
        logger.error(
            "Configuration error: Key '%s' not found in the config dictionary. "
            "Check the config.yaml file. Returning None.",
            ke
        )

        return None

    except Exception as e:
        logger.exception(
            "Fatal error: Unhandled error in update_oni_data. Returning None."
        )
        return None




def run_extract_noaa_data(config: dict, start_date: datetime.date, end_date: datetime.date) -> Optional[datetime.date]:

    logger.info(
        "Running NOAA ONI Historical data extraction process"
    )


    try:
       


   
        latest_processed_date = update_oni_data(config, start_date, end_date)

        if latest_processed_date:
            logger.info(
                "NOAA ONI Historical data processed. Latest date: %s",
                latest_processed_date
            )
        else:
            logger.warning(
                "No NOAA ONI Historical data processed or updated within the requested range."
            )
        logger.info("="*80)
        return latest_processed_date
    except Exception:
        logger.exception(
            "Fatal error during NOAA ONI Historical orchestration execution. Returning None."
        )
        return None
