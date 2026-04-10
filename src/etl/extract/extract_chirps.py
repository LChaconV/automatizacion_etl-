
import os
import datetime
import requests
import gzip
import time
from pathlib import Path
from typing import Optional, List, Tuple
import yaml
import logging
logger = logging.getLogger(__name__)


try:
    from src.utils.config_loader import load_config
except ImportError:
    def load_config(path):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)

def _download_and_unzip_chirps_file(
    full_url: str,
    local_gz_path: Path,
    local_tif_path: Path,
    max_attempts: int,
    delay_seconds: int
) -> Optional[Path]:

    if local_tif_path.exists():
        logger.info(
            "STATUS: CHIRPS file already exists locally: %s. Skipping download.",
            local_tif_path
        )
        return local_tif_path

    for attempt in range(max_attempts):
        try:
            
            logger.info(
                "Attempt %s/%s: Downloading %s...",
                attempt + 1,
                max_attempts,
                full_url
            )

            response = requests.get(full_url, stream=True, timeout=120)
            response.raise_for_status()

            with open(local_gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("STATUS: GZ file downloaded: %s", local_gz_path)

            with gzip.open(local_gz_path, 'rb') as f_in:
                with open(local_tif_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            

            logger.info("STATUS: TIF file decompressed: %s", local_tif_path)

            os.remove(local_gz_path)

            logger.info("STATUS: Temporary GZ file removed: %s", local_gz_path)


            return local_tif_path

        

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None

            if status_code == 404:
                logger.warning(
                    "404 Not Found: CHIRPS file not yet available at the URL. Retrying if attempts remain."
                )
            else:
                logger.error(
                    "Unexpected HTTP error (%s): %s. Retrying if attempts remain.",
                    status_code,
                    e
                )

        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Connection error: Could not connect to the server. Retrying if attempts remain. %s",
                e
            )

        except requests.exceptions.Timeout as e:
            logger.error(
                "Timeout error: The request exceeded the time limit. Retrying if attempts remain. %s",
                e
            )

        except Exception as e:
            logger.exception(
                "An unexpected error occurred during download or decompression. Retrying if attempts remain."
            )

        if attempt < max_attempts - 1:
            logger.info("Waiting %s seconds before retrying...", delay_seconds)
            time.sleep(delay_seconds)
        else:
            logger.error(
                "FAILED: Maximum of %s retries reached for %s. Skipping this file.",
                max_attempts,
                full_url
            )
            return None



def extract_chirps_data_range(
    config: dict,
    start_date: datetime.date, 
    end_date: Optional[datetime.date] = None
) -> Tuple[List[Path], Optional[datetime.date]]:

    downloaded_files_paths = []
    latest_downloaded_date_in_run = None

    try:
        chirps_source_config = config['extraction_params']['chirps_source']
        download_retries_config = config['extraction_params']['chirps_download_retries']

        PROJECT_ROOT_FOR_THIS_SCRIPT = Path(__file__).resolve().parents[3]

        raw_base_dir_relative = config['data']['raw_base_dir']
        download_sub_dir_relative = chirps_source_config['download_sub_dir']
        download_dir = PROJECT_ROOT_FOR_THIS_SCRIPT / raw_base_dir_relative / download_sub_dir_relative
        os.makedirs(download_dir, exist_ok=True)

        max_attempts = download_retries_config['max_attempts']
        delay_hours = download_retries_config['delay_hours']
        delay_seconds = delay_hours * 3600

        if end_date is None:
            end_date = datetime.date.today()

        if start_date > end_date:
            logger.warning( "start_date is after end_date. No files will be downloaded.")
            return [], None 

        logger.info(
            "--- Starting CHIRPS extraction for range: %s-%02d to %s-%02d ---",
            start_date.year,
            start_date.month,
            end_date.year,
            end_date.month
        )

        current_iter_date = datetime.date(start_date.year, start_date.month, 1)

        while current_iter_date <= end_date.replace(day=1):
            year_to_fetch = current_iter_date.year
            month_to_fetch = f"{current_iter_date.month:02d}"

            filename_gz = f"{chirps_source_config['filename_prefix']}{year_to_fetch}.{month_to_fetch}{chirps_source_config['filename_suffix']}"
            full_url = f"{chirps_source_config['base_url']}{filename_gz}"
            local_gz_path = download_dir / filename_gz
            local_tif_path = download_dir / filename_gz.replace(".gz", "")
            logger.info(
                "Processing CHIRPS for: %s-%02d",
                year_to_fetch,
                month_to_fetch
            )

            logger.info("URL: %s", full_url)

            logger.info("Local destination (TIF): %s", local_tif_path)
                

            downloaded_path = _download_and_unzip_chirps_file(
                full_url, local_gz_path, local_tif_path, max_attempts, delay_seconds
            )

            if downloaded_path:
                downloaded_files_paths.append(downloaded_path)
                
                if not latest_downloaded_date_in_run or current_iter_date > latest_downloaded_date_in_run:
                    latest_downloaded_date_in_run = current_iter_date # Keep track of the actual date of the downloaded file

          
            if current_iter_date.month == 12:
                current_iter_date = datetime.date(current_iter_date.year + 1, 1, 1)
            else:
                current_iter_date = datetime.date(current_iter_date.year, current_iter_date.month + 1, 1)

        logger.info(
            "--- CHIRPS extraction completed for the range. Total files downloaded/existing: %s ---",
            len(downloaded_files_paths)
        )

        return downloaded_files_paths, latest_downloaded_date_in_run

    except KeyError as e:
        logger.error(
            "Configuration error: Key '%s' not found in config.yaml. Returning empty list.",
            e
        )
        return [], None

    except Exception as e:
        logger.exception(
            "A critical error occurred in the extract_chirps_data_range function. Returning empty list."
        )
        return [], None



def run_extract_chirps(
    config: dict,
    start_date: datetime.date,
    end_date: Optional[datetime.date] = None
) -> Tuple[List[Path], Optional[datetime.date]]:

    logger.info("Running CHIRPS data extraction process")

    try:
        logger.info(
            "Executing CHIRPS data extraction | From: %s | To: %s",
            start_date,
            end_date if end_date else "current day"
        )

        downloaded_chirps_files, latest_processed_date = extract_chirps_data_range(
            config, start_date, end_date
        )

        if downloaded_chirps_files:
            logger.info(
                "Success: %s CHIRPS files downloaded/verified",
                len(downloaded_chirps_files)
            )
            for f_path in downloaded_chirps_files:
                logger.info("CHIRPS file: %s", f_path.name)
        else:
            logger.info("No CHIRPS files were downloaded or verified")

        logger.info("=" * 80)

        return downloaded_chirps_files, latest_processed_date

    except Exception:
        logger.exception(
            "A fatal error occurred during CHIRPS orchestration execution"
        )
        return [], None



if __name__ == "__main__":
    print("--- Running extract_chirps_data.py as a standalone script for testing ---")

    PROJECT_ROOT_FOR_TESTING = Path(__file__).resolve().parents[3]
    config_file_path_example = PROJECT_ROOT_FOR_TESTING / "config" / "config.yaml"

   
    try:
        config_example = load_config(config_file_path_example)
        
        # Dummy status manager for standalone test
        _test_status_file = PROJECT_ROOT_FOR_TESTING / config_example['data']['status_dir'] / "last_processed_dates.yaml"
        _test_status_file.parent.mkdir(parents=True, exist_ok=True)
        
        def _test_get_last_processed_date(key):
            if _test_status_file.exists():
                try:
                    with open(_test_status_file, 'r') as f:
                        data = yaml.safe_load(f) or {}
                        if key in data:
                            return datetime.datetime.strptime(data[key], '%Y-%m-%d').date()
                except: pass
            return None
        
        def _test_update_last_processed_date(key, date):
            data = {}
            if _test_status_file.exists():
                try:
                    with open(_test_status_file, 'r') as f:
                        data = yaml.safe_load(f) or {}
                except: pass
            data[key] = date.strftime('%Y-%m-%d')
            with open(_test_status_file, 'w') as f:
                yaml.safe_dump(data, f)
            print(f"TEST STATUS: Updated {key} to {date} in {_test_status_file}")
        
        start_date_override_example = datetime.date(2025, 1, 1)
        end_date_example = datetime.date(2025, 2, 1) # Example end date
        downloaded_chirps_files_test, latest_date_test = run_extract_chirps(config_example, start_date_override_example, end_date_example)
        if downloaded_chirps_files_test:
            print(f"\nSuccess: Downloaded/verified {len(downloaded_chirps_files_test)} files (forced start).")
            if latest_date_test:
                _test_update_last_processed_date('chirps', latest_date_test)
        else:
            print("\nNo files downloaded/verified (forced start).")

    except Exception as e:
        print(f"A fatal error occurred during standalone execution: {e}")
        import traceback
        traceback.print_exc()