

import datetime
import logging
import requests
from bs4 import BeautifulSoup
import time
from typing import Optional, Dict, List, Tuple
import pandas as pd
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

def build_enso_url(base_url: str, year: int, month: int) -> str:

    query_param = "?enso_tab=enso-sst_table"
    
    current_date = datetime.date.today()
    target_date = datetime.date(year, month, 1)
    
   
    if target_date.year == current_date.year and target_date.month == current_date.month:
        url_path = "current/"
    else:
   
        first_day_of_previous_month = target_date.replace(day=1) 
        month_name = first_day_of_previous_month.strftime('%B')
        url_path = f"{first_day_of_previous_month.year}-{month_name}-quick-look/"



    full_url = f"{base_url}{url_path}{query_param}"
    logger.debug(
    "Constructed URL for %s-%02d: %s",
    year,
    month,
    full_url
    )

    return full_url

def scrape_webpage_with_retries(
    url: str,
    max_retries: int = 3,
    delay_seconds: int = 5,
    timeout_seconds: int = 10,
    headers: Optional[Dict[str, str]] = None
) -> Optional[BeautifulSoup]:

    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    for attempt in range(1, max_retries + 1):
        logger.info(
            "Attempt %s/%s: Fetching %s",
            attempt,
            max_retries,
            url
        )

        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            logger.info(
                "Successfully fetched %s on attempt %s. Parsing content...",
                url,
                attempt
            )

            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except requests.exceptions.HTTPError as http_err:
            logger.error("HTTP error occurred: %s (Status Code: %s)", http_err, response.status_code)
            
        except requests.exceptions.ConnectionError as conn_err:
            logger.error("Connection error occurred: %s", conn_err)
        except requests.exceptions.Timeout as timeout_err:
            logger.error("Timeout error occurred: %s", timeout_err)
        except requests.exceptions.RequestException as req_err:
            logger.error("An unexpected request error occurred: %s", req_err)
        except Exception as e:
            logger.error("An unknown error occurred during fetching: %s", e)

        if attempt < max_retries:
            logger.info("Retrying in %s seconds...", delay_seconds)
            time.sleep(delay_seconds)
        else:
            logger.error("Max retries reached (%s). Failed to fetch %s.", max_retries, url)
            return None
    return None

def get_oni_index_value(base_url: str, target_year: int, target_month: int) -> List[Tuple[str, float]]:

    target_url = build_enso_url(base_url, target_year, target_month)
    soup = scrape_webpage_with_retries(target_url)

    if not soup:
        logger.error("Failed to fetch content for %s.", target_url)
        return []

    oni_table = soup.find("table", id="modelsTable")
    if not oni_table:
        logger.error("ONI table with ID 'modelsTable' not found on the page.")
        return []

        

    target_row_month = None
    thead = oni_table.find("thead")
    if thead:

        rows = thead.find_all("tr")
        if len(rows) > 1:
            target_row_month = rows[1]
    
    if not target_row_month:
        logger.error("Header row with months not found in the table's thead.")
        return []

    target_row = None
    tbody = oni_table.find("tbody")
    if tbody:
        for row in tbody.find_all("tr"):
            th_tag = row.find("th") # Busca cualquier 'th' en la fila
            if th_tag and "Average, All models" in th_tag.get_text(strip=True):
                target_row = row
                break
    
    if not target_row:
        logger.error("'Average, All models' row not found in the table's tbody.")
        return []
    
 
    header_cells = [th.get_text(strip=True) for th in target_row_month.find_all("th") if th.get_text(strip=True)][1:]
    data_cells = [td.get_text(strip=True) for td in target_row.find_all("td")]

    results = []
    try:
      
        periodo = header_cells[0]
        valor_str = data_cells[0]

        if valor_str and valor_str.replace('.', '', 1).replace('-', '', 1).isdigit():
            valor = float(valor_str)
            results.append((periodo, valor))
    except (ValueError, IndexError):
        logger.warning("Failed to convert the first value in the row ‘Average, All models’.")
        return []

            
    return results
def extract_oni_prediction_df(config: dict, target_year: int, target_month: int) -> Optional[pd.DataFrame]:

    try:
        extraction_date = datetime.date(target_year, target_month, 1)
        extraction_str = extraction_date.strftime('%Y-%m')

        base_url = config['extraction_params']['oni_prediction_extraction']['base_url']
        preds: List[Tuple[str, float]] = get_oni_index_value(base_url, target_year, target_month)

        if not preds:
            logger.warning(f"[PREDICT EXTRACT] No predictions found for {extraction_str}.")
            return None

        # Mapeo del periodo trimestral a mes representativo
        month_map = {
            'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12,
            'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5,
            'MJJ': 6, 'JJA': 7
        }

        rows = []
        for period, value in preds:
            m = month_map.get(period, 0)
            if m == 0:
                logger.debug(f"[PREDICT EXTRACT] Unknown period code '{period}', skipping.")
                continue

            # Si el mes representativo < mes de extracción, el periodo cae en el siguiente año
            year_assigned = target_year + 1 if m < target_month else target_year

            rows.append({
                "year": int(year_assigned),
                "month": int(m),
                #"date_extraction": extraction_str,
                "prediction_period": period,
                "prediction_oni": float(value),
                
            })

        if not rows:
            logger.warning(f"[PREDICT EXTRACT] Built 0 rows for {extraction_str}.")
            return None

        df = pd.DataFrame(rows)
        logger.info(f"[PREDICT EXTRACT] Built {len(df)} row(s) for {extraction_str}.")
        

        return df

    except KeyError as ke:
        logger.error(f"[PREDICT EXTRACT] Missing config key: {ke}")
        return None
    except Exception as e:
        logger.exception(f"[PREDICT EXTRACT] Unhandled error for {target_year}-{target_month}: {e}")
        return None

if __name__ == "__main__":
    # -----------------------------------------
    # MAIN DE PRUEBA PARA EJECUTAR EL EXTRACT
    # -----------------------------------------
    print("\n=== TEST: extract_oni_prediction_df ===")

    from pathlib import Path
    from src.utils.config_loader import load_config
    
    
    project_root = Path(__file__).resolve().parents[3] 
    print(f"Project root detected: {project_root}")


    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)
    config["project_root"] = str(project_root)

    # fecha a probar
    test_year = 2025
    test_month = 12

    print(f"Extracting ONI prediction for {test_year}-{test_month}...")

  
    df_raw = extract_oni_prediction_df(config, test_year, test_month)


    if df_raw is None or df_raw.empty:
        print("\n  Extract returned no data.")
    else:
        print("\nRaw extract output:")
        print(df_raw)
        print("\nDone.")
    

    df_raw.to_csv("C:/Users/laura/OneDrive/TESIS/ETL_LauraChacon/ETL_code/data/raw/oni_prediction_extract.csv", index=False)
