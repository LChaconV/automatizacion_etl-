

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

# Mapeo período trimestral móvil -> mes representativo (mes central del
# trimestre). Coincide con la convención estándar ENOS: la letra central
# del código de 3 letras es el mes central (ej. JAS -> Agosto).
MONTH_MAP = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
    'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12,
}
PERIOD_BY_MONTH = {v: k for k, v in MONTH_MAP.items()}


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


def _get_oni_index_value_from_html(base_url: str, target_year: int, target_month: int) -> List[Tuple[str, float]]:
    """
    Intento primario (versión original): scrapea la tabla de texto
    id="modelsTable" de la página HTML pública de IRI y lee la fila
    "Average, All models". Solo devuelve el primer período/valor de esa
    fila (igual que la versión original).

    Puede devolver [] si la tabla, la fila o el valor no se encuentran
    (ej. porque IRI rediseñó la página) — en ese caso, extract_oni_prediction_df()
    cae al fallback de JSON, ver _get_oni_index_value_from_json().
    """
    target_url = build_enso_url(base_url, target_year, target_month)
    soup = scrape_webpage_with_retries(target_url)

    if not soup:
        logger.warning("Failed to fetch content for %s.", target_url)
        return []

    oni_table = soup.find("table", id="modelsTable")
    if not oni_table:
        logger.warning("ONI table with ID 'modelsTable' not found on the page.")
        return []

    target_row_month = None
    thead = oni_table.find("thead")
    if thead:
        rows = thead.find_all("tr")
        if len(rows) > 1:
            target_row_month = rows[1]

    if not target_row_month:
        logger.warning("Header row with months not found in the table's thead.")
        return []

    target_row = None
    tbody = oni_table.find("tbody")
    if tbody:
        for row in tbody.find_all("tr"):
            th_tag = row.find("th")  # Busca cualquier 'th' en la fila
            if th_tag and "Average, All models" in th_tag.get_text(strip=True):
                target_row = row
                break

    if not target_row:
        logger.warning("'Average, All models' row not found in the table's tbody.")
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
        logger.warning("Failed to convert the first value in the row 'Average, All models'.")
        return []

    return results


def build_plumes_json_url(json_base_url: str, year: int, month: int) -> str:
    """
    Construye la URL del API JSON que alimenta el gráfico de pronóstico
    ENOS de IRI (plumes_json). 'month' es 1-indexado (1=enero); el API
    espera 0-indexado (0=enero), igual que JS Date.getMonth().
    """
    month_0indexed = month - 1
    return f"{json_base_url}{year}/{month_0indexed}"


def _fetch_plumes_json(
    json_base_url: str,
    target_year: int,
    target_month: int,
    max_retries: int = 3,
    delay_seconds: int = 5,
    timeout_seconds: int = 15,
) -> Optional[dict]:

    url = build_plumes_json_url(json_base_url, target_year, target_month)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(1, max_retries + 1):
        logger.info("Attempt %s/%s: Fetching %s", attempt, max_retries, url)
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            # El API devuelve 500 cuando se pide un mes futuro sin pronóstico
            # vigente todavía (no es un error transitorio: no tiene sentido reintentar).
            logger.error("HTTP error fetching %s: %s", url, http_err)
            if response.status_code >= 400 and response.status_code < 500:
                return None
        except requests.exceptions.RequestException as req_err:
            logger.error("Request error fetching %s: %s", url, req_err)
        except ValueError as json_err:
            logger.error("Invalid JSON from %s: %s", url, json_err)
            return None

        if attempt < max_retries:
            logger.info("Retrying in %s seconds...", delay_seconds)
            time.sleep(delay_seconds)
        else:
            logger.error("Max retries reached (%s). Failed to fetch %s.", max_retries, url)
            return None
    return None


def _get_oni_index_value_from_json(json_base_url: str, target_year: int, target_month: int) -> List[Tuple[str, float]]:
    """
    Fallback: obtiene el pronóstico ONI consensuado ("Average, All models",
    equivalente a 'averages.total' en el API JSON de IRI) para los 9
    trimestres móviles siguientes al mes de emisión, empezando por el
    trimestre que contiene a ese mes (ej. emisión en junio -> MJJ es el
    primer período). A diferencia del intento primario (HTML), este
    devuelve los 9 períodos disponibles, no solo el primero.
    """
    data = _fetch_plumes_json(json_base_url, target_year, target_month)
    if not data:
        logger.error(
            "No se pudo obtener el pronóstico ENOS para %s-%02d desde el API JSON de IRI.",
            target_year, target_month,
        )
        return []

    valores = data.get("averages", {}).get("total")
    if not valores:
        logger.error(
            "La respuesta del API JSON no contiene 'averages.total' para %s-%02d.",
            target_year, target_month,
        )
        return []

    resultados = []
    for i, valor in enumerate(valores):
        if valor is None or valor == -999:
            continue
        mes_periodo = ((target_month - 1 + i) % 12) + 1
        periodo = PERIOD_BY_MONTH[mes_periodo]
        resultados.append((periodo, float(valor)))

    return resultados


def get_oni_index_value(
    base_url: str,
    target_year: int,
    target_month: int,
    json_base_url: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Intenta primero el scraping HTML original (tabla 'modelsTable'). Si no
    la encuentra (la página ya no la tiene, o cambió de estructura), cae
    al fallback de API JSON ('plumes_json'), que es lo que el sitio de IRI
    usa actualmente para alimentar el gráfico de pronóstico.
    """
    resultados = _get_oni_index_value_from_html(base_url, target_year, target_month)
    if resultados:
        return resultados

    if not json_base_url:
        logger.warning(
            "Tabla HTML no disponible y no se configuró json_fallback_base_url; "
            "no se puede obtener el pronóstico para %s-%02d.",
            target_year, target_month,
        )
        return []

    logger.warning(
        "Tabla HTML no disponible para %s-%02d, usando fallback de API JSON.",
        target_year, target_month,
    )
    return _get_oni_index_value_from_json(json_base_url, target_year, target_month)


def extract_oni_prediction_df(config: dict, target_year: int, target_month: int) -> Optional[pd.DataFrame]:

    try:
        extraction_date = datetime.date(target_year, target_month, 1)
        extraction_str = extraction_date.strftime('%Y-%m')

        extraction_cfg = config['extraction_params']['oni_prediction_extraction']
        base_url = extraction_cfg['base_url']
        json_base_url = extraction_cfg.get('json_fallback_base_url')

        preds: List[Tuple[str, float]] = get_oni_index_value(
            base_url, target_year, target_month, json_base_url=json_base_url
        )

        if not preds:
            logger.warning(f"[PREDICT EXTRACT] No predictions found for {extraction_str}.")
            return None

        rows = []
        for period, value in preds:
            m = MONTH_MAP.get(period, 0)
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
