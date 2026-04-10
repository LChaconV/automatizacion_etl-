# simple_tabular_model.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm



# -------------------------
# Utilidades
# -------------------------
def linear_decreasing_weights(n_years: int) -> np.ndarray:
    """
    Pesos lineales decrecientes (más peso al año más reciente):
      n=5 -> [5,4,3,2,1] normalizados
    """
    if n_years <= 0:
        raise ValueError("n_years debe ser > 0")
    w = np.arange(n_years, 0, -1, dtype=float)
    w = w / w.sum()
    return w


def month_int(m: int | str) -> int:
    mi = int(m)
    if mi < 1 or mi > 12:
        raise ValueError("month debe estar entre 1 y 12")
    return mi


# -------------------------
# Lecturas
# -------------------------
def read_chirps_year(chirps_root: Path, year: int) -> pd.DataFrame:
    p = chirps_root / f"year={year}" / f"fact_chirps_muni_{year}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    df = pd.read_parquet(p)

    required = {"date", "muni_code", "precip_mean_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en CHIRPS {p.name}: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["muni_code"] = df["muni_code"].astype(str)

    # Un registro por muni/mes (por si hay duplicados, promediamos)
    df = (
        df.groupby(["year", "month", "muni_code"], as_index=False)
        .agg(precip_mean_mm=("precip_mean_mm", "mean"))
    )
    return df


def read_oni_year(oni_root: Path, year: int) -> pd.DataFrame:
    p = oni_root / f"year={year}" / f"noaa_oni_{year}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    df = pd.read_parquet(p)

    required = {"date", "value_oni"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en ONI {p.name}: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = (
        df.groupby(["year", "month"], as_index=False)
        .agg(value_oni=("value_oni", "mean"))
    )
    return df


def read_oni_pred_year(oni_pred_root: Path, issue_year: int) -> pd.DataFrame:
    """
    Lee ONI pronosticado para un 'issue_year' (año de emisión del pronóstico).
    Estructura esperada:
      oni_pred_root/year=YYYY/noaa_oni_pred_YYYY.parquet
    Columnas esperadas: date, prediction_oni
    """
    p = oni_pred_root / f"year={issue_year}" / f"noaa_oni_pred_{issue_year}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    df = pd.read_parquet(p)

    required = {"date", "prediction_oni"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en ONI predicho {p.name}: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def get_predicted_oni_value(
    oni_pred_root: Path,
    target_year: int,
    target_month: int,
    issue_years: List[int],
) -> float:
    """
    Busca el ONI pronosticado (prediction_oni) para (target_year, target_month)
    recorriendo una lista de issue_years (por ejemplo [2024, 2025]).
    Devuelve el primer match encontrado.
    """
    for issue_year in issue_years:
        df = read_oni_pred_year(oni_pred_root, issue_year)
        sel = df.loc[
            (df["year"] == target_year) & (df["month"] == target_month),
            "prediction_oni",
        ]
        if not sel.empty:
            return float(sel.iloc[0])

    years_txt = ", ".join(map(str, issue_years))
    raise ValueError(
        f"No se encontró ONI pronosticado para {target_year}-{target_month:02d} "
        f"en issue_years=[{years_txt}]"
    )


# -------------------------
# Cálculos del modelo
# -------------------------
def compute_weighted_avg_for_year_month(
    chirps_root: Path,
    target_year: int,
    target_month: int,
    n_years: int,
) -> pd.DataFrame:
    """
    Promedio ponderado por muni_code para (target_year, target_month),
    usando los años: target_year-1 ... target_year-n_years (mismo mes).
    Maneja faltantes por muni renormalizando pesos localmente.
    """
    weights = linear_decreasing_weights(n_years)  # i=0 es year=target_year-1 (más peso)
    years = [target_year - (i + 1) for i in range(n_years)]  # reciente -> antiguo

    # Leer y pivotear por muni
    frames = []
    for y in years:
        df_y = read_chirps_year(chirps_root, y)
        df_y = df_y[df_y["month"] == target_month][["muni_code", "precip_mean_mm"]].copy()
        df_y.rename(columns={"precip_mean_mm": f"y_{y}"}, inplace=True)
        frames.append(df_y)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="muni_code", how="outer")

    value_cols = [f"y_{y}" for y in years]
    V = merged[value_cols].to_numpy(dtype="float64")  # (n_muni, n_years)

    W = weights.reshape(1, -1)  # (1, n_years)
    valid = ~np.isnan(V)

    w_eff = valid * W
    w_sum = w_eff.sum(axis=1)  # (n_muni,)

    with np.errstate(divide="ignore", invalid="ignore"):
        w_norm = np.where(w_sum.reshape(-1, 1) > 0, w_eff / w_sum.reshape(-1, 1), np.nan)

    avg = np.nansum(V * w_norm, axis=1)

    out = pd.DataFrame(
        {
            "year": target_year,
            "month": target_month,
            "muni_code": merged["muni_code"].astype(str),
            "avg_precip_mm": avg,
            "years_available": valid.sum(axis=1).astype(int),
            "expected_years": n_years,
        }
    )
    return out


def build_training_errors(
    chirps_root: Path,
    target_month: int,
    train_years: List[int],
    n_years: int,
) -> pd.DataFrame:
    """
    Para cada año y en train_years (todos del mismo mes):
      error = real_precip(year,month) - avg_precip(year,month) (avg usa años previos)
    Devuelve columnas: year, month, muni_code, real_precip_mm, avg_precip_mm, error_mm
    """
    all_rows = []
    for y in train_years:
        # promedio para "año y" usando años anteriores (y-1 ... y-n_years)
        avg_df = compute_weighted_avg_for_year_month(chirps_root, y, target_month, n_years)

        real_df = read_chirps_year(chirps_root, y)
        real_df = real_df[(real_df["month"] == target_month)][
            ["year", "month", "muni_code", "precip_mean_mm"]
        ].copy()
        real_df.rename(columns={"precip_mean_mm": "real_precip_mm"}, inplace=True)

        m = real_df.merge(avg_df[["muni_code", "avg_precip_mm"]], on="muni_code", how="left")
        m["error_mm"] = m["real_precip_mm"] - m["avg_precip_mm"]
        all_rows.append(m)

    out = pd.concat(all_rows, ignore_index=True)
    return out



def fit_error_models_by_entity(
    train_df: pd.DataFrame,
    oni_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajusta por muni_code: error_mm = alpha + beta * oni
    Calcula p-valor para prueba de hipótesis sobre beta.
    """
    df = train_df.merge(oni_df, on=["year", "month"], how="left")
    df = df.dropna(subset=["error_mm", "value_oni"]).copy()

    rows = []
    for muni, g in df.groupby("muni_code"):
        if len(g) < 5:  # Subimos a 5 para que la estadística tenga sentido
            continue
            
        X = g["value_oni"].to_numpy()
        y = g["error_mm"].to_numpy()
        
        # Statsmodels necesita agregar la constante manualmente para el alpha
        X_with_const = sm.add_constant(X)
        
        try:
            model_sm = sm.OLS(y, X_with_const).fit()
            
            # El p-valor del coeficiente del ONI (índice 1, ya que 0 es la constante)
            p_val = model_sm.pvalues[1] if len(model_sm.pvalues) > 1 else np.nan
            r_sq = model_sm.rsquared
            
            rows.append({
                "muni_code": str(muni),
                "alpha": float(model_sm.params[0]),
                "beta": float(model_sm.params[1]) if len(model_sm.params) > 1 else 0.0,
                "p_value": p_val,
                "r_squared": r_sq,
                "n_obs": int(len(g)),
                "is_significant": p_val < 0.05  # Nivel de confianza del 95%
            })
        except:
            continue

    return pd.DataFrame(rows)

def predict_interval(
    avg_target_df: pd.DataFrame,
    models_df: pd.DataFrame,
    oni_target_value: float,
) -> pd.DataFrame:
    """
    error_hat = abs(alpha + beta * oni_target)
    interval = [max(0, avg - error_hat), avg + error_hat]
    """
    m = avg_target_df.merge(models_df, on="muni_code", how="left")

    # Si una entidad no tiene modelo (alpha/beta NaN), dejamos NaN (o puedes fallback a 0)
    m["error_hat"] = np.abs(m["alpha"] + m["beta"] * oni_target_value)
    m.loc[m["alpha"].isna() | m["beta"].isna(), "error_hat"] = np.nan

    m["precip_min"] = np.maximum(0.0, m["avg_precip_mm"] - m["error_hat"])
    m["precip_max"] = m["avg_precip_mm"] + m["error_hat"]
    m["oni_used"] = oni_target_value
    return m




# -------------------------
# Ejecución ejemplo
# -------------------------
# -------------------------
# Config de rutas (Windows)
# -------------------------
DEFAULT_CHIRPS_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\chirps_municipal"
DEFAULT_ONI_HIST_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\noaa_historical"
DEFAULT_ONI_PRED_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\noaa_prediction"

def run_example():
    # Objetivo
    target_year = 2025
    target_month = 12

    # Ventanas
    n_years = 10       # para promedio ponderado (baseline)
    train_years = 10   # cantidad de años para entrenar error~ONI (histórico)

    # Rutas
    chirps_root = Path(DEFAULT_CHIRPS_DIR)
    oni_hist_root = Path(DEFAULT_ONI_HIST_DIR)
    oni_pred_root = Path(DEFAULT_ONI_PRED_DIR)

    out_dir = Path("model_outputs_tabular")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Promedio ponderado (baseline) para ENERO 2025 (usa 2015-2024)
    avg_target = compute_weighted_avg_for_year_month(
        chirps_root, target_year, target_month, n_years
    )

    # 2) ONI histórico para entrenar (2015-2024 + opcional 2025 si existiera)
    train_years_list = list(range(target_year - train_years, target_year))  # 2015..2024
    oni_years_needed = sorted(set(train_years_list))  # solo histórico para entrenamiento
    oni_all_hist = pd.concat(
        [read_oni_year(oni_hist_root, y) for y in oni_years_needed],
        ignore_index=True,
    )

    # 3) ONI pronosticado para el mes objetivo (ENERO 2025)
    #    Intentamos primero con pronóstico emitido en 2024; si no está, probamos 2025.
    oni_target_value = get_predicted_oni_value(
        oni_pred_root=oni_pred_root,
        target_year=target_year,
        target_month=target_month,
        issue_years=[2024, 2025],
    )

    # 4) Errores históricos (para el mes objetivo, en años de entrenamiento)
    train_errors = build_training_errors(
        chirps_root,
        target_month,
        train_years=train_years_list,
        n_years=n_years,
    )

    # 5) Modelo error ~ ONI (entrenado con ONI histórico)
    models = fit_error_models_by_entity(train_errors, oni_all_hist)

    # 6) Predicción (intervalo) usando ONI PRONOSTICADO para enero 2025
    prediction = predict_interval(avg_target, models, oni_target_value)

    # 7) Guardar salidas
    avg_target.to_parquet(out_dir / "avg_input.parquet", index=False)
    train_errors.to_parquet(out_dir / "train_errors.parquet", index=False)
    models.to_parquet(out_dir / "models.parquet", index=False)
    prediction.to_parquet(out_dir / "prediction_interval.parquet", index=False)

    print("Modelo ejecutado correctamente")
    print(f"ONI pronosticado usado para {target_year}-{target_month:02d}: {oni_target_value}")
  

if __name__ == "__main__":
    # 🔒 No se ejecuta nada al importar el módulo
    # ▶ Solo se ejecuta si corres este archivo directamente
    run_example()
