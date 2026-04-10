
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =========================================================
# Utilidades (demostración funcional)
# =========================================================
def linear_decreasing_weights(n_years: int) -> np.ndarray:

    if n_years <= 0:
        raise ValueError("n_years debe ser > 0")
    w = np.arange(n_years, 0, -1, dtype=float)
    return w / w.sum()


def month_int(m: int | str) -> int:
    mi = int(m)
    if mi < 1 or mi > 12:
        raise ValueError("month debe estar entre 1 y 12")
    return mi


def _require_cols(df: pd.DataFrame, required: set[str], where: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {where}: {missing}")


def _log(msg: str) -> None:
    print(f"[demo] {msg}")


# =========================================================
# Lecturas (adaptadas a tu salida ETL)
# =========================================================
def read_chirps_year(chirps_root: Path, year: int) -> pd.DataFrame:

    p = chirps_root / f"year={year}" / f"fact_chirps_muni_{year}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")

    df = pd.read_parquet(p)
    _require_cols(df, {"date", "muni_code", "precip_mean_mm"}, f"CHIRPS {p.name}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["muni_code"] = df["muni_code"].astype(str)


    df = (
        df.groupby(["year", "month", "muni_code"], as_index=False)
        .agg(precip_mean_mm=("precip_mean_mm", "mean"))
    )
    return df


def read_oni_hist_year(oni_root: Path, year: int) -> pd.DataFrame:

    p = oni_root / f"year={year}" / f"noaa_oni_{year}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")

    df = pd.read_parquet(p)
    _require_cols(df, {"date", "value_oni"}, f"ONI hist {p.name}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df = (
        df.groupby(["year", "month"], as_index=False)
        .agg(value_oni=("value_oni", "mean"))
    )
    return df


def read_oni_pred_issue_year(oni_pred_root: Path, issue_year: int) -> pd.DataFrame:

    p = oni_pred_root / f"year={issue_year}" / f"noaa_oni_pred_{issue_year}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")

    df = pd.read_parquet(p)
    _require_cols(df, {"date", "prediction_oni"}, f"ONI pred {p.name}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def get_oni_target_value(
    oni_hist_root: Path,
    oni_pred_root: Optional[Path],
    target_year: int,
    target_month: int,
    issue_years: List[int],
) -> Tuple[float, str]:

    # 1) Predicho
    if oni_pred_root is not None:
        for issue_year in issue_years:
            try:
                dfp = read_oni_pred_issue_year(oni_pred_root, issue_year)
            except FileNotFoundError:
                continue

            sel = dfp.loc[
                (dfp["year"] == target_year) & (dfp["month"] == target_month),
                "prediction_oni",
            ]
            if not sel.empty and pd.notna(sel.iloc[0]):
                return float(sel.iloc[0]), f"predicho(issue_year={issue_year})"

    # 2) Histórico
    try:
        dfh = read_oni_hist_year(oni_hist_root, target_year)
        sel = dfh.loc[
            (dfh["year"] == target_year) & (dfh["month"] == target_month),
            "value_oni",
        ]
        if not sel.empty and pd.notna(sel.iloc[0]):
            return float(sel.iloc[0]), "historico"
    except FileNotFoundError:
        pass

    raise ValueError(
        f"No se encontró ONI para {target_year}-{target_month:02d} "
        f"(ni predicho en {issue_years}, ni histórico)."
    )


# =========================================================
# Baseline: promedio ponderado por municipio (mismo mes, años previos)
# =========================================================
def compute_weighted_avg_for_year_month(
    chirps_root: Path,
    target_year: int,
    target_month: int,
    n_years: int,
) -> pd.DataFrame:

    target_month = month_int(target_month)

    weights = linear_decreasing_weights(n_years) 
    years = [target_year - (i + 1) for i in range(n_years)] 

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
    V = merged[value_cols].to_numpy(dtype="float64")  

    W = weights.reshape(1, -1)
    valid = ~np.isnan(V)

    w_eff = valid * W
    w_sum = w_eff.sum(axis=1)

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


# =========================================================
# Entrenamiento "ligero": error del baseline explicado por ONI
# =========================================================
def build_training_errors(
    chirps_root: Path,
    target_month: int,
    train_years: List[int],
    n_years: int,
) -> pd.DataFrame:

    target_month = month_int(target_month)

    all_rows = []
    for y in train_years:
        avg_df = compute_weighted_avg_for_year_month(chirps_root, y, target_month, n_years)

        real_df = read_chirps_year(chirps_root, y)
        real_df = real_df[real_df["month"] == target_month][
            ["year", "month", "muni_code", "precip_mean_mm"]
        ].copy()
        real_df.rename(columns={"precip_mean_mm": "real_precip_mm"}, inplace=True)

        m = real_df.merge(avg_df[["muni_code", "avg_precip_mm"]], on="muni_code", how="left")
        m["error_mm"] = m["real_precip_mm"] - m["avg_precip_mm"]
        all_rows.append(m)

    return pd.concat(all_rows, ignore_index=True)


def fit_error_models_by_entity(
    train_df: pd.DataFrame,
    oni_df: pd.DataFrame,
    min_obs: int = 5,
) -> pd.DataFrame:

    df = train_df.merge(oni_df, on=["year", "month"], how="left")
    df = df.dropna(subset=["error_mm", "value_oni"]).copy()

    rows = []
    for muni, g in df.groupby("muni_code"):
        if len(g) < min_obs:
            continue

        X = g["value_oni"].to_numpy()
        y = g["error_mm"].to_numpy()
        Xc = sm.add_constant(X)

        try:
            model = sm.OLS(y, Xc).fit()
            p_val = model.pvalues[1] if len(model.pvalues) > 1 else np.nan

            rows.append(
                {
                    "muni_code": str(muni),
                    "alpha": float(model.params[0]),
                    "beta": float(model.params[1]) if len(model.params) > 1 else 0.0,
                    "p_value": float(p_val) if pd.notna(p_val) else np.nan,
                    "r_squared": float(model.rsquared),
                    "n_obs": int(len(g)),
                    "is_significant": bool(p_val < 0.05) if pd.notna(p_val) else False,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


# =========================================================
# Predicción funcional: baseline + ajuste ONI (intervalo)
# =========================================================
def predict_interval(
    avg_target_df: pd.DataFrame,
    models_df: pd.DataFrame,
    oni_target_value: float,
    fallback_error_mm: float = 0.0,
) -> pd.DataFrame:

    m = avg_target_df.merge(models_df, on="muni_code", how="left")

    m["error_hat"] = np.abs(m["alpha"] + m["beta"] * oni_target_value)

    no_model = m["alpha"].isna() | m["beta"].isna()
    m.loc[no_model, "error_hat"] = fallback_error_mm

    m["precip_min"] = np.maximum(0.0, m["avg_precip_mm"] - m["error_hat"])
    m["precip_max"] = m["avg_precip_mm"] + m["error_hat"]
    m["oni_used"] = float(oni_target_value)
    return m


# =========================================================
# Ejecución demostrativa (ETL -> outputs)
# =========================================================
def run_demo(
    chirps_root: Path,
    oni_hist_root: Path,
    oni_pred_root: Optional[Path],
    target_year: int,
    target_month: int,
    n_years_baseline: int = 10,
    train_years_back: int = 10,
    issue_years: Optional[List[int]] = None,
    out_dir: Path = Path("model_outputs_demo"),
) -> None:

    issue_years = issue_years or [target_year - 1, target_year]
    target_month = month_int(target_month)

    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Target: {target_year}-{target_month:02d}")
    _log(f"Baseline: n_years={n_years_baseline}")
    _log(f"Train error~ONI: últimos {train_years_back} años (mismo mes)")


    avg_target = compute_weighted_avg_for_year_month(
        chirps_root, target_year, target_month, n_years_baseline
    )
    _log(f"Baseline listo: {len(avg_target):,} municipios")

   
    train_years_list = list(range(target_year - train_years_back, target_year))
    oni_hist = pd.concat(
        [read_oni_hist_year(oni_hist_root, y) for y in sorted(set(train_years_list))],
        ignore_index=True,
    )
    _log(f"ONI histórico cargado: {oni_hist['year'].min()}..{oni_hist['year'].max()}")

    oni_target_value, oni_source = get_oni_target_value(
        oni_hist_root=oni_hist_root,
        oni_pred_root=oni_pred_root,
        target_year=target_year,
        target_month=target_month,
        issue_years=issue_years,
    )
    _log(f"ONI usado ({oni_source}) = {oni_target_value:.3f}")


    train_errors = build_training_errors(
        chirps_root=chirps_root,
        target_month=target_month,
        train_years=train_years_list,
        n_years=n_years_baseline,
    )
    _log(f"Tabla de errores: {len(train_errors):,} filas")


    models = fit_error_models_by_entity(train_errors, oni_hist, min_obs=5)
    _log(f"Modelos ajustados: {len(models):,} municipios (con >=5 obs)")


    prediction = predict_interval(avg_target, models, oni_target_value, fallback_error_mm=0.0)


    avg_target.to_parquet(out_dir / "avg_baseline.parquet", index=False)
    train_errors.to_parquet(out_dir / "train_errors.parquet", index=False)
    models.to_parquet(out_dir / "error_models.parquet", index=False)
    prediction.to_parquet(out_dir / "prediction_interval.parquet", index=False)

  
    summary = {
        "target_year": target_year,
        "target_month": target_month,
        "n_years_baseline": n_years_baseline,
        "train_years_back": train_years_back,
        "oni_used": oni_target_value,
        "oni_source": oni_source,
        "n_munis_baseline": int(len(avg_target)),
        "n_munis_with_model": int(len(models)),
        "out_dir": str(out_dir.resolve()),
    }
    pd.DataFrame([summary]).to_json(out_dir / "run_summary.json", orient="records", indent=2)
    _log(f"Outputs guardados en: {out_dir.resolve()}")


# =========================================================
# Config por defecto 
# =========================================================
DEFAULT_CHIRPS_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\chirps_municipal"
DEFAULT_ONI_HIST_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\noaa_historical"
DEFAULT_ONI_PRED_DIR = r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon\ETL_code\data\processed\noaa_prediction"


def run_example():
    
    target_year = 2025
    target_month = 12

    run_demo(
        chirps_root=Path(DEFAULT_CHIRPS_DIR),
        oni_hist_root=Path(DEFAULT_ONI_HIST_DIR),
        oni_pred_root=Path(DEFAULT_ONI_PRED_DIR), 
        target_year=target_year,
        target_month=target_month,
        n_years_baseline=10,
        train_years_back=10,
        issue_years=[2024, 2025], 
        out_dir=Path("model_outputs_demo"),
    )


if __name__ == "__main__":
    run_example()
