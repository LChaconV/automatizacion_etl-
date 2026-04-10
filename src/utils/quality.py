from __future__ import annotations
import pandas as pd


class QualityError(Exception):
    """Raised when a dataset fails quality checks."""


def check_noaa_oni(
    df: pd.DataFrame,
    min_val: float = -5.0,
    max_val: float = 5.0,
    require_month_day_eq_1: bool = True,
) -> None:

    required = {"date", "value_oni"}
    missing = required - set(df.columns)
    if missing:
        raise QualityError(f"Missing required columns: {missing}")

    if df["date"].isna().any() or df["value_oni"].isna().any():
        raise QualityError("Nulls detected in 'date' or 'value_oni'.")

    dates = pd.to_datetime(df["date"])
    values = pd.to_numeric(df["value_oni"], errors="coerce")

    if values.isna().any():
        raise QualityError("Non-numeric values found in 'value_oni'.")

    out_of_range = (values < min_val) | (values > max_val)
    if out_of_range.any():
        bad = df.loc[out_of_range, ["date", "value_oni"]].head(5)
        raise QualityError(f"'value_oni' out of range [{min_val}, {max_val}]. Examples:\n{bad}")


    if require_month_day_eq_1:
        bad_day = dates.dt.day.ne(1)
        if bad_day.any():
            bad = df.loc[bad_day, ["date"]].head(5)
            raise QualityError("Some 'date' values are not the 1st day of month. Examples:\n" + bad.to_string(index=False))


    if dates.duplicated().any():
        dup = df.loc[dates.duplicated(), ["date"]].head(5)
        raise QualityError("Duplicate 'date' values found. Examples:\n" + dup.to_string(index=False))

def check_chirps_stats(
    meta_payload: dict,
    min_ge: float = 0.0,
    mean_le: float = 1000.0,
    require_crs: bool = True,
    require_nodata: bool = True,
) -> None:
    
    if "stats" not in meta_payload or "meta" not in meta_payload:
        raise QualityError("Invalid CHIRPS meta payload: missing 'stats' or 'meta' keys.")

    stats = meta_payload["stats"]
    meta = meta_payload["meta"]

    for k in ("min", "max", "mean", "std"):
        if k not in stats:
            raise QualityError(f"Missing stats key: '{k}'")

    for k in ("width", "height", "pixel_size_x", "pixel_size_y"):
        if k not in meta:
            raise QualityError(f"Missing meta key: '{k}'")

    if require_crs and not meta.get("crs"):
        raise QualityError("CRS is required but missing/empty.")

    if require_nodata and meta.get("nodata") is None:
        raise QualityError("NoData value is required but missing.")

   
    if stats["min"] < min_ge:
        raise QualityError(f"CHIRPS stat_min={stats['min']} < {min_ge} (invalid).")

    if stats["mean"] > mean_le:
        raise QualityError(f"CHIRPS stat_mean={stats['mean']} > {mean_le} (suspiciously high).")

    if int(meta["width"]) <= 0 or int(meta["height"]) <= 0:
        raise QualityError(f"Invalid raster shape: width={meta['width']}, height={meta['height']}.")

