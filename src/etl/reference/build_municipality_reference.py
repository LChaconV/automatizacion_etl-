from pathlib import Path
import logging

import pandas as pd
import geopandas as gpd
import yaml

logger = logging.getLogger(__name__)


def build_municipality_reference_geojson(config: dict) -> None:

    project_root = Path(config["project_root"])


    # 2) Construir paths completos
    csv_path = project_root /config['data']['base_dir']/config['data']['reference']['divipola_csv']
    geojson_path = project_root /config['data']['base_dir']/config['data']['reference']["municipalities_geojson"]
    output_parquet_path = project_root / config['data']['base_dir']/config['data']['reference']["dim_municipality_geoparquet"]

    # 3) (Opcional pero recomendable) verificar existencia de entrada
    if not csv_path.exists():
        raise FileNotFoundError(f"Administrative CSV not found at: {csv_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"Municipalities GeoJSON not found at: {geojson_path}")

    logger.info(f"Building municipality reference from:\n  CSV: {csv_path}\n  GEOJSON: {geojson_path}")

    logger.info(f"Loading administrative CSV from: {csv_path}")
    # Nota: ajusta encoding si es necesario ('latin-1', 'utf-8', 'utf-8-sig')
    df_csv = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Estandarizar nombres de columnas del CSV
    # Deben coincidir exactamente con tus encabezados reales:
    # "Código Departamento","Código Municipio","Nombre Departamento","Nombre Municipio"
    col_map = {
        "Código Departamento": "dept_code_raw",
        "Código Municipio": "muni_code_raw",
        "Nombre Departamento": "dept_name",
        "Nombre Municipio": "muni_name",
    }

    missing_cols = [c for c in col_map.keys() if c not in df_csv.columns]
    if missing_cols:
        raise ValueError(
            f"Expected columns {missing_cols} not found in CSV. "
            f"Available columns: {list(df_csv.columns)}"
        )

    df_csv = df_csv.rename(columns=col_map)

    # Normalizar códigos como strings con ceros a la izquierda
    df_csv["dept_code"] = df_csv["dept_code_raw"].astype(str).str.zfill(2)
    df_csv["muni_code"] = df_csv["muni_code_raw"].astype(str).str.zfill(5)

    # Quedarnos solo con columnas necesarias y eliminar duplicados por muni_code
    df_csv_dim = (
        df_csv[["muni_code", "dept_code", "muni_name", "dept_name"]]
        .drop_duplicates(subset=["muni_code"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Administrative dimension: {len(df_csv_dim)} unique municipalities from CSV."
    )

    # -----------------------------------------------------
    # 2) Cargar GeoJSON con geometría de municipios
    # -----------------------------------------------------
    logger.info(f"Loading municipalities GeoJSON from: {geojson_path}")
    gdf_geo = gpd.read_file(geojson_path)

    # Verificación básica de columnas esperadas
    # Según tu descripción: ['MpCodigo', 'MpNombre', ..., 'Depto', 'geometry']
    if "MpCodigo" not in gdf_geo.columns:
        raise ValueError(
            f"'MpCodigo' column not found in GeoJSON. "
            f"Available columns: {list(gdf_geo.columns)}"
        )

    # Crear muni_code normalizado desde MpCodigo
    gdf_geo["muni_code"] = gdf_geo["MpCodigo"].astype(str).str.zfill(5)

    logger.info(
        f"GeoJSON municipalities: {len(gdf_geo)} features before merge."
    )

    # -----------------------------------------------------
    # 3) Unir CSV (códigos + nombres) con GeoJSON (geometría)
    # -----------------------------------------------------
    gdf_merged = gdf_geo.merge(
        df_csv_dim, on="muni_code", how="inner", validate="m:1"
    )

    logger.info(
        "Merged GeoDataFrame: "
        f"{len(gdf_merged)} municipalities after inner join on 'muni_code'."
    )

    # Aviso si hay municipios en CSV que no están en el GeoJSON
    missing_in_geo = set(df_csv_dim["muni_code"]) - set(gdf_merged["muni_code"])
    if missing_in_geo:
        logger.warning(
            f"{len(missing_in_geo)} municipality codes from CSV were not found "
            f"in GeoJSON (example: {list(missing_in_geo)[:5]}...)"
        )

    # Mantener solo columnas finales deseadas
    final_cols = ["muni_code", "dept_code", "muni_name", "dept_name", "geometry"]
    for c in final_cols:
        if c not in gdf_merged.columns:
            raise ValueError(
                f"Expected column '{c}' not found after merge. "
                f"Available columns: {list(gdf_merged.columns)}"
            )

    gdf_final = gdf_merged[final_cols].copy()

    # Opcional: ordenar por dept_code, muni_code
    gdf_final = gdf_final.sort_values(
        ["dept_code", "muni_code"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # 4) Guardar parquet final
    # -----------------------------------------------------
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing unified municipality reference GeoParquet to: {output_parquet_path}")

    # Guarda geometría y atributos en formato columnar
    gdf_final.to_parquet(output_parquet_path, index=False)

    logger.info("Municipality reference GeoParquet successfully created.")

from src.utils.config_loader import load_config

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    config = load_config(project_root / "config" / "config.yaml")
    config["project_root"] = str(project_root)

    build_municipality_reference_geojson(config)
