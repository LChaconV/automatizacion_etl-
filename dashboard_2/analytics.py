"""
analytics.py
------------
Cálculos analíticos del dashboard:
  1. Correlación cruzada ONI ↔ precipitación (lags 0–12)
  2. Detección de anomalías por percentiles P5/P95 por municipio-mes
  3. Climatología mensual por fase ONI
  4. KPIs resumen para el panel de métricas
"""

import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────
# 1. CORRELACIÓN CRUZADA Y REZAGO ÓPTIMO
# ─────────────────────────────────────────────

@st.cache_data
def calcular_correlacion_lags(
    df: pd.DataFrame,
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    Calcula la correlación de Pearson entre ONI (desplazado) y precipitación
    para lags de 0 a max_lag meses.

    El ONI se desplaza hacia atrás (shift positivo) para preguntar:
    '¿el ONI de hace N meses predice la precipitación de hoy?'

    Parámetros
    ----------
    df       : DataFrame con columnas 'date', 'precip_mm', 'oni'
               (ya combinado y filtrado por municipio o nivel nacional)
    max_lag  : número máximo de meses de rezago a evaluar.
               Default = 6. Con 264 meses disponibles (22 años), lags > 6
               pierden significancia estadística (p > 0.05) en municipios
               del Eje Cafetero y la mayoría de la región Andina. Si se
               amplía la cobertura temporal del dataset, se puede aumentar
               este valor con precaución.

    Retorna
    -------
    DataFrame con columnas:
        lag        int    meses de rezago
        correlacion float  correlación de Pearson
        p_valor    float  significancia estadística
        abs_corr   float  valor absoluto (para ordenar)
    """
    from scipy.stats import pearsonr

    serie = df.sort_values("date")[["date", "precip_mm", "oni"]].dropna()

    filas = []
    for lag in range(0, max_lag + 1):
        oni_lag = serie["oni"].shift(lag)
        mascara = oni_lag.notna()
        x = oni_lag[mascara].values
        y = serie["precip_mm"][mascara].values
        if len(x) < 20:
            continue
        r, p = pearsonr(x, y)
        filas.append({"lag": lag, "correlacion": r, "p_valor": p, "abs_corr": abs(r)})

    resultado = pd.DataFrame(filas)
    return resultado


def lag_optimo(df_lags: pd.DataFrame) -> dict:
    """
    Retorna el lag con mayor correlación absoluta y sus estadísticos.
    """
    if df_lags.empty:
        return {}
    fila = df_lags.loc[df_lags["abs_corr"].idxmax()]
    return {
        "lag":          int(fila["lag"]),
        "correlacion":  round(float(fila["correlacion"]), 3),
        "p_valor":      round(float(fila["p_valor"]), 4),
        "significativo": fila["p_valor"] < 0.05,
    }


# ─────────────────────────────────────────────
# 2. ANOMALÍAS POR PERCENTILES P5/P95 POR MES
# ─────────────────────────────────────────────

@st.cache_data
def detectar_anomalias(
    df: pd.DataFrame,
    clima_ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Marca cada observación mensual como anomalía comparándola con los
    percentiles P5/P95 de la climatología de referencia OMM 1991–2020.

    Los umbrales son FIJOS e independientes del filtro de período del
    dashboard. El filtro de período solo controla qué observaciones se
    muestran, no cómo se calculan los umbrales.

    Parámetros
    ----------
    df        : DataFrame filtrado por municipio (o nacional), con columnas
                'muni_code', 'mes', 'precip_mm'
    clima_ref : salida de data_loader.cargar_climatologia_referencia()
                columnas: muni_code, mes, p5, p50, p95
                Para la vista Nacional se usa el promedio de p5/p50/p95
                de todos los municipios por mes.

    Tipos de anomalía resultantes:
        'déficit'   precipitación < P5 del mes (referencia OMM)
        'exceso'    precipitación > P95 del mes (referencia OMM)
        'normal'    dentro del rango

    Retorna
    -------
    df original con columnas adicionales:
        p5, p50, p95      umbrales de referencia OMM del mes
        anomalia          str: 'déficit' | 'exceso' | 'normal'
        desvio_pct        float: % de desvío respecto a P50
    """
    es_nacional = df["muni_code"].iloc[0] == "NACIONAL"

    if es_nacional:
        # Vista nacional: promediar los percentiles de todos los municipios
        clima = (
            clima_ref.groupby("mes")[["p5", "p50", "p95"]]
            .mean()
            .reset_index()
        )
        resultado = df.merge(clima, on="mes", how="left")
    else:
        muni = df["muni_code"].iloc[0]
        clima = clima_ref[clima_ref["muni_code"] == muni][["mes", "p5", "p50", "p95"]]
        if clima.empty:
            raise ValueError(
                f"No se encontró climatología de referencia para muni_code={muni}. "
                "Verifica que construir_climatologia_referencia.py se ejecutó correctamente."
            )
        resultado = df.merge(clima, on="mes", how="left")

    condiciones = [
        resultado["precip_mm"] < resultado["p5"],
        resultado["precip_mm"] > resultado["p95"],
    ]
    resultado["anomalia"] = np.select(condiciones, ["déficit", "exceso"], default="normal")

    resultado["desvio_pct"] = (
        (resultado["precip_mm"] - resultado["p50"]) / resultado["p50"] * 100
    ).round(1)

    return resultado


# ─────────────────────────────────────────────
# 3. CLIMATOLOGÍA MENSUAL POR FASE ONI
# ─────────────────────────────────────────────

@st.cache_data
def climatologia_por_fase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la precipitación media mensual (ciclo anual) para cada fase ONI.

    Parámetros
    ----------
    df : DataFrame combinado con 'mes', 'precip_mm', 'fase_oni'

    Retorna
    -------
    DataFrame con columnas: mes, fase_oni, precip_media, precip_p25, precip_p75
    (incluye percentiles para mostrar la dispersión en el gráfico)
    """
    grp = (
        df.groupby(["mes", "fase_oni"])["precip_mm"]
        .agg(
            precip_media="mean",
            precip_p25=lambda x: x.quantile(0.25),
            precip_p75=lambda x: x.quantile(0.75),
            n_meses="count",
        )
        .reset_index()
    )
    # Mantener solo fases con suficientes datos
    grp = grp[grp["n_meses"] >= 3]
    return grp


# ─────────────────────────────────────────────
# 4. CORRELACIÓN POR MUNICIPIO (para el mapa)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Calculando correlaciones por municipio...")
def correlacion_todos_municipios(
    df_combinado: pd.DataFrame,
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    Calcula la correlación ONI ↔ precipitación en el lag óptimo
    para todos los municipios. Usado para colorear el mapa coroplético.

    max_lag = 6 por la misma razón que en calcular_correlacion_lags.
    Ver nota en esa función.

    Retorna
    -------
    DataFrame con columnas: muni_code, correlacion, lag, p_valor
    """
    registros = []
    for muni, grupo in df_combinado.groupby("muni_code"):
        lags = calcular_correlacion_lags(grupo, max_lag=max_lag)
        if lags.empty:
            continue
        mejor = lag_optimo(lags)
        registros.append({
            "muni_code":   muni,
            "correlacion": mejor.get("correlacion"),
            "lag":         mejor.get("lag"),
            "p_valor":     mejor.get("p_valor"),
        })
    return pd.DataFrame(registros)


# ─────────────────────────────────────────────
# 5. KPIs RESUMEN
# ─────────────────────────────────────────────

def calcular_kpis(
    df: pd.DataFrame,
    df_lags: pd.DataFrame,
    df_anomalias: pd.DataFrame,
    df_oni: pd.DataFrame,
) -> dict:
    """
    Calcula los 4 KPIs del panel superior del dashboard.

    Retorna
    -------
    dict con claves:
        precip_promedio_mm   float
        oni_actual           float
        fase_oni_actual      str
        correlacion_optima   float
        lag_optimo_meses     int
        n_anomalias          int
    """
    precip_promedio = round(df["precip_mm"].mean(), 1)

    # ONI más reciente disponible
    oni_reciente = df_oni.sort_values("date").iloc[-1]
    oni_actual    = round(float(oni_reciente["oni"]), 2)
    fase_actual   = oni_reciente["fase_oni"]

    mejor_lag  = lag_optimo(df_lags)
    corr_optima = mejor_lag.get("correlacion", None)
    lag_meses   = mejor_lag.get("lag", None)

    n_anomalias = int((df_anomalias["anomalia"] != "normal").sum())

    return {
        "precip_promedio_mm":  precip_promedio,
        "oni_actual":          oni_actual,
        "fase_oni_actual":     fase_actual,
        "correlacion_optima":  corr_optima,
        "lag_optimo_meses":    lag_meses,
        "n_anomalias":         n_anomalias,
    }
