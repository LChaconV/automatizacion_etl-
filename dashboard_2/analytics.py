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

def _autocorr_lag1(serie: np.ndarray) -> float:
    """Autocorrelación lag-1 de una serie (para corrección de n efectivo)."""
    if len(serie) < 4:
        return 0.0
    r = np.corrcoef(serie[:-1], serie[1:])[0, 1]
    return 0.0 if np.isnan(r) else r


def _p_valor_efectivo(r: float, n: int, n_eff: float) -> float:
    """
    Recalcula el p-valor de una correlación de Pearson usando el tamaño de
    muestra efectivo (n_eff) en vez del nominal (n), para corregir el sesgo
    de autocorrelación serial (Bretherton et al. 1999).

    Sin esta corrección, pearsonr() asume observaciones independientes,
    lo que infla artificialmente la significancia cuando ONI y precipitación
    tienen autocorrelación mes a mes (eventos ENOS duran varios meses).
    """
    from scipy.stats import t as t_dist

    n_eff = max(3.0, min(n_eff, n))
    if abs(r) >= 1:
        return 0.0
    grados_libertad = n_eff - 2
    if grados_libertad <= 0:
        return 1.0
    t_stat = r * np.sqrt(grados_libertad) / np.sqrt(1 - r ** 2)
    return float(2 * t_dist.sf(abs(t_stat), df=grados_libertad))


@st.cache_data
def calcular_correlacion_lags(
    df: pd.DataFrame,
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    Calcula la correlación de Pearson entre ONI (desplazado) y la anomalía
    de precipitación (deseasonalizada) para lags de 0 a max_lag meses.

    El ONI se desplaza hacia atrás (shift positivo) para preguntar:
    '¿el ONI de hace N meses predice la anomalía de precipitación de hoy?'

    Se usa 'anomalia_mm' (precip_mm - p50 climatológico del mes, calculado
    en detectar_anomalias()) en vez de 'precip_mm' crudo. Correlacionar el
    valor crudo mezcla el ciclo estacional (gran amplitud, nada que ver con
    ENOS) con la señal interanual real, lo que distorsiona la correlación.
    El ONI no requiere este ajuste: ya es en sí mismo un índice de anomalía
    de temperatura superficial del mar.

    El p-valor se corrige por autocorrelación serial: tanto el ONI como la
    precipitación tienen memoria mes a mes (un evento ENOS dura varios
    meses), lo que viola el supuesto de independencia de pearsonr() e
    infla artificialmente la significancia. Se recalcula usando un tamaño
    de muestra efectivo (Bretherton et al. 1999).

    Parámetros
    ----------
    df       : DataFrame con columnas 'date', 'anomalia_mm', 'oni'
               (salida de detectar_anomalias(), ya combinado y filtrado
               por municipio o nivel nacional)
    max_lag  : número máximo de meses de rezago a evaluar.
               Default = 6. Con 264 meses disponibles (22 años), lags > 6
               pierden significancia estadística (p > 0.05) en municipios
               del Eje Cafetero y la mayoría de la región Andina. Si se
               amplía la cobertura temporal del dataset, se puede aumentar
               este valor con precaución.

    Retorna
    -------
    DataFrame con columnas:
        lag           int    meses de rezago
        correlacion   float  correlación de Pearson (sobre anomalía deseasonalizada)
        p_valor       float  p-valor corregido por autocorrelación (n efectivo)
        p_valor_crudo float  p-valor sin corrección, solo para referencia/comparación
        n             int    observaciones usadas
        n_efectivo    float  tamaño de muestra efectivo tras corregir autocorrelación
        slope         float  pendiente de la regresión anomalia_mm ~ oni_lag (mm por °C de ONI)
        intercept     float  intercepto de esa regresión
        abs_corr      float  valor absoluto (para ordenar)
    """
    from scipy.stats import pearsonr

    serie = df.sort_values("date")[["date", "anomalia_mm", "oni"]].dropna()

    filas = []
    for lag in range(0, max_lag + 1):
        oni_lag = serie["oni"].shift(lag)
        mascara = oni_lag.notna()
        x = oni_lag[mascara].values
        y = serie["anomalia_mm"][mascara].values
        n = len(x)
        if n < 20:
            continue
        r, p_crudo = pearsonr(x, y)

        r1x = _autocorr_lag1(x)
        r1y = _autocorr_lag1(y)
        n_efectivo = n * (1 - r1x * r1y) / (1 + r1x * r1y) if (1 + r1x * r1y) != 0 else n
        p_efectivo = _p_valor_efectivo(r, n, n_efectivo)

        slope, intercept = np.polyfit(x, y, 1)

        filas.append({
            "lag": lag,
            "correlacion": r,
            "p_valor": p_efectivo,
            "p_valor_crudo": p_crudo,
            "n": n,
            "n_efectivo": round(n_efectivo, 1),
            "slope": slope,
            "intercept": intercept,
            "abs_corr": abs(r),
        })

    resultado = pd.DataFrame(filas)
    return resultado


def lag_optimo(df_lags: pd.DataFrame) -> dict:
    """
    Retorna el lag con mayor correlación absoluta y sus estadísticos.
    El p-valor reportado ya está corregido por autocorrelación.
    """
    if df_lags.empty:
        return {}
    fila = df_lags.loc[df_lags["abs_corr"].idxmax()]
    return {
        "lag":            int(fila["lag"]),
        "correlacion":    round(float(fila["correlacion"]), 3),
        "p_valor":        round(float(fila["p_valor"]), 4),
        "p_valor_crudo":  round(float(fila["p_valor_crudo"]), 4),
        "significativo":  fila["p_valor"] < 0.05,
        "slope":          float(fila["slope"]),
        "intercept":      float(fila["intercept"]),
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

    P50 es la MEDIANA mensual del período de referencia, no la media. Se
    eligió la mediana (en vez de la "normal" en sentido estricto de la OMM,
    que es la media aritmética) porque la precipitación mensual suele tener
    distribución asimétrica y la mediana es más robusta frente a valores
    extremos. Por eso en el dashboard y la tesis debe hablarse de "mediana
    (P50) de referencia OMM 1991–2020", no de "normal OMM" sin más —ese
    término técnicamente designa la media, que también está disponible en
    clima_ref (columna 'media') si se quisiera usar en su lugar.

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
        'déficit'   precipitación < P5 del mes (mediana de referencia OMM)
        'exceso'    precipitación > P95 del mes (mediana de referencia OMM)
        'normal'    dentro del rango

    Soporta tanto un único municipio (o la vista 'NACIONAL') como un
    DataFrame con varios municipios mezclados (ej. todo df_combinado),
    necesario para deseasonalizar antes de correlacionar municipio por
    municipio en el mapa.

    Retorna
    -------
    df original con columnas adicionales:
        p5, p50, p95      umbrales de referencia OMM 1991–2020 del mes (p50 = mediana)
        anomalia          str: 'déficit' | 'exceso' | 'normal'
        desvio_pct        float: % de desvío respecto a la mediana (P50)
        anomalia_mm       float: precip_mm - p50 (anomalía deseasonalizada en mm,
                          respecto a la mediana de referencia, no a la media)
    """
    municipios_presentes = df["muni_code"].unique()

    if list(municipios_presentes) == ["NACIONAL"]:
        # Vista nacional: promediar los percentiles de todos los municipios
        clima = (
            clima_ref.groupby("mes")[["p5", "p50", "p95"]]
            .mean()
            .reset_index()
        )
        resultado = df.merge(clima, on="mes", how="left")
    else:
        clima = clima_ref[clima_ref["muni_code"].isin(municipios_presentes)][
            ["muni_code", "mes", "p5", "p50", "p95"]
        ]
        if clima.empty:
            raise ValueError(
                f"No se encontró climatología de referencia para muni_code en "
                f"{list(municipios_presentes)[:5]}... "
                "Verifica que construir_climatologia_referencia.py se ejecutó correctamente."
            )
        resultado = df.merge(clima, on=["muni_code", "mes"], how="left")

    condiciones = [
        resultado["precip_mm"] < resultado["p5"],
        resultado["precip_mm"] > resultado["p95"],
    ]
    resultado["anomalia"] = np.select(condiciones, ["déficit", "exceso"], default="normal")

    resultado["desvio_pct"] = (
        (resultado["precip_mm"] - resultado["p50"]) / resultado["p50"] * 100
    ).round(1)

    resultado["anomalia_mm"] = resultado["precip_mm"] - resultado["p50"]

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

def _correccion_fdr(p_valores: pd.Series, alpha: float = 0.05) -> tuple[pd.Series, pd.Series]:
    """
    Corrección de Benjamini-Hochberg (FDR) para comparaciones múltiples.

    Sin esto, evaluar ~1.100 municipios de forma independiente a α=0.05
    produce decenas de "falsos positivos" solo por azar. El procedimiento
    ordena los p-valores y ajusta el umbral de decisión según el rango,
    sin tocar los valores de correlación/lag individuales de cada municipio.

    Retorna
    -------
    (p_valor_fdr, significativo_fdr) — mismas posiciones que p_valores
    """
    p = p_valores.values
    n = len(p)
    orden = np.argsort(p)
    p_ordenado = p[orden]
    rangos = np.arange(1, n + 1)

    # p-valor ajustado: monótono, método estándar de Benjamini-Hochberg-Yekutieli
    p_ajustado_ordenado = np.minimum.accumulate((p_ordenado * n / rangos)[::-1])[::-1]
    p_ajustado_ordenado = np.clip(p_ajustado_ordenado, 0, 1)

    p_ajustado = np.empty(n)
    p_ajustado[orden] = p_ajustado_ordenado

    significativo = p_ajustado < alpha

    return (
        pd.Series(p_ajustado, index=p_valores.index),
        pd.Series(significativo, index=p_valores.index),
    )


@st.cache_data(show_spinner="Calculando correlaciones por municipio...")
def correlacion_todos_municipios(
    df_combinado: pd.DataFrame,
    clima_ref: pd.DataFrame,
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    Calcula la correlación ONI ↔ anomalía de precipitación (deseasonalizada)
    en el lag óptimo para todos los municipios, con p-valor corregido por
    autocorrelación y significancia ajustada por comparaciones múltiples
    (FDR). Usado para colorear el mapa coroplético.

    max_lag = 6 por la misma razón que en calcular_correlacion_lags.
    Ver nota en esa función.

    Retorna
    -------
    DataFrame con columnas:
        muni_code, correlacion, lag, p_valor, p_valor_fdr, significativo_fdr,
        slope, intercept
    """
    df_anomalias = detectar_anomalias(df_combinado, clima_ref)

    registros = []
    for muni, grupo in df_anomalias.groupby("muni_code"):
        lags = calcular_correlacion_lags(grupo, max_lag=max_lag)
        if lags.empty:
            continue
        mejor = lag_optimo(lags)
        registros.append({
            "muni_code":   muni,
            "correlacion": mejor.get("correlacion"),
            "lag":         mejor.get("lag"),
            "p_valor":     mejor.get("p_valor"),
            "slope":       mejor.get("slope"),
            "intercept":   mejor.get("intercept"),
        })

    resultado = pd.DataFrame(registros)
    if resultado.empty:
        return resultado

    resultado["p_valor_fdr"], resultado["significativo_fdr"] = _correccion_fdr(
        resultado["p_valor"]
    )
    return resultado


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


# ─────────────────────────────────────────────
# 6. RIESGO PROSPECTIVO (ONI PREDICHO)
# ─────────────────────────────────────────────

def proyectar_riesgo_prospectivo(
    df_oni_pred: pd.DataFrame,
    mejor_lag: dict,
    fecha_referencia: pd.Timestamp,
) -> pd.DataFrame:
    """
    Proyecta el impacto esperado en precipitación a partir del pronóstico
    ONI vigente, usando la sensibilidad ya validada (lag óptimo histórico,
    deseasonalizada y con p-valor corregido) del municipio seleccionado.

    No reemplaza el cálculo histórico de correlación: este solo aplica
    hacia adelante una relación ya estimada con datos observados.

    Como el ETL de pronóstico no conserva la fecha de emisión de cada
    predicción (ver cargar_oni_prediccion()), el horizonte (lead time) se
    aproxima como la distancia en meses entre 'fecha_referencia' (el último
    mes con ONI observado) y el mes objetivo de cada pronóstico. La
    confianza asignada es una categorización cualitativa basada en la
    habilidad de pronóstico de ENOS reportada por CPC/IRI en verificaciones
    históricas (alta a 1-3 meses, moderada a 4-6, baja más allá de 6-7
    meses por la "barrera de primavera"), no una incertidumbre calculada
    a partir del spread real del ensamble de modelos (no disponible aquí).

    Parámetros
    ----------
    df_oni_pred       : salida de data_loader.cargar_oni_prediccion()
    mejor_lag         : salida de lag_optimo() para el municipio/vista activa
    fecha_referencia  : fecha del último ONI observado (ancla del horizonte)

    Retorna
    -------
    DataFrame con columnas:
        date, prediction_period, oni_predicho, lead_meses,
        confianza_pronostico, mes_impacto_esperado, anomalia_esperada_mm,
        relacion_significativa
    """
    if df_oni_pred.empty:
        return pd.DataFrame()

    df = df_oni_pred[df_oni_pred["date"] > fecha_referencia].copy().sort_values("date")
    if df.empty:
        return df

    df["lead_meses"] = (
        (df["date"].dt.year - fecha_referencia.year) * 12
        + (df["date"].dt.month - fecha_referencia.month)
    )

    def _confianza(lead: int) -> str:
        if lead <= 3:
            return "Alta"
        elif lead <= 6:
            return "Media"
        return "Baja"

    df["confianza_pronostico"] = df["lead_meses"].apply(_confianza)

    lag       = mejor_lag.get("lag")
    slope     = mejor_lag.get("slope")
    intercept = mejor_lag.get("intercept")

    if lag is not None and slope is not None:
        df["mes_impacto_esperado"] = df["date"] + pd.DateOffset(months=int(lag))
        df["anomalia_esperada_mm"] = (slope * df["oni_predicho"] + intercept).round(1)
        df["relacion_significativa"] = bool(mejor_lag.get("significativo", False))
    else:
        df["mes_impacto_esperado"]   = pd.NaT
        df["anomalia_esperada_mm"]   = np.nan
        df["relacion_significativa"] = False

    return df.reset_index(drop=True)
