
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

    from scipy.stats import pearsonr, spearmanr

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
        rho_s, p_crudo_s = spearmanr(x, y)

        r1x = _autocorr_lag1(x)
        r1y = _autocorr_lag1(y)
        n_efectivo = n * (1 - r1x * r1y) / (1 + r1x * r1y) if (1 + r1x * r1y) != 0 else n
        p_efectivo = _p_valor_efectivo(r, n, n_efectivo)
        p_efectivo_s = _p_valor_efectivo(rho_s, n, n_efectivo)

        slope, intercept = np.polyfit(x, y, 1)

        filas.append({
            "lag": lag,
            "correlacion": r,
            "p_valor": p_efectivo,
            "p_valor_crudo": p_crudo,
            "spearman": rho_s,
            "spearman_p_valor": p_efectivo_s,
            "spearman_p_valor_crudo": p_crudo_s,
            "n": n,
            "n_efectivo": round(n_efectivo, 1),
            "slope": slope,
            "intercept": intercept,
            "abs_corr": abs(r),
        })

    resultado = pd.DataFrame(filas)
    return resultado


def lag_optimo(df_lags: pd.DataFrame) -> dict:

    if df_lags.empty:
        return {}
    fila = df_lags.loc[df_lags["abs_corr"].idxmax()]
    significativo = fila["p_valor"] < 0.05
    spearman_significativo = fila["spearman_p_valor"] < 0.05
    concuerda_spearman = (
        np.sign(fila["correlacion"]) == np.sign(fila["spearman"])
        and significativo == spearman_significativo
    )
    return {
        "lag":                    int(fila["lag"]),
        "correlacion":            round(float(fila["correlacion"]), 3),
        "p_valor":                round(float(fila["p_valor"]), 4),
        "p_valor_crudo":          round(float(fila["p_valor_crudo"]), 4),
        "significativo":          significativo,
        "spearman":               round(float(fila["spearman"]), 3),
        "spearman_p_valor":       round(float(fila["spearman_p_valor"]), 4),
        "spearman_significativo": spearman_significativo,
        "concuerda_spearman":     bool(concuerda_spearman),
        "slope":                  float(fila["slope"]),
        "intercept":              float(fila["intercept"]),
    }


# ─────────────────────────────────────────────
# 2. ANOMALÍAS POR PERCENTILES P5/P95 POR MES
# ─────────────────────────────────────────────

@st.cache_data
def detectar_anomalias(
    df: pd.DataFrame,
    clima_ref: pd.DataFrame,
) -> pd.DataFrame:
    
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

    df_anomalias = detectar_anomalias(df_combinado, clima_ref)

    registros = []
    for muni, grupo in df_anomalias.groupby("muni_code"):
        lags = calcular_correlacion_lags(grupo, max_lag=max_lag)
        if lags.empty:
            continue
        mejor = lag_optimo(lags)
        registros.append({
            "muni_code":         muni,
            "correlacion":       mejor.get("correlacion"),
            "lag":               mejor.get("lag"),
            "p_valor":           mejor.get("p_valor"),
            "spearman":          mejor.get("spearman"),
            "spearman_p_valor":  mejor.get("spearman_p_valor"),
            "slope":             mejor.get("slope"),
            "intercept":         mejor.get("intercept"),
        })

    resultado = pd.DataFrame(registros)
    if resultado.empty:
        return resultado

    resultado["p_valor_fdr"], resultado["significativo_fdr"] = _correccion_fdr(
        resultado["p_valor"]
    )
    resultado["spearman_p_valor_fdr"], resultado["spearman_significativo_fdr"] = _correccion_fdr(
        resultado["spearman_p_valor"]
    )
    resultado["concuerda_spearman"] = (
        np.sign(resultado["correlacion"]) == np.sign(resultado["spearman"])
    ) & (resultado["significativo_fdr"] == resultado["spearman_significativo_fdr"])

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

    if df_oni_pred.empty:
        return pd.DataFrame()

    df = df_oni_pred[df_oni_pred["date"] > fecha_referencia].copy().sort_values("date")
    pronostico_vencido = df.empty

    if pronostico_vencido:
        df = df_oni_pred.sort_values("date").tail(9).copy()
        if df.empty:
            return df

    df["lead_meses"] = (
        (df["date"].dt.year - fecha_referencia.year) * 12
        + (df["date"].dt.month - fecha_referencia.month)
    )
    df["pronostico_vencido"] = pronostico_vencido

    def _confianza(lead: int) -> str:
        if pronostico_vencido:
            return "Vencido"
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
