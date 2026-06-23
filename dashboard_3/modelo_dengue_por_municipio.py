"""
modelo_dengue_81001.py
----------------------
Modelo predictivo de casos de dengue para municipio 81001 (Arauca)
usando regresión binomial negativa.

Variables:
    - Estacionalidad: mes del año (dummies, enero = referencia)
    - Predictor climático: ONI con lag 0 o lag 1

Dos versiones:
    Modelo A: ONI lag 0 (mismo mes)
    Modelo B: ONI lag 1 (mes anterior)

Partición:
    Entrenamiento : 2013–2021
    Validación    : 2022–2023

Métricas de comparación:
    AIC, MAE, correlación predicho vs observado en validación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from scipy.stats import spearmanr

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

DATASET_PATH = (
    r"C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon"
    r"\ETL_code\dashboard_3\processed\dataset_52001_2013_2023.parquet"
)

ANIO_TRAIN_FIN = 2021
ANIO_VAL_INI   = 2022

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        10,
})

# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el dataset y verifica columnas requeridas."""
    df = pd.read_parquet(ruta)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Derivar mes y anio desde date si no existen
    if "mes" not in df.columns:
        df["mes"] = df["date"].dt.month
    if "anio" not in df.columns:
        df["anio"] = df["date"].dt.year
        
    requeridas = ["date", "casos", "value_oni", "mes", "anio"]
    faltantes  = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el dataset: {faltantes}")



    print(f"Dataset cargado: {len(df)} registros · "
          f"{df['anio'].min()}–{df['anio'].max()}")
    return df


def preparar_features(df: pd.DataFrame, lag_oni: int) -> pd.DataFrame:
    """
    Construye el DataFrame de features:
        - Dummies de mes (enero = referencia, se omite)
        - ONI con el lag especificado
        - Intercepto

    Parámetros
    ----------
    df      : dataset completo
    lag_oni : rezago del ONI en meses (0 o 1)
    """
    df = df.copy()

    # ONI con rezago
    df["oni_lag"] = df["value_oni"].shift(lag_oni)

    # Dummies de mes (enero = referencia → se omite con drop_first=True)
    dummies_mes = pd.get_dummies(df["mes"], prefix="mes", drop_first=True)
    # Asegurar que sean int
    dummies_mes = dummies_mes.astype(int)

    # Combinar features
    features = pd.concat([dummies_mes, df["oni_lag"]], axis=1)
    features = sm.add_constant(features)

    # Eliminar filas con NaN (por el lag)
    mask    = df["oni_lag"].notna()
    X       = features[mask].reset_index(drop=True)
    y       = df.loc[mask, "casos"].reset_index(drop=True)
    fechas  = df.loc[mask, "date"].reset_index(drop=True)
    anios   = df.loc[mask, "anio"].reset_index(drop=True)

    return X, y, fechas, anios


def ajustar_modelo(X_train: pd.DataFrame,
                   y_train: pd.Series) -> sm.NegativeBinomial:
    """
    Ajusta un modelo de regresión binomial negativa.

    Retorna el modelo ajustado (ResultsWrapper de statsmodels).
    """
    modelo = sm.NegativeBinomial(y_train, X_train)
    resultado = modelo.fit(disp=False, maxiter=200)
    return resultado


def evaluar_modelo(resultado, X_val: pd.DataFrame,
                   y_val: pd.Series) -> dict:
    """
    Evalúa el modelo en el conjunto de validación.

    Retorna dict con:
        predicciones : array de casos predichos
        mae          : error absoluto medio
        correlacion  : Spearman entre predicho y observado
        p_valor      : significancia de la correlación
    """
    predicciones = resultado.predict(X_val)

    mae = float(np.mean(np.abs(predicciones - y_val)))
    r, p = spearmanr(predicciones, y_val)

    return {
        "predicciones": predicciones,
        "mae":          round(mae, 2),
        "correlacion":  round(r, 4),
        "p_valor":      round(p, 4),
    }


def graficar_resultados(fechas_val: pd.Series,
                        y_val: pd.Series,
                        res_a: dict,
                        res_b: dict) -> None:
    """
    Grafica casos observados vs predichos para los dos modelos
    en el período de validación.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        "Validación del modelo · Municipio 81001 · 2022–2023",
        fontsize=13, fontweight="bold"
    )

    for ax, res, lag, color in zip(
        axes,
        [res_a, res_b],
        [0, 1],
        ["#185FA5", "#E24B4A"],
    ):
        ax.bar(fechas_val, y_val,
               color="gray", alpha=0.4, width=20, label="Observado")
        ax.plot(fechas_val, res["predicciones"],
                color=color, linewidth=2, marker="o",
                markersize=4, label=f"Predicho (lag {lag})")
        ax.set_ylabel("Casos de dengue")
        ax.set_title(
            f"Modelo lag {lag} · MAE = {res['mae']:.2f} · "
            f"Spearman r = {res['correlacion']:.3f} "
            f"(p = {res['p_valor']:.4f})",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def graficar_coeficientes(resultado_a, resultado_b) -> None:
    """
    Grafica los coeficientes del ONI y su intervalo de confianza
    para los dos modelos.
    """
    fig, ax = plt.subplots(figsize=(7, 3))

    for i, (res, lag, color) in enumerate(zip(
        [resultado_a, resultado_b], [0, 1], ["#185FA5", "#E24B4A"]
    )):
        coef = res.params["oni_lag"]
        ci   = res.conf_int().loc["oni_lag"]
        ax.errorbar(
            x=coef, y=i,
            xerr=[[coef - ci[0]], [ci[1] - coef]],
            fmt="o", color=color, markersize=8, linewidth=2,
            label=f"Lag {lag}",
        )

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Lag 0", "Lag 1"])
    ax.set_xlabel("Coeficiente ONI (escala log)")
    ax.set_title("Efecto del ONI sobre casos de dengue · IC 95%",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def preparar_features_solo_oni(df: pd.DataFrame, lag_oni: int) -> tuple:
    """
    Construye features usando SOLO el ONI como predictor (sin estacionalidad).
    Útil para aislar el efecto del ONI independientemente del ciclo anual.
    """
    df = df.copy()
    df["oni_lag"] = df["value_oni"].shift(lag_oni)

    features = sm.add_constant(df[["oni_lag"]])
    mask   = df["oni_lag"].notna()
    X      = features[mask].reset_index(drop=True)
    y      = df.loc[mask, "casos"].reset_index(drop=True)
    fechas = df.loc[mask, "date"].reset_index(drop=True)
    anios  = df.loc[mask, "anio"].reset_index(drop=True)

    return X, y, fechas, anios


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    # ── Carga ─────────────────────────────────────────────────────────────
    df = cargar_datos(DATASET_PATH)

    # ── Preparar features para cada versión ───────────────────────────────
    print("\nPreparando features...")
    X_a, y_a, fechas_a, anios_a = preparar_features(df, lag_oni=0)
    X_b, y_b, fechas_b, anios_b = preparar_features(df, lag_oni=1)

    # ── Partición train / validación ──────────────────────────────────────
    mask_train_a = anios_a <= ANIO_TRAIN_FIN
    mask_val_a   = anios_a >= ANIO_VAL_INI
    mask_train_b = anios_b <= ANIO_TRAIN_FIN
    mask_val_b   = anios_b >= ANIO_VAL_INI

    X_train_a, y_train_a = X_a[mask_train_a], y_a[mask_train_a]
    X_val_a,   y_val_a   = X_a[mask_val_a],   y_a[mask_val_a]
    X_train_b, y_train_b = X_b[mask_train_b], y_b[mask_train_b]
    X_val_b,   y_val_b   = X_b[mask_val_b],   y_b[mask_val_b]

    print(f"\nEntrenamiento : {(anios_a[mask_train_a]).min()}–"
          f"{(anios_a[mask_train_a]).max()} "
          f"({mask_train_a.sum()} meses)")
    print(f"Validación    : {(anios_a[mask_val_a]).min()}–"
          f"{(anios_a[mask_val_a]).max()} "
          f"({mask_val_a.sum()} meses)")

    # ── Ajuste ────────────────────────────────────────────────────────────
    print("\nAjustando Modelo A (lag 0)...")
    resultado_a = ajustar_modelo(X_train_a, y_train_a)

    print("Ajustando Modelo B (lag 1)...")
    resultado_b = ajustar_modelo(X_train_b, y_train_b)

    # ── Evaluación ────────────────────────────────────────────────────────
    res_a = evaluar_modelo(resultado_a, X_val_a, y_val_a)
    res_b = evaluar_modelo(resultado_b, X_val_b, y_val_b)

    # ── Resumen comparativo ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 55)
    print(f"{'Métrica':<25} {'Lag 0':>12} {'Lag 1':>12}")
    print("-" * 55)
    print(f"{'AIC (entrenamiento)':<25} "
          f"{resultado_a.aic:>12.2f} {resultado_b.aic:>12.2f}")
    print(f"{'MAE (validación)':<25} "
          f"{res_a['mae']:>12.2f} {res_b['mae']:>12.2f}")
    print(f"{'Spearman r (validación)':<25} "
          f"{res_a['correlacion']:>12.4f} {res_b['correlacion']:>12.4f}")
    print(f"{'p-valor Spearman':<25} "
          f"{res_a['p_valor']:>12.4f} {res_b['p_valor']:>12.4f}")
    print("=" * 55)

    # Coeficiente ONI en cada modelo
    print(f"\nCoeficiente ONI lag 0 : "
          f"{resultado_a.params['oni_lag']:.4f} "
          f"(p = {resultado_a.pvalues['oni_lag']:.4f})")
    print(f"Coeficiente ONI lag 1 : "
          f"{resultado_b.params['oni_lag']:.4f} "
          f"(p = {resultado_b.pvalues['oni_lag']:.4f})")

    # ── Prueba: modelo solo ONI (sin estacionalidad) ─────────────────────
    print("\n" + "=" * 55)
    print("PRUEBA: MODELO SOLO ONI (sin estacionalidad)")
    print("=" * 55)

    for lag in [0, 1]:
        X_s, y_s, fechas_s, anios_s = preparar_features_solo_oni(df, lag_oni=lag)
        mask_tr = anios_s <= ANIO_TRAIN_FIN
        mask_vl = anios_s >= ANIO_VAL_INI

        res_solo = ajustar_modelo(X_s[mask_tr], y_s[mask_tr])
        eval_solo = evaluar_modelo(res_solo, X_s[mask_vl], y_s[mask_vl])

        print(f"\nLag {lag}:")
        print(f"  AIC entrenamiento : {res_solo.aic:.2f}")
        print(f"  MAE validación    : {eval_solo['mae']:.2f}")
        print(f"  Spearman r        : {eval_solo['correlacion']:.4f} "
              f"(p = {eval_solo['p_valor']:.4f})")
        print(f"  Coef. ONI         : {res_solo.params['oni_lag']:.4f} "
              f"(p = {res_solo.pvalues['oni_lag']:.4f})")

    # ── Gráficos ──────────────────────────────────────────────────────────
    fechas_val = fechas_a[mask_val_a].reset_index(drop=True)
    graficar_resultados(fechas_val, y_val_a.reset_index(drop=True),
                        res_a, res_b)
    graficar_coeficientes(resultado_a, resultado_b)

    return resultado_a, resultado_b, res_a, res_b


if __name__ == "__main__":
    resultado_a, resultado_b, res_a, res_b = main()