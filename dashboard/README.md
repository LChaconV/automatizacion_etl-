# Dashboard de Exposición Climática · Colombia

**Laura Andrea Chacón Velásquez · Tesis Maestría Ciencia de Datos**  
Escuela Colombiana de Ingeniería Julio Garavito · 2025

---

## Descripción

Dashboard interactivo en Streamlit que muestra el índice de exposición climática
por departamento en Colombia, combinando tres fuentes del pipeline ETL:

| Fuente | Datos | Uso en el dashboard |
|--------|-------|---------------------|
| **CHIRPS** | Precipitación satelital mensual por municipio | Anomalías, variabilidad, score de riesgo |
| **NOAA** | Índice Oceánico de El Niño (ONI) histórico | Fases El Niño / La Niña / Neutro |
| **IRI** | Predicciones trimestrales del ONI | Proyección de riesgo próximos 3 meses |

## Estructura esperada de los datos

```
ETL_code/
├── data/
│   ├── processed/
│   │   ├── chirps_municipal/
│   │   │   └── year=2004/fact_chirps_muni_2004.parquet
│   │   ├── noaa_historical/
│   │   │   └── year=1950/noaa_oni_1950.parquet
│   │   └── noaa_prediction/
│   │       └── year=2024/noaa_oni_pred_2024.parquet
└── dashboard/
    ├── app.py               ← este archivo
    ├── requirements.txt
    └── divipola.parquet     ← geometría municipal (geometry, id_mun, name_mun, id_dept)
```

## Instalación

```bash
# Desde la carpeta del dashboard
pip install -r requirements.txt
```

## Ejecución

```bash
# Desde la carpeta ETL_code/ o ajustar BASE_PATH en app.py
streamlit run dashboard/app.py
```

El dashboard abre automáticamente en http://localhost:8501

---

## Funcionalidades del dashboard

### Tab 1 — Anomalías de precipitación
- Serie temporal mensual de anomalía vs. promedio histórico (1981–2020)
- ONI superpuesto con bandas de color por fase (El Niño / La Niña / Neutro)
- Gráfico de correlación: anomalía × score de riesgo por fase ONI
- Distribución de fases ONI en el período seleccionado
- Heatmap de anomalía por departamento × tiempo

### Tab 2 — Semáforo de riesgo
- Mapa coroplético a nivel departamento o municipio
- Ranking visual por nivel: Bajo / Medio / Alto / Crítico
- Gráfico de barras horizontales con score por departamento
- Tabla exportable con todos los indicadores

### Tab 3 — Proyección 3 meses
- Cards con las últimas 3 predicciones ONI del IRI
- Score de riesgo proyectado por departamento
- Gráfico comparativo: riesgo actual vs. proyectado
- Supuestos del modelo explicados

---

## Fórmula del Score de Exposición Climática (0-100)

```
z_score        = (precip - media_historica) / std_historica   [por municipio-mes]
componente_z   = |z_score| × 25  [máx 60]
componente_ONI = 25 si El Niño y z < -0.5  (sequía + El Niño)
               = 20 si La Niña y z >  0.5  (exceso + La Niña)
               =  8 si fase activa sin anomalía alineada
               =  0 si fase Neutra

score = min(100, componente_z + componente_ONI)
```

Niveles:
- **Bajo** (0–30): exposición normal
- **Medio** (31–55): anomalía moderada o fase ONI activa
- **Alto** (56–75): anomalía significativa alineada con ONI
- **Crítico** (76–100): anomalía extrema potenciada por ONI
