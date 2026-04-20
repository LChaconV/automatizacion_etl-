# ETL automatizado datos meteorológicos

## Descripción del Proyecto
Este proyecto es un sistema de ETL para la obtención de datos de precipitacion de CHIRPS e indice del ONI

## Requisitos y Configuración Inicial
Para asegurar la correcta ejecución del sistema, asegúrese de tener configurados los siguientes archivos.

### 1. Restablecer Fechas de Procesamiento
Antes de la primera ejecución, es necesario establecer las fechas de inicio en el archivo de estado. Edite el archivo `data/processed/status/last_processed_dates.yaml`
El contenido del archivo debe ser similar a este:

```yaml
chirps: '2015-01-01'
noaa_oni_historical: '2015-01-01'
noaa_oni_prediction: '2015-01-01'
```

### 2. Archivos Geoespaciales Obligatorios
Verifique que los siguientes archivos GeoJSON existan y contengan polígonos con el Sistema de Referencia de Coordenadas EPSG:4326 (WGS 84):

- **Polígono de Colombia**:
` data/processed/maps/colombia_polygon/colombia_WGS84.geojson`
- **Polígonos de Municipios**:
` data/processed/maps/municipality_polygons_WGS84.geojson`
Descarga aquí: https://drive.google.com/drive/folders/1dBW9wQDtdjDa77XRFxh91TAyLXRg7ZEW?usp=sharing
## Uso y Ejecución del sistema

La ejecución del sistema se lleva a cabo en en Windows desde el Programador de tareas (Task Scheduler)
- Presiona Win + R
- Escribe: taskschd.msc
- En el panel derecho: "Crear tarea básica..."
- Asigna un nombre, por ejemplo: ETL meteorologico Mensual
- Define la frecuencia de ejecución
- Selecciona: "Iniciar un programa"
- Configurar:
  -  Programa o script:cmd.exe
  -  Agregar argumentos: por ejemplo /c "C:\Users\laura\TESIS\ETL_LauraChacon\ETL_code\run_etl.bat"
- Guardas la tarea
