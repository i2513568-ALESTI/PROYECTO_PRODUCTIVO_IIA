# 🚴 Sistema de Predicción de Ventas - Bike Gear

Aplicación Streamlit para predecir demanda e ingresos utilizando modelos de Machine Learning entrenados.

## 📋 Requisitos Previos

Asegúrate de tener los siguientes archivos en el mismo directorio que `app.py`:

- `modelo_demanda_final.pkl` - Modelo Random Forest para predecir cantidad vendida
- `modelo_ingresos_final.pkl` - Modelo Gradient Boosting para predecir ingresos
- `scaler.pkl` - Escalador StandardScaler usado en el entrenamiento
- `datos_simulados_bike_gear_realista.csv` - Dataset (opcional, para valores por defecto)

## 🚀 Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 💻 Ejecución

Para ejecutar la aplicación Streamlit:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📊 Uso

1. **Configura los parámetros** en la barra lateral:
   - Categoría del producto
   - Condición climática
   - Precio base
   - Descuento aplicado (%)
   - Precio de competencia
   - Stock disponible
   - Evento deportivo (checkbox)

2. **Haz clic en "Realizar Predicción"** para obtener:
   - Cantidad predicha de unidades
   - Ingresos totales predichos
   - Ingresos reales considerando el stock disponible
   - Alertas si el stock es insuficiente

## 🎯 Modelos

- **Modelo de Demanda**: Random Forest Regressor (R² ≈ 0.89)
- **Modelo de Ingresos**: Gradient Boosting Regressor (R² ≈ 0.97)

## ⚠️ Notas Importantes

- Los modelos fueron entrenados con datos específicos. Asegúrate de que los valores de entrada estén dentro de los rangos del dataset de entrenamiento.
- Si no tienes los archivos `.pkl`, necesitas ejecutar el notebook `Mejora_Proyecto_Productivo_IDL3.ipynb` primero para generarlos.
