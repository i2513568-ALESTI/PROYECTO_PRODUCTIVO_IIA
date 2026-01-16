import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Ventas - Bike Gear",
    page_icon="🚴",
    layout="wide"
)

# Título principal
st.title("🚴 Sistema de Predicción de Ventas - Bike Gear")
st.markdown("---")

# Función para cargar modelos
@st.cache_resource
def load_models():
    """Carga los modelos y el scaler desde archivos .pkl"""
    try:
        modelo_demanda = joblib.load('modelo_demanda_final.pkl')
        modelo_ingresos = joblib.load('modelo_ingresos_final.pkl')
        scaler = joblib.load('scaler.pkl')
        return modelo_demanda, modelo_ingresos, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Error: No se encontró el archivo {e.filename}")
        st.info("💡 Asegúrate de que los archivos .pkl estén en el mismo directorio que app.py")
        return None, None, None

# Cargar modelos
modelo_demanda, modelo_ingresos, scaler = load_models()

# Verificar si los modelos se cargaron correctamente
if modelo_demanda is None or modelo_ingresos is None or scaler is None:
    st.stop()

# Cargar datos para obtener valores únicos de categorías
@st.cache_data
def load_data():
    """Carga el dataset para obtener valores únicos"""
    try:
        df = pd.read_csv('datos_simulados_bike_gear_realista.csv', sep=';', decimal=',')
        return df
    except:
        return None

# Crear label encoders (cacheados)
@st.cache_resource
def create_encoders(df):
    """Crea los label encoders basados en los datos"""
    if df is None:
        return None, None
    
    le_categoria = LabelEncoder()
    le_categoria.fit(df['Categoria'].astype(str))
    
    le_clima = LabelEncoder()
    le_clima.fit(df['Clima'].astype(str))
    
    return le_categoria, le_clima

df = load_data()
le_categoria, le_clima = create_encoders(df)

# Sidebar para entrada de datos
st.sidebar.header("📊 Parámetros de Entrada")

# Valores únicos para los selectboxes
if df is not None:
    categorias = sorted(df['Categoria'].unique().tolist())
    climas = sorted(df['Clima'].unique().tolist())
    precio_base_min = float(df['Precio_Base'].min())
    precio_base_max = float(df['Precio_Base'].max())
    precio_base_promedio = float(df['Precio_Base'].mean())
    
    precio_comp_min = float(df['Precio_Competencia'].min())
    precio_comp_max = float(df['Precio_Competencia'].max())
    precio_comp_promedio = float(df['Precio_Competencia'].mean())
    
    stock_min = int(df['Stock_Previo_Dia'].min())
    stock_max = int(df['Stock_Previo_Dia'].max())
    stock_promedio = int(df['Stock_Previo_Dia'].mean())
else:
    # Valores por defecto si no se puede cargar el CSV
    categorias = ['Accesorios', 'Bicicletas', 'Componentes', 'Ropa']
    climas = ['Soleado', 'Nublado', 'Lluvioso']
    precio_base_min = 50.0
    precio_base_max = 3000.0
    precio_base_promedio = 500.0
    precio_comp_min = 50.0
    precio_comp_max = 3000.0
    precio_comp_promedio = 500.0
    stock_min = 0
    stock_max = 200
    stock_promedio = 50

# Inputs del usuario
categoria = st.sidebar.selectbox("Categoría del Producto", categorias)
clima = st.sidebar.selectbox("Condición Climática", climas)
precio_base = st.sidebar.number_input(
    "Precio Base (S/.)",
    min_value=precio_base_min,
    max_value=precio_base_max,
    value=precio_base_promedio,
    step=10.0
)
descuento = st.sidebar.slider(
    "Descuento Aplicado (%)",
    min_value=0,
    max_value=50,
    value=10,
    step=5
)
precio_competencia = st.sidebar.number_input(
    "Precio de Competencia (S/.)",
    min_value=precio_comp_min,
    max_value=precio_comp_max,
    value=precio_comp_promedio,
    step=10.0
)
stock_disponible = st.sidebar.number_input(
    "Stock Disponible",
    min_value=stock_min,
    max_value=stock_max,
    value=stock_promedio,
    step=1
)
evento_deportivo = st.sidebar.checkbox("¿Hay Evento Deportivo?", value=False)

# Botón de predicción
if st.sidebar.button("🔮 Realizar Predicción", type="primary"):
    
    # Preparar datos para predicción
    # Codificar las variables categóricas usando los encoders cacheados
    if le_categoria is not None and le_clima is not None:
        try:
            categoria_encoded = le_categoria.transform([categoria])[0]
            clima_encoded = le_clima.transform([clima])[0]
        except ValueError:
            # Si la categoría no está en el encoder, usar índice como fallback
            st.warning("⚠️ Categoría o clima no encontrado en el dataset. Usando valores por defecto.")
            categoria_encoded = categorias.index(categoria) if categoria in categorias else 0
            clima_encoded = climas.index(clima) if clima in climas else 0
    else:
        # Valores por defecto si no hay encoders
        categoria_encoded = categorias.index(categoria) if categoria in categorias else 0
        clima_encoded = climas.index(clima) if clima in climas else 0
    
    # Convertir descuento de porcentaje a decimal
    descuento_decimal = descuento / 100.0
    
    # Convertir evento deportivo a int
    evento_int = 1 if evento_deportivo else 0
    
    # Crear DataFrame con las features (en el mismo orden que se entrenó)
    # Según el notebook, las features son:
    # ['Precio_Base', 'Descuento_Aplicado', 'Stock_Previo_Dia', 'Precio_Competencia', 
    #  'Categoria_encoded', 'Clima_encoded', 'Evento_Deportivo_int']
    
    # Nota: El notebook menciona 'Canal_encoded' pero luego solo usa 7 features
    # Vamos a usar las 7 que se mencionan en el output
    
    datos_entrada = pd.DataFrame({
        'Precio_Base': [precio_base],
        'Descuento_Aplicado': [descuento_decimal],
        'Stock_Previo_Dia': [stock_disponible],
        'Precio_Competencia': [precio_competencia],
        'Categoria_encoded': [categoria_encoded],
        'Clima_encoded': [clima_encoded],
        'Evento_Deportivo_int': [evento_int]
    })
    
    # Escalar los datos
    datos_escalados = scaler.transform(datos_entrada)
    
    # Realizar predicciones
    prediccion_cantidad = modelo_demanda.predict(datos_escalados)[0]
    prediccion_ingresos = modelo_ingresos.predict(datos_escalados)[0]
    
    # Asegurar que la cantidad no sea negativa
    prediccion_cantidad = max(0, prediccion_cantidad)
    
    # Calcular precio final unitario
    precio_final_unitario = precio_base * (1 - descuento_decimal)
    
    # Calcular ingresos reales (considerando stock disponible)
    cantidad_vendida_real = min(prediccion_cantidad, stock_disponible)
    ingresos_reales = cantidad_vendida_real * precio_final_unitario
    
    # Calcular ventas perdidas por falta de stock
    ventas_perdidas = max(0, prediccion_cantidad - stock_disponible)
    dinero_perdido = ventas_perdidas * precio_final_unitario
    
    # Mostrar resultados
    st.success("✅ Predicción completada exitosamente!")
    
    # Crear columnas para mostrar resultados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📦 Cantidad Predicha",
            value=f"{prediccion_cantidad:.1f} unidades",
            delta=f"{prediccion_cantidad - stock_disponible:.1f} vs stock"
        )
    
    with col2:
        st.metric(
            label="💰 Ingresos Predichos",
            value=f"S/. {prediccion_ingresos:,.2f}",
            delta=f"S/. {prediccion_ingresos - ingresos_reales:,.2f}"
        )
    
    with col3:
        st.metric(
            label="💵 Ingresos Reales (con stock)",
            value=f"S/. {ingresos_reales:,.2f}",
            delta=f"{cantidad_vendida_real:.0f} unidades vendidas"
        )
    
    # Información adicional
    st.markdown("---")
    st.subheader("📊 Detalles de la Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Parámetros de Entrada")
        st.write(f"- **Categoría:** {categoria}")
        st.write(f"- **Clima:** {clima}")
        st.write(f"- **Precio Base:** S/. {precio_base:,.2f}")
        st.write(f"- **Descuento:** {descuento}%")
        st.write(f"- **Precio Final:** S/. {precio_final_unitario:,.2f}")
        st.write(f"- **Precio Competencia:** S/. {precio_competencia:,.2f}")
        st.write(f"- **Stock Disponible:** {stock_disponible} unidades")
        st.write(f"- **Evento Deportivo:** {'Sí' if evento_deportivo else 'No'}")
    
    with col2:
        st.markdown("### Resultados de la Predicción")
        st.write(f"- **Demanda Potencial:** {prediccion_cantidad:.1f} unidades")
        st.write(f"- **Cantidad Vendible (con stock):** {cantidad_vendida_real:.0f} unidades")
        st.write(f"- **Ingresos Totales Predichos:** S/. {prediccion_ingresos:,.2f}")
        st.write(f"- **Ingresos Reales (limitado por stock):** S/. {ingresos_reales:,.2f}")
        
        if ventas_perdidas > 0:
            st.warning(f"⚠️ **Ventas Perdidas:** {ventas_perdidas:.1f} unidades (S/. {dinero_perdido:,.2f})")
        else:
            st.success("✅ Stock suficiente para cubrir la demanda")
    
    # Alerta de stock
    if prediccion_cantidad > stock_disponible:
        st.error(f"🔴 **ALERTA:** La demanda predicha ({prediccion_cantidad:.1f} unidades) excede el stock disponible ({stock_disponible} unidades). Se recomienda aumentar el inventario.")
    else:
        st.info(f"🟢 **Stock Adecuado:** El inventario actual es suficiente para cubrir la demanda predicha.")

# Información sobre los modelos
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.info(
    """
    **Modelos Utilizados:**
    - 🎯 **Demanda:** Random Forest Regressor
    - 💰 **Ingresos:** Gradient Boosting Regressor
    
    **Precisión:**
    - R² Demanda: ~0.89
    - R² Ingresos: ~0.97
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Proyecto Productivo IDL3 - Sistema de Predicción de Ventas</p>
    </div>
    """,
    unsafe_allow_html=True
)
