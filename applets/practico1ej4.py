import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Configuración de la página
st.set_page_config(page_title="Applet de Regresión Lineal", layout="wide")

st.title("📈 Applet Interactivo: Regresión Lineal")
st.markdown("Esta herramienta permite explorar la relación entre la complejidad de los datos, el ruido estadístico y la capacidad de generalización de un modelo de regresión lineal simple.")

# --- BARRA LATERAL (CONTROLES INTERACTIVOS) ---
st.sidebar.header("⚙️ (a) Generación de Datos")
N = st.sidebar.slider("Número de puntos (N)", min_value=10, max_value=10000, value=150, step=10)

st.sidebar.subheader("Función Base: y = mx + b")
m = st.sidebar.slider("Pendiente (m)", min_value=-10.0, max_value=10.0, value=3.0, step=0.1)
b = st.sidebar.slider("Intersección (b)", min_value=-20.0, max_value=20.0, value=5.0, step=0.5)

x_min = st.sidebar.number_input("x_min", value=0.0)
x_max = st.sidebar.number_input("x_max", value=50.0)

st.sidebar.header("🌫️ (b) Modelado del Ruido")
sigma = st.sidebar.slider("Desviación estándar del ruido (σ)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
# Semilla para observar la inestabilidad en la pregunta (b)
seed = st.sidebar.number_input("Semilla del Ruido (Seed)", value=42, step=1)

st.sidebar.header("✂️ (c) Partición de Datos")
test_size = st.sidebar.slider("Proporción de Prueba (Test %)", min_value=0.1, max_value=0.9, value=0.3, step=0.05)

# --- LÓGICA DE DATOS Y MODELADO ---
# 1. Generar los datos
np.random.seed(seed)
# Generamos X aleatorio uniforme para que la partición train/test esté bien distribuida en el dominio
X = np.random.uniform(x_min, x_max, N).reshape(-1, 1)
# Ruido normal ε ~ N(0, σ²)
epsilon = np.random.normal(0, sigma, N).reshape(-1, 1)
# Función verdadera más ruido
y_true = m * X + b
y = y_true + epsilon

# 2. Dividir los datos en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# 3. Entrenar el modelo (Ajuste)
model = LinearRegression()
model.fit(X_train, y_train)

m_hat = model.coef_[0][0]
b_hat = model.intercept_[0]

# 4. Predicciones
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 5. Cálculo de Métricas (MSE)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# --- (d) VISUALIZACIÓN Y MÉTRICAS ---
col1, col2 = st.columns([3, 1])

with col1:
    # Crear gráfico interactivo con Plotly
    fig = go.Figure()

    # Puntos de Entrenamiento
    fig.add_trace(go.Scatter(x=X_train.flatten(), y=y_train.flatten(), mode='markers',
                             name='Entrenamiento (Train)', marker=dict(color='#1f77b4', size=8, opacity=0.7)))
    
    # Puntos de Prueba
    fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test.flatten(), mode='markers',
                             name='Prueba (Test)', marker=dict(color='#ff7f0e', size=8, opacity=0.8)))

    # Línea generadora verdadera (sin ruido)
    x_line = np.array([x_min, x_max]).reshape(-1, 1)
    y_line_true = m * x_line + b
    fig.add_trace(go.Scatter(x=x_line.flatten(), y=y_line_true.flatten(), mode='lines',
                             name='Recta Verdadera', line=dict(color='gray', width=2, dash='dash')))

    # Línea del modelo ajustado (predicción)
    y_line_pred = model.predict(x_line)
    fig.add_trace(go.Scatter(x=x_line.flatten(), y=y_line_pred.flatten(), mode='lines',
                             name='Recta de Ajuste (Modelo)', line=dict(color='red', width=3)))

    fig.update_layout(title="Scatter Plot y Rectas de Ajuste", xaxis_title="X", yaxis_title="Y", height=500)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Métricas de Error")
    st.info(f"**MSE Entrenamiento:**\n\n{mse_train:.2f}")
    st.warning(f"**MSE Prueba:**\n\n{mse_test:.2f}")
    
    st.markdown("---")
    st.subheader("Parámetros Ajustados")
    st.write(f"**Pendiente estimada ($\\hat{{m}}$):** {m_hat:.3f}")
    st.write(f"**Intersección estimada ($\\hat{{b}}$):** {b_hat:.3f}")
    st.write(f"*(Pendiente real: {m}, Intersección real: {b})*")

# --- RESPUESTAS A LAS PREGUNTAS FUNDAMENTADAS ---
st.markdown("---")
st.header("📝 Respuestas Fundamentadas")

with st.expander("a) Al fijar σ y aumentar N, ¿cómo evoluciona la brecha entre el MSE_train y el MSE_test?"):
    st.markdown("""
    **Respuesta:** Al fijar el ruido ($\\sigma$) y aumentar progresivamente la cantidad de datos ($N$), **la brecha entre el $MSE_{train}$ y el $MSE_{test}$ tiende a cerrarse** (disminuye). 
    
    * **¿Por qué sucede esto?** Cuando $N$ es pequeño, el modelo puede ajustarse a particularidades aleatorias del ruido en el conjunto de entrenamiento (memorización parcial), lo que resulta en un $MSE_{train}$ artificialmente bajo y un $MSE_{test}$ alto. Al aumentar $N$, el modelo ya no puede "memorizar" el ruido porque este se cancela estadísticamente. 
    * Ambos errores convergerán asintóticamente hacia un mismo valor: **la varianza del ruido irreducible ($\\sigma^2$)**.
    """)

with st.expander("b) Con un N pequeño, ¿qué sucede con la estabilidad de la pendiente calculada al variar la semilla del ruido?"):
    st.markdown("""
    **Respuesta:** Con un $N$ pequeño (por ejemplo $N=10$), **la estabilidad de la pendiente calculada ($\\hat{{m}}$) es muy baja**. Esto significa que al variar la semilla del ruido en la barra lateral, observarás que la recta de ajuste roja salta y cambia drásticamente de inclinación en cada iteración.
    
    * **Fundamento:** Con pocos puntos, los valores atípicos generados por la distribución de ruido tienen un peso enorme en la función de costo de los Mínimos Cuadrados Ordinarios. Esto se traduce en una **alta varianza** en las estimaciones de los parámetros ($\\hat{{m}}$ y $\\hat{{b}}$). El modelo es muy sensible a la muestra específica con la que fue entrenado.
    """)

with st.expander("c) ¿Es posible observar overfitting sobre datos puramente lineales con un modelo lineal? (Basado en Sesgo-Varianza)"):
    st.markdown("""
    **Respuesta:** **Estrictamente hablando en términos de complejidad de modelo, NO.** El *overfitting* (sobreajuste) ocurre cuando utilizamos un modelo con excesiva complejidad (altos grados de libertad, bajo sesgo pero alta varianza, como un polinomio de grado 10) para ajustar datos simples, terminando por modelar el ruido. Aquí, la naturaleza de los datos es lineal y el modelo es lineal: **el sesgo del modelo es cero** (tiene la capacidad perfecta de representar el fenómeno subyacente).
    
    **Sin embargo, sí se puede observar una *falsa apariencia* de overfitting (Brecha de Generalización)** cuando $N$ es muy pequeño respecto a la intensidad del ruido ($\\sigma$). 
    * En un escenario de $N$ muy pequeño y ruido alto, el modelo lineal intentará acomodarse a esos pocos puntos ruidosos. Esto generará un $MSE_{train}$ menor que el $MSE_{test}$. 
    * En el marco de **Sesgo-Varianza**, este error en prueba no viene dado por intentar usar una función demasiado flexible (no podemos usar algo más simple que una recta), sino que proviene puramente de la **alta Varianza de los parámetros estimada** debido al escaso tamaño muestral. Se sufre de falta de datos, no de exceso de flexibilidad del algoritmo.
    """)
