"""
Applet interactivo del Perceptrón en 2D.

Ejecutar:
    streamlit run applets/perceptron.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def safe_range(lo: float, hi: float, padding: float = 0.5) -> tuple[float, float]:
    """Garantiza lo < hi para sliders de Streamlit."""
    lo, hi = float(lo), float(hi)
    if lo >= hi:
        hi = lo + padding
    return lo, hi


def init_session_state() -> None:
    defaults = {
        "points": [],
        "history": [],
        "step_info": [],
        "trained": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def points_to_arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.empty((0, 2)), np.empty(0)
    x = np.array([[p["x1"], p["x2"]] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    return x, y


def train_perceptron(
    x: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    eta: float,
    max_updates: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Entrena el perceptrón online y guarda un snapshot de pesos tras cada actualización.

    Regla: w <- w + eta * y * [1, x1, x2] si y * (w · x_aug) <= 0
    """
    w = w_init.astype(float).copy()
    history: list[np.ndarray] = [w.copy()]
    step_info: list[dict] = [{"epoch": 0, "point_idx": None, "misclassified": False}]

    if len(x) == 0:
        return history, step_info

    n_updates = 0
    epoch = 0
    while n_updates < max_updates:
        epoch += 1
        any_update = False
        for i in range(len(x)):
            x_aug = np.array([1.0, x[i, 0], x[i, 1]])
            if y[i] * (w @ x_aug) <= 0:
                w = w + eta * y[i] * x_aug
                history.append(w.copy())
                step_info.append(
                    {
                        "epoch": epoch,
                        "point_idx": i,
                        "misclassified": True,
                        "x1": float(x[i, 0]),
                        "x2": float(x[i, 1]),
                        "y": float(y[i]),
                    }
                )
                n_updates += 1
                any_update = True
                if n_updates >= max_updates:
                    break
        if not any_update:
            break

    return history, step_info


def boundary_line(
    w: np.ndarray,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Recta de decisión w0 + w1*x1 + w2*x2 = 0."""
    w0, w1, w2 = w
    x1_min, x1_max = x_lim
    x2_min, x2_max = y_lim

    if abs(w2) < 1e-10:
        if abs(w1) < 1e-10:
            return None
        x_vert = -w0 / w1
        return np.array([x_vert, x_vert]), np.array([x2_min, x2_max])

    x1 = np.linspace(x1_min, x1_max, 200)
    x2 = -(w0 + w1 * x1) / w2
    return x1, x2


def prediction(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    return np.sign(x_aug @ w)


def make_figure(
    points: list[dict],
    w: np.ndarray,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    highlight_idx: int | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Regiones de clasificación (fondo suave)
    grid_n = 40
    gx = np.linspace(x_lim[0], x_lim[1], grid_n)
    gy = np.linspace(y_lim[0], y_lim[1], grid_n)
    xx, yy = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    scores = prediction(w, grid_pts).reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            x=gx,
            y=gy,
            z=scores,
            colorscale=[[0, "rgba(214, 39, 40, 0.12)"], [1, "rgba(31, 119, 180, 0.12)"]],
            showscale=False,
            contours=dict(start=-1, end=1, size=2, coloring="fill"),
            line=dict(width=0),
            hoverinfo="skip",
            name="regiones",
        )
    )

    pos = [p for p in points if p["y"] > 0]
    neg = [p for p in points if p["y"] < 0]

    if pos:
        fig.add_trace(
            go.Scatter(
                x=[p["x1"] for p in pos],
                y=[p["x2"] for p in pos],
                mode="markers",
                name="Clase +1",
                marker=dict(color="#1f77b4", size=12, line=dict(width=1, color="white")),
            )
        )
    if neg:
        fig.add_trace(
            go.Scatter(
                x=[p["x1"] for p in neg],
                y=[p["x2"] for p in neg],
                mode="markers",
                name="Clase −1",
                marker=dict(color="#d62728", size=12, line=dict(width=1, color="white")),
            )
        )

    if highlight_idx is not None and 0 <= highlight_idx < len(points):
        hp = points[highlight_idx]
        fig.add_trace(
            go.Scatter(
                x=[hp["x1"]],
                y=[hp["x2"]],
                mode="markers",
                name="Punto actualizado",
                marker=dict(
                    color="#ff7f0e",
                    size=18,
                    symbol="circle-open",
                    line=dict(width=3, color="#ff7f0e"),
                ),
                showlegend=True,
            )
        )

    line = boundary_line(w, x_lim, y_lim)
    if line is not None:
        x1_line, x2_line = line
        fig.add_trace(
            go.Scatter(
                x=x1_line,
                y=x2_line,
                mode="lines",
                name="Frontera de decisión",
                line=dict(color="#2ca02c", width=3),
            )
        )

    # Vector normal (dirección de w en el plano x1-x2)
    _, w1, w2 = w
    norm = np.hypot(w1, w2)
    if norm > 1e-8:
        scale = 0.35 * min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])
        fig.add_trace(
            go.Scatter(
                x=[0, w1 / norm * scale],
                y=[0, w2 / norm * scale],
                mode="lines+markers",
                name="Vector w (proyección)",
                line=dict(color="#9467bd", width=2, dash="dot"),
                marker=dict(size=6, color="#9467bd"),
            )
        )

    fig.update_layout(
        title="Plano 2D: puntos y frontera de decisión",
        xaxis_title="x₁",
        yaxis_title="x₂",
        xaxis=dict(range=list(x_lim), constrain="domain"),
        yaxis=dict(range=list(y_lim), scaleanchor="x", scaleratio=1),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="closest",
    )
    return fig


def retrain() -> None:
    x, y = points_to_arrays(st.session_state.points)
    w_init = np.array(
        [
            st.session_state.get("w0_init", 0.0),
            st.session_state.get("w1_init", 0.0),
            st.session_state.get("w2_init", 1.0),
        ],
        dtype=float,
    )
    eta = float(st.session_state.get("eta", 1.0))
    max_updates = int(st.session_state.get("max_updates", 200))

    history, step_info = train_perceptron(x, y, w_init, eta, max_updates)
    st.session_state.history = history
    st.session_state.step_info = step_info
    st.session_state.trained = True
    st.session_state.step_idx = 0


init_session_state()

st.set_page_config(page_title="Perceptrón 2D", layout="wide")
st.title("Perceptrón en 2D")
st.markdown(
    "Agregá puntos etiquetados en el plano, entrená el perceptrón paso a paso y observá "
    "cómo se mueve la **frontera de decisión** "
    r"($w_0 + w_1 x_1 + w_2 x_2 = 0$) tras cada actualización "
    r"$\mathbf{w} \leftarrow \mathbf{w} + \eta\, y\, [1, x_1, x_2]$ cuando un punto está mal clasificado."
)

# --- Sidebar: datos y hiperparámetros ---
st.sidebar.header("Datos")
x_plot_min = st.sidebar.number_input("x₁ mín", value=-5.0, step=0.5, key="x_min")
x_plot_max = st.sidebar.number_input("x₁ máx", value=5.0, step=0.5, key="x_max")
y_plot_min = st.sidebar.number_input("x₂ mín", value=-5.0, step=0.5, key="y_min")
y_plot_max = st.sidebar.number_input("x₂ máx", value=5.0, step=0.5, key="y_max")

st.sidebar.subheader("Agregar punto")
x1_lo, x1_hi = safe_range(x_plot_min, x_plot_max)
x2_lo, x2_hi = safe_range(y_plot_min, y_plot_max)
new_x1 = st.sidebar.slider("x₁", x1_lo, x1_hi, min(max(x1_lo, 0.0), x1_hi), step=0.1)
new_x2 = st.sidebar.slider("x₂", x2_lo, x2_hi, min(max(x2_lo, 0.0), x2_hi), step=0.1)
new_label = st.sidebar.radio(
    "Clase",
    options=[1, -1],
    format_func=lambda v: "+1" if v == 1 else "−1",
    horizontal=True,
)

if st.sidebar.button("Agregar punto", use_container_width=True):
    st.session_state.points.append({"x1": new_x1, "x2": new_x2, "y": float(new_label)})
    retrain()

st.sidebar.subheader("Ejemplos rápidos")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Separables", use_container_width=True):
        st.session_state.points = [
            {"x1": -2.0, "x2": -1.0, "y": -1.0},
            {"x1": -1.5, "x2": 2.0, "y": -1.0},
            {"x1": 1.0, "x2": -2.0, "y": 1.0},
            {"x1": 2.5, "x2": 1.5, "y": 1.0},
            {"x1": 0.0, "x2": 0.5, "y": 1.0},
        ]
        retrain()
with col_b:
    if st.button("XOR", use_container_width=True):
        st.session_state.points = [
            {"x1": -1.0, "x2": -1.0, "y": 1.0},
            {"x1": 1.0, "x2": 1.0, "y": 1.0},
            {"x1": -1.0, "x2": 1.0, "y": -1.0},
            {"x1": 1.0, "x2": -1.0, "y": -1.0},
        ]
        retrain()

if st.sidebar.button("Borrar todos los puntos", use_container_width=True):
    st.session_state.points = []
    retrain()

st.sidebar.header("Entrenamiento")
st.session_state.eta = st.sidebar.slider("η (tasa de aprendizaje)", 0.01, 2.0, 1.0, 0.01)
st.session_state.max_updates = st.sidebar.slider("Máx. actualizaciones", 10, 500, 200, 10)
st.sidebar.caption("Pesos iniciales w = [w₀, w₁, w₂]")
st.session_state.w0_init = st.sidebar.number_input("w₀", value=0.0, step=0.1)
st.session_state.w1_init = st.sidebar.number_input("w₁", value=0.0, step=0.1)
st.session_state.w2_init = st.sidebar.number_input("w₂", value=1.0, step=0.1)

if st.sidebar.button("Reentrenar desde cero", use_container_width=True):
    retrain()

# --- Tabla de puntos ---
if st.session_state.points:
    pts_df = pd.DataFrame(st.session_state.points)
    pts_df.index.name = "#"
    st.dataframe(pts_df, use_container_width=True, hide_index=False)

    del_cols = st.columns(len(st.session_state.points))
    for i, col in enumerate(del_cols):
        with col:
            if st.button(f"Quitar {i}", key=f"del_{i}"):
                st.session_state.points.pop(i)
                retrain()
                st.rerun()

# --- Controles de paso ---
history = st.session_state.history
if not history:
    history = [np.array([0.0, 0.0, 1.0])]
    st.session_state.history = history

if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0

max_step = len(history) - 1
st.session_state.step_idx = min(st.session_state.step_idx, max_step)

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])
with ctrl1:
    if st.button("⏮ Inicio"):
        st.session_state.step_idx = 0
with ctrl2:
    if st.button("◀ Paso anterior"):
        st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
with ctrl3:
    if st.button("Paso siguiente ▶"):
        st.session_state.step_idx = min(max_step, st.session_state.step_idx + 1)
with ctrl4:
    if max_step > 0:
        st.session_state.step_idx = st.slider(
            "Paso del algoritmo",
            0,
            max_step,
            st.session_state.step_idx,
        )
    else:
        st.caption("Paso 0 — agregá puntos y entrená para ver actualizaciones.")
        st.session_state.step_idx = 0

w_current = history[st.session_state.step_idx]
step_meta = (
    st.session_state.step_info[st.session_state.step_idx]
    if st.session_state.step_idx < len(st.session_state.step_info)
    else {}
)
highlight = step_meta.get("point_idx")

x_lim = (float(x_plot_min), float(x_plot_max))
y_lim = (float(y_plot_min), float(y_plot_max))
fig = make_figure(st.session_state.points, w_current, x_lim, y_lim, highlight_idx=highlight)
st.plotly_chart(fig, use_container_width=True)

# --- Panel informativo ---
info_l, info_r = st.columns(2)
with info_l:
    st.subheader("Pesos actuales")
    st.latex(
        rf"w_0 = {w_current[0]:.4f},\quad "
        rf"w_1 = {w_current[1]:.4f},\quad "
        rf"w_2 = {w_current[2]:.4f}"
    )
    st.markdown(
        f"**Frontera:** `{w_current[0]:.3f} + {w_current[1]:.3f}·x₁ + {w_current[2]:.3f}·x₂ = 0`"
    )
    if st.session_state.step_idx == 0:
        st.info("Paso 0: pesos iniciales (antes de actualizar).")
    elif step_meta.get("misclassified"):
        st.warning(
            f"Actualización #{st.session_state.step_idx}: "
            f"punto #{step_meta['point_idx']} mal clasificado "
            f"({step_meta['x1']:.2f}, {step_meta['x2']:.2f}), y = {step_meta['y']:+.0f}. "
            f"Época {step_meta['epoch']}."
        )
    else:
        st.success("Sin más errores en la última pasada: convergencia alcanzada.")

with info_r:
    st.subheader("Estado")
    st.write(f"**Puntos:** {len(st.session_state.points)}")
    st.write(f"**Actualizaciones registradas:** {max_step}")
    if st.session_state.points:
        x_arr, y_arr = points_to_arrays(st.session_state.points)
        preds = prediction(w_current, x_arr)
        n_err = int(np.sum(preds != y_arr))
        st.write(f"**Errores con pesos actuales:** {n_err} / {len(y_arr)}")
        if n_err == 0 and st.session_state.step_idx > 0:
            st.success("Todos los puntos clasificados correctamente.")

st.markdown("---")
with st.expander("Cómo leer el gráfico"):
    st.markdown(
        """
- **Azul / rojo:** clases +1 y −1.
- **Verde:** frontera donde la salida del perceptrón cambia de signo.
- **Fondo tenue:** región que el modelo asigna a cada clase con los pesos actuales.
- **Naranja:** punto que provocó la última actualización de pesos.
- **Púrpura punteado:** dirección del vector de pesos (w₁, w₂); la frontera es perpendicular a él.

Probá el preset **XOR**: los datos no son linealmente separables, el algoritmo seguirá
actualizando pesos hasta alcanzar el máximo de pasos sin converger.
        """
    )
