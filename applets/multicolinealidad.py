import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_dataset(n, k, x_min, x_max, noise_std, seed, cat_strength=1.0):
    """
    Dataset sintético para demostrar multicolinealidad por One-Hot Encoding.

    Estructura:
    - x: numérica
    - cat: categórica con K niveles
    - y: clase binaria muestreada desde una probabilidad logística (sigmoid)
    """

    rng = np.random.default_rng(seed)

    cat_labels = [f"C{i}" for i in range(k)]
    # Mantener categorías uniformes para que todas tengan presencia.
    cat_idx = rng.integers(0, k, size=n)
    cat = np.array([cat_labels[i] for i in cat_idx], dtype=object)

    x = rng.uniform(x_min, x_max, size=n)

    # Efectos "verdaderos" lineales en el logit.
    beta0 = rng.normal(0.0, 1.0)
    beta_x = rng.normal(0.0, 1.0)
    beta_cat = rng.normal(0.0, 1.0, size=k) * cat_strength

    eps = rng.normal(0.0, 1.0, size=n)
    z = beta0 + beta_x * x + beta_cat[cat_idx] + noise_std * eps
    p = sigmoid(z)
    y = rng.binomial(1, p)

    df = pd.DataFrame({"x": x, "cat": cat, "y": y})
    return df, cat_labels


def _make_ohe(drop_setting, cat_labels):
    # Compatibilidad entre versiones de scikit-learn (sparse_output vs sparse).
    try:
        return OneHotEncoder(
            categories=[cat_labels],
            drop=drop_setting,
            sparse_output=False,
            handle_unknown="ignore",
        )
    except TypeError:
        return OneHotEncoder(
            categories=[cat_labels],
            drop=drop_setting,
            sparse=False,
            handle_unknown="ignore",
        )


def compute_vif(X, col_names):
    """
    VIF aproximado usando R^2 de regresión OLS (con lstsq, robusto a singularidades).
    Para el caso con multicolinealidad exacta, suele dar `inf` (o valores enormes).
    """

    X = np.asarray(X, dtype=float)
    n, p = X.shape

    out = []
    for j in range(p):
        x_j = X[:, j]
        X_others = np.delete(X, j, axis=1)

        # Regresión de x_j sobre el resto (OLS) usando mínimos cuadrados.
        beta, *_ = np.linalg.lstsq(X_others, x_j, rcond=None)
        x_j_hat = X_others @ beta

        resid = x_j - x_j_hat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((x_j - np.mean(x_j)) ** 2))

        if ss_tot < 1e-14:
            # Columna casi constante.
            r2 = 1.0 if ss_res < 1e-14 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        denom = 1.0 - r2
        if denom <= 1e-12:
            vif = float("inf")
        else:
            vif = 1.0 / denom

        out.append({"term": col_names[j], "vif": vif})

    return pd.DataFrame(out)


def fit_linear_probability_with_ohe(df, drop_setting, cat_labels, test_size, seed_model):
    y = df["y"]
    # `stratify` puede fallar si alguna clase no aparece (posible si el noise es 0 y/o la señal es muy fuerte).
    stratify = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed_model,
        shuffle=True,
        stratify=stratify,
    )

    ohe = _make_ohe(drop_setting=drop_setting, cat_labels=cat_labels)
    ohe.fit(train_df[["cat"]])

    X_train_cat = ohe.transform(train_df[["cat"]])
    X_test_cat = ohe.transform(test_df[["cat"]])

    x_train_num = train_df[["x"]].to_numpy()
    x_test_num = test_df[["x"]].to_numpy()

    X_train = np.hstack([x_train_num, X_train_cat])
    X_test = np.hstack([x_test_num, X_test_cat])

    y_train = train_df["y"].to_numpy(dtype=float)
    y_test = test_df["y"].to_numpy(dtype=float)

    model = LogisticRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_test_bin = (y_pred_test >= 0.5).astype(int)
    acc = accuracy_score(y_test.astype(int), y_pred_test_bin)

    intercept = float(model.intercept_[0])
    coefs = model.coef_[0].reshape(-1)

    ohe_feature_names = list(ohe.get_feature_names_out(["cat"]))
    coef_names = ["x"] + ohe_feature_names

    # Diagnóstico de multicolinealidad: rango y condición usando una columna de intercept.
    X_train_design = np.column_stack([np.ones(len(X_train)), X_train])
    design_rank = int(np.linalg.matrix_rank(X_train_design))
    design_cond = float(np.linalg.cond(X_train_design))

    # VIF: típicamente se reporta para variables explicativas (sin intercept).
    X_no_intercept = X_train  # columnas: [x, dummies...]
    vif_df = compute_vif(X_no_intercept, coef_names)

    # Para comparar coeficientes, devolvemos también el nombre del término "intercept".
    coef_with_intercept = pd.Series(
        data=[intercept] + coefs.tolist(), index=["intercept"] + coef_names
    )

    return {
        "model": model,
        "drop_setting": drop_setting,
        "accuracy": float(acc),
        "intercept": intercept,
        "coef_series": coef_with_intercept,
        "coef_names": coef_names,
        "design_rank": design_rank,
        "design_cond": design_cond,
        "vif_df": vif_df,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }


def label_term(term):
    # Para que el gráfico no sea demasiado largo.
    if term == "intercept":
        return "intercept"
    if term == "x":
        return "x"
    if term.startswith("cat_"):
        return term.replace("cat_", "cat:")
    return term


st.set_page_config(page_title="Multicolinealidad con OHE", layout="wide")
st.title("Multicolinealidad en práctica (One-Hot Encoding) 🧩")

st.markdown(
    "Esta app compara dos modelos de regresión logística: "
    "`drop='first'` (menos dummies) vs `drop=None` (incluye un dummy extra). "
    "La idea es que con `drop=None` + intercept (Ordenada al Origen) aparece una **dependencia lineal exacta**, "
    "lo que vuelve los coeficientes poco interpretables o incluso numéricamente inestables. "
    "Esto se debe a que las variables categóricas son codificadas como dummies, y si alguna de ellas es una combinación lineal de las otras, se produce multicolinealidad."
    "Esto se puede detectar con el VIF (Variance Inflation Factor), que mide la correlación entre una variable y las demás."
    " Pueden ver un poco más sobre multicolinealidad en el blog [The Most Overlooked Problem With One-Hot Encoding](https://blog.dailydoseofds.com/p/the-most-overlooked-problem-with)"
)

st.sidebar.header("Controles ⚙️")
N = st.sidebar.slider("N (muestras)", min_value=50, max_value=4000, value=800, step=50)
K = st.sidebar.slider("K (categorías)", min_value=2, max_value=10, value=4, step=1)

x_min = st.sidebar.number_input("x_min", value=-2.0, step=0.5)
x_max = st.sidebar.number_input("x_max", value=2.0, step=0.5)

noise_std = st.sidebar.slider("noise_std (estocasticidad)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
cat_strength = st.sidebar.slider("cat_strength (fuerza categórica)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
st.sidebar.caption(
    "Controla qué tan fuerte es el efecto real de la variable categórica `cat` sobre la probabilidad de la clase `y` "
    "(en el generador se multiplica el vector de efectos de categorías por `cat_strength`)."
)

seed_data = st.sidebar.number_input("seed_data", value=42, step=1)
seed_model = st.sidebar.number_input("seed_model", value=123, step=1)
test_size = st.sidebar.slider("test_size", min_value=0.1, max_value=0.5, value=0.25, step=0.05)

df, cat_labels = generate_dataset(
    n=N,
    k=K,
    x_min=float(x_min),
    x_max=float(x_max),
    noise_std=float(noise_std),
    seed=int(seed_data),
    cat_strength=float(cat_strength),
)

cat_counts = df["cat"].value_counts().sort_index()

left, right = st.columns([2, 1])
with left:
    st.subheader("Dataset (vista rápida) 📊")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

with right:
    st.subheader("Frecuencia por categoría 📎")
    st.write(cat_counts.rename_axis("cat").to_frame("count"))

fig_scatter = go.Figure()
fig_scatter.add_trace(
    go.Scatter(
        x=df.loc[df["y"] == 0, "x"],
        y=df.loc[df["y"] == 0, "y"],
        mode="markers",
        name="y=0",
        marker=dict(size=7, opacity=0.65),
    )
)
fig_scatter.add_trace(
    go.Scatter(
        x=df.loc[df["y"] == 1, "x"],
        y=df.loc[df["y"] == 1, "y"],
        mode="markers",
        name="y=1",
        marker=dict(size=7, opacity=0.85),
    )
)
fig_scatter.update_layout(
    title="x vs y (y binaria), esto muestra que vamos a necesitar algo más que x para poder predecir y",
    xaxis_title="x",
    yaxis_title="y",
    height=420,
    legend=dict(orientation="h"),
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

tabs = st.tabs(["drop='first'", "drop=None (dummy extra) + comparación"])

res_first = None
res_none = None

with tabs[0]:
    res_first = fit_linear_probability_with_ohe(
        df=df,
        drop_setting="first",
        cat_labels=cat_labels,
        test_size=float(test_size),
        seed_model=int(seed_model),
    )
    st.subheader("Diagnóstico 🔍")
    diag_first_df = pd.DataFrame(
        [
            {
                "drop": "first",
                "train_size": res_first["train_size"],
                "test_size": res_first["test_size"],
                "accuracy": res_first["accuracy"],
                "design_rank(incl_intercept)": res_first["design_rank"],
                "design_cond": res_first["design_cond"],
            }
        ]
    )
    st.write(diag_first_df.set_index("drop").T)

    st.subheader("VIF (sin intercept) 📈")
    st.markdown(
        "**¿Qué es VIF? (Variance Inflation Factor)** El VIF mide cuánto se “infla” la varianza del estimador de una variable "
        "cuando esa variable es (linealmente) explicada por las demás. Para cada columna $j$, se calcula un $R_j^2$ al "
        "hacer regresión sobre esa columna contra el resto y luego `VIF_j = 1 / (1 - R_j^2)`. Si hay multicolinealidad fuerte, "
        "`R_j^2` se acerca a 1 y el VIF crece mucho (e incluso puede irse a `inf` si existe dependencia lineal exacta). "
        "Referencia: [Wikipedia - Variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor)."
    )
    st.dataframe(res_first["vif_df"], use_container_width=True, hide_index=True)

    st.subheader("Coeficientes del modelo 🧮")
    coef_df = res_first["coef_series"].rename("coef").to_frame()
    coef_df["term"] = coef_df.index.map(label_term)
    st.dataframe(coef_df.reset_index(drop=True), use_container_width=True, hide_index=True)

with tabs[1]:
    res_none = fit_linear_probability_with_ohe(
        df=df,
        drop_setting=None,
        cat_labels=cat_labels,
        test_size=float(test_size),
        seed_model=int(seed_model),
    )

    st.subheader("Diagnóstico 🔍")
    diag_none_df = pd.DataFrame(
        [
            {
                "drop": "drop=None",
                "train_size": res_none["train_size"],
                "test_size": res_none["test_size"],
                "accuracy": res_none["accuracy"],
                "design_rank(incl_intercept)": res_none["design_rank"],
                "design_cond": res_none["design_cond"],
            }
        ]
    )
    st.write(diag_none_df.set_index("drop").T)

    st.subheader("VIF (sin intercept) 📈")
    st.markdown(
        "**¿Qué es VIF? (Variance Inflation Factor)** El VIF mide cuánto se “infla” la varianza del estimador de una variable "
        "cuando esa variable es (linealmente) explicada por las demás. Para cada columna $j$, se calcula un $R_j^2$ al "
        "hacer regresión sobre esa columna contra el resto y luego `VIF_j = 1 / (1 - R_j^2)`. Si hay multicolinealidad fuerte, "
        "`R_j^2` se acerca a 1 y el VIF crece mucho (e incluso puede irse a `inf` si existe dependencia lineal exacta). "
        "Referencia: [Wikipedia - Variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor)."
    )
    st.dataframe(res_none["vif_df"], use_container_width=True, hide_index=True)

    st.markdown(
        "**Qué esperar:** con `drop=None` hay un dummy extra, "
        "y con intercept aparecen dummies cuya suma replica la constante. "
        "Por eso el rango del diseño baja (multicolinealidad exacta)."
    )

    st.subheader("Comparación de coeficientes (drop='first' vs drop=None) ⚖️")
    all_terms = sorted(set(res_first["coef_series"].index).union(set(res_none["coef_series"].index)))
    coef_first_aligned = [res_first["coef_series"].get(t, np.nan) for t in all_terms]
    coef_none_aligned = [res_none["coef_series"].get(t, np.nan) for t in all_terms]

    coef_comp = pd.DataFrame(
        {
            "term": [label_term(t) for t in all_terms],
            "coef_drop_first": coef_first_aligned,
            "coef_drop_none": coef_none_aligned,
        }
    )
    st.dataframe(coef_comp, use_container_width=True, hide_index=True)

    # Gráfico: barras lado a lado.
    fig_coef = go.Figure()
    fig_coef.add_trace(
        go.Bar(
            x=coef_comp["term"],
            y=coef_comp["coef_drop_first"],
            name="drop='first'",
            opacity=0.75,
        )
    )
    fig_coef.add_trace(
        go.Bar(
            x=coef_comp["term"],
            y=coef_comp["coef_drop_none"],
            name="drop=None (dummy extra)",
            opacity=0.75,
        )
    )
    fig_coef.update_layout(
        barmode="group",
        title="Coeficientes comparados (incluye intercept)",
        xaxis_title="término",
        yaxis_title="coeficiente",
        height=440,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_coef, use_container_width=True)

st.markdown("---")
st.info(
    "ℹ️ Nota: aunque los coeficientes cambien mucho con `drop=None`, la métrica de predicción "
    "(accuracy con umbral 0.5) puede verse relativamente similar. "
    "La multicolinealidad afecta principalmente la interpretabilidad y la estabilidad numérica de los coeficientes."
)
