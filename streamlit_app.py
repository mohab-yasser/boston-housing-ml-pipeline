from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "boston_rf_model.joblib"
DATA_PATH = BASE_DIR / "housing.csv"

FEATURE_COLUMNS = ["RM", "LSTAT", "PTRATIO"]
TARGET_COLUMN = "MEDV"

APP_ICON = "\U0001F3E0"
CHART_ICON = "\U0001F4CA"
INFO_TAB_LABEL = "\u2139\ufe0f About Project"

MODEL_METRICS = {
    "Final Test R^2": "0.829",
    "Cross Validation R^2": "0.840",
    "MAE": "$48,555.72",
    "RMSE": "$68,146.87",
}

MODEL_COMPARISON = pd.DataFrame(
    {
        "Model": [
            "Baseline",
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Tuned Decision Tree",
            "Tuned Random Forest",
        ],
        "Score": [-0.015, 0.706, 0.764, 0.855, 0.792, 0.840],
        "Evaluation": [
            "Validation R^2",
            "Validation R^2",
            "Validation R^2",
            "Validation R^2",
            "CV R^2",
            "CV R^2",
        ],
    }
)

CLIENT_EXAMPLES = {
    "Client 1": {"RM": 5.0, "LSTAT": 17.0, "PTRATIO": 15.0},
    "Client 2": {"RM": 4.0, "LSTAT": 32.0, "PTRATIO": 22.0},
    "Client 3": {"RM": 8.0, "LSTAT": 3.0, "PTRATIO": 12.0},
}

FEATURE_DETAILS = {
    "RM": {
        "label": "Average Rooms",
        "description": "Average number of rooms per dwelling.",
        "default": 6.24,
        "hard_min": 1.0,
        "hard_max": 12.0,
        "step": 0.01,
    },
    "LSTAT": {
        "label": "Lower-Status Population",
        "description": "Percentage of lower-status population in the neighborhood.",
        "default": 12.94,
        "hard_min": 0.0,
        "hard_max": 60.0,
        "step": 0.01,
    },
    "PTRATIO": {
        "label": "Student-Teacher Ratio",
        "description": "Ratio of students to teachers in nearby schools.",
        "default": 18.52,
        "hard_min": 5.0,
        "hard_max": 30.0,
        "step": 0.01,
    },
}


st.set_page_config(
    page_title="Boston Housing Intelligence",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --bg: #0b1020;
            --panel: rgba(18, 26, 48, 0.88);
            --panel-soft: rgba(31, 42, 70, 0.72);
            --border: rgba(148, 163, 184, 0.22);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --warning: #f59e0b;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 28rem),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.10), transparent 26rem),
                linear-gradient(135deg, #070b16 0%, #111827 46%, #0b1020 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: rgba(8, 13, 25, 0.96);
            border-right: 1px solid var(--border);
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.20);
        }

        .hero {
            padding: 2.1rem 2.3rem;
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.76)),
                linear-gradient(120deg, rgba(56, 189, 248, 0.18), rgba(34, 197, 94, 0.12));
            box-shadow: 0 26px 70px rgba(0, 0, 0, 0.30);
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2.2rem, 5vw, 4.6rem);
            letter-spacing: 0;
            line-height: 1.02;
        }

        .hero .subtitle {
            margin-top: 0.55rem;
            color: #bae6fd;
            font-size: 1.35rem;
            font-weight: 650;
        }

        .hero .body {
            margin-top: 0.9rem;
            max-width: 780px;
            color: #cbd5e1;
            font-size: 1.02rem;
            line-height: 1.6;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.20);
        }

        .soft-card {
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
        }

        .result-card {
            border: 1px solid rgba(56, 189, 248, 0.38);
            border-radius: 22px;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.78), rgba(15, 23, 42, 0.94));
            box-shadow: 0 28px 80px rgba(14, 165, 233, 0.18);
        }

        .result-label {
            color: #bae6fd;
            text-transform: uppercase;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.08em;
        }

        .result-value {
            margin-top: 0.25rem;
            color: #f8fafc;
            font-size: clamp(2.2rem, 5vw, 4rem);
            font-weight: 800;
            letter-spacing: 0;
        }

        .pill {
            display: inline-block;
            margin-top: 0.85rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.36);
            color: #bbf7d0;
            font-weight: 700;
            font-size: 0.86rem;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.15rem;
            min-height: 142px;
        }

        .metric-card strong {
            display: block;
            color: #f8fafc;
            font-size: 1.75rem;
            margin-top: 0.35rem;
        }

        .metric-card span {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .timeline-item {
            border-left: 2px solid rgba(56, 189, 248, 0.55);
            padding: 0.15rem 0 0.85rem 1rem;
            color: #dbeafe;
        }

        .timeline-item:last-child {
            padding-bottom: 0;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.35);
            background: rgba(15, 23, 42, 0.84);
            color: #e0f2fe;
            font-weight: 700;
        }

        .stButton > button:hover {
            border-color: rgba(34, 197, 94, 0.55);
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH.name}", False

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
        version_warning = any(
            issubclass(item.category, InconsistentVersionWarning) for item in caught
        )
        if not hasattr(model, "predict"):
            return None, "The loaded object does not expose a predict() method.", version_warning
        return model, None, version_warning
    except Exception as exc:
        return None, f"Could not load the saved model: {exc}", False


@st.cache_data
def load_housing_data():
    if not DATA_PATH.exists():
        return None, f"Dataset file not found: {DATA_PATH.name}"

    try:
        data = pd.read_csv(DATA_PATH)
    except Exception as exc:
        return None, f"Could not load housing.csv: {exc}"

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        return None, f"housing.csv is missing required columns: {', '.join(missing_columns)}"

    return data, None


def format_currency(value):
    return f"${value:,.2f}"


def initialize_inputs():
    for feature_name, details in FEATURE_DETAILS.items():
        key = feature_name.lower()
        st.session_state.setdefault(key, details["default"])
    st.session_state.setdefault("prediction_requested", False)
    st.session_state.setdefault("selected_example", None)


def apply_example(example_name):
    values = CLIENT_EXAMPLES[example_name]
    for feature_name, value in values.items():
        st.session_state[feature_name.lower()] = value
    st.session_state["selected_example"] = example_name
    st.session_state["prediction_requested"] = True


def medv_stats(data):
    prices = data[TARGET_COLUMN]
    return {
        "min": float(prices.min()),
        "mean": float(prices.mean()),
        "median": float(prices.median()),
        "max": float(prices.max()),
        "q1": float(prices.quantile(0.25)),
        "q3": float(prices.quantile(0.75)),
        "p90": float(prices.quantile(0.90)),
    }


def price_category(prediction, stats):
    if prediction < stats["q1"]:
        return "Lower Quartile"
    if prediction <= stats["q3"]:
        return "Middle 50%"
    if prediction <= stats["p90"]:
        return "Upper Quartile"
    return "Top Decile"


def out_of_range_messages(values, data):
    messages = []
    for feature_name, value in values.items():
        observed_min = float(data[feature_name].min())
        observed_max = float(data[feature_name].max())
        if value < observed_min or value > observed_max:
            messages.append(
                f"{feature_name}={value:.2f} is outside the training range "
                f"({observed_min:.2f} to {observed_max:.2f})."
            )
    return messages


def render_metric_card(label, value, description):
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
            <div class="small-muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(data, model_warning):
    with st.sidebar:
        st.markdown("### Boston Housing Price Prediction")
        st.caption("Educational ML portfolio demo · historical data")
        st.divider()
        st.metric("Model", "Random Forest")
        st.metric("Features", "3")
        st.metric("Dataset", f"{len(data):,} records")
        st.metric("Test R^2", "0.829")
        st.divider()
        st.markdown(
            """
            **Inputs**

            `RM`, `LSTAT`, `PTRATIO`
            """
        )
        if model_warning:
            st.warning(
                "The model was saved with a different scikit-learn version. "
                "Cross-version loading is not supported; use scikit-learn 1.6.0."
            )
        st.info(
            "Historical, modified Boston Housing dataset; LSTAT is a sensitive socioeconomic proxy. "
            "Predictions are for educational demonstration only, not real property valuation."
        )


def model_comparison_chart():
    chart_data = MODEL_COMPARISON.copy()
    chart_data["Score Label"] = chart_data["Score"].map(lambda value: f"{value:.3f}")
    colors = [
        "#64748b",
        "#60a5fa",
        "#f59e0b",
        "#22c55e",
        "#a78bfa",
        "#38bdf8",
    ]

    fig = px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Model",
        color_discrete_sequence=colors,
        text="Score Label",
        hover_data={"Evaluation": True, "Score": ":.3f", "Score Label": False},
        title="Model Comparison — Different Evaluation Protocols",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis_title="R^2 Score",
        xaxis_title=None,
        yaxis=dict(range=[-0.08, 0.90]),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def correlation_heatmap(data):
    corr = data[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=460,
    )
    return fig


def scatter_with_trend(data, feature_name):
    fig = px.scatter(
        data,
        x=feature_name,
        y=TARGET_COLUMN,
        opacity=0.72,
        title=f"{feature_name} vs MEDV",
        labels={
            "RM": "Average Rooms",
            "LSTAT": "Lower-Status Population (%)",
            "PTRATIO": "Student-Teacher Ratio",
            TARGET_COLUMN: "House Price ($)",
        },
    )

    x_values = data[feature_name].to_numpy()
    y_values = data[TARGET_COLUMN].to_numpy()
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    y_line = slope * x_line + intercept

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Trend",
            line=dict(color="#f97316", width=3),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=430,
    )
    fig.update_traces(marker=dict(size=8, color="#38bdf8"), selector=dict(mode="markers"))
    return fig


def predict_price(model, values):
    input_data = pd.DataFrame(
        {
            "RM": [values["RM"]],
            "LSTAT": [values["LSTAT"]],
            "PTRATIO": [values["PTRATIO"]],
        },
        columns=FEATURE_COLUMNS,
    )
    return float(model.predict(input_data)[0]), input_data


model, model_error, model_version_warning = load_model()
data, data_error = load_housing_data()

if data_error:
    st.error(data_error)
    st.stop()

if model_error:
    st.error(model_error)
    st.stop()

initialize_inputs()
render_sidebar(data, model_version_warning)
stats = medv_stats(data)

st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_ICON} Boston Housing Intelligence</h1>
        <div class="subtitle">AI-Powered House Price Prediction</div>
        <div class="body">
            Estimate Boston housing values using the final tuned Random Forest regression model from the project.
            The model uses room count, neighborhood lower-status population, and student-teacher ratio to predict
            historical median home value in adjusted dollars. This is a learning demo, not a real-estate appraisal.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predictor, tab_performance, tab_insights, tab_about = st.tabs(
    ["Price Predictor", "Model Performance", "Data Insights", INFO_TAB_LABEL]
)

with tab_predictor:
    st.subheader("Price Predictor")
    st.caption("Choose values manually or start from one of the project client examples.")

    st.markdown("#### Try Example Properties")
    example_columns = st.columns(3)
    for column, example_name in zip(example_columns, CLIENT_EXAMPLES):
        values = CLIENT_EXAMPLES[example_name]
        with column:
            st.markdown(
                f"""
                <div class="soft-card">
                    <strong>{example_name}</strong>
                    <div class="small-muted">
                        RM {values["RM"]:.0f} | LSTAT {values["LSTAT"]:.0f}% | PTRATIO {values["PTRATIO"]:.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                f"Use {example_name}",
                key=f"use_{example_name.lower().replace(' ', '_')}",
                on_click=apply_example,
                args=(example_name,),
                width="stretch",
            )

    st.markdown("#### Property Inputs")
    input_columns = st.columns(3)
    for column, feature_name in zip(input_columns, FEATURE_COLUMNS):
        details = FEATURE_DETAILS[feature_name]
        observed_min = float(data[feature_name].min())
        observed_max = float(data[feature_name].max())
        with column:
            st.number_input(
                details["label"],
                min_value=details["hard_min"],
                max_value=details["hard_max"],
                step=details["step"],
                key=feature_name.lower(),
                help=details["description"],
            )
            st.caption(details["description"])
            st.caption(f"Training range: {observed_min:.2f} to {observed_max:.2f}")

    current_values = {
        "RM": float(st.session_state["rm"]),
        "LSTAT": float(st.session_state["lstat"]),
        "PTRATIO": float(st.session_state["ptratio"]),
    }

    if st.button("Predict House Price", type="primary", width="stretch"):
        st.session_state["prediction_requested"] = True
        st.session_state["selected_example"] = None

    range_messages = out_of_range_messages(current_values, data)
    if range_messages:
        st.warning(
            "One or more inputs are outside the observed training range. "
            "This is extrapolation, so the estimate may be less reliable.\n\n"
            + "\n".join(f"- {message}" for message in range_messages)
        )

    if st.session_state["prediction_requested"]:
        try:
            prediction, input_data = predict_price(model, current_values)
            delta = prediction - stats["mean"]
            delta_direction = "above" if delta >= 0 else "below"
            category = price_category(prediction, stats)
            progress_value = (prediction - stats["min"]) / (stats["max"] - stats["min"])
            progress_value = max(0.0, min(1.0, progress_value))

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated Property Value</div>
                    <div class="result-value">{format_currency(prediction)}</div>
                    <div class="pill">{category}</div>
                    <div class="small-muted" style="margin-top: 0.9rem;">
                        {format_currency(abs(delta))} {delta_direction} the dataset average of {format_currency(stats["mean"])}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(progress_value)

            context_columns = st.columns(4)
            context_columns[0].metric("Dataset Min", format_currency(stats["min"]))
            context_columns[1].metric("Dataset Average", format_currency(stats["mean"]))
            context_columns[2].metric("Dataset Median", format_currency(stats["median"]))
            context_columns[3].metric("Dataset Max", format_currency(stats["max"]))

            with st.expander("Prediction input sent to the model", expanded=False):
                st.dataframe(input_data, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Prediction failed. Please check the inputs and model file. Details: {exc}")
    else:
        st.markdown(
            """
            <div class="card">
                <strong>Ready for prediction</strong>
                <div class="small-muted">
                    Enter property values or choose an example, then run the saved Random Forest model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_performance:
    st.subheader("Model Performance")
    st.caption("Reported metrics come from the final notebook. The Streamlit app does not retrain the model.")

    metric_columns = st.columns(4)
    metric_descriptions = {
        "Final Test R^2": "Explained variation on the held-out test set.",
        "Cross Validation R^2": "Average development score used for model selection.",
        "MAE": "Average absolute prediction error in dollars.",
        "RMSE": "Penalizes larger prediction errors more strongly than MAE.",
    }
    for column, (label, value) in zip(metric_columns, MODEL_METRICS.items()):
        with column:
            render_metric_card(label, value, metric_descriptions[label])

    st.caption("Baseline and untuned models use one validation split; tuned-model scores use 5-fold cross-validation. These are different evaluation protocols and are not directly comparable.")
    st.plotly_chart(model_comparison_chart(), width="stretch")

    st.markdown(
        """
        <div class="card">
            <strong>Selection rule</strong>
            <div class="small-muted">
                The tuned Random Forest was selected because it had the strongest cross-validation performance.
                The held-out test set was inspected only after model selection to estimate final generalization.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_insights:
    st.subheader("Data Insights")

    data_columns = st.columns(4)
    data_columns[0].metric("Rows", f"{len(data):,}")
    data_columns[1].metric("Predictors", "3")
    data_columns[2].metric("Missing Values", f"{int(data.isna().sum().sum()):,}")
    data_columns[3].metric("Duplicate Rows", f"{int(data.duplicated().sum()):,}")

    st.plotly_chart(correlation_heatmap(data), width="stretch")

    st.markdown(
        """
        <div class="card">
            <strong>Key correlation insights</strong>
            <div class="small-muted">
                More rooms generally correspond to higher house value. Higher LSTAT generally corresponds to
                lower house value, and higher PTRATIO also tends to correspond to lower house value.
                LSTAT has the strongest absolute correlation with MEDV. Correlation does not prove causation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scatter_tabs = st.tabs(["RM vs MEDV", "LSTAT vs MEDV", "PTRATIO vs MEDV"])
    for scatter_tab, feature_name in zip(scatter_tabs, FEATURE_COLUMNS):
        with scatter_tab:
            st.plotly_chart(scatter_with_trend(data, feature_name), width="stretch")

with tab_about:
    st.subheader("About Project")

    info_columns = st.columns(4)
    info_columns[0].metric("Problem Type", "Regression")
    info_columns[1].metric("Best Model", "Tuned RF")
    info_columns[2].metric("Features", "RM, LSTAT, PTRATIO")
    info_columns[3].metric("Target", "MEDV")

    st.markdown(
        """
        <div class="card">
            <strong>Project workflow</strong>
            <div class="small-muted">
                Supervised Machine Learning - Regression project using three predictors to estimate house prices.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workflow_steps = [
        "Data Collection",
        "Exploratory Data Analysis",
        "Data Cleaning",
        "Feature Analysis",
        "Train / Validation / Test Split",
        "Baseline Model",
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Learning Curves",
        "Bias vs Variance Analysis",
        "Cross Validation",
        "GridSearchCV",
        "Best Model Selection",
        "Final Test Evaluation",
        "Deployment with Streamlit",
    ]
    st.markdown(
        "".join(f'<div class="timeline-item">{step}</div>' for step in workflow_steps),
        unsafe_allow_html=True,
    )