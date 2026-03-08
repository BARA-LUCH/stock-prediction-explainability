"""
explainability/shap_analysis.py
Generates SHAP explanations for XGBoost and Random Forest models.
Produces feature importance, waterfall charts, and summary plots.
"""

import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def get_shap_values(result: dict) -> dict:
    """
    Compute SHAP values for a trained tree-based model (XGBoost or Random Forest).
    Returns explainer and SHAP values.
    """
    model = result["model"]
    X_train = result["X_train"]
    X_test = result["X_test"]
    features = result["features"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, Random Forest returns list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "X_test": X_test,
        "features": features,
        "base_value": float(explainer.expected_value)
            if not isinstance(explainer.expected_value, np.ndarray)
            else float(explainer.expected_value[1]),
    }


def plot_feature_importance(shap_data: dict, model_name: str, top_n: int = 15) -> go.Figure:
    """
    Bar chart of mean absolute SHAP values (global feature importance).
    """
    shap_values = shap_data["shap_values"]
    features = shap_data["features"]

    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": features, "SHAP Importance": mean_abs})
    df = df.sort_values("SHAP Importance", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=df["SHAP Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(
            color=df["SHAP Importance"],
            colorscale="Blues",
            showscale=False
        )
    ))

    fig.update_layout(
        title=f"🔍 Feature Importance — {model_name} (SHAP)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
    )
    return fig


def plot_waterfall(shap_data: dict, model_name: str, sample_idx: int = 0) -> go.Figure:
    """
    Waterfall chart showing how each feature pushed the prediction
    up or down for a single data point.
    """
    shap_values = shap_data["shap_values"][sample_idx]
    features = shap_data["features"]
    base_value = shap_data["base_value"]

    # Sort by absolute impact
    indices = np.argsort(np.abs(shap_values))[-12:]
    sv = shap_values[indices]
    ft = [features[i] for i in indices]

    colors = ["#E74C3C" if v > 0 else "#2E86C1" for v in sv]
    labels = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in sv]

    fig = go.Figure(go.Bar(
        x=sv,
        y=ft,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
    ))

    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title=f"📊 SHAP Waterfall — {model_name} (Sample #{sample_idx})<br><sup>Base value: {base_value:.3f}</sup>",
        xaxis_title="SHAP Value (impact on prediction)",
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
    )
    return fig


def plot_shap_scatter(shap_data: dict, model_name: str, feature: str) -> go.Figure:
    """
    Scatter plot of SHAP value vs feature value for a single feature.
    Shows how feature value magnitude affects prediction.
    """
    features = shap_data["features"]
    if feature not in features:
        return None

    feat_idx = features.index(feature)
    x = shap_data["X_test"][:, feat_idx]
    y = shap_data["shap_values"][:, feat_idx]

    fig = px.scatter(
        x=x, y=y,
        color=y,
        color_continuous_scale="RdBu",
        labels={"x": feature, "y": "SHAP Value"},
        title=f"SHAP Dependence Plot — {feature} ({model_name})",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
    return fig


def get_top_features(shap_data: dict, top_n: int = 5) -> list:
    """Return names of top N most important features."""
    shap_values = shap_data["shap_values"]
    features = shap_data["features"]
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs)[::-1][:top_n]
    return [features[i] for i in indices]


def generate_shap_insight(shap_data: dict, model_name: str) -> str:
    """Generate a human-readable insight from SHAP analysis."""
    top_features = get_top_features(shap_data, top_n=5)
    shap_values = shap_data["shap_values"]
    features = shap_data["features"]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:3]
    top_pct = mean_abs[top_idx] / mean_abs.sum() * 100

    lines = [
        f"**{model_name} — Key Drivers:**",
        f"- Top 3 features account for **{top_pct.sum():.1f}%** of prediction weight",
    ]
    for i, idx in enumerate(top_idx):
        lines.append(f"- **{features[idx]}** contributed {top_pct[i]:.1f}%")

    return "\n".join(lines)
