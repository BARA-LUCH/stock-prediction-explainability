"""
app.py — Stock Price Prediction + Explainability Dashboard
Multi-market (US, Israeli, Crypto), 3 models, SHAP analysis
Built by Bara Luch
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Stock Prediction + SHAP",
    page_icon="📈",
    layout="wide"
)

# ── Imports ───────────────────────────────────────────────────────────────────
from data.fetcher import (
    fetch_single_asset, ALL_ASSETS, MARKET_CATEGORIES,
    US_STOCKS, ISRAELI_STOCKS, CRYPTO, ETFs
)
from features.engineer import engineer_features, FEATURE_COLUMNS
from models.trainer import train_all_models, compare_models
from explainability.shap_analysis import (
    get_shap_values, plot_feature_importance,
    plot_waterfall, get_top_features, generate_shap_insight
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Prediction + Explainability")
st.markdown("**Multi-market ML pipeline** — US Stocks · Israeli Stocks · Crypto · 3 Models · SHAP Analysis")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    market = st.selectbox("Market", list(MARKET_CATEGORIES.keys()))
    tickers_in_market = MARKET_CATEGORIES[market]
    ticker_labels = {t: f"{t} — {ALL_ASSETS.get(t, t)}" for t in tickers_in_market}
    ticker = st.selectbox("Asset", list(ticker_labels.keys()), format_func=lambda x: ticker_labels[x])

    horizon = st.slider("Prediction Horizon (days)", 1, 20, 5,
                        help="How many days ahead to predict direction")

    st.markdown("---")
    st.markdown("**Models:** XGBoost · Random Forest · LSTM")
    st.markdown("**Features:** 38 technical indicators")
    st.markdown("**Explainability:** SHAP values")
    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("Built by **Bara Luch**")
    st.markdown("[GitHub](https://github.com/BARA-LUCH) · [LinkedIn](https://linkedin.com/in/bara-luch)")

# ── Main Pipeline ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"📥 Fetching {ticker} data..."):
        df_raw = fetch_single_asset(ticker, years=5)

    if df_raw is None or df_raw.empty:
        st.error(f"❌ Could not fetch data for {ticker}. Try another asset.")
        st.stop()

    with st.spinner("⚙️ Engineering features..."):
        df_feat = engineer_features(df_raw, horizon=horizon)

    if len(df_feat) < 200:
        st.warning(f"⚠️ Only {len(df_feat)} rows after feature engineering. Results may be limited.")

    with st.spinner("🤖 Training 3 models (this takes ~1-2 minutes)..."):
        results = train_all_models(df_feat)

    st.success(f"✅ Analysis complete for **{ticker} — {ALL_ASSETS.get(ticker, '')}**")
    st.session_state["results"] = results
    st.session_state["df_raw"] = df_raw
    st.session_state["df_feat"] = df_feat
    st.session_state["ticker"] = ticker
    st.session_state["horizon"] = horizon

# ── Display Results ───────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    df_raw = st.session_state["df_raw"]
    df_feat = st.session_state["df_feat"]
    ticker = st.session_state["ticker"]
    horizon = st.session_state["horizon"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price & Data", "🏆 Model Comparison", "🔍 SHAP Explainability",
        "📉 Predictions vs Actual", "📋 Raw Data"
    ])

    # ── Tab 1: Price Chart ────────────────────────────────────────────────────
    with tab1:
        st.subheader(f"📊 {ticker} — Price History (5 Years)")

        col1, col2, col3, col4 = st.columns(4)
        latest = df_raw["Close"].iloc[-1]
        start_price = df_raw["Close"].iloc[0]
        returns_5y = (latest / start_price - 1) * 100
        volatility = df_raw["Close"].pct_change().std() * np.sqrt(252) * 100

        col1.metric("Latest Price", f"${latest:.2f}")
        col2.metric("5Y Return", f"{returns_5y:+.1f}%")
        col3.metric("Annual Volatility", f"{volatility:.1f}%")
        col4.metric("Data Points", f"{len(df_raw):,}")

        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df_raw.index,
            open=df_raw["Open"],
            high=df_raw["High"],
            low=df_raw["Low"],
            close=df_raw["Close"],
            name=ticker,
            increasing_line_color="#2ECC71",
            decreasing_line_color="#E74C3C"
        )])
        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=450,
            xaxis_rangeslider_visible=False,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        fig2 = px.bar(
            x=df_raw.index, y=df_raw["Volume"],
            title="Trading Volume",
            labels={"x": "Date", "y": "Volume"},
            color=df_raw["Volume"],
            color_continuous_scale="Blues"
        )
        fig2.update_layout(height=250, showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Model Comparison ───────────────────────────────────────────────
    with tab2:
        st.subheader("🏆 Model Comparison")
        comp = results["comparison"]

        # Metrics table
        st.dataframe(
            comp.style.highlight_max(axis=0, color="#D5F5E3")
                      .format("{:.4f}"),
            use_container_width=True
        )

        # Best model callout
        best_model = comp["ROC-AUC"].idxmax()
        best_auc = comp["ROC-AUC"].max()
        st.success(f"🥇 Best model: **{best_model}** with ROC-AUC = **{best_auc:.4f}**")

        # Bar chart comparison
        metrics_to_plot = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"]
        comp_plot = comp[metrics_to_plot].reset_index().melt(id_vars="Model")
        fig3 = px.bar(
            comp_plot, x="variable", y="value", color="Model",
            barmode="group",
            title="Model Performance Comparison",
            labels={"variable": "Metric", "value": "Score"},
            color_discrete_sequence=["#2E86C1", "#E74C3C", "#2ECC71"]
        )
        fig3.update_layout(height=400, plot_bgcolor="white", yaxis_range=[0, 1])
        st.plotly_chart(fig3, use_container_width=True)

        # Train/test info
        st.info(f"Training samples: **{results['train_size']:,}** · Test samples: **{results['test_size']:,}** · Prediction horizon: **{horizon} days**")

    # ── Tab 3: SHAP ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("🔍 SHAP Explainability")
        st.markdown("*Understand WHY each model makes its predictions*")

        model_choice = st.selectbox("Select model for SHAP analysis", ["XGBoost", "Random Forest"])
        model_key = "xgboost" if model_choice == "XGBoost" else "random_forest"
        result = results[model_key]

        with st.spinner("Computing SHAP values..."):
            shap_data = get_shap_values(result)

        # Insight text
        insight = generate_shap_insight(shap_data, model_choice)
        st.info(insight)

        col_a, col_b = st.columns(2)

        with col_a:
            fig_imp = plot_feature_importance(shap_data, model_choice)
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_b:
            sample_idx = st.slider("Waterfall sample index", 0, min(50, len(result["X_test"]) - 1), 0)
            fig_wf = plot_waterfall(shap_data, model_choice, sample_idx)
            st.plotly_chart(fig_wf, use_container_width=True)

        # Top features
        top_feats = get_top_features(shap_data, top_n=5)
        st.markdown(f"**Top 5 predictive features:** {' · '.join([f'`{f}`' for f in top_feats])}")

    # ── Tab 4: Predictions ────────────────────────────────────────────────────
    with tab4:
        st.subheader("📉 Predictions vs Actual")

        model_choice2 = st.selectbox("Model", ["XGBoost", "Random Forest", "LSTM"], key="pred_model")
        key_map = {"XGBoost": "xgboost", "Random Forest": "random_forest", "LSTM": "lstm"}
        res = results[key_map[model_choice2]]

        y_test = res["y_test"]
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]

        # Probability line chart
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=y_prob, mode="lines", name="Predicted Probability", line=dict(color="#2E86C1")))
        fig4.add_trace(go.Scatter(y=y_test, mode="lines", name="Actual Direction", line=dict(color="#E74C3C", dash="dot")))
        fig4.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Decision boundary")
        fig4.update_layout(
            title=f"{model_choice2} — Predicted Probability vs Actual Direction",
            xaxis_title="Test Sample",
            yaxis_title="Value",
            height=400,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Confusion matrix
        cm = res["metrics"]["Confusion Matrix"]
        cm_df = pd.DataFrame(cm,
            index=["Actual Down", "Actual Up"],
            columns=["Predicted Down", "Predicted Up"]
        )
        fig5 = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                         title=f"Confusion Matrix — {model_choice2}")
        fig5.update_layout(height=350)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Tab 5: Raw Data ───────────────────────────────────────────────────────
    with tab5:
        st.subheader("📋 Feature-Engineered Data")
        st.dataframe(df_feat.tail(100), use_container_width=True)
        csv = df_feat.to_csv().encode("utf-8")
        st.download_button("⬇️ Download Full Dataset (CSV)", csv,
                           file_name=f"{ticker}_features.csv", mime="text/csv")

else:
    # Landing state
    st.info("👈 Select a market and asset from the sidebar, then click **Run Analysis**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Assets Available", "45+")
    col2.metric("Years of Data", "5")
    col3.metric("Technical Indicators", "38")

    st.markdown("""
    ### How it works
    1. **Select** any stock, crypto, or ETF from 3 markets
    2. **Run Analysis** — fetches live data, engineers 38 features
    3. **3 models** train simultaneously (XGBoost, Random Forest, LSTM)
    4. **SHAP** explains exactly why each prediction was made
    5. **Compare** model performance with interactive charts
    """)
