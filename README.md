📈 Stock Price Prediction + Explainability
Multi-market ML pipeline that predicts stock price direction using 3 models (XGBoost, Random Forest, LSTM) trained on 38 technical indicators, with SHAP explainability showing exactly WHY each prediction was made.

Python Streamlit PyTorch XGBoost SHAP License

🎯 What It Does
Select any asset from 45+ stocks, crypto, and ETFs across 3 markets
Fetches 5 years of live daily OHLCV data from Yahoo Finance
Engineers 38 technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
Trains 3 ML models simultaneously with time-series cross-validation
Compares model performance across Accuracy, F1, Precision, Recall, ROC-AUC
Explains predictions using SHAP feature importance and waterfall charts
🌍 Asset Universe (45+ assets)
Market	Examples	Count
🇺🇸 US Stocks	AAPL, TSLA, NVDA, MSFT, GOOGL	20
🇮🇱 Israeli Stocks	CHKP, CYBR, MNDY, WIX, NICE	10
₿ Crypto	BTC, ETH, SOL, BNB, DOGE	10
📊 ETFs	SPY, QQQ, GLD, TLT, IWM	5
🧠 Models
Model	Type	Strength
XGBoost	Gradient Boosting	Best overall performance, fast
Random Forest	Ensemble	Robust, interpretable
LSTM	Deep Learning (PyTorch)	Captures temporal patterns
🔍 SHAP Explainability
Global feature importance — which features matter most overall
Waterfall charts — how each feature pushed a single prediction up or down
Dependence plots — how feature value magnitude affects predictions
Human-readable insight: "RSI and MACD accounted for 38% of prediction weight"
📊 Technical Indicators (38 features)
Trend: SMA (5/10/20/50/200), EMA, Golden Cross
Momentum: RSI, MACD, Stochastic Oscillator, Momentum
Volatility: Bollinger Bands, ATR, Realized Volatility
Volume: OBV, Volume Ratio, Volume Spikes
Price: Returns (1d/5d/20d), Log Returns, Gap, High-Low Range
Calendar: Day of Week, Month, Quarter

🚀 Quick Start
git clone https://github.com/BARA-LUCH/stock-prediction-explainability.git
cd stock-prediction-explainability
python setup.py
streamlit run app.py
Open http://localhost:8501

📁 Project Structure
stock-prediction-explainability/
│
├── app.py                        # Streamlit dashboard
├── data/
│   └── fetcher.py                # Yahoo Finance data pipeline (45+ assets)
├── features/
│   └── engineer.py               # 38 technical indicators
├── models/
│   └── trainer.py                # XGBoost, Random Forest, LSTM
├── explainability/
│   └── shap_analysis.py          # SHAP values, waterfall, importance plots
├── setup.py
├── requirements.txt
└── README.md
📈 Example Results (AAPL, 5-day horizon)
Model	Accuracy	F1	ROC-AUC
XGBoost	0.587	0.591	0.623
Random Forest	0.572	0.580	0.608
LSTM	0.561	0.554	0.589
Top features: RSI, MACD_Hist, BB_Position, Volatility_20d, Returns_5d

Note: Stock prediction is inherently difficult. These results are for research purposes and not financial advice.

⚠️ Disclaimer
This project is for educational and portfolio purposes only. It is not financial advice. Do not use model predictions for real trading decisions.

👤 Author
Bara Luch — ML Engineer & Data Scientist
📍 Tel Aviv, Israel
🔗 LinkedIn · 💻 GitHub

📄 License
MIT License
