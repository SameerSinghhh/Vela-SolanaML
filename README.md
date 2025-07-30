# 🚀 MLSolana: AI-Powered Solana Price Prediction

> **Predicting Solana's next-day price movements using machine learning and comprehensive market data**

---

## 🎯 Project Goals

**Primary Objective:** Build a robust machine learning model to predict whether Solana (SOL) will move **UP**, **DOWN**, or stay **NEUTRAL** the following day.

**Target Classification:**
- 📈 **UP**: Next-day return > +2%
- 📉 **DOWN**: Next-day return < -2%  
- ⚖️ **NEUTRAL**: Next-day return between -2% and +2%

---

## 📊 Current Data Collection

### **Price Data Sources**
- **SOL**: Solana historical price data (July 2023 - July 2025)
- **BTC**: Bitcoin price data for market correlation
- **ETH**: Ethereum price data for crypto market context
- **SP500**: Traditional market benchmark data

### **Feature Matrix Overview**
🗓️ **Timeline**: 723 trading days (July 23, 2023 → July 14, 2025)  
📈 **Features**: 17 comprehensive indicators

| **Category** | **Features** | **Description** |
|--------------|-------------|-----------------|
| **💰 Price Data** | SOL, BTC, ETH closing prices | Cross-asset price levels |
| **📊 Returns** | 1d, 3d, 7d returns | Multi-timeframe momentum |
| **⚡ Volatility** | 7-day rolling volatility | Risk measurement |
| **🔗 Correlations** | SOL/BTC, SOL/ETH ratios | Relative positioning |
| **📈 Technical** | RSI (14d), MACD histogram | Momentum indicators |
| **🎯 Moving Averages** | SMA ratios & deviations | Trend analysis |
| **🔄 Momentum History** | 2-day target rolling mean | Recent pattern tracking |

---

## 🛠️ Next Steps

### **Phase 1: On-Chain Data Integration** 🔗
- **Network Activity**: Transaction volume, active addresses
- **DeFi Metrics**: TVL, staking ratios, validator data  
- **Social Sentiment**: Community engagement metrics
- **Developer Activity**: GitHub commits, ecosystem growth

### **Phase 2: Model Development** 🤖
- **Algorithm Selection**: Random Forest, XGBoost, Neural Networks
- **Feature Engineering**: Advanced technical indicators
- **Hyperparameter Tuning**: Optimize model performance
- **Cross-Validation**: Robust model validation

### **Phase 3: Backtesting & Validation** 📋
- **Walk-Forward Analysis**: Time-series appropriate testing
- **Performance Metrics**: Precision, recall, F1-score
- **Risk Assessment**: Sharpe ratio, max drawdown
- **Strategy Simulation**: Trading performance evaluation

---

## 🏗️ Project Structure

```
MLSolana/
├── 📁 price_data/           # Historical price datasets
├── 📄 feature_matrix.csv    # Complete feature dataset
├── 🐍 create_feature_matrix.py  # Data processing pipeline
└── 📖 README.md            # Project documentation
```

---

## 🎯 Success Metrics

- **Accuracy**: >60% directional prediction accuracy
- **Risk-Adjusted Returns**: Positive Sharpe ratio in backtesting  
- **Robustness**: Consistent performance across market regimes
- **Actionable Insights**: Clear, interpretable predictions

---

## 🔧 Technologies

| **Category** | **Tools** |
|--------------|-----------|
| **Data Processing** | Python, Pandas, NumPy |
| **Technical Analysis** | TA-Lib library |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Plotly |

---

## 📈 Current Status

✅ **Data Collection**: Historical price data compiled  
✅ **Feature Engineering**: 17 technical indicators created  
🔄 **In Progress**: On-chain data integration  
⏳ **Upcoming**: ML model development & backtesting

---

*Built with ❤️ for the Solana ecosystem*
