import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Fetch Live Sensex Data
def get_live_sensex():
    sensex = yf.download('^BSESN', period='7d', interval='5m')
    sensex.reset_index(inplace=True)
    if sensex.empty:
        return None
    return sensex[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Datetime')

# Feature Engineering
def add_features(df):
    df['MA9'] = df['Close'].rolling(9).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    df.dropna(inplace=True)
    return df

# Prepare dataset for ML
def prepare_data(df):
    df = add_features(df)
    df['Target'] = df['Close'].shift(-1)  # Predicting next close price
    df.dropna(inplace=True)
    return df

# Train & Test Model
def train_model(df):
    X = df[['MA9', 'MA20', 'Volatility']]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    return model, mae

# Trading Signal based on Moving Averages
def get_trade_signal_ma(df):
    if len(df) < 2:
        return "⚖️ Hold"
    
    prev_candle = df.iloc[-2]
    last_candle = df.iloc[-1]
    
    prev_ma9, prev_ma20 = float(prev_candle['MA9']), float(prev_candle['MA20'])
    last_ma9, last_ma20, last_close = float(last_candle['MA9']), float(last_candle['MA20']), float(last_candle['Close'])

    if (prev_ma9 < prev_ma20) and (last_ma9 > last_ma20) and (last_close > last_ma9):
        return "📈 Buy Call Option"
    elif (prev_ma9 > prev_ma20) and (last_ma9 < last_ma20) and (last_close < last_ma9):
        return "📉 Buy Put Option"
    return "⚖️ Hold"

# Get Live Option Chain Data (Replace with real API)
def get_option_chain():
    strike_prices = np.arange(64000, 67000, 500)
    call_prices = np.random.uniform(200, 1000, len(strike_prices))
    put_prices = np.random.uniform(200, 1000, len(strike_prices))
    return pd.DataFrame({'Strike Price': strike_prices, 'Call Price': call_prices, 'Put Price': put_prices})

# Streamlit UI
st.set_page_config(page_title="Sensex Live Predictor", layout="wide")
st.title("📈 Sensex Live Market Predictor")

if st.button("🔍 Predict Live Market"):
    data = get_live_sensex()

    if data is None or data.empty:
        st.error("No live data available.")
    else:
        data = add_features(data)
        trade_signal = get_trade_signal_ma(data)

        st.markdown(f"### 🏦 Trade Suggestion: `{trade_signal}`")

        # Fetch Option Chain Data
        option_chain = get_option_chain()

        col1, col2, col3 = st.columns(3)

        # Sensex Price Chart
        with col1:
            st.subheader("📊 Sensex Price Chart")
            fig = go.Figure()
            
            # Enhanced Candlestick Chart for TradingView Style
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                increasing=dict(line=dict(color='green', width=2), fillcolor='green'),
                decreasing=dict(line=dict(color='red', width=2), fillcolor='red'),
                name="Candlesticks"
            ))

            # Moving Averages
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA9'],
                mode='lines', name="MA9",
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA20'],
                mode='lines', name="MA20",
                line=dict(color='orange', width=2)
            ))

            # Layout Enhancements
            fig.update_layout(
                height=500, width=800,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                title="Sensex Price Chart with Candlesticks & Moving Averages",
                yaxis_title="Price",
                xaxis_title="Time",
                plot_bgcolor='#131722',
                paper_bgcolor='#131722',
                font=dict(color='white')
            )

            st.plotly_chart(fig)

        # Put Option Chart
        with col2:
            st.subheader("📉 Put Option Prices")
            fig2 = go.Figure(data=[go.Bar(
                x=option_chain['Strike Price'], y=option_chain['Put Price'],
                marker=dict(color='red'), name='Put Price'
            )])
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2)

        # Call Option Chart
        with col3:
            st.subheader("📈 Call Option Prices")
            fig3 = go.Figure(data=[go.Bar(
                x=option_chain['Strike Price'], y=option_chain['Call Price'],
                marker=dict(color='green'), name='Call Price'
            )])
            fig3.update_layout(template="plotly_dark")
            st.plotly_chart(fig3)

# Streamlit UI - Evaluate Model Accuracy
if st.button("🔍 Evaluate Model Accuracy"):
    data = get_live_sensex()

    if data is None or data.empty:
        st.error("No live data available.")
    else:
        prepared_data = prepare_data(data)
        model, mae = train_model(prepared_data)
        
        if mae < 50:
            st.success(f"✅ Excellent Accuracy! Model MAE: {mae:.2f} points 🎯")
        elif mae < 80:
            st.warning(f"⚠️ Good Accuracy! Model MAE: {mae:.2f} points ✅")
        else:
            st.error(f"🚨 Needs Improvement! Model MAE: {mae:.2f} points ❌")

        
