import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# JavaScript for persistence
components.html("""
<script>
    const savedTrades = localStorage.getItem('qestc_trades');
    const savedCount = localStorage.getItem('qestc_trade_count');
    const savedPro = localStorage.getItem('qestc_pro');
    if (savedTrades) window.parent.postMessage({type: 'load_trades', data: savedTrades}, "*");
    if (savedCount) window.parent.postMessage({type: 'load_count', data: savedCount}, "*");
    if (savedPro) window.parent.postMessage({type: 'load_pro', data: savedPro}, "*");

    window.addEventListener('message', function(event) {
        if (event.data.type === 'save_trades') localStorage.setItem('qestc_trades', event.data.data);
        if (event.data.type === 'save_count') localStorage.setItem('qestc_trade_count', event.data.data);
        if (event.data.type === 'save_pro') localStorage.setItem('qestc_pro', event.data.data);
    });
</script>
""", height=0)

# Session state
if 'token_balance' not in st.session_state:
    st.session_state.token_balance = 0
if 'trade_count' not in st.session_state:
    st.session_state.trade_count = 0
if 'max_free_trades' not in st.session_state:
    st.session_state.max_free_trades = 5
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'is_pro' not in st.session_state:
    st.session_state.is_pro = False
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = "BTC"
if 'last_price_fetch' not in st.session_state:
    st.session_state.last_price_fetch = time.time() - 70
if 'current_prices' not in st.session_state:
    st.session_state.current_prices = {"BTC": 65000, "ETH": 1900, "SOL": 80}
if 'historical_prices' not in st.session_state:
    st.session_state.historical_prices = {}
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Fetch live prices
def fetch_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        prices = {
            "BTC": data["bitcoin"]["usd"],
            "ETH": data["ethereum"]["usd"],
            "SOL": data["solana"]["usd"]
        }
        st.session_state.current_prices = prices
        st.session_state.last_price_fetch = time.time()
        return prices
    except Exception as e:
        st.warning(f"Price fetch failed: {str(e)}. Using last known prices.")
        return st.session_state.current_prices

if time.time() - st.session_state.last_price_fetch > 60 or st.session_state.auto_refresh:
    fetch_prices()

# Fetch historical prices (30 days daily)
def fetch_historical_prices(asset):
    asset_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "ADA": "cardano"}
    coin_id = asset_map.get(asset, "bitcoin")
    key = f"{coin_id}_30d"
    if key in st.session_state.historical_prices and time.time() - st.session_state.historical_prices[key]['last_fetch'] < 3600:
        return st.session_state.historical_prices[key]['data']

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('timestamp', inplace=True)
        prices = prices.resample('D').last().dropna()
        st.session_state.historical_prices[key] = {'data': prices, 'last_fetch': time.time()}
        return prices
    except Exception as e:
        st.warning(f"Historical data fetch failed: {str(e)}. Using mock data.")
        dates = pd.date_range(end=datetime.datetime.now(), periods=30, freq='D')
        prices = pd.DataFrame({'price': np.random.uniform(60000, 70000, 30)}, index=dates)
        return prices

# Real linear regression prediction
def get_prediction(asset):
    prices = fetch_historical_prices(asset)
    if len(prices) < 10:
        return {"score": 50.0, "signal": "Hold", "reason": "Insufficient historical data", "metrics": {}, "chart_data": None}

    prices['time'] = np.arange(len(prices))
    X = prices['time'].values.reshape(-1, 1)
    y = prices['price'].values

    model = LinearRegression()
    model.fit(X, y)

    next_time = len(prices)
    predicted_price = model.predict([[next_time]])[0]
    current_price = y[-1]

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    r2 = r2_score(y, y_pred) * 100

    score = r2
    signal = "Buy" if predicted_price > current_price * 1.005 else "Sell" if predicted_price < current_price * 0.995 else "Hold"
    reason = f"Predicted next day: ${predicted_price:,.2f} (vs current ${current_price:,.2f}). Model fit: R² {r2:.1f}%"

    chart_df = prices.copy()
    chart_df['predicted'] = np.nan
    chart_df.loc[chart_df.index[-1], 'predicted'] = predicted_price

    return {
        "score": score,
        "signal": signal,
        "reason": reason,
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2},
        "chart_data": chart_df,
        "predicted_price": predicted_price
    }

# Mock simulate trade
def simulate_trade(asset, prediction):
    outcome = "Win" if np.random.rand() > 0.3 else "Loss"
    profit = np.random.uniform(-5, 15) if outcome == "Win" else np.random.uniform(-15, -5)
    return {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asset": asset,
        "signal": prediction["signal"],
        "outcome": outcome,
        "profit_pct": profit
    }

# Sidebar
st.set_page_config(page_title="QESTC Simulator", layout="wide")
st.sidebar.header("QESTC Controls")

st.session_state.selected_asset = st.sidebar.selectbox("Select Asset", ["BTC", "ETH", "SOL", "XRP", "ADA"])
token_purchase = st.sidebar.number_input("Buy CRYPT Tokens ($1 = 5 tokens)", min_value=0, step=1)
if st.sidebar.button("Purchase Tokens"):
    st.session_state.token_balance += token_purchase * 5
    st.sidebar.success(f"Added {token_purchase * 5} CRYPT tokens! Balance: {st.session_state.token_balance}")

st.sidebar.metric("CRYPT Balance", st.session_state.token_balance)
st.sidebar.metric("Free Trades Left", max(0, st.session_state.max_free_trades - st.session_state.trade_count))

st.sidebar.checkbox("Auto-refresh live prices", value=st.session_state.auto_refresh, key="auto_refresh")

# Pro Activation
if not st.session_state.is_pro:
    pro_key = st.sidebar.text_input("Pro Key (for unlimited)", type="password")
    if st.sidebar.button("Activate Pro"):
        if pro_key == "TESTKEY123":
            st.session_state.is_pro = True
            st.sidebar.success("Pro Activated! Unlimited simulations unlocked.")
        else:
            st.sidebar.error("Invalid key.")
else:
    st.sidebar.success("Pro Active – Unlimited Simulations")

# Main UI
st.title("QESTC Predictive Simulator")
st.markdown("**Simulation-only platform** – No real money traded. Test strategies risk-free. Real linear regression prediction model on 30-day history.")

with st.expander("Quick Start Guide", expanded=True):
    st.markdown("""
    1. Buy CRYPT tokens ($1 = 5 tokens) for extra simulations.
    2. Select an asset.
    3. View real prediction (based on 30-day daily history) → Run simulated trade.
    4. See results in ledger, model accuracy metrics, and price/profit charts.
    5. Enable auto-refresh for live price updates.
    """)

# Live Price Display
prices = fetch_prices()
selected_price = prices.get(st.session_state.selected_asset, 65000)
col1, col2, col3 = st.columns(3)
col1.metric(f"{st.session_state.selected_asset} Price", f"${selected_price:,.2f}", "Live from CoinGecko")

prediction = get_prediction(st.session_state.selected_asset)
col2.metric("Prediction Score (R²)", f"{prediction['score']:.1f}%")
col3.metric("Signal", prediction['signal'])

st.info(prediction['reason'])

# Model Metrics
with st.expander("Model Accuracy Metrics (on 30-day history)", expanded=False):
    metrics = prediction.get('metrics', {})
    if metrics:
        colm1, colm2, colm3, colm4 = st.columns(4)
        colm1.metric("MAE", f"${metrics['MAE']:,.2f}")
        colm2.metric("RMSE", f"${metrics['RMSE']:,.2f}")
        colm3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        colm4.metric("R²", f"{metrics['R2']:.1f}%")
    else:
        st.info("No metrics available yet.")

# Price History Chart
with st.expander("Price History & Prediction", expanded=True):
    chart_data = prediction.get('chart_data')
    if chart_data is not None and not chart_data.empty:
        fig = px.line(chart_data, x=chart_data.index, y='price', title=f"{st.session_state.selected_asset} 30-Day Price History")
        fig.add_scatter(x=[chart_data.index[-1]], y=[prediction.get('predicted_price', chart_data['price'].iloc[-1])],
                        mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Predicted Next Day')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical chart available.")

# Executor
if st.button("Run Simulated Trade"):
    if st.session_state.is_pro:
        pass
    elif st.session_state.trade_count < st.session_state.max_free_trades:
        st.session_state.trade_count += 1
    elif st.session_state.token_balance > 0:
        st.session_state.token_balance -= 1
    else:
        st.warning("No free trades or tokens left. Buy CRYPT or activate Pro.")
        st.stop()

    result = simulate_trade(st.session_state.selected_asset, prediction)
    st.session_state.trade_history.insert(0, result)
    st.success(f"Simulated: {result['outcome']} ({result['profit_pct']:+.2f}%)")

# Ledger
st.subheader("Trade Ledger")
if st.session_state.trade_history:
    df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(df.style.format({"profit_pct": "{:+.2f}%"}))
else:
    st.info("Run a simulated trade to see results.")

# Profit History Chart
with st.expander("Simulated Profit History", expanded=False):
    if st.session_state.trade_history:
        df = pd.DataFrame(st.session_state.trade_history)
        fig_profit = px.bar(df, x='time', y='profit_pct', color='outcome',
                            title="Profit/Loss per Simulated Trade",
                            labels={'profit_pct': 'Profit/Loss (%)'},
                            color_discrete_map={"Win": "green", "Loss": "red"})
        st.plotly_chart(fig_profit, use_container_width=True)
    else:
        st.info("No trades yet.")

# Cumulative Profit Chart
with st.expander("Cumulative Profit/Loss", expanded=False):
    if st.session_state.trade_history:
        df = pd.DataFrame(st.session_state.trade_history)
        df['cumulative_profit'] = df['profit_pct'].cumsum()
        fig_cum = px.line(df, x='time', y='cumulative_profit', title="Running Total Profit/Loss")
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info("No trades yet.")

# Trade Stats Summary
with st.expander("Trade Statistics", expanded=False):
    if st.session_state.trade_history:
        df = pd.DataFrame(st.session_state.trade_history)
        total_trades = len(df)
        wins = len(df[df['outcome'] == 'Win'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_profit = df['profit_pct'].mean()
        total_pnl = df['profit_pct'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Avg Profit per Trade", f"{avg_profit:+.2f}%")
        col4.metric("Total PnL", f"{total_pnl:+.2f}%")
    else:
        st.info("No trades yet.")

st.caption("NOT FINANCIAL ADVICE. Simulation tool only. © 2026 Paul de Bruyn.")
