import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import streamlit.components.v1 as components
import requests
import requests
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

if time.time() - st.session_state.last_price_fetch > 60:
    fetch_prices()

# Mock prediction
def get_prediction(asset):
    score = np.random.uniform(70, 99)
    signal = "Buy" if np.random.rand() > 0.5 else "Sell"
    return {"score": score / 100, "signal": signal, "reason": f"Mock: {asset} trend"}

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
st.markdown("**Simulation-only platform** – No real money traded. Test strategies risk-free. Prices live from CoinGecko.")

with st.expander("Quick Start Guide", expanded=True):
    st.markdown("""
    1. Buy CRYPT tokens ($1 = 5 tokens) for extra simulations.
    2. Select an asset.
    3. View live prediction → Run simulated trade.
    4. See results in ledger.
    """)

# Live Price Display
prices = fetch_prices()
selected_price = prices.get(st.session_state.selected_asset, 65000)
col1, col2, col3 = st.columns(3)
col1.metric(f"{st.session_state.selected_asset} Price", f"${selected_price:,.2f}", "Live from CoinGecko")
prediction = get_prediction(st.session_state.selected_asset)
col2.metric("Prediction Score", f"{prediction['score']:.1%}")
selected_price = prices.get(st.session_state.selected_asset, 65000)
col1, col2, col3 = st.columns(3)
col3.metric("Signal", prediction['signal'])

# Executor - FREE TRADES DEPLETES FIRST
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

st.caption("NOT FINANCIAL ADVICE. Simulation tool only. © 2026 Paul de Bruyn.")
