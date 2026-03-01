# app.py - Cryptonomer SEO Optimizer (updated March 2026 version)

import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import serpapi
from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, DateRange, Metric, Dimension
)

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────

st.set_page_config(
    page_title="Cryptonomer SEO Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# LOAD SECRETS SAFELY
# ────────────────────────────────────────────────

def get_secret(key_name, fallback=None):
    try:
        return st.secrets[key_name]
    except (KeyError, TypeError):
        try:
            # fallback for nested structure
            return st.secrets["secrets"][key_name]
        except (KeyError, TypeError):
            return os.getenv(key_name, fallback)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SERPAPI_KEY    = get_secret("SERPAPI_API_KEY")
GA4_PROPERTY_ID = get_secret("GA4_PROPERTY_ID")

# ────────────────────────────────────────────────
# CLIENT INITIALIZATION WITH ERROR HANDLING
# ────────────────────────────────────────────────

openai_client = None
ga_client = None

if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"OpenAI client initialization failed: {str(e)}")
else:
    st.warning("OpenAI API key missing or invalid format → GPT features disabled")

if GA4_PROPERTY_ID:
    try:
        creds_info = st.secrets.get("gcp_service_account", {})
        if creds_info:
            credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/analytics.readonly"]
            )
            ga_client = BetaAnalyticsDataClient(credentials=credentials)
        else:
            st.warning("gcp_service_account not found in secrets → GA4 disabled")
    except Exception as e:
        st.error(f"GA4 client failed: {str(e)}")
else:
    st.info("GA4 property ID not set → GA4 features disabled")

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────

with st.sidebar:
    st.title("Cryptonomer SEO Tool")
    st.markdown("**Version:** Updated 2026-03 • Stable")

    keywords_text = st.text_area(
        "Keywords (one per line)",
        value="blockchain decoding\ncrypto trading strategies\nelliptic curve cryptography\nmodular arithmetic crypto",
        height=140
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    analyze_btn = st.button("Run Full Analysis", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# MAIN CONTENT
# ────────────────────────────────────────────────

st.title("SEO & Content Optimizer")
st.caption("Crypto-focused keyword analysis, SERP snippets, clustering & math learning demo")

if analyze_btn:
    keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]
    if not keywords:
        st.error("Please enter at least one keyword.")
        st.stop()

    with st.spinner("Processing analysis (this may take 10–40 seconds)..."):

        tabs = st.tabs(["GA4", "SERP & Snippets", "Clusters", "Math Learning"])

        # ── Tab 1: GA4 ────────────────────────────────────────
        with tabs[0]:
            if ga_client and GA4_PROPERTY_ID:
                try:
                    request = RunReportRequest(
                        property=f'properties/{GA4_PROPERTY_ID}',
                        date_ranges=[DateRange(
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d')
                        )],
                        dimensions=[Dimension(name="pagePath")],
                        metrics=[Metric(name="activeUsers")]
                    )
                    response = ga_client.run_report(request=request)
                    rows = []
                    for row in response.rows:
                        rows.append({
                            "page": row.dimension_values[0].value,
                            "users": int(row.metric_values[0].value)
                        })
                    df_ga = pd.DataFrame(rows)
                    if not df_ga.empty:
                        st.subheader("Top Pages (by active users)")
                        st.dataframe(df_ga.sort_values("users", ascending=False).head(10))
                    else:
                        st.info("No data returned for selected date range.")
                except Exception as e:
                    st.error(f"GA4 query failed: {str(e)}")
            else:
                st.info("GA4 not configured.")

        # ── Tab 2: SERP & Snippets ─────────────────────────────
        with tabs[1]:
            if SERPAPI_KEY and openai_client:
                serp_data = {}
                for kw in keywords[:5]:  # limit to avoid quota burn
                    try:
                        params = {"q": kw, "api_key": SERPAPI_KEY, "num": 4}
                        results = serpapi.search(params).get("organic_results", [])
                        serp_data[kw] = [
                            {"title": r.get("title", ""), "snippet": r.get("snippet", "")}
                            for r in results
                        ]
                    except Exception as e:
                        st.warning(f"SERP failed for '{kw}': {str(e)}")

                if serp_data:
                    st.subheader("Current Top SERP Snippets")
                    for kw, items in serp_data.items():
                        with st.expander(f"Keyword: {kw}"):
                            for i, item in enumerate(items, 1):
                                st.markdown(f"**{i}. {item['title']}**  \n{item['snippet']}")

                    # GPT optimization
                    try:
                        prompt = (
                            "You are a senior SEO copywriter for crypto content. "
                            "Optimize these SERP snippets for higher CTR while keeping them natural "
                            "and keyword-rich:\n\n" + json.dumps(serp_data, indent=2)
                        )
                        resp = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.65,
                            max_tokens=900
                        )
                        st.subheader("GPT-Optimized Snippet Suggestions")
                        st.markdown(resp.choices[0].message.content)
                    except Exception as e:
                        st.error(f"GPT snippet optimization failed: {str(e)}")
            else:
                st.info("SERPAPI_KEY or OpenAI key missing → SERP analysis skipped.")

        # ── Tab 3: Keyword Clusters ────────────────────────────
        with tabs[2]:
            if len(keywords) >= 2:
                try:
                    # Very simple clustering on dummy rank simulation
                    ranks = {kw: i + 1 for i, kw in enumerate(keywords)}
                    df = pd.DataFrame({"keyword": keywords, "mock_rank": list(ranks.values())})
                    kmeans = KMeans(n_clusters=min(3, len(keywords)), n_init=10)
                    df["cluster"] = kmeans.fit_predict(df[["mock_rank"]])
                    st.subheader("Keyword Clustering (simple)")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Clustering failed: {str(e)}")
            else:
                st.info("Need ≥ 2 keywords for clustering demo.")

        # ── Tab 4: Continual Learning Demo (Math / Crypto relevant) ──
        with tabs[3]:
            st.subheader("Simple Continual Learning – Modular Arithmetic Example")

            class TinyNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(2, 1)

                def forward(self, x):
                    return self.fc(x)

            model = TinyNet()
            optimizer = optim.Adam(model.parameters(), lr=0.02)
            loss_fn = nn.MSELoss()

            # Mock crypto-mod data: a % b → remainder
            x = torch.tensor([[7.0, 3.0], [13.0, 5.0], [23.0, 7.0]], dtype=torch.float32)
            y = torch.tensor([[1.0], [3.0], [2.0]], dtype=torch.float32)  # 7%3=1, 13%5=3, 23%7=2

            losses = []
            for _ in range(180):
                pred = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            final_loss = losses[-1] if losses else 999
