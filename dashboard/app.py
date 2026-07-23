"""
Real-time Traffic Dashboard for MARL 2x2 Grid — Streamlit
Reads live metrics from logs/marl_metrics.json and historical progression from logs/marl_history.jsonl.
Provides real-time visualization of multi-intersection coordination, queue lengths, and waiting times.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import os

st.set_page_config(
    page_title="MARL 2x2 Grid Traffic Control Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark/vibrant aesthetics
st.markdown("""
<style>
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-badge-green {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-yellow {
        background-color: #F59E0B;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚦 Multi-Agent RL (2×2 Grid) — Real-Time Green Wave Dashboard")
st.markdown("Monitoring coordinated traffic light AI (`J00`, `J01`, `J10`, `J11`) executing **Independent Learners DQN** with cooperative observation signals.")

METRICS_FILE = os.path.join("logs", "marl_metrics.json")
HISTORY_FILE = os.path.join("logs", "marl_history.jsonl")
TL_IDS = ["J00", "J01", "J10", "J11"]

def load_data():
    latest = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                latest = json.load(f)
        except Exception:
            pass
            
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        step = data.get("__meta__", {}).get("step", 0)
                        row = {"Step": step}
                        for tid in TL_IDS:
                            if tid in data:
                                row[f"{tid}_Queue"] = data[tid].get("queue", 0)
                                row[f"{tid}_Wait"] = data[tid].get("waiting_time", 0)
                                row[f"{tid}_Phase"] = data[tid].get("phase", 0)
                        history.append(row)
        except Exception:
            pass
    return latest, pd.DataFrame(history)

# Sidebar controls
st.sidebar.header("⚙️ Dashboard Settings")
auto_refresh = st.sidebar.checkbox("🟢 Live Auto-Refresh (2s)", value=True)
refresh_btn = st.sidebar.button("🔄 Manual Refresh")

# Main content
latest, df_hist = load_data()

if not latest and df_hist.empty:
    st.info("⌛ No MARL training data found yet. Start training by running `python agent/train_marl.py` in your terminal to see live metrics appear here!")
else:
    meta = latest.get("__meta__", {})
    current_step = meta.get("step", 0)
    total_steps = meta.get("total_timesteps", 500000)
    
    # Top Progress Bar
    progress_pct = min(1.0, current_step / max(1, total_steps))
    st.sidebar.progress(progress_pct, text=f"Training Progress: {current_step:,} / {total_steps:,} steps")

    # Key KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    for idx, (col, tid) in enumerate(zip([col1, col2, col3, col4], TL_IDS)):
        with col:
            t_data = latest.get(tid, {})
            q = t_data.get("queue", 0.0)
            w = t_data.get("waiting_time", 0.0)
            p = t_data.get("phase", 0)
            
            p_label = "🟢 North-South Green" if p == 0 else ("🟡 Yellow" if p in [1, 3] else "🟢 East-West Green")
            st.metric(label=f"Intersection {tid} Queue", value=f"{q:.1f} veh", delta=f"{w:.1f}s wait")
            st.caption(f"Current Phase: **{p_label}** (Phase {p})")

    st.markdown("---")

    # Historical Charts
    if not df_hist.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📈 Real-Time Queue Length across Intersections")
            q_cols = [f"{tid}_Queue" for tid in TL_IDS if f"{tid}_Queue" in df_hist.columns]
            fig_q = px.line(
                df_hist, x="Step", y=q_cols,
                labels={"value": "Halting Vehicles (Queue)", "Step": "Training Timesteps", "variable": "Intersection"},
                color_discrete_sequence=["#38BDF8", "#34D399", "#FBBF24", "#F87171"]
            )
            fig_q.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
            st.plotly_chart(fig_q, use_container_width=True)
            
        with c2:
            st.subheader("⏱️ Total Waiting Time across Intersections")
            w_cols = [f"{tid}_Wait" for tid in TL_IDS if f"{tid}_Wait" in df_hist.columns]
            fig_w = px.line(
                df_hist, x="Step", y=w_cols,
                labels={"value": "Cumulative Waiting Time (s)", "Step": "Training Timesteps", "variable": "Intersection"},
                color_discrete_sequence=["#60A5FA", "#A7F3D0", "#FDE68A", "#FCA5A5"]
            )
            fig_w.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
            st.plotly_chart(fig_w, use_container_width=True)

        # Green Wave Grid Map Overview
        st.subheader("🗺️ 2×2 Grid Intersection Topology & Live Pressure")
        grid_cols = st.columns([1, 2, 2, 1])
        with grid_cols[1]:
            q00 = latest.get("J00", {}).get("queue", 0)
            q01 = latest.get("J01", {}).get("queue", 0)
            st.markdown(f"""
            <div style="background:#1E293B; padding:15px; border-radius:10px; text-align:center; border:1px solid #38BDF8;">
                <h4>North-West (J00)</h4>
                <p>Queue: <b>{q00:.1f}</b> | Phase: {latest.get('J00', {}).get('phase', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        with grid_cols[2]:
            st.markdown(f"""
            <div style="background:#1E293B; padding:15px; border-radius:10px; text-align:center; border:1px solid #34D399;">
                <h4>North-East (J01)</h4>
                <p>Queue: <b>{q01:.1f}</b> | Phase: {latest.get('J01', {}).get('phase', 0)}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        grid_cols_b = st.columns([1, 2, 2, 1])
        with grid_cols_b[1]:
            q10 = latest.get("J10", {}).get("queue", 0)
            st.markdown(f"""
            <div style="background:#1E293B; padding:15px; border-radius:10px; text-align:center; border:1px solid #FBBF24;">
                <h4>South-West (J10)</h4>
                <p>Queue: <b>{q10:.1f}</b> | Phase: {latest.get('J10', {}).get('phase', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        with grid_cols_b[2]:
            q11 = latest.get("J11", {}).get("queue", 0)
            st.markdown(f"""
            <div style="background:#1E293B; padding:15px; border-radius:10px; text-align:center; border:1px solid #F87171;">
                <h4>South-East (J11)</h4>
                <p>Queue: <b>{q11:.1f}</b> | Phase: {latest.get('J11', {}).get('phase', 0)}</p>
            </div>
            """, unsafe_allow_html=True)

if auto_refresh:
    time.sleep(2)
    st.rerun()
