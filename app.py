import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import simpy, random
from sklearn.ensemble import RandomForestRegressor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Water Digital Twin",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    /* ── Water wave background ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f0f8ff;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%2390e0ef' fill-opacity='0.3' d='M0,160L48,176C96,192,192,224,288,218.7C384,213,480,171,576,165.3C672,160,768,192,864,197.3C960,203,1056,181,1152,165.3C1248,149,1344,139,1392,133.3L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E"),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%2300b4d8' fill-opacity='0.15' d='M0,96L48,112C96,128,192,160,288,160C384,160,480,128,576,122.7C672,117,768,139,864,154.7C960,171,1056,181,1152,170.7C1248,160,1344,128,1392,112L1440,96L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: bottom;
        background-size: cover;
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #e0f7fa 100%);
        border-right: 2px solid #90e0ef;
        box-shadow: 4px 0 20px rgba(0,180,216,0.1);
    }

    /* ── Alert boxes ── */
    .alert-danger {
        background: linear-gradient(135deg, rgba(255,82,82,0.1), rgba(255,50,50,0.05));
        border: 1px solid rgba(255,82,82,0.5);
        border-left: 5px solid #ff5252;
        border-radius: 12px;
        padding: 16px 20px;
        color: #c62828;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .alert-success {
        background: linear-gradient(135deg, rgba(0,200,83,0.1), rgba(0,180,60,0.05));
        border: 1px solid rgba(0,200,83,0.5);
        border-left: 5px solid #00c853;
        border-radius: 12px;
        padding: 16px 20px;
        color: #1b5e20;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .alert-warning {
        background: linear-gradient(135deg, rgba(255,171,0,0.12), rgba(255,150,0,0.05));
        border: 1px solid rgba(255,171,0,0.5);
        border-left: 5px solid #ffab00;
        border-radius: 12px;
        padding: 16px 20px;
        color: #e65100;
        font-weight: 600;
        font-size: 1.05rem;
    }

    /* ── Hero title ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0077b6, #00b4d8, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-shadow: none;
    }
    .hero-sub {
        color: #5b8fa8;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* ── Section headers ── */
    .section-header {
        color: #0077b6;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 2px solid #90e0ef;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #caf0f8;
        border-top: 4px solid #00b4d8;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(0,180,216,0.12);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); }
    div[data-testid="stMetric"] label { color: #5b8fa8 !important; font-weight: 600; }
    div[data-testid="stMetric"] div   { color: #0077b6 !important; font-weight: 700; }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 18px rgba(0,119,182,0.35);
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #023e8a, #0077b6);
        box-shadow: 0 6px 24px rgba(0,119,182,0.5);
        transform: translateY(-2px);
    }

    /* ── Sidebar labels ── */
    .stSelectbox label, .stSlider label { color: #0077b6 !important; font-weight: 600; }

    /* ── Sidebar title ── */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #023e8a; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,180,216,0.1); }
    thead tr th { background-color: #e0f7fa !important; color: #0077b6 !important; font-weight: 700 !important; }

    /* ── Water droplet bubbles (decorative) ── */
    .water-bubble {
        position: fixed;
        border-radius: 50%;
        background: rgba(0,180,216,0.08);
        animation: float 8s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    /* ── General text ── */
    p, span, div { color: #1a3a4a; }

    /* ── Hide branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>

<!-- Decorative water bubbles -->
<div class="water-bubble" style="width:180px;height:180px;top:10%;left:5%;animation-delay:0s;"></div>
<div class="water-bubble" style="width:100px;height:100px;top:40%;left:90%;animation-delay:2s;"></div>
<div class="water-bubble" style="width:60px;height:60px;top:70%;left:15%;animation-delay:4s;"></div>
<div class="water-bubble" style="width:130px;height:130px;top:20%;left:75%;animation-delay:1s;"></div>
""", unsafe_allow_html=True)


# ── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # FIX 1: Robust path resolution — works from any directory / any machine
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "water_dataset_ml_ready.csv"),
        os.path.join(script_dir, "water_dataset_ml_ready (2).csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        st.error(
            "❌ CSV file not found. Place `water_dataset_ml_ready.csv` in the "
            "same folder as app.py and restart."
        )
        st.stop()

    # FIX 2: Normalise ward names — strip whitespace, fix capitalisation issues
    df["Ward_Name"] = df["Ward_Name"].str.strip()
    df["Ward_Name"] = df["Ward_Name"].str[0].str.upper() + df["Ward_Name"].str[1:]

    # FIX 3: Merge known duplicate / typo ward names into canonical spellings
    WARD_ALIASES = {
        "Singsandra":  "Singasandra",   # typo duplicate
        "Jayangar":    "Jayanagar",     # typo duplicate
    }
    df["Ward_Name"] = df["Ward_Name"].replace(WARD_ALIASES)

    # FIX 4: Parse dates properly (DD-MM-YYYY or YYYY-MM-DD → datetime)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    return df


features = [
    'Ward_Number', 'Month', 'Quarter', 'Population',
    'Connections', 'Connection_Density',
    'Consumption', 'Supply',
    'Per_Capita_Consumption', 'Supply_Per_Connection',
    'Supply_Demand_Ratio'
]


# ── Water Redistribution Engine ────────────────────────────────────────────────
def compute_redistribution(df):
    """
    Identifies today's surplus and deficit wards (using latest available data
    per ward) and computes an optimal water diversion plan.

    Surplus ward  : Supply > Demand  →  has water to give
    Deficit ward  : Supply < Demand  →  needs extra water

    Returns:
        surplus_df   – wards with available surplus (MLD)
        deficit_df   – wards with unmet demand (MLD)
        diversion_df – matched diversion recommendations
    """
    # FIX 5: Use proper datetime sort (date column is now parsed to datetime)
    latest = (
        df.sort_values("Date", ascending=False)
        .groupby("Ward_Name")
        .first()
        .reset_index()
    )

    latest["Surplus_MLD"] = (latest["Supply"] - latest["Consumption"]).clip(lower=0)
    latest["Deficit_MLD"] = (latest["Consumption"] - latest["Supply"]).clip(lower=0)

    surplus_wards = (
        latest[latest["Surplus_MLD"] > 0][
            ["Ward_Name", "Supply", "Consumption", "Surplus_MLD", "Supply_Demand_Ratio"]
        ]
        .sort_values("Surplus_MLD", ascending=False)
        .reset_index(drop=True)
    )

    deficit_wards = (
        latest[latest["Deficit_MLD"] > 0][
            ["Ward_Name", "Supply", "Consumption", "Deficit_MLD", "Supply_Demand_Ratio"]
        ]
        .sort_values("Deficit_MLD", ascending=False)
        .reset_index(drop=True)
    )

    # Greedy matching: assign surplus to deficit in descending order of severity
    diversions = []
    remaining_surplus = surplus_wards["Surplus_MLD"].tolist()
    surplus_names     = surplus_wards["Ward_Name"].tolist()

    for _, def_row in deficit_wards.iterrows():
        needed   = def_row["Deficit_MLD"]
        assigned = 0.0
        sources  = []

        for i, sur_name in enumerate(surplus_names):
            if needed <= 0:
                break
            available = remaining_surplus[i]
            if available <= 0:
                continue
            give = min(available, needed)
            sources.append(f"{sur_name} ({give:.1f} MLD)")
            remaining_surplus[i] -= give
            assigned += give
            needed   -= give

        if assigned > 0:
            diversions.append({
                "Deficit Ward"        : def_row["Ward_Name"],
                "Shortfall (MLD)"     : round(def_row["Deficit_MLD"], 2),
                "Diverted (MLD)"      : round(assigned, 2),
                "Remaining Gap (MLD)" : round(def_row["Deficit_MLD"] - assigned, 2),
                "Source Wards"        : " | ".join(sources),
                "Coverage (%)"        : round(assigned / def_row["Deficit_MLD"] * 100, 1),
            })

    diversion_df = pd.DataFrame(diversions) if diversions else pd.DataFrame()

    # Update surplus table with remaining after diversion
    surplus_wards["After Diversion (MLD)"] = [round(r, 2) for r in remaining_surplus]

    return surplus_wards, deficit_wards, diversion_df


@st.cache_resource
def train_model(df):
    # FIX 12: Match notebook — filter anomalies before training so both
    # environments produce identical predictions.
    # Anomaly = Consumption exceeds mean + 2 × std (same threshold as notebook).
    threshold = df["Consumption"].mean() + 2 * df["Consumption"].std()
    df_normal = df[df["Consumption"] <= threshold].copy()
    X = df_normal[features]
    y = df_normal["Demand_Gap"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


df    = load_data()
model = train_model(df)


# ── WNTR Hydraulic Simulation ──────────────────────────────────────────────────
def run_wntr_simulation(ward_name, supply_mld, demand_mld):
    import wntr

    # FIX 6: Clean ward name for safe node naming (remove ALL special chars)
    safe_name = re.sub(r"[^A-Za-z0-9]", "_", ward_name)[:10]
    demand_m3s = demand_mld * 0.01157

    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("BWSSB_Source",         base_head=50.0,    coordinates=(0, 0))
    wn.add_junction(safe_name + "_E",  base_demand=demand_m3s * 0.30, elevation=10.0, coordinates=(100,  0))
    wn.add_junction(safe_name + "_Z1", base_demand=demand_m3s * 0.35, elevation=8.0,  coordinates=(200, 100))
    wn.add_junction(safe_name + "_Z2", base_demand=demand_m3s * 0.35, elevation=8.0,  coordinates=(200,-100))
    wn.add_pipe("P1", "BWSSB_Source",   safe_name + "_E",  length=1000, diameter=0.5, roughness=100)
    wn.add_pipe("P2", safe_name + "_E", safe_name + "_Z1", length=500,  diameter=0.3, roughness=100)
    wn.add_pipe("P3", safe_name + "_E", safe_name + "_Z2", length=500,  diameter=0.3, roughness=100)
    wn.options.time.duration          = 3600
    wn.options.time.hydraulic_timestep = 3600

    try:
        sim     = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()
        pressure_df = results.node["pressure"]
        flowrate_df = results.link["flowrate"]
        avg_pressure = float(abs(pressure_df.iloc[:, 1:]).mean().mean())
        avg_pressure = max(5.0, min(avg_pressure, 60.0))
        avg_flow_mld = float(abs(flowrate_df).mean().mean()) / 0.01157
    except Exception:
        ratio        = supply_mld / demand_mld if demand_mld > 0 else 1
        avg_pressure = round(ratio * 30, 2)
        avg_flow_mld = round(supply_mld * 0.85, 2)

    return {
        "pressure"           : round(avg_pressure, 2),
        "flow_mld"           : round(avg_flow_mld, 2),
        "demand_satisfaction": round(min(supply_mld / demand_mld, 1.0) * 100, 1) if demand_mld > 0 else 100,
        "shortage_mld"       : round(max(demand_mld - supply_mld, 0), 2),
        "status"             : "Balanced" if supply_mld >= demand_mld else "Shortage",
    }


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.6rem">💧 Controls</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">Ward Selection</p>', unsafe_allow_html=True)

    # FIX 7: Sort ward list alphabetically so all wards (incl. Herohalli, Sidedahalli) appear
    all_wards = sorted(df["Ward_Name"].unique())
    zone = st.selectbox("Select Ward", all_wards, label_visibility="collapsed")

    # FIX 8: Safe row lookup — use first available record, never crash on empty
    ward_rows = df[df["Ward_Name"] == zone]
    if ward_rows.empty:
        st.error(f"No data found for ward: {zone}")
        st.stop()
    row = ward_rows.iloc[0].copy()

    st.markdown("---")
    st.markdown('<p class="section-header">Simulation Parameters</p>', unsafe_allow_html=True)

    supply_change = st.slider(
        "📈 Supply Change (MLD)",
        min_value=-50, max_value=50, value=0, step=1,
        help="Adjust the water supply in million litres per day"
    )
    demand_change = st.slider(
        "📉 Demand Change (MLD)",
        min_value=-50, max_value=50, value=0, step=1,
        help="Adjust the water demand/consumption in million litres per day"
    )

    st.markdown("---")
    run        = st.button("🚀 Run What-If Simulation")
    run_simpy  = st.button("⚙️ Run 24-Hour SimPy Twin")
    run_wntr   = st.button("🔧 Run WNTR Hydraulic Sim")
    run_redist = st.button("🔀 Run Water Redistribution")

    st.markdown("---")
    st.markdown("""
            <div style="color:#5b8fa8; font-size:0.8rem; line-height:1.6">
            <b style="color:#0077b6">About</b><br>
            This digital twin uses four simulators:<br>
            <b>1. What-If Engine</b> — ML-powered scenario simulation<br>
            <b>2. SimPy</b> — Discrete event 24-hour water system simulation<br>
            <b>3. WNTR</b> — Hydraulic simulation for pressure, flow, and infrastructure analysis<br>
            <b>4. Redistribution Engine</b> — Identifies surplus wards and diverts water to deficit wards<br>
            </div>
            """, unsafe_allow_html=True)


# ── MAIN CONTENT ───────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div style="padding: 10px 0 20px 0">
    <p class="hero-title">💧 Urban Water Digital Twin</p>
    <p class="hero-sub">AI-powered water supply simulation · Bengaluru Ward Analytics</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🏘️ Ward", zone)
with col2:
    st.metric("👥 Population", f"{int(row['Population']):,}")
with col3:
    st.metric("🔌 Connections", f"{int(row['Connections']):,}")
with col4:
    ratio = row["Supply_Demand_Ratio"]
    st.metric("⚖️ Supply/Demand Ratio", f"{ratio:.2f}", delta=f"{'Surplus' if ratio >= 1 else 'Deficit'}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Two column layout ──
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<p class="section-header">📊 Ward Overview</p>', unsafe_allow_html=True)

    # FIX 9: Sort by Month number so Jan→Dec order is always correct
    ward_data = (
        df[df["Ward_Name"] == zone]
        .groupby("Month")[["Supply", "Consumption"]]
        .mean()
        .reset_index()
        .sort_values("Month")          # ← crucial fix
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    ward_data["Month_Name"] = ward_data["Month"].map(month_names)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ward_data["Month_Name"], y=ward_data["Supply"],
        name="Supply", marker_color="#00b4d8",
        marker_line_color="#64ffda", marker_line_width=0.5
    ))
    fig.add_trace(go.Bar(
        x=ward_data["Month_Name"], y=ward_data["Consumption"],
        name="Demand", marker_color="#ef476f",
        marker_line_color="#ff8fa3", marker_line_width=0.5
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="rgba(255,255,255,0.7)",
        paper_bgcolor="rgba(255,255,255,0)",
        font_color="#1a3a4a",
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#1a3a4a"),
        xaxis=dict(gridcolor="#caf0f8", tickfont_color="#5b8fa8",
                   categoryorder="array",                  # ← keep Jan-Dec order
                   categoryarray=list(month_names.values())),
        yaxis=dict(gridcolor="#caf0f8", tickfont_color="#5b8fa8", title="MLD"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280
    )
    # FIX 10: use_container_width instead of invalid width="stretch"
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<p class="section-header">🗺️ Imbalance by Season</p>', unsafe_allow_html=True)

    season_data = (
        df[df["Ward_Name"] == zone]
        .groupby("Season")["Imbalance_Score"]
        .mean()
        .reset_index()
    )
    colors = {"Winter": "#00b4d8", "Summer": "#ef476f", "Monsoon": "#06d6a0", "Post-Monsoon": "#ffd166"}
    fig2 = go.Figure(go.Bar(
        x=season_data["Season"],
        y=season_data["Imbalance_Score"],
        marker_color=[colors.get(s, "#64ffda") for s in season_data["Season"]],
        text=season_data["Imbalance_Score"].round(2),
        textposition="outside",
        textfont_color="#1a3a4a"
    ))
    fig2.update_layout(
        plot_bgcolor="rgba(255,255,255,0.7)",
        paper_bgcolor="rgba(255,255,255,0)",
        font_color="#1a3a4a",
        xaxis=dict(gridcolor="#caf0f8", tickfont_color="#5b8fa8"),
        yaxis=dict(gridcolor="#caf0f8", tickfont_color="#5b8fa8", title="Avg Imbalance Score"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Simulation Result ──────────────────────────────────────────────────────────
if run:
    new_supply = row["Supply"] + supply_change
    new_demand = row["Consumption"] + demand_change

    sim = row.copy()
    sim["Supply"]                 = new_supply
    sim["Consumption"]            = new_demand
    sim["Per_Capita_Consumption"] = new_demand / row["Population"]   if row["Population"]  > 0 else 0
    sim["Supply_Per_Connection"]  = new_supply / row["Connections"]  if row["Connections"] > 0 else 0
    sim["Supply_Demand_Ratio"]    = new_supply / new_demand          if new_demand != 0    else 0

    input_df   = pd.DataFrame([sim[features]])
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.markdown('<p class="section-header">🔬 Simulation Results</p>', unsafe_allow_html=True)

    st.markdown("### 📊 Before vs After Comparison")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("💧 Original Supply", f"{row['Supply']:.1f} MLD")
    with c2:
        st.metric("💧 New Supply", f"{new_supply:.1f} MLD")

    c3, c4 = st.columns(2)
    with c3:
        st.metric("🔄 Original Demand", f"{row['Consumption']:.1f} MLD")
    with c4:
        st.metric("🔄 New Demand", f"{new_demand:.1f} MLD")

    st.markdown("")

    r1, r2, r3, r4 = st.columns(4)
    new_ratio = new_supply / new_demand if new_demand != 0 else 0
    with r1:
        st.metric("💧 New Supply",      f"{new_supply:.1f} MLD",  delta=f"{supply_change:+.0f} MLD")
    with r2:
        st.metric("🔄 New Demand",      f"{new_demand:.1f} MLD",  delta=f"{demand_change:+.0f} MLD")
    with r3:
        st.metric("⚖️ New S/D Ratio",   f"{new_ratio:.2f}",       delta=f"{new_ratio - ratio:.2f}")
    with r4:
        st.metric("🤖 Predicted Gap",   f"{prediction:.2f} MLD")

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction > 10:
        st.markdown(f'<div class="alert-danger">🚨 Critical Water Shortage Predicted &nbsp;|&nbsp; Demand Gap: <b>{prediction:.2f} MLD</b> — Immediate intervention required.</div>', unsafe_allow_html=True)
    elif prediction > 0:
        st.markdown(f'<div class="alert-warning">⚠️ Mild Water Shortage Predicted &nbsp;|&nbsp; Demand Gap: <b>{prediction:.2f} MLD</b> — Monitor closely.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-success">✅ Supply is Balanced &nbsp;|&nbsp; Demand Gap: <b>{prediction:.2f} MLD</b> — No shortage predicted.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_ratio,
            delta={"reference": 1.0, "valueformat": ".2f"},
            title={"text": "Supply / Demand Ratio", "font": {"color": "#1a3a4a", "size": 14}},
            number={"font": {"color": "#64ffda", "size": 36}},
            gauge={
                "axis": {"range": [0, 2], "tickcolor": "#5b8fa8", "tickfont": {"color": "#5b8fa8"}},
                "bar": {"color": "#00b4d8"},
                "bgcolor": "rgba(255,255,255,0.5)",
                "bordercolor": "#caf0f8",
                "steps": [
                    {"range": [0,    0.75], "color": "rgba(239,71,111,0.3)"},
                    {"range": [0.75, 1.0],  "color": "rgba(255,209,102,0.3)"},
                    {"range": [1.0,  2.0],  "color": "rgba(6,214,160,0.3)"},
                ],
                "threshold": {"line": {"color": "#64ffda", "width": 3}, "value": 1.0},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            font_color="#1a3a4a",
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g2:
        categories  = ["Supply (MLD)", "Demand (MLD)", "S/D Ratio × 50"]
        before_vals = [row["Supply"], row["Consumption"], ratio * 50]
        after_vals  = [new_supply,    new_demand,         new_ratio * 50]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Before", x=categories, y=before_vals,
                                   marker_color="rgba(100,255,218,0.5)",
                                   marker_line_color="#64ffda", marker_line_width=1))
        fig_comp.add_trace(go.Bar(name="After", x=categories, y=after_vals,
                                   marker_color="rgba(0,180,216,0.7)",
                                   marker_line_color="#00b4d8", marker_line_width=1))
        fig_comp.update_layout(
            barmode="group",
            title={"text": "Before vs After Simulation", "font": {"color": "#1a3a4a", "size": 14}},
            plot_bgcolor="rgba(255,255,255,0.7)",
            paper_bgcolor="rgba(255,255,255,0)",
            font_color="#1a3a4a",
            legend=dict(bgcolor="rgba(255,255,255,0.8)"),
            xaxis=dict(gridcolor="#caf0f8", tickfont_color="#5b8fa8"),
            yaxis=dict(gridcolor="#caf0f8", tickfont_color="#5b8fa8"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=260
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ── SimPy 24-hour Simulation ───────────────────────────────────────────────────
def run_simpy_twin(ward_row, hours=24, supply_change=0, demand_change=0):
    """SimPy Discrete Event Simulator — runs water system hour by hour."""
    log = []
    base_supply = ward_row["Supply"]      + supply_change
    base_demand = ward_row["Consumption"] + demand_change

    def water_process(env):
        while True:
            supply = base_supply + random.uniform(-10, 10)
            demand = base_demand + random.uniform(-20, 20)
            sim = ward_row.copy()
            sim["Supply"]                 = supply
            sim["Consumption"]            = demand
            sim["Per_Capita_Consumption"] = demand / ward_row["Population"]  if ward_row["Population"]  > 0 else 0
            sim["Supply_Per_Connection"]  = supply / ward_row["Connections"] if ward_row["Connections"] > 0 else 0
            sim["Supply_Demand_Ratio"]    = supply / demand                  if demand                  > 0 else 0
            gap = model.predict(pd.DataFrame([sim[features]], columns=features))[0]
            log.append({"Hour": env.now, "Supply": round(supply, 2),
                        "Demand": round(demand, 2), "Gap": round(gap, 2)})
            yield env.timeout(1)

    env = simpy.Environment()
    env.process(water_process(env))
    env.run(until=hours)
    return pd.DataFrame(log)


if run_simpy:
    st.markdown("---")
    st.markdown('<p class="section-header">⚙️ SimPy Digital Twin — 24-Hour Simulation</p>', unsafe_allow_html=True)
    st.info(f"Running 24-hour discrete event simulation for **{zone}**...")

    sim_df = run_simpy_twin(row, hours=24, supply_change=supply_change, demand_change=demand_change)

    s1, s2, s3 = st.columns(3)
    s1.metric("🚨 Critical Hours", int((sim_df["Gap"] > 20).sum()))
    s2.metric("⚠️ Warning Hours",  int(((sim_df["Gap"] > 10) & (sim_df["Gap"] <= 20)).sum()))
    s3.metric("✅ Normal Hours",   int((sim_df["Gap"] <= 10).sum()))

    st.markdown("<br>", unsafe_allow_html=True)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=sim_df["Hour"], y=sim_df["Supply"],  name="Supply",
                                line=dict(color="#0077b6", width=2), mode="lines+markers"))
    fig_s.add_trace(go.Scatter(x=sim_df["Hour"], y=sim_df["Demand"],  name="Demand",
                                line=dict(color="#ef476f", width=2), mode="lines+markers"))
    fig_s.add_trace(go.Scatter(x=sim_df["Hour"], y=sim_df["Gap"],     name="Predicted Gap",
                                line=dict(color="#ffd166", width=2, dash="dot"), mode="lines"))
    fig_s.add_hline(y=20, line_dash="dash", line_color="red",    annotation_text="Critical (20 MLD)")
    fig_s.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Warning (10 MLD)")
    fig_s.update_layout(
        title=f"24-Hour SimPy Simulation — {zone}",
        plot_bgcolor="rgba(255,255,255,0.7)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#1a3a4a", height=380, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(title="Hour", gridcolor="#caf0f8"),
        yaxis=dict(title="MLD",  gridcolor="#caf0f8")
    )
    st.plotly_chart(fig_s, use_container_width=True)

    with st.expander("📋 View 24-Hour Simulation Log"):
        st.dataframe(sim_df, use_container_width=True)


# ── WNTR Hydraulic Simulation ──────────────────────────────────────────────────
if run_wntr:
    st.markdown("---")
    st.markdown('<p class="section-header">🔧 WNTR Hydraulic Simulation</p>', unsafe_allow_html=True)
    st.info(f"Running hydraulic simulation for **{zone}**...")

    try:
        wr = run_wntr_simulation(zone, row["Supply"] + supply_change, row["Consumption"] + demand_change)

        w1, w2, w3, w4 = st.columns(4)
        w1.metric("💧 Avg Pressure",       f"{wr['pressure']} m")
        w2.metric("🌊 Simulated Flow",      f"{wr['flow_mld']} MLD")
        w3.metric("✅ Demand Satisfaction", f"{wr['demand_satisfaction']}%")
        w4.metric("📊 Physical Shortage",   f"{wr['shortage_mld']} MLD")

        st.markdown("<br>", unsafe_allow_html=True)

        fig_w = go.Figure(go.Indicator(
            mode="gauge+number",
            value=wr["pressure"],
            title={"text": "Water Pressure at Ward (metres)", "font": {"color": "#1a3a4a", "size": 14}},
            number={"font": {"color": "#0077b6", "size": 36}, "suffix": " m"},
            gauge={
                "axis": {"range": [0, 60]},
                "bar":  {"color": "#00b4d8"},
                "bgcolor": "rgba(255,255,255,0.5)",
                "steps": [
                    {"range": [0,  15], "color": "rgba(239,71,111,0.3)"},
                    {"range": [15, 30], "color": "rgba(255,209,102,0.3)"},
                    {"range": [30, 60], "color": "rgba(6,214,160,0.3)"},
                ],
                "threshold": {"line": {"color": "#0077b6", "width": 3}, "value": 30},
            }
        ))
        fig_w.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#1a3a4a",
                             height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_w, use_container_width=True)

        if wr["status"] == "Balanced":
            st.markdown(f'<div class="alert-success">✅ WNTR Hydraulic Status: Balanced — Demand fully satisfied at {wr["demand_satisfaction"]}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-danger">🚨 WNTR Hydraulic Status: Shortage — {wr["shortage_mld"]} MLD physical deficit detected</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"WNTR simulation error: {e}. Make sure wntr is installed: pip install wntr")


# ── Water Redistribution Engine ───────────────────────────────────────────────
if run_redist:
    st.markdown("---")
    st.markdown('<p class="section-header">🔀 Water Redistribution — Surplus to Deficit Diversion</p>', unsafe_allow_html=True)
    st.info("Analysing all wards to identify surplus and deficit zones and computing an optimal diversion plan...")

    surplus_df, deficit_df, diversion_df = compute_redistribution(df)

    total_surplus  = surplus_df["Surplus_MLD"].sum()
    total_deficit  = deficit_df["Deficit_MLD"].sum()
    total_diverted = diversion_df["Diverted (MLD)"].sum() if not diversion_df.empty else 0
    gap_after      = total_deficit - total_diverted

    # ── Summary KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💚 Surplus Wards",  f"{len(surplus_df)}",       help="Wards where supply exceeds demand today")
    k2.metric("🔴 Deficit Wards",  f"{len(deficit_df)}",       help="Wards where demand exceeds supply today")
    k3.metric("💧 Total Surplus",  f"{total_surplus:.1f} MLD", help="Water available for diversion")
    k4.metric("📉 Total Deficit",  f"{total_deficit:.1f} MLD", help="Total unmet demand across all deficit wards")

    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    d1.metric("🔀 Total Diverted", f"{total_diverted:.1f} MLD",
              delta=f"{total_diverted/total_deficit*100:.1f}% of deficit covered" if total_deficit > 0 else "N/A")
    d2.metric("⚠️ Remaining Gap",  f"{gap_after:.1f} MLD",
              delta=f"{-gap_after:.1f} reduction", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📋 Diversion Plan")
    if not diversion_df.empty:

        def colour_coverage(val):
            if val >= 90:
                return "background-color: rgba(6,214,160,0.25); color:#1b5e20; font-weight:600"
            elif val >= 50:
                return "background-color: rgba(255,209,102,0.3); color:#e65100; font-weight:600"
            else:
                return "background-color: rgba(239,71,111,0.2); color:#c62828; font-weight:600"

        styled = diversion_df.style.map(colour_coverage, subset=["Coverage (%)"])
        st.dataframe(styled, use_container_width=True)

        # ── Visualisations ──
        v1, v2 = st.columns(2)

        with v1:
            st.markdown("#### 📊 Top Surplus Wards (available water)")
            top_sur = surplus_df.head(15)
            fig_sur = go.Figure(go.Bar(
                x=top_sur["Surplus_MLD"],
                y=top_sur["Ward_Name"],
                orientation="h",
                marker_color="#06d6a0",
                marker_line_color="#00b4d8",
                marker_line_width=0.5,
                text=top_sur["Surplus_MLD"].round(1).astype(str) + " MLD",
                textposition="outside",
                textfont_color="#1a3a4a"
            ))
            fig_sur.update_layout(
                plot_bgcolor="rgba(255,255,255,0.7)", paper_bgcolor="rgba(255,255,255,0)",
                font_color="#1a3a4a", height=420, margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(title="Surplus (MLD)", gridcolor="#caf0f8"),
                yaxis=dict(tickfont=dict(size=10))
            )
            st.plotly_chart(fig_sur, use_container_width=True)

        with v2:
            st.markdown("#### 🔴 Top Deficit Wards (water needed)")
            top_def = deficit_df.head(15)
            fig_def = go.Figure(go.Bar(
                x=top_def["Deficit_MLD"],
                y=top_def["Ward_Name"],
                orientation="h",
                marker_color="#ef476f",
                marker_line_color="#c62828",
                marker_line_width=0.5,
                text=top_def["Deficit_MLD"].round(1).astype(str) + " MLD",
                textposition="outside",
                textfont_color="#1a3a4a"
            ))
            fig_def.update_layout(
                plot_bgcolor="rgba(255,255,255,0.7)", paper_bgcolor="rgba(255,255,255,0)",
                font_color="#1a3a4a", height=420, margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(title="Deficit (MLD)", gridcolor="#caf0f8"),
                yaxis=dict(tickfont=dict(size=10))
            )
            st.plotly_chart(fig_def, use_container_width=True)

        # ── Coverage waterfall ──
        st.markdown("#### 💧 Diversion Coverage per Deficit Ward")
        top_plan   = diversion_df.head(20).copy()
        bar_colors = ["#06d6a0" if c >= 90 else "#ffd166" if c >= 50 else "#ef476f"
                      for c in top_plan["Coverage (%)"]]
        fig_cov = go.Figure(go.Bar(
            x=top_plan["Deficit Ward"],
            y=top_plan["Coverage (%)"],
            marker_color=bar_colors,
            text=top_plan["Coverage (%)"].astype(str) + "%",
            textposition="outside",
            textfont_color="#1a3a4a"
        ))
        fig_cov.add_hline(y=100, line_dash="dash", line_color="#0077b6",
                           annotation_text="Full coverage (100%)")
        fig_cov.update_layout(
            plot_bgcolor="rgba(255,255,255,0.7)", paper_bgcolor="rgba(255,255,255,0)",
            font_color="#1a3a4a", height=360, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(tickangle=-35, tickfont=dict(size=9), gridcolor="#caf0f8"),
            yaxis=dict(title="Coverage (%)", range=[0, 115], gridcolor="#caf0f8")
        )
        st.plotly_chart(fig_cov, use_container_width=True)

        # ── Alert for currently selected ward ──
        ward_surplus  = surplus_df[surplus_df["Ward_Name"] == zone]
        ward_deficit  = deficit_df[deficit_df["Ward_Name"] == zone]

        # FIX 11: Use re.escape so ward names with special chars don't crash str.contains
        if not diversion_df.empty:
            ward_diverted_from = diversion_df[
                diversion_df["Source Wards"].str.contains(re.escape(zone), na=False)
            ]
            ward_diverted_to = diversion_df[diversion_df["Deficit Ward"] == zone]
        else:
            ward_diverted_from = pd.DataFrame()
            ward_diverted_to   = pd.DataFrame()

        st.markdown("---")
        st.markdown(f"### 🏘️ Status for Selected Ward: **{zone}**")

        if not ward_surplus.empty:
            surplus_val = ward_surplus.iloc[0]["Surplus_MLD"]
            after_val   = ward_surplus.iloc[0].get("After Diversion (MLD)", surplus_val)
            st.markdown(
                f'<div class="alert-success">💚 <b>{zone}</b> has a surplus of <b>{surplus_val:.2f} MLD</b> today — '
                f'remaining after diversion: <b>{after_val:.2f} MLD</b>.</div>',
                unsafe_allow_html=True
            )
        elif not ward_deficit.empty:
            deficit_val  = ward_deficit.iloc[0]["Deficit_MLD"]
            covered      = ward_diverted_to.iloc[0]["Diverted (MLD)"]  if not ward_diverted_to.empty else 0
            coverage     = ward_diverted_to.iloc[0]["Coverage (%)"]    if not ward_diverted_to.empty else 0
            sources_txt  = ward_diverted_to.iloc[0]["Source Wards"]    if not ward_diverted_to.empty else "None"
            alert_class  = "warning" if coverage >= 50 else "danger"
            st.markdown(
                f'<div class="alert-{alert_class}">'
                f'🔴 <b>{zone}</b> has a deficit of <b>{deficit_val:.2f} MLD</b> today. '
                f'Diversion covers <b>{covered:.2f} MLD ({coverage}%)</b>.<br>'
                f'<small>Source wards: {sources_txt}</small></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="alert-success">✅ <b>{zone}</b> is balanced — supply meets demand today. No diversion needed.</div>',
                unsafe_allow_html=True
            )

        # ── Expandable full tables ──
        with st.expander("📋 View Full Surplus Wards Table"):
            st.dataframe(surplus_df, use_container_width=True)
        with st.expander("📋 View Full Deficit Wards Table"):
            st.dataframe(deficit_df, use_container_width=True)
        with st.expander("📋 View Full Diversion Plan"):
            st.dataframe(diversion_df, use_container_width=True)

    else:
        st.markdown(
            '<div class="alert-success">✅ No redistribution needed — all wards are currently balanced.</div>',
            unsafe_allow_html=True
        )


# ── Raw data expander ──────────────────────────────────────────────────────────
with st.expander("📋 View Raw Ward Data"):
    st.dataframe(
        df[df["Ward_Name"] == zone].reset_index(drop=True),
        use_container_width=True
    )
