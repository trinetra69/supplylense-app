"""
SupplyLense — Predictive Demand Forecasting Dashboard
Full interactive Streamlit dashboard with 7 panels.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
SYNTH_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
FEEDBACK_DIR = os.path.join(BASE_DIR, 'data', 'feedback')
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Page Config & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SupplyLense — Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.sub-header {
    color: #a0aec0;
    font-size: 1.1rem;
    font-weight: 300;
    margin-top: -10px;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.kpi-card:hover {
    border-color: rgba(102,126,234,0.6);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(102,126,234,0.2);
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-label {
    color: #a0aec0;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.gold-badge { color: #ffd700; font-weight: 700; }
.silver-badge { color: #c0c0c0; font-weight: 700; }
.bronze-badge { color: #cd7f32; font-weight: 700; }
.error-badge { color: #ff4444; font-weight: 700; }

.section-divider {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(102,126,234,0.4) 50%, transparent 100%);
    margin: 30px 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(102,126,234,0.1);
    border-radius: 8px;
    padding: 8px 20px;
    border: 1px solid rgba(102,126,234,0.2);
    color: #a0aec0;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-color: transparent !important;
}

div[data-testid="stMetric"] {
    background: rgba(102,126,234,0.08);
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetric"] label {
    color: #a0aec0 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
}

.waste-alert {
    background: linear-gradient(135deg, rgba(255,68,68,0.15) 0%, rgba(255,107,107,0.15) 100%);
    border: 1px solid rgba(255,68,68,0.4);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

h1, h2, h3 { color: #e2e8f0 !important; }
p, span, label { color: #cbd5e0; }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all datasets and results."""
    data = {}
    
    # Synthetic datasets
    for name in ['sales', 'weather', 'holidays', 'product_metadata', 'competitor_promos', 'macro_signals']:
        path = os.path.join(SYNTH_DIR, f'{name}.parquet')
        if os.path.exists(path):
            data[name] = pd.read_csv("data/results/sample.csv")(path)
    
    # Results
    for name in ['forecasts', 'leaderboard']:
        path = os.path.join(RESULTS_DIR, f'{name}.parquet')
        if os.path.exists(path):
            data[name] = pd.read_csv("data/results/sample.csv")(path)
    
    # Hierarchy
    hier_path = os.path.join(RESULTS_DIR, 'hierarchy.json')
    if os.path.exists(hier_path):
        with open(hier_path) as f:
            data['hierarchy'] = json.load(f)
    
    # Pipeline metadata
    meta_path = os.path.join(RESULTS_DIR, 'pipeline_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data['metadata'] = json.load(f)
    
    # Store config
    config_path = os.path.join(BASE_DIR, 'data', 'store_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            data['store_config'] = json.load(f)
    
    return data


def check_pipeline_run():
    """Check if the pipeline has been run."""
    return os.path.exists(os.path.join(RESULTS_DIR, 'forecasts.parquet'))


# ──────────────────────────────────────────────
# Plotly Theme
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#a0aec0'),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
    hoverlabel=dict(bgcolor='#1a1a2e', font_color='#e2e8f0', bordercolor='#667eea'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)'),
)

COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#48bb78',
    'warning': '#ed8936',
    'danger': '#fc8181',
    'info': '#63b3ed',
    'pred_50': '#667eea',
    'pred_band': 'rgba(102,126,234,0.15)',
    'actual': '#48bb78',
    'scenario': '#ed8936',
}


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h2 style="text-align:center;">📊 SupplyLense</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#a0aec0; font-size:0.8rem;">Predictive Demand Forecasting<br>& Supply Chain Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(255,255,255,0.1);">', unsafe_allow_html=True)
    
    if check_pipeline_run():
        data = load_data()
        
        st.markdown("#### 🏪 Store Profile")
        config = data.get('store_config', {})
        st.info(f"**{config.get('city', 'Ahmedabad')}** — {config.get('area_type', 'Urban')} {config.get('urban_density', 'High-density')}")
        st.caption(f"Store: {config.get('store_type', 'General Grocery')}")
        st.caption(f"ID: {config.get('store_id', 'STORE_AHM_001')}")
        
        st.markdown('<hr style="border-color:rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        
        meta = data.get('metadata', {})
        st.markdown("#### ⚙️ Pipeline Info")
        st.caption(f"Products: {meta.get('n_products', 12)}")
        st.caption(f"Training days: {meta.get('n_days', 1500)}+")
        st.caption(f"Pipeline time: {meta.get('pipeline_run_time_seconds', 0):.0f}s")
        st.caption(f"Models: NeuralProphet · TFT · LightGBM")
        
        st.markdown('<hr style="border-color:rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        
        hier = data.get('hierarchy', {})
        st.markdown("#### 🎯 Coherence Score")
        coherence = hier.get('coherence_score', 1.0)
        st.progress(coherence, text=f"{coherence*100:.0f}% — Bottom-Up")
        
    else:
        # st.warning("Pipeline not yet run...")


# ──────────────────────────────────────────────
# ONBOARDING CHECK
# ──────────────────────────────────────────────
if not check_pipeline_run():
    st.markdown('<h1 class="main-header">SupplyLense</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predictive Demand Forecasting & Supply Chain Intelligence</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Welcome to SupplyLense!")
    st.markdown("""
    This platform provides AI-powered demand forecasting for kirana stores and neighbourhood grocery outlets.
    
    **To get started:**
    1. Open a terminal in this directory
    2. Run: `python run_pipeline.py`
    3. Wait for the pipeline to complete (~5-10 minutes)
    4. Refresh this page
    """)
    
    # Quick onboarding wizard
    st.markdown("---")
    st.markdown("### 📋 Store Onboarding")
    
    tab1, tab2, tab3 = st.tabs(["1️⃣ Store Profile", "2️⃣ Product Setup", "3️⃣ Sales Seed"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            area = st.radio("Area Type", ["Urban", "Rural"])
            if area == "Urban":
                density = st.radio("Urban Density", ["High-density", "Medium-density", "Low-density"])
            else:
                rural_type = st.radio("Rural Type", ["Peri-urban", "Deep rural"])
        with col2:
            city = st.text_input("City / Region", "Ahmedabad")
            store_type = st.selectbox("Store Type", ["General Grocery", "Dairy-focused", "Snacks & Beverages", "Mixed"])
            since = st.date_input("Operating Since", datetime(2020, 1, 1))
    
    with tab2:
        st.markdown("##### Default Product List (12 products)")
        products = pd.DataFrame([
            {"Product": "Full-cream Milk (500ml)", "Category": "Dairy", "Shelf Life": 2, "Stock": 100, "Price (₹)": 30},
            {"Product": "Buttermilk (200ml)", "Category": "Dairy", "Shelf Life": 3, "Stock": 80, "Price (₹)": 15},
            {"Product": "Curd (400g)", "Category": "Dairy", "Shelf Life": 5, "Stock": 60, "Price (₹)": 40},
            {"Product": "Paneer (200g)", "Category": "Dairy", "Shelf Life": 7, "Stock": 30, "Price (₹)": 80},
            {"Product": "Butter (100g)", "Category": "Dairy", "Shelf Life": 30, "Stock": 25, "Price (₹)": 55},
            {"Product": "Biscuits (100g)", "Category": "Snacks", "Shelf Life": 180, "Stock": 100, "Price (₹)": 20},
            {"Product": "Namkeen (200g)", "Category": "Snacks", "Shelf Life": 90, "Stock": 50, "Price (₹)": 40},
            {"Product": "Sweets / Mithai (250g)", "Category": "Sweets", "Shelf Life": 3, "Stock": 20, "Price (₹)": 120},
            {"Product": "Cold Drink (600ml)", "Category": "Beverages", "Shelf Life": 270, "Stock": 80, "Price (₹)": 40},
            {"Product": "Packaged Water (1L)", "Category": "Beverages", "Shelf Life": 365, "Stock": 120, "Price (₹)": 20},
            {"Product": "Bread Loaf (400g)", "Category": "Staples", "Shelf Life": 4, "Stock": 50, "Price (₹)": 35},
            {"Product": "Eggs (tray of 6)", "Category": "Staples", "Shelf Life": 14, "Stock": 40, "Price (₹)": 42},
        ])
        st.dataframe(products, use_container_width=True, hide_index=True)
    
    with tab3:
        st.radio("Historical Sales Source", ["Auto-Generate (Recommended)", "Upload CSV"], key="sales_source")
        if st.session_state.get("sales_source") == "Upload CSV":
            st.file_uploader("Upload CSV (date, product_id, units_sold)", type=['csv'])
        else:
            st.success("✅ The system will auto-generate a realistic 1500+ day sales history based on your store profile.")
    
    st.stop()


# ──────────────────────────────────────────────
# MAIN DASHBOARD (after pipeline run)
# ──────────────────────────────────────────────
data = load_data()

# Header
st.markdown('<h1 class="main-header">SupplyLense</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive Demand Forecasting & Supply Chain Intelligence — Ahmedabad, Gujarat</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# KPI Cards
# ──────────────────────────────────────────────
sales_df = data.get('sales', pd.DataFrame())
forecasts_df = data.get('forecasts', pd.DataFrame())
leaderboard_df = data.get('leaderboard', pd.DataFrame())
products_df = data.get('product_metadata', pd.DataFrame())
hier = data.get('hierarchy', {})
meta = data.get('metadata', {})

# Compute KPIs
avg_wmape = leaderboard_df[leaderboard_df['status'] == 'OK']['wmape'].astype(float).mean() if not leaderboard_df.empty else 0
n_products = len(products_df) if not products_df.empty else 12
best_model_counts = leaderboard_df[leaderboard_df['is_winner'] == True]['model_name'].value_counts() if not leaderboard_df.empty else pd.Series()
top_model = best_model_counts.index[0] if len(best_model_counts) > 0 else 'LightGBM'

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{avg_wmape:.1f}%</div>
        <div class="kpi-label">Avg WMAPE</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{n_products}</div>
        <div class="kpi-label">Products Tracked</div>
    </div>""", unsafe_allow_html=True)
with col3:
    coherence = hier.get('coherence_score', 1.0) * 100
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{coherence:.0f}%</div>
        <div class="kpi-label">Coherence Score</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">3</div>
        <div class="kpi-label">Models Trained</div>
    </div>""", unsafe_allow_html=True)

# Product-wise 30-Day Demand Table
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("### 📦 30-Day Demand Forecast — Product Wise")
if not forecasts_df.empty and not products_df.empty:
    pw_demand = forecasts_df.groupby('product_id').agg(
        total_demand=('pred_50', 'sum'),
        lower_bound=('pred_10', 'sum'),
        upper_bound=('pred_90', 'sum'),
        model=('model', 'first'),
    ).reset_index()
    pw_demand = pw_demand.merge(products_df[['product_id', 'product_name', 'category']], on='product_id', how='left')
    pw_demand = pw_demand[['product_id', 'product_name', 'category', 'total_demand', 'lower_bound', 'upper_bound', 'model']]
    pw_demand.columns = ['ID', 'Product', 'Category', 'Total Demand (30d)', 'Lower Bound', 'Upper Bound', 'Best Model']
    pw_demand = pw_demand.sort_values('Total Demand (30d)', ascending=False)
    for col in ['Total Demand (30d)', 'Lower Bound', 'Upper Bound']:
        pw_demand[col] = pw_demand[col].apply(lambda x: f"{x:,.0f}")
    st.dataframe(pw_demand, use_container_width=True, hide_index=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_forecast, tab_leaderboard, tab_hierarchy, tab_simulation, tab_feedback = st.tabs([
    "📈 Forecast Explorer",
    "🏆 Model Leaderboard",
    "🏗️ Hierarchy Panel",
    "🔮 What-If Simulation",
    "💬 Feedback & Alerts",
])


# ═══════════════════════════════════════════════
# TAB 1: FORECAST EXPLORER
# ═══════════════════════════════════════════════
with tab_forecast:
    st.markdown("### 📈 Product Forecast Explorer")
    
    col_sel, col_horizon = st.columns([3, 1])
    with col_sel:
        product_options = {}
        if not products_df.empty:
            for _, p in products_df.iterrows():
                product_options[f"{p['product_id']} — {p['product_name']}"] = p['product_id']
        selected_product_label = st.selectbox("Select Product", list(product_options.keys()), key="forecast_product")
        selected_pid = product_options.get(selected_product_label, 'P001')
    with col_horizon:
        horizon = st.selectbox("Forecast Horizon", [7, 14, 30], index=2, key="forecast_horizon")
    
    # Get product's actual sales + forecast on matching dates
    if not sales_df.empty and not forecasts_df.empty:
        prod_sales = sales_df[sales_df['product_id'] == selected_pid].copy()
        prod_sales['date'] = pd.to_datetime(prod_sales['date'])
        prod_sales = prod_sales.sort_values('date')
        
        prod_forecast = forecasts_df[forecasts_df['product_id'] == selected_pid].copy()
        prod_forecast['date'] = pd.to_datetime(prod_forecast['date'])
        fc = prod_forecast.head(horizon)
        
        # Get actual sales for the same dates as forecasts
        fc_dates = set(fc['date'].values)
        actual_on_fc_dates = prod_sales[prod_sales['date'].isin(fc_dates)].sort_values('date')
        
        # Merge actual and predicted on date
        merged = fc.merge(actual_on_fc_dates[['date', 'units_sold']], on='date', how='left')
        
        fig = go.Figure()
        
        # Actual sales on forecast dates
        fig.add_trace(go.Scatter(
            x=merged['date'], y=merged['units_sold'],
            mode='lines+markers', name='Actual Sales',
            line=dict(color=COLORS['actual'], width=2.5),
            marker=dict(size=7, symbol='circle'),
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=merged['date'], y=merged['pred_50'],
            mode='lines+markers', name='Predicted (50th pct)',
            line=dict(color=COLORS['pred_50'], width=2.5, dash='dot'),
            marker=dict(size=7, symbol='diamond'),
        ))
        
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Actual vs Predicted — {selected_product_label}",
            xaxis_title="Date", yaxis_title="Units",
            height=450,
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        col_table, col_stats = st.columns([2, 1])
        with col_table:
            st.markdown("##### 📋 Actual vs Predicted Details")
            display_fc = merged[['date', 'units_sold', 'pred_50', 'pred_10', 'pred_90', 'model']].copy()
            display_fc.columns = ['Date', 'Actual', 'Predicted', 'Lower (10th)', 'Upper (90th)', 'Model']
            display_fc['Date'] = display_fc['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_fc, use_container_width=True, hide_index=True)
        
        with col_stats:
            st.markdown("##### 📊 Quick Stats")
            avg_forecast = fc['pred_50'].mean()
            total_forecast = fc['pred_50'].sum()
            avg_actual = actual_on_fc_dates['units_sold'].mean() if not actual_on_fc_dates.empty else 0
            
            st.metric("Avg Daily Predicted", f"{avg_forecast:.0f} units")
            st.metric("Avg Daily Actual", f"{avg_actual:.0f} units")
            st.metric("Total Predicted ({} days)".format(horizon), f"{total_forecast:.0f} units")
            
            if avg_actual > 0:
                error = abs(avg_forecast - avg_actual) / avg_actual * 100
                st.metric("Error %", f"{error:.1f}%")


# ═══════════════════════════════════════════════
# TAB 2: MODEL LEADERBOARD
# ═══════════════════════════════════════════════
with tab_leaderboard:
    st.markdown("### 🏆 Model Leaderboard")
    
    if not leaderboard_df.empty:
        view_mode = st.radio("View Mode", ["All Products (Average)", "Per Product"], horizontal=True, key="lb_view")
        
        if view_mode == "All Products (Average)":
            # Aggregate across products
            agg_lb = leaderboard_df[leaderboard_df['status'] == 'OK'].groupby('model_name').agg({
                'wmape': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                'rmse': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                'pinball_composite': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                'coverage': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                'composite_score': lambda x: pd.to_numeric(x, errors='coerce').mean(),
            }).reset_index().sort_values('composite_score')
            
            # Add badges
            badges = ['🥇', '🥈', '🥉']
            agg_lb['Rank'] = [badges[i] if i < 3 else '' for i in range(len(agg_lb))]
            
            display_cols = ['Rank', 'model_name', 'wmape', 'rmse', 'pinball_composite', 'coverage', 'composite_score']
            display_lb = agg_lb[display_cols].copy()
            display_lb.columns = ['Rank', 'Model', 'WMAPE (%)', 'RMSE', 'Pinball Loss', 'Coverage', 'Composite Score']
            
            for col in ['WMAPE (%)', 'RMSE', 'Pinball Loss', 'Coverage', 'Composite Score']:
                display_lb[col] = display_lb[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "—")
            
            st.dataframe(display_lb, use_container_width=True, hide_index=True)
            
            # Win counts chart
            win_counts = leaderboard_df[leaderboard_df['is_winner'] == True]['model_name'].value_counts()
            if len(win_counts) > 0:
                fig_wins = go.Figure(go.Bar(
                    x=win_counts.index, y=win_counts.values,
                    marker_color=[COLORS['primary'], COLORS['secondary'], COLORS['warning']][:len(win_counts)],
                    text=win_counts.values, textposition='outside',
                ))
                fig_wins.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Best Model Wins Across Products",
                    xaxis_title="Model", yaxis_title="Number of Products Won",
                    height=350,
                )
                st.plotly_chart(fig_wins, use_container_width=True)
        
        else:
            # Per-product view
            prod_select = st.selectbox("Select Product", 
                                       [f"{r['product_id']} — {r['product_name']}" for _, r in leaderboard_df.drop_duplicates('product_id').iterrows()],
                                       key="lb_product")
            pid = prod_select.split(" — ")[0].strip()
            
            prod_lb = leaderboard_df[leaderboard_df['product_id'] == pid].copy()
            
            for col in ['wmape', 'rmse', 'pinball_composite', 'coverage', 'composite_score']:
                prod_lb[col] = pd.to_numeric(prod_lb[col], errors='coerce')
            
            prod_lb = prod_lb.sort_values('composite_score')
            
            badges = ['🥇', '🥈', '🥉']
            prod_lb['Rank'] = [badges[i] if i < 3 else '' for i in range(len(prod_lb))]
            
            display_cols = ['Rank', 'model_name', 'wmape', 'rmse', 'pinball_composite', 'coverage', 'composite_score', 'status']
            avail_cols = [c for c in display_cols if c in prod_lb.columns]
            
            st.dataframe(prod_lb[avail_cols], use_container_width=True, hide_index=True)
            
            # Metric comparison radar chart
            valid_models = prod_lb[prod_lb['status'] == 'OK']
            if len(valid_models) >= 2:
                categories_radar = ['WMAPE', 'RMSE', 'Pinball', 'Coverage']
                fig_radar = go.Figure()
                
                colors_list = [COLORS['primary'], COLORS['warning'], COLORS['success']]
                for idx, (_, row) in enumerate(valid_models.iterrows()):
                    vals = [
                        float(row.get('wmape', 0)),
                        float(row.get('rmse', 0)),
                        float(row.get('pinball_composite', 0)),
                        float(row.get('coverage', 0)) * 100,
                    ]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=categories_radar, fill='toself',
                        name=row['model_name'],
                        line_color=colors_list[idx % len(colors_list)],
                        fillcolor=f"rgba({','.join(str(int(colors_list[idx % len(colors_list)].lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.1)",
                    ))
                
                fig_radar.update_layout(
                    **PLOTLY_LAYOUT,
                    polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, color='#a0aec0')),
                    title=f"Model Comparison — {prod_select}",
                    height=400,
                )
                st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("No leaderboard data available. Run the pipeline first.")


# ═══════════════════════════════════════════════
# TAB 3: HIERARCHY PANEL
# ═══════════════════════════════════════════════
with tab_hierarchy:
    st.markdown("### 🏗️ Hierarchical Forecast View")
    st.markdown("**Bottom-Up Architecture**: SKU → Category → Store Total. Coherence guaranteed by construction.")
    
    col_hier1, col_hier2 = st.columns([2, 1])
    
    with col_hier1:
        # Store-level total demand chart
        if not forecasts_df.empty:
            store_daily = forecasts_df.groupby('date').agg({
                'pred_50': 'sum', 'pred_10': 'sum', 'pred_90': 'sum'
            }).reset_index()
            store_daily['date'] = pd.to_datetime(store_daily['date'])
            
            fig_store = go.Figure()
            fig_store.add_trace(go.Bar(
                x=store_daily['date'], y=store_daily['pred_50'],
                name='Store Total Demand',
                marker_color=COLORS['primary'],
                opacity=0.8,
            ))
            fig_store.add_trace(go.Scatter(
                x=store_daily['date'], y=store_daily['pred_90'],
                mode='lines', name='Upper Bound (90th)',
                line=dict(color=COLORS['warning'], width=1, dash='dash'),
            ))
            fig_store.update_layout(
                **PLOTLY_LAYOUT,
                title="Store-Level Total Daily Demand Forecast",
                xaxis_title="Date", yaxis_title="Total Units",
                height=400,
            )
            st.plotly_chart(fig_store, use_container_width=True)
    
    with col_hier2:
        # Category contribution donut chart
        contributions = hier.get('category_contributions', {})
        if contributions:
            cats = list(contributions.keys())
            vals = list(contributions.values())
            
            donut_colors = ['#667eea', '#764ba2', '#48bb78', '#ed8936', '#fc8181']
            
            fig_donut = go.Figure(go.Pie(
                labels=cats, values=vals,
                hole=0.55,
                marker_colors=donut_colors[:len(cats)],
                textinfo='label+percent',
                textfont=dict(color='white', size=11),
            ))
            fig_donut.update_layout(
                **PLOTLY_LAYOUT,
                title="Category Share (30-Day)",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
    
    # Category drill-down
    st.markdown("##### 📂 Category Drill-Down")
    if not forecasts_df.empty and not products_df.empty:
        fc_with_meta = forecasts_df.merge(products_df[['product_id', 'category', 'product_name']], on='product_id', how='left')
        
        categories = fc_with_meta['category'].unique()
        for cat in sorted(categories):
            with st.expander(f"📦 {cat}", expanded=False):
                cat_products = fc_with_meta[fc_with_meta['category'] == cat]
                cat_summary = cat_products.groupby(['product_id', 'product_name']).agg({
                    'pred_50': 'sum', 'pred_10': 'sum', 'pred_90': 'sum'
                }).reset_index()
                cat_summary.columns = ['Product ID', 'Product', 'Total Demand (50th)', 'Lower Bound', 'Upper Bound']
                st.dataframe(cat_summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 4: WHAT-IF SIMULATION
# ═══════════════════════════════════════════════
with tab_simulation:
    st.markdown("### 🔮 What-If Simulation Engine")
    st.markdown("Configure scenarios and see how they shift demand forecasts vs baseline.")
    
    sim_product_label = st.selectbox("Select Product for Simulation", list(product_options.keys()), key="sim_product")
    sim_pid = product_options.get(sim_product_label, 'P001')
    
    scenario_type = st.selectbox("Scenario Type", [
        "Festival / Holiday Spike",
        "Sudden Weather Event",
        "Competitor Discount",
        "Supply Disruption",
        "Price Change",
    ])
    
    col_params, col_result = st.columns([1, 2])
    
    with col_params:
        st.markdown("##### ⚙️ Scenario Parameters")
        
        sim_params = {}
        
        if scenario_type == "Festival / Holiday Spike":
            event_type = st.selectbox("Event", ["Diwali", "Holi", "Eid", "Navratri", "Christmas", "IPL Final", "Custom"])
            sim_params['event_type'] = event_type
            sim_params['intensity'] = st.slider("Intensity Multiplier", 1.0, 4.0, 2.0, 0.1)
            sim_params['start_idx'] = st.slider("Start Day (in forecast)", 0, 25, 5)
            sim_params['duration_days'] = st.number_input("Duration (days)", 1, 15, 5)
            sim_params['pre_event_days'] = st.number_input("Pre-event window (days)", 0, 7, 3)
            
        elif scenario_type == "Sudden Weather Event":
            sim_params['event_type'] = st.selectbox("Event Type", ["Heatwave", "Heavy Rain", "Cold Snap", "Dense Fog"])
            sim_params['severity'] = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
            sim_params['start_idx'] = st.slider("Start Day", 0, 25, 3)
            sim_params['duration_days'] = st.number_input("Duration (days)", 1, 14, 5)
            
        elif scenario_type == "Competitor Discount":
            sim_params['discount_pct'] = st.selectbox("Discount Level", [10, 15, 20, 25, 30])
            sim_params['affected_categories'] = st.multiselect("Affected Categories", 
                                                                ["Dairy", "Snacks", "Beverages", "Staples", "Sweets"],
                                                                default=["Dairy"])
            sim_params['duration_days'] = st.number_input("Duration (days)", 1, 30, 7)
            sim_params['proximity'] = st.selectbox("Competitor Proximity", 
                                                    ["Same street", "Same neighbourhood", "Same zone"])
            sim_params['start_idx'] = st.slider("Start Day", 0, 25, 0)
            
        elif scenario_type == "Supply Disruption":
            sim_params['disruption_type'] = st.selectbox("Type", ["Transport strike", "Supplier shortage", "Logistics delay"])
            sim_params['severity'] = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
            sim_params['start_idx'] = st.slider("Start Day", 0, 25, 5)
            sim_params['duration_days'] = st.number_input("Duration (days)", 1, 14, 5)
            
        elif scenario_type == "Price Change":
            sim_params['price_change_pct'] = st.slider("Price Change (%)", -30, 30, 10)
            sim_params['start_idx'] = st.slider("Effective From Day", 0, 25, 0)
    
    with col_result:
        st.markdown("##### 📊 Simulation Result")
        
        # Get baseline forecast
        prod_fc = forecasts_df[forecasts_df['product_id'] == sim_pid] if not forecasts_df.empty else pd.DataFrame()
        
        if not prod_fc.empty:
            baseline = prod_fc['pred_50'].values[:30]
            
            # Get product info
            prod_info_row = products_df[products_df['product_id'] == sim_pid]
            product_info = {}
            if not prod_info_row.empty:
                product_info = prod_info_row.iloc[0].to_dict()
            
            # Apply scenario
            try:
                from src.simulation.scenarios import run_scenario, compute_scenario_impact
                
                scenario_pred = run_scenario(baseline, scenario_type, sim_params, product_info)
                
                # Plot comparison
                dates_fc = prod_fc['date'].values[:30]
                fig_sim = go.Figure()
                
                fig_sim.add_trace(go.Scatter(
                    x=dates_fc, y=baseline,
                    mode='lines', name='Baseline Forecast',
                    line=dict(color=COLORS['actual'], width=2),
                ))
                fig_sim.add_trace(go.Scatter(
                    x=dates_fc, y=scenario_pred,
                    mode='lines', name=f'Scenario: {scenario_type}',
                    line=dict(color=COLORS['scenario'], width=3, dash='dot'),
                    fill='tonexty', fillcolor='rgba(237,137,54,0.1)',
                ))
                
                fig_sim.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Baseline vs Scenario Forecast",
                    xaxis_title="Date", yaxis_title="Units",
                    height=350,
                )
                st.plotly_chart(fig_sim, use_container_width=True)
                
                # Impact metrics
                unit_price = product_info.get('unit_price_inr', 30)
                impact = compute_scenario_impact(baseline, scenario_pred, unit_price)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    delta_color = "normal" if impact['demand_change_units'] >= 0 else "inverse"
                    st.metric("Demand Change", f"{impact['demand_change_units']:+.0f} units", 
                             f"{impact['demand_change_pct']:+.1f}%")
                with c2:
                    st.metric("Revenue Impact", f"₹{impact['revenue_impact_inr']:+,.0f}")
                with c3:
                    st.metric("Reorder Qty", f"{impact['recommended_reorder_qty']} units")
                    
            except Exception as e:
                st.error(f"Simulation error: {e}")
        else:
            st.warning("No forecast data available for this product.")


# ═══════════════════════════════════════════════
# TAB 5: FEEDBACK & ALERTS
# ═══════════════════════════════════════════════
with tab_feedback:
    st.markdown("### 💬 User Feedback & Expiry Alerts")
    
    feedback_tab, waste_tab = st.tabs(["📝 Feedback Panel", "⚠️ Expiry Waste Alerts"])
    
    with feedback_tab:
        st.markdown("##### Rate Past Predictions & Submit Corrections")
        st.markdown("Your feedback improves future forecasts through model retraining.")
        
        fb_product = st.selectbox("Product", list(product_options.keys()), key="fb_product")
        fb_pid = product_options.get(fb_product, 'P001')
        
        # Show recent predictions
        prod_fc = forecasts_df[forecasts_df['product_id'] == fb_pid] if not forecasts_df.empty else pd.DataFrame()
        
        fb_path = os.path.join(FEEDBACK_DIR, 'feedback_log.csv')
        
        if not prod_fc.empty:
            st.markdown("##### Recent Predictions — Enter Actual Sales")
            col_fb1, col_fb2 = st.columns([2, 1])
            with col_fb1:
                feedback_entries = []
                for i, (_, row) in enumerate(prod_fc.head(5).iterrows()):
                    with st.container():
                        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                        with c1:
                            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                            st.text(f"📅 {date_str}")
                        with c2:
                            st.text(f"Predicted: {row['pred_50']:.0f}")
                        with c3:
                            rating = st.radio("", ["👍", "👎"], key=f"rating_{fb_pid}_{i}", horizontal=True, label_visibility="collapsed")
                        with c4:
                            actual = st.number_input("Actual", min_value=0, value=int(row['pred_50']), key=f"actual_{fb_pid}_{i}", label_visibility="collapsed")
                        feedback_entries.append({
                            'date': date_str,
                            'product_id': fb_pid,
                            'predicted': row['pred_50'],
                            'actual': actual,
                            'rating': 'up' if rating == '👍' else 'down',
                            'submitted_at': datetime.now().isoformat(),
                        })
            
            with col_fb2:
                st.markdown("##### Feedback Stats")
                # Load existing feedback
                if os.path.exists(fb_path):
                    fb_log = pd.read_csv(fb_path)
                    n_corrections = len(fb_log)
                    thumbs_up = (fb_log['rating'] == 'up').sum()
                    thumbs_down = (fb_log['rating'] == 'down').sum()
                    avg_error = abs(fb_log['predicted'] - fb_log['actual']).mean()
                else:
                    fb_log = pd.DataFrame()
                    n_corrections = 0
                    thumbs_up = 0
                    thumbs_down = 0
                    avg_error = 0
                
                st.metric("Total Corrections", n_corrections)
                st.metric("👍 Accurate", int(thumbs_up))
                st.metric("👎 Needs Improvement", int(thumbs_down))
                if n_corrections > 0:
                    st.metric("Avg Error (units)", f"{avg_error:.1f}")
                
                # Submit button — actually saves
                if st.button("💾 Submit Feedback", type="primary", key="submit_fb"):
                    new_fb = pd.DataFrame(feedback_entries)
                    if os.path.exists(fb_path):
                        existing = pd.read_csv(fb_path)
                        combined = pd.concat([existing, new_fb], ignore_index=True)
                        combined = combined.drop_duplicates(subset=['date', 'product_id'], keep='last')
                    else:
                        combined = new_fb
                    combined.to_csv(fb_path, index=False)
                    st.success(f"✅ Saved {len(new_fb)} corrections! Total: {len(combined)}")
                    st.rerun()
                
                st.markdown("---")
                
                # Retrain button — triggers pipeline
                if n_corrections >= 3:
                    if st.button("🔄 Retrain Models", type="secondary", key="retrain_btn"):
                        st.warning("⏳ Retraining pipeline started... This takes ~5 min. Refresh the page when done.")
                        import subprocess
                        subprocess.Popen(
                            ['python', 'run_pipeline.py'],
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                        )
                        st.info("Pipeline running in background. Refresh page in a few minutes to see updated results.")
                else:
                    st.button("🔄 Retrain Models", disabled=True, key="retrain_btn")
                    st.caption(f"Submit {3 - n_corrections} more corrections to enable retraining")
            
            # Show feedback history
            if os.path.exists(fb_path) and n_corrections > 0:
                st.markdown("---")
                st.markdown("##### 📋 Feedback History")
                fb_display = pd.read_csv(fb_path).tail(20)
                fb_display = fb_display[['date', 'product_id', 'predicted', 'actual', 'rating', 'submitted_at']]
                fb_display.columns = ['Date', 'Product', 'Predicted', 'Actual', 'Rating', 'Submitted']
                fb_display['Error'] = abs(fb_display['Predicted'] - fb_display['Actual'])
                fb_display['Rating'] = fb_display['Rating'].map({'up': '👍', 'down': '👎'})
                st.dataframe(fb_display, use_container_width=True, hide_index=True)
    
    with waste_tab:
        st.markdown("##### ⚠️ Expiry Waste Alert System")
        st.markdown("Products where current stock exceeds forecasted demand within shelf life window.")
        
        if not products_df.empty and not forecasts_df.empty:
            perishables = products_df[products_df['is_perishable'] == True]
            
            alerts = []
            for _, prod in perishables.iterrows():
                pid = prod['product_id']
                shelf_life = prod['shelf_life_days']
                
                prod_fc = forecasts_df[forecasts_df['product_id'] == pid]
                if not prod_fc.empty:
                    demand_in_shelf_life = prod_fc['pred_50'].head(shelf_life).sum()
                    
                    # Assume current stock from last sales data
                    if not sales_df.empty:
                        last_stock = sales_df[sales_df['product_id'] == pid].sort_values('date').tail(1)
                        current_stock = last_stock['stock_available'].values[0] if not last_stock.empty else 50
                    else:
                        current_stock = 50
                    
                    if current_stock > demand_in_shelf_life * 1.2:
                        waste_units = max(0, current_stock - demand_in_shelf_life)
                        waste_value = waste_units * prod['unit_cost_inr']
                        
                        # Suggested discount to clear stock
                        excess_pct = (waste_units / current_stock * 100) if current_stock > 0 else 0
                        suggested_discount = min(30, max(5, excess_pct * 0.5))
                        
                        alerts.append({
                            'Product': prod['product_name'],
                            'Shelf Life': f"{shelf_life} days",
                            'Current Stock': int(current_stock),
                            'Forecasted Demand': f"{demand_in_shelf_life:.0f}",
                            'Expected Waste': f"{waste_units:.0f} units",
                            'Waste Value': f"₹{waste_value:.0f}",
                            'Suggested Discount': f"{suggested_discount:.0f}%",
                        })
            
            if alerts:
                for alert in alerts:
                    st.markdown(f"""<div class="waste-alert">
                        <strong>⚠️ {alert['Product']}</strong> — Shelf life: {alert['Shelf Life']}<br>
                        Stock: {alert['Current Stock']} | Demand: {alert['Forecasted Demand']} | 
                        <span style="color:#fc8181;">Waste: {alert['Expected Waste']} (₹{alert['Waste Value']})</span><br>
                        💡 Suggested markdown discount: <strong>{alert['Suggested Discount']}</strong> to clear excess stock
                    </div>""", unsafe_allow_html=True)
                
                st.markdown("---")
                st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
            else:
                st.success("✅ No waste alerts! Current stock levels are well-aligned with demand forecasts.")
        else:
            st.info("Run the pipeline to generate waste alerts.")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.75rem; padding:10px;">
    <strong>SupplyLense</strong> — Predictive Demand Forecasting & Supply Chain Intelligence<br>
    Built for Prama Innovations Hackathon, DA-IICT 2025<br>
    3 Models • 1500+ Days • 5 Scenarios • 4 Metrics
</div>
""", unsafe_allow_html=True)
