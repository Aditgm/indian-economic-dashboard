import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objects as go
import numpy as np
from config import INDIAN_INDICATORS, CATEGORIES, DATE_RANGES
import time
import logging
from functools import wraps
from typing import Dict, List, Optional, Tuple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def monitor_performance(func_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func_name} executed in {execution_time:.2f}s")

            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            st.session_state.performance_metrics[func_name] = execution_time
            
            return result
        return wrapper
    return decorator
CACHE_TTL = 300 
SAMPLE_CACHE_TTL = 3600

st.set_page_config(
    page_title="Indian Economic Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)
COLORS = {
    "saffron": "#FF9933",
    "white": "#FFFFFF", 
    "green": "#138808",
    
    "dark_bg": "#0E1117",
    "dark_surface": "#1E2028",
    "dark_surface_light": "#2D3748",
    "dark_text_primary": "#FFFFFF",
    "dark_text_secondary": "#A0AEC0",
    "dark_accent": "#4299E1",
    
    "light_bg": "#FFFFFF",
    "light_surface": "#F7FAFC",
    "light_surface_dark": "#EDF2F7",
    "light_text_primary": "#1A202C",
    "light_text_secondary": "#4A5568",
    "light_accent": "#3182CE",
    
    "nifty_color": "#E53E3E",      
    "inr_color": "#38A169",
    "warning": "#D69E2E",
    "success": "#48BB78",              
    "error": "#E53E3E",            
    "info": "#3182CE",

    "grid_dark": "#2D3748",
    "grid_light": "#E2E8F0",
    "border_dark": "#4A5568", 
    "border_light": "#CBD5E0"
}
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global App Styling */
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: {COLORS['dark_bg']};
        color: {COLORS['dark_text_primary']};
    }}
    
    /* Main Header - High Contrast Indian Flag Theme */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['saffron']} 0%, {COLORS['white']} 50%, {COLORS['green']} 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 2px solid {COLORS['border_dark']};
    }}
    
    .main-header h1 {{
        color: #000000 !important;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.8);
        letter-spacing: -0.5px;
    }}
    
    .main-header p {{
        color: #1A202C !important;
        margin: 0.5rem 0 0 0;
        font-weight: 600;
        font-size: 1.1rem;
    }}
    
    /* Section Headers */
    .stSubheader {{
        color: {COLORS['dark_text_primary']} !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        margin-bottom: 1rem !important;
    }}
    
    /* Metric Cards - High Contrast */
    div[data-testid="metric-container"] {{
        background: linear-gradient(135deg, {COLORS['dark_surface']} 0%, {COLORS['dark_surface_light']} 100%) !important;
        border: 2px solid {COLORS['border_dark']} !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        margin: 1rem 0 !important;
    }}
    
    div[data-testid="metric-container"] label {{
        color: {COLORS['dark_text_secondary']} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}
    
    div[data-testid="metric-container"] [data-testid="metric-value"] {{
        color: {COLORS['dark_text_primary']} !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }}
    
    div[data-testid="metric-container"] [data-testid="metric-delta"] {{
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}
    
    /* Chart Containers */
    .stPlotlyChart {{
        background-color: {COLORS['dark_surface']} !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        border: 2px solid {COLORS['border_dark']} !important;
        margin: 1rem 0 !important;
        padding: 0.5rem !important;
    }}
    
    /* Sidebar Styling */
    .css-1d391kg {{
        background-color: {COLORS['dark_surface']} !important;
        border-right: 3px solid {COLORS['saffron']} !important;
    }}
    
    .css-1d391kg .stMarkdown {{
        color: {COLORS['dark_text_primary']} !important;
    }}
    
    /* Info Boxes */
    .info-box {{
        background: linear-gradient(135deg, {COLORS['info']}15 0%, {COLORS['info']}25 100%);
        border: 2px solid {COLORS['info']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: {COLORS['dark_text_primary']};
    }}
    
    .success-box {{
        background: linear-gradient(135deg, {COLORS['success']}15 0%, {COLORS['success']}25 100%);
        border: 2px solid {COLORS['success']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: {COLORS['dark_text_primary']};
    }}
    
    .warning-box {{
        background: linear-gradient(135deg, {COLORS['warning']}15 0%, {COLORS['warning']}25 100%);
        border: 2px solid {COLORS['warning']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: {COLORS['dark_text_primary']};
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background-color: {COLORS['dark_surface']} !important;
        border: 2px solid {COLORS['border_dark']} !important;
        border-radius: 10px !important;
        color: {COLORS['dark_text_primary']} !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: {COLORS['dark_surface_light']} !important;
        border: 1px solid {COLORS['border_dark']} !important;
        color: {COLORS['dark_text_primary']} !important;
    }}
    
    /* DataFrame Styling */
    .dataframe {{
        background-color: {COLORS['dark_surface']} !important;
        color: {COLORS['dark_text_primary']} !important;
        border: 1px solid {COLORS['border_dark']} !important;
        border-radius: 8px !important;
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['saffron']} 0%, {COLORS['green']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 153, 51, 0.3);
    }}
</style>
""", unsafe_allow_html=True)
def calculate_moving_averages(data, windows):
    """Calculate Simple Moving Averages for given time windows"""
    ma_data = pd.DataFrame(index=data.index)
    for window in windows:
        if len(data) >= window:
            ma_data[f'MA_{window}'] = data.rolling(window=window, min_periods=window).mean()
        else:
            ma_data[f'MA_{window}'] = None
    return ma_data

def calculate_technical_indicators(data, indicator_key):
    """Calculate various technical indicators for a given data series"""
    results = {}
    if len(data) > 1:
        daily_returns = data.pct_change()
        results['volatility'] = daily_returns.std() * 100 * np.sqrt(252)
        results['total_return'] = ((data.iloc[-1] / data.iloc[0]) - 1) * 100
        results['max_drawdown'] = ((data.cummax() - data) / data.cummax()).max() * 100
    
    return results

def create_professional_chart(data, title, color, ma_windows=None, chart_height=400):
    """Create enhanced chart with optional technical indicators"""
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean_data.index,
        y=clean_data.values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=clean_data.index,
        y=clean_data.values,
        mode='lines',
        line=dict(color=color, width=3, shape='linear'),
        name=title,
        hovertemplate='<b>%{y:,.2f}</b><br>%{x|%d %b %Y}<br><extra></extra>'
    ))
    if ma_windows and len(ma_windows) > 0:
        ma_data = calculate_moving_averages(clean_data, ma_windows)
        ma_colors = ['#FFD700', '#FFA500', '#FF6347', '#9370DB']
        
        for i, window in enumerate(ma_windows):
            if f'MA_{window}' in ma_data.columns and not ma_data[f'MA_{window}'].isna().all():
                fig.add_trace(go.Scatter(
                    x=ma_data.index,
                    y=ma_data[f'MA_{window}'],
                    mode='lines',
                    line=dict(
                        color=ma_colors[i % len(ma_colors)], 
                        width=2,
                        dash='dash'
                    ),
                    name=f'MA-{window}',
                    hovertemplate=f'MA {window}: %{{y:,.2f}}<br>%{{x|%d %b %Y}}<extra></extra>'
                ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color=COLORS['dark_text_primary'], family="Inter"),
            x=0.5,
            y=0.95
        ),
        plot_bgcolor=COLORS['dark_surface'],
        paper_bgcolor=COLORS['dark_surface'],
        font=dict(color=COLORS['dark_text_primary'], family="Inter"),
        
        xaxis=dict(
            gridcolor=COLORS['grid_dark'],
            linecolor=COLORS['border_dark'],
            tickfont=dict(color=COLORS['dark_text_secondary'], size=11),
            showgrid=True,
            gridwidth=1,
            zeroline=False
        ),
        
        yaxis=dict(
            gridcolor=COLORS['grid_dark'],
            linecolor=COLORS['border_dark'],
            tickfont=dict(color=COLORS['dark_text_secondary'], size=11),
            showgrid=True,
            gridwidth=1,
            zeroline=False
        ),
        
        margin=dict(l=60, r=60, t=60, b=50),
        hovermode='x unified',
        height=chart_height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.1)',
            bordercolor=COLORS['border_dark'],
            borderwidth=1
        )
    )
    
    return fig
def create_correlation_heatmap(df, selected_indicators):
    """Create an interactive correlation heatmap"""
    if len(df.columns) < 2:
        return go.Figure()
    returns_df = df.pct_change().dropna()
    corr_matrix = returns_df.corr()
    labels = [INDIAN_INDICATORS[col]['name'] for col in corr_matrix.columns if col in INDIAN_INDICATORS]
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmin=-1, 
        zmax=1,
        zmid=0,
        hoverongaps=False,
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(
            title="Correlation Coefficient",
            titlefont=dict(color=COLORS['dark_text_primary']),
            tickfont=dict(color=COLORS['dark_text_secondary'])
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Indicator Correlation Matrix</b><br><sub>Based on Daily Percentage Changes</sub>",
            font=dict(size=18, color=COLORS['dark_text_primary']),
            x=0.5
        ),
        plot_bgcolor=COLORS['dark_surface'],
        paper_bgcolor=COLORS['dark_surface'],
        font=dict(color=COLORS['dark_text_primary']),
        height=600,
        margin=dict(l=100, r=100, t=100, b=50),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(color=COLORS['dark_text_secondary']),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['dark_text_secondary']),
            showgrid=False
        )
    )
    
    return fig
# ===== FIXED: Correlation Analysis Functions =====
def create_correlation_heatmap(df, selected_indicators):
    """Create an interactive correlation heatmap"""
    if len(df.columns) < 2:
        return go.Figure()
    returns_df = df.pct_change().dropna()
    corr_matrix = returns_df.corr()
    labels = [INDIAN_INDICATORS[col]['name'] for col in corr_matrix.columns if col in INDIAN_INDICATORS]
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmin=-1, 
        zmax=1,
        zmid=0,
        hoverongaps=False,
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(
            title=dict(
                text="Correlation Coefficient",
                font=dict(
                    color=COLORS['dark_text_primary'],
                    size=14,
                    family="Inter"
                )
            ),
            tickfont=dict(
                color=COLORS['dark_text_secondary'],
                size=12
            ),
            thickness=20,
            len=0.8,
            outlinecolor=COLORS['border_dark'],
            outlinewidth=1
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Indicator Correlation Matrix</b><br><sub>Based on Daily Percentage Changes</sub>",
            font=dict(size=18, color=COLORS['dark_text_primary']),
            x=0.5
        ),
        plot_bgcolor=COLORS['dark_surface'],
        paper_bgcolor=COLORS['dark_surface'],
        font=dict(color=COLORS['dark_text_primary']),
        height=600,
        margin=dict(l=100, r=100, t=100, b=50),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(color=COLORS['dark_text_secondary']),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['dark_text_secondary']),
            showgrid=False
        )
    )
    
    return fig


def get_correlation_insights(df, selected_indicators):
    """Generate text insights about correlations"""
    if len(df.columns) < 2:
        return []
    returns_df = df.pct_change().dropna()
    corr_matrix = returns_df.corr()
    insights = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            indicator1 = INDIAN_INDICATORS[corr_matrix.columns[i]]['name']
            indicator2 = INDIAN_INDICATORS[corr_matrix.columns[j]]['name']
            
            if corr_val > 0.7:
                insights.append(f"🔗 **Strong positive correlation** between {indicator1} and {indicator2} ({corr_val:.2f})")
            elif corr_val < -0.7:
                insights.append(f"🔄 **Strong negative correlation** between {indicator1} and {indicator2} ({corr_val:.2f})")
    
    return insights

@st.cache_data
def get_indian_market_data(selected_indicators=None, days_back=365):
    """Fetch data for selected indicators with custom time period"""
    
    if selected_indicators is None:
        selected_indicators = ["NIFTY_50", "INR_USD"]
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    data_dict = {}
    status_info = []
    
    for indicator_key in selected_indicators:
        if indicator_key in INDIAN_INDICATORS:
            try:
                indicator = INDIAN_INDICATORS[indicator_key]
                ticker = indicator["ticker"]
                
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty and len(hist) >= 10:
                    clean_data = hist['Close'].fillna(method='ffill').fillna(method='bfill')
                    data_dict[indicator_key] = clean_data
                    status_info.append(f"✅ {indicator['name']}: {len(clean_data)} data points loaded")
                else:
                    status_info.append(f"⚠️ {indicator['name']}: Insufficient data")
                    
            except Exception as e:
                status_info.append(f"❌ {indicator['name']}: {str(e)[:50]}...")
    
    if data_dict:
        df = pd.DataFrame(data_dict)
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df, status_info
    else:
        return pd.DataFrame(), status_info
@st.cache_data
def create_realistic_sample_data(selected_indicators=None, days_back=365):
    """Create realistic sample data for selected indicators"""
    
    if selected_indicators is None:
        selected_indicators = ["NIFTY_50", "INR_USD"]
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    data_dict = {}
    sample_params = {
        "NIFTY_50": {"base": 22000, "volatility": 0.02},
        "INR_USD": {"base": 83.0, "volatility": 0.003},
        "BANK_NIFTY": {"base": 45000, "volatility": 0.025},
        "GOLD": {"base": 2000, "volatility": 0.015},
        "CRUDE_OIL": {"base": 75, "volatility": 0.03},
        "RELIANCE": {"base": 2500, "volatility": 0.02},
        "TCS": {"base": 3500, "volatility": 0.018},
        "HDFC_BANK": {"base": 1600, "volatility": 0.02}
    }
    
    for indicator_key in selected_indicators:
        params = sample_params.get(indicator_key, {"base": 1000, "volatility": 0.02})
        returns = np.random.normal(0, params["volatility"], len(dates))
        prices = [params["base"]]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        data_dict[indicator_key] = pd.Series(prices[1:], index=dates)
    
    return pd.DataFrame(data_dict)

st.sidebar.header("📊 Dashboard Controls")

st.sidebar.subheader("📅 Time Period")
selected_range = st.sidebar.selectbox(
    "Select time period:",
    options=list(DATE_RANGES.keys()),
    index=3, 
    help="Choose how far back to fetch data"
)
days_back = DATE_RANGES[selected_range]

st.sidebar.subheader("📈 Select Indicators")
selected_indicators = []

for category, indicators in CATEGORIES.items():
    st.sidebar.markdown(f"**{category}:**")
    
    for indicator_key in indicators:
        indicator = INDIAN_INDICATORS[indicator_key]
        default_selected = indicator_key in ['NIFTY_50', 'INR_USD']
        
        is_selected = st.sidebar.checkbox(
            f"{indicator['name']}",
            value=default_selected,
            key=f"cb_{indicator_key}",
            help=indicator['description']
        )
        
        if is_selected:
            selected_indicators.append(indicator_key)
st.sidebar.subheader("🎛️ Display Mode")
display_mode = st.sidebar.radio(
    "Choose display style:",
    ["Individual Charts", "Grid View"],
    help="How to display selected indicators"
)

st.sidebar.subheader("ℹ️ Selection Info")
st.sidebar.info(f"""
**Selected Period:** {selected_range}
**Indicators:** {len(selected_indicators)} selected
**Display:** {display_mode}
""")

st.sidebar.markdown(f"""
<div style="background: linear-gradient(135deg, {COLORS['dark_surface']} 0%, {COLORS['dark_surface_light']} 100%); 
           padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; 
           border: 2px solid {COLORS['saffron']};">
    <h3 style="color: {COLORS['saffron']}; margin-top: 0; font-weight: 600;">📊 Market Overview</h3>
    <div style="color: {COLORS['dark_text_primary']}; line-height: 1.6;">
        <p><strong style="color: {COLORS['nifty_color']};">📈 NIFTY 50</strong><br>
        Benchmark index of top 50 Indian companies</p>
        <p><strong style="color: {COLORS['inr_color']};">💱 INR/USD</strong><br>
        Indian Rupee to US Dollar exchange rate</p>
        <p><strong style="color: {COLORS['info']};">⏰ Trading Hours</strong><br>
        Mon-Fri: 9:15 AM - 3:30 PM IST</p>
    </div>
</div>
""", unsafe_allow_html=True)

current_time = datetime.datetime.now()
market_open = current_time.weekday() < 5 and 9.25 <= current_time.hour + current_time.minute/60 <= 15.5

st.sidebar.markdown(f"""
<div style="background: {'linear-gradient(135deg, ' + COLORS['success'] + '15 0%, ' + COLORS['success'] + '25 100%)' if market_open else 'linear-gradient(135deg, ' + COLORS['warning'] + '15 0%, ' + COLORS['warning'] + '25 100%)'};
           border: 2px solid {COLORS['success'] if market_open else COLORS['warning']};
           border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: {COLORS['dark_text_primary']}; font-weight: 600; text-align: center;">
        <strong>🕐 {current_time.strftime('%d %B %Y')}</strong><br>
        <span style="font-size: 1.1em;">{current_time.strftime('%I:%M %p IST')}</span><br>
        <span style="color: {COLORS['success'] if market_open else COLORS['warning']};">
        {'🟢 Market Open' if market_open else '🔴 Market Closed'}
        </span>
    </p>
</div>
""", unsafe_allow_html=True)

use_live_data = st.sidebar.toggle("📡 Use Live Data", value=True, help="Switch between live and sample data")

st.markdown("""
<div class="main-header">
    <h1>🇮🇳 Indian Economic Dashboard Pro</h1>
    <p>Multi-Indicator Analysis & Real-time Market Data</p>
</div>
""", unsafe_allow_html=True)

if not selected_indicators:
    st.warning("👆 Please select at least one indicator from the sidebar to begin analysis.")
    st.stop()
with st.spinner(f'🔄 Loading {len(selected_indicators)} indicators for {selected_range}...'):
    
    if use_live_data:
        df, status_messages = get_indian_market_data(selected_indicators, days_back)
        for msg in status_messages:
            if "✅" in msg:
                st.success(msg)
            elif "⚠️" in msg:
                st.warning(msg)
            elif "❌" in msg:
                st.error(msg)
        
        if df.empty:
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="color: {COLORS['warning']}; margin-top: 0;">⚠️ Live Data Temporarily Unavailable</h4>
                <p>Switching to high-quality sample data for demonstration.</p>
                <p><strong>Possible reasons:</strong> Market closed, API limits, or connectivity issues.</p>
            </div>
            """, unsafe_allow_html=True)
            df = create_realistic_sample_data(selected_indicators, days_back)
            data_source = "Sample Data"
        else:
            data_source = "Live Market Data"
    else:
        df = create_realistic_sample_data(selected_indicators, days_back)
        data_source = "Sample Data"
if not df.empty:
    st.markdown(f"""
    <div class="success-box">
        <h4 style="color: {COLORS['success']}; margin-top: 0;">✅ Dashboard Ready</h4>
        <p><strong>Source:</strong> {data_source} • <strong>Records:</strong> {len(df):,} • <strong>Updated:</strong> {df.index[-1].strftime('%d %B %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("📊 Portfolio Overview")
    metric_cols = st.columns(min(len(selected_indicators), 4))
    
    for i, indicator_key in enumerate(selected_indicators[:4]):
        if indicator_key in df.columns:
            indicator = INDIAN_INDICATORS[indicator_key]
            current_val = df[indicator_key].iloc[-1]
            prev_val = df[indicator_key].iloc[-2] if len(df) > 1 else current_val
            change = current_val - prev_val
            pct_change = (change / prev_val) * 100 if prev_val != 0 else 0
            
            with metric_cols[i]:
                st.metric(
                    label=indicator['name'],
                    value=f"{current_val:.2f}",
                    delta=f"{pct_change:+.2f}%"
                )
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 **Technical Analysis**", 
        "🔗 **Correlation Matrix**", 
        "📊 **Market Analytics**", 
        "💾 **Data Export**"
    ])
    with tab1:
        st.header("📈 Technical Analysis")
        st.markdown("**Analyze trends with price charts and moving averages**")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            show_ma = st.checkbox("📊 Show Moving Averages", value=False)
            
        with col2:
            ma_windows = []
            if show_ma:
                ma_windows = st.multiselect(
                    'Select MA Periods:',
                    options=[5, 10, 20, 50, 100, 200],
                    default=[20, 50],
                    help="Moving Average periods in days"
                )
        if display_mode == "Individual Charts":
            for indicator_key in selected_indicators:
                if indicator_key in df.columns:
                    indicator = INDIAN_INDICATORS[indicator_key]
                    with st.expander(f"📊 {indicator['name']} - Technical Analysis", expanded=True):
                        st.markdown(f"**{indicator['description']}**")
                        chart = create_professional_chart(
                            df[indicator_key], 
                            indicator['name'], 
                            indicator['color'],
                            ma_windows if show_ma else None
                        )
                        st.plotly_chart(chart, use_container_width=True)
                        tech_indicators = calculate_technical_indicators(df[indicator_key], indicator_key)
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            current = df[indicator_key].iloc[-1]
                            st.metric("Current Value", f"{current:.2f} {indicator['unit']}")
                        
                        with metric_col2:
                            if 'volatility' in tech_indicators:
                                st.metric("Volatility (Annual)", f"{tech_indicators['volatility']:.1f}%")
                        
                        with metric_col3:
                            if 'total_return' in tech_indicators:
                                st.metric("Total Return", f"{tech_indicators['total_return']:+.1f}%")
        
        elif display_mode == "Grid View":
            st.subheader("Grid View - Technical Charts")
            
            cols_per_row = 2
            rows_needed = (len(selected_indicators) + cols_per_row - 1) // cols_per_row
            
            for row in range(rows_needed):
                cols = st.columns(cols_per_row)
                
                for col_idx in range(cols_per_row):
                    indicator_idx = row * cols_per_row + col_idx
                    
                    if indicator_idx < len(selected_indicators):
                        indicator_key = selected_indicators[indicator_idx]
                        
                        if indicator_key in df.columns:
                            indicator = INDIAN_INDICATORS[indicator_key]
                            
                            with cols[col_idx]:
                                chart = create_professional_chart(
                                    df[indicator_key], 
                                    indicator['name'], 
                                    indicator['color'],
                                    ma_windows if show_ma else None,
                                    chart_height=350
                                )
                                st.plotly_chart(chart, use_container_width=True)
    with tab2:
        st.header("🔗 Correlation Analysis")
        st.markdown("**Understand relationships between different economic indicators**")
        
        if len(selected_indicators) >= 2:
            corr_chart = create_correlation_heatmap(df, selected_indicators)
            st.plotly_chart(corr_chart, use_container_width=True)
            st.subheader("💡 Key Insights")
            insights = get_correlation_insights(df, selected_indicators)
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("📊 **Moderate correlations detected** - No extremely strong relationships found among selected indicators.")
            with st.expander("❓ Understanding Correlation Values", expanded=False):
                st.markdown("""
                **Correlation Coefficient Guide:**
                - **+0.7 to +1.0**: Strong positive correlation (move together)
                - **+0.3 to +0.7**: Moderate positive correlation  
                - **-0.3 to +0.3**: Weak/no correlation (independent movement)
                - **-0.7 to -0.3**: Moderate negative correlation
                - **-1.0 to -0.7**: Strong negative correlation (move opposite)
                
                **Financial Interpretation:**
                - High correlation = Similar risk/return patterns
                - Low correlation = Better diversification potential
                - Negative correlation = Natural hedge opportunities
                """)
                
        else:
            st.warning("⚠️ Please select at least 2 indicatorsa to analyze correlations.")
    with tab3:
        st.header("📊 Market Analytics Dashboard")
        st.markdown("**Statistical analysis and performance metrics**")
        analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
        
        with analytics_col1:
            st.metric("📈 Data Points", f"{len(df):,}")
        
        with analytics_col2:
            period = (df.index[-1] - df.index[0]).days
            st.metric("📅 Analysis Period", f"{period} days")
        
        with analytics_col3:
            avg_indicators = len(selected_indicators)
            st.metric("📊 Indicators Tracked", f"{avg_indicators}")
        
        with analytics_col4:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("🎯 Data Completeness", f"{completeness:.1f}%")
        st.subheader("📋 Statistical Analysis")
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.markdown("### Recent Performance (Last 10 Days)")
            recent_df = df.tail(10).copy()
            recent_df.index = recent_df.index.strftime('%Y-%m-%d')
            display_df = recent_df.copy()
            display_df.columns = [INDIAN_INDICATORS[col]['name'] for col in display_df.columns if col in INDIAN_INDICATORS]
            
            st.dataframe(display_df, use_container_width=True)
        
        with stats_col2:
            st.markdown("### Statistical Summary")
            stats = df.describe()
            stats.columns = [INDIAN_INDICATORS[col]['name'] for col in stats.columns if col in INDIAN_INDICATORS]
            
            st.dataframe(stats, use_container_width=True)
    with tab4:
        st.header("💾 Data Export & Analysis Tools")
        st.markdown("**Download data for external analysis**")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.subheader("📊 Export Raw Data")
            export_df = df.copy()
            export_df.columns = [INDIAN_INDICATORS[col]['name'] for col in export_df.columns if col in INDIAN_INDICATORS]
            
            csv = export_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f'indian_economic_indicators_{selected_range.lower().replace(" ", "_")}_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                help="Download complete dataset for analysis in Excel, Python, etc."
            )
            
            st.info(f"**File contains:** {len(df)} rows × {len(df.columns)} indicators covering {selected_range}")
        
        with export_col2:
            st.subheader("🔍 Data Preview")
            st.dataframe(
                export_df.head(5), 
                use_container_width=True
            )
            st.markdown("#### Data Quality Report")
            quality_info = []
            
            for col in df.columns:
                if col in INDIAN_INDICATORS:
                    null_pct = (df[col].isnull().sum() / len(df)) * 100
                    quality_info.append({
                        'Indicator': INDIAN_INDICATORS[col]['name'],
                        'Completeness': f"{100-null_pct:.1f}%",
                        'Data Points': len(df[col].dropna())
                    })
            
            quality_df = pd.DataFrame(quality_info)
            st.dataframe(quality_df, use_container_width=True)

else:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['error']}15 0%, {COLORS['error']}25 100%);
               border: 2px solid {COLORS['error']}; border-radius: 12px; 
               padding: 2rem; text-align: center; margin: 2rem 0;">
        <h3 style="color: {COLORS['error']};">❌ Data Loading Failed</h3>
        <p style="color: {COLORS['dark_text_primary']};">Unable to load market data. Please check your connection and try again.</p>
    </div>
    """, unsafe_allow_html=True)
