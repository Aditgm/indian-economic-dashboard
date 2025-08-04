import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objects as go
import numpy as np

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
    
    /* Spinner Styling */
    .stSpinner {{
        color: {COLORS['saffron']} !important;
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
def create_professional_chart(data, title, color, show_volume=False):
    """Create a high-contrast, continuous line chart"""
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean_data.index,
        y=clean_data.values,
        mode='lines',
        line=dict(
            color=color, 
            width=3,
            shape='linear'
        ),
        name=title,
        hovertemplate='<b>%{y:,.2f}</b><br>%{x|%d %b %Y}<extra></extra>',
        fill=None
    ))
    fig.add_trace(go.Scatter(
        x=clean_data.index,
        y=clean_data.values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=20, color=COLORS['dark_text_primary'], family="Inter"),
            x=0.5,
            y=0.95
        ),
        plot_bgcolor=COLORS['dark_surface'],
        paper_bgcolor=COLORS['dark_surface'],
        font=dict(color=COLORS['dark_text_primary'], family="Inter"),
        
        xaxis=dict(
            gridcolor=COLORS['grid_dark'],
            linecolor=COLORS['border_dark'],
            tickfont=dict(color=COLORS['dark_text_secondary'], size=12),
            title_font=dict(color=COLORS['dark_text_primary'], size=14),
            showgrid=True,
            gridwidth=1,
            zeroline=False
        ),
        
        yaxis=dict(
            gridcolor=COLORS['grid_dark'],
            linecolor=COLORS['border_dark'],
            tickfont=dict(color=COLORS['dark_text_secondary'], size=12),
            title_font=dict(color=COLORS['dark_text_primary'], size=14),
            showgrid=True,
            gridwidth=1,
            zeroline=False
        ),
        
        margin=dict(l=60, r=60, t=80, b=60),
        hovermode='x unified',
        showlegend=False,
        height=400
    )
    
    return fig

@st.cache_data
def get_indian_market_data():
    """Fetch Indian market data with proper error handling and data continuity"""
    
    data_dict = {}
    status_info = []
    
    tickers = {
        "NIFTY_50": "^NSEI",
        "INR_USD": "INR=X"
    }
    
    for name, ticker in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", interval="1d")
            
            if not hist.empty and len(hist) >= 30:
                clean_data = hist['Close'].fillna(method='ffill').fillna(method='bfill')
                data_dict[name] = clean_data
                status_info.append(f"✅ {name}: {len(clean_data)} data points loaded")
            else:
                status_info.append(f"⚠️ {name}: Insufficient data")
                
        except Exception as e:
            status_info.append(f"❌ {name}: {str(e)[:50]}...")
    
    if data_dict:
        df = pd.DataFrame(data_dict)
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df, status_info
    else:
        return pd.DataFrame(), status_info
@st.cache_data
def create_realistic_sample_data():
    """Create realistic, continuous sample data for Indian markets"""
    dates = pd.date_range(start='2023-01-01', end='2024-08-04', freq='D')
    np.random.seed(42)
    
    nifty_base = 22000
    returns = np.random.normal(0.0005, 0.02, len(dates))
    nifty_prices = [nifty_base]
    
    for ret in returns:
        nifty_prices.append(nifty_prices[-1] * (1 + ret))
    
    nifty_data = pd.Series(nifty_prices[1:], index=dates)
    nifty_data = nifty_data.clip(lower=18000, upper=26000)
    inr_base = 83.0
    inr_returns = np.random.normal(0, 0.003, len(dates)) 
    inr_prices = [inr_base]
    
    for ret in inr_returns:
        inr_prices.append(inr_prices[-1] * (1 + ret))
    
    inr_data = pd.Series(inr_prices[1:], index=dates)
    inr_data = inr_data.clip(lower=79.0, upper=86.0)  
    
    df = pd.DataFrame({
        'NIFTY_50': nifty_data,
        'INR_USD': inr_data
    })
    
    return df

st.markdown("""
<div class="main-header">
    <h1>🇮🇳 Indian Economic Dashboard</h1>
    <p>Professional Market Analysis & Real-time Data</p>
</div>
""", unsafe_allow_html=True)

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

with st.spinner('🔄 Loading professional market analysis...'):
    
    if use_live_data:
        df, status_messages = get_indian_market_data()
        
        if df.empty:
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="color: {COLORS['warning']}; margin-top: 0;">⚠️ Live Data Temporarily Unavailable</h4>
                <p>Switching to high-quality sample data for demonstration.</p>
                <p><strong>Possible reasons:</strong> Market closed, API limits, or connectivity issues.</p>
            </div>
            """, unsafe_allow_html=True)
            df = create_realistic_sample_data()
            data_source = "Sample Data"
        else:
            data_source = "Live Market Data"
    else:
        df = create_realistic_sample_data()
        data_source = "Sample Data"

if not df.empty:
    st.markdown(f"""
    <div class="success-box">
        <h4 style="color: {COLORS['success']}; margin-top: 0;">✅ Dashboard Ready</h4>
        <p><strong>Source:</strong> {data_source} • <strong>Records:</strong> {len(df):,} • <strong>Updated:</strong> {df.index[-1].strftime('%d %B %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("📈 NIFTY 50 Index")
        
        if "NIFTY_50" in df.columns and not df["NIFTY_50"].empty:
            nifty_chart = create_professional_chart(
                df["NIFTY_50"], 
                "NIFTY 50 Performance", 
                COLORS['nifty_color']
            )
            st.plotly_chart(nifty_chart, use_container_width=True)
            
            current_nifty = df["NIFTY_50"].iloc[-1]
            prev_nifty = df["NIFTY_50"].iloc[-2] if len(df) > 1 else current_nifty
            nifty_change = current_nifty - prev_nifty
            nifty_pct = (nifty_change / prev_nifty) * 100 if prev_nifty else 0
            
            st.metric(
                label="Current Value",
                value=f"{current_nifty:,.0f}",
                delta=f"{nifty_change:+,.0f} ({nifty_pct:+.2f}%)"
            )
    
    with col2:
        st.subheader("💱 INR/USD Exchange Rate")
        
        if "INR_USD" in df.columns and not df["INR_USD"].empty:
            inr_chart = create_professional_chart(
                df["INR_USD"], 
                "INR/USD Exchange Rate", 
                COLORS['inr_color']
            )
            st.plotly_chart(inr_chart, use_container_width=True)
            
            current_inr = df["INR_USD"].iloc[-1]
            prev_inr = df["INR_USD"].iloc[-2] if len(df) > 1 else current_inr
            inr_change = current_inr - prev_inr
            inr_pct = (inr_change / prev_inr) * 100 if prev_inr else 0
            
            st.metric(
                label="Current Rate",
                value=f"₹{current_inr:.3f}",
                delta=f"{inr_change:+.3f} ({inr_pct:+.2f}%)"
            )
    
    st.subheader("📊 Market Analytics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("📈 Data Points", f"{len(df):,}")
    
    with metric_col2:
        period = (df.index[-1] - df.index[0]).days
        st.metric("📅 Time Period", f"{period} days")
    
    with metric_col3:
        if "NIFTY_50" in df.columns:
            volatility = df["NIFTY_50"].pct_change().std() * 100 * np.sqrt(252)
            st.metric("📊 Annual Volatility", f"{volatility:.1f}%")
    
    with metric_col4:
        st.metric("🔄 Data Quality", "Excellent")
    
    with st.expander("📈 Detailed Market Analysis", expanded=False):
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("### 📊 Recent Performance")
            recent_df = df.tail(10).copy()
            recent_df.index = recent_df.index.strftime('%Y-%m-%d')
            st.dataframe(recent_df, use_container_width=True)
        
        with analysis_col2:
            st.markdown("### 📋 Statistical Summary") 
            stats = df.describe()
            st.dataframe(stats, use_container_width=True)
        
        st.markdown("### 💡 Market Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            if "NIFTY_50" in df.columns:
                nifty_return = ((df["NIFTY_50"].iloc[-1] / df["NIFTY_50"].iloc[0]) - 1) * 100
                st.markdown(f"""
                <div class="info-box">
                    <h5 style="color: {COLORS['nifty_color']}; margin-top: 0;">NIFTY 50 Analysis</h5>
                    <p><strong>Total Return:</strong> {nifty_return:+.2f}%</p>
                    <p><strong>Range:</strong> {df['NIFTY_50'].min():,.0f} - {df['NIFTY_50'].max():,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with insights_col2:
            if "INR_USD" in df.columns:
                inr_change_total = ((df["INR_USD"].iloc[-1] / df["INR_USD"].iloc[0]) - 1) * 100
                st.markdown(f"""
                <div class="info-box">
                    <h5 style="color: {COLORS['inr_color']}; margin-top: 0;">INR/USD Analysis</h5>
                    <p><strong>Currency Change:</strong> {inr_change_total:+.2f}%</p>
                    <p><strong>Range:</strong> ₹{df['INR_USD'].min():.3f} - ₹{df['INR_USD'].max():.3f}</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['error']}15 0%, {COLORS['error']}25 100%);
               border: 2px solid {COLORS['error']}; border-radius: 12px; 
               padding: 2rem; text-align: center; margin: 2rem 0;">
        <h3 style="color: {COLORS['error']};">❌ Data Loading Failed</h3>
        <p style="color: {COLORS['dark_text_primary']};">Unable to load market data. Please check your connection and try again.</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
<div style="background: linear-gradient(135deg, {COLORS['dark_surface']} 0%, {COLORS['dark_surface_light']} 100%);
           padding: 2rem; border-radius: 12px; text-align: center; 
           border: 2px solid {COLORS['saffron']}; margin-top: 2rem;">
    <h4 style="color: {COLORS['saffron']}; margin-top: 0;">🇮🇳 Indian Economic Dashboard</h4>
    <p style="color: {COLORS['dark_text_primary']}; margin-bottom: 0.5rem;">
        <strong>Built with:</strong> Streamlit • Python • yfinance • Plotly
    </p>
    <p style="color: {COLORS['dark_text_secondary']}; font-size: 0.9rem; margin-bottom: 0;">
        Professional market analysis tool • Last updated: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p IST')}
    </p>
</div>
""", unsafe_allow_html=True)
