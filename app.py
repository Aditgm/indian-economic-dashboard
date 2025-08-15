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
import concurrent.futures
from threading import Lock
import difflib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_performance(func_name: str):
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

def get_kaleido():
    try:
        import kaleido
        return True
    except ImportError:
        return False

BSE_STOCKS = {
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT", "ticker": "TCS.NS"},
    "INFY.NS": {"name": "Infosys Limited", "sector": "IT", "ticker": "INFY.NS"},
    "WIPRO.NS": {"name": "Wipro Limited", "sector": "IT", "ticker": "WIPRO.NS"},
    "HCLTECH.NS": {"name": "HCL Technologies", "sector": "IT", "ticker": "HCLTECH.NS"},
    "TECHM.NS": {"name": "Tech Mahindra", "sector": "IT", "ticker": "TECHM.NS"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Banking", "ticker": "HDFCBANK.NS"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Banking", "ticker": "ICICIBANK.NS"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "Banking", "ticker": "SBIN.NS"},
    "AXISBANK.NS": {"name": "Axis Bank", "sector": "Banking", "ticker": "AXISBANK.NS"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Banking", "ticker": "KOTAKBANK.NS"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG", "ticker": "HINDUNILVR.NS"},
    "ITC.NS": {"name": "ITC Limited", "sector": "FMCG", "ticker": "ITC.NS"},
    "NESTLEIND.NS": {"name": "Nestle India", "sector": "FMCG", "ticker": "NESTLEIND.NS"},
    "BRITANNIA.NS": {"name": "Britannia Industries", "sector": "FMCG", "ticker": "BRITANNIA.NS"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Auto", "ticker": "MARUTI.NS"},
    "TATAMOTORS.NS": {"name": "Tata Motors", "sector": "Auto", "ticker": "TATAMOTORS.NS"},
    "M&M.NS": {"name": "Mahindra & Mahindra", "sector": "Auto", "ticker": "M&M.NS"},
    "BAJAJ-AUTO.NS": {"name": "Bajaj Auto", "sector": "Auto", "ticker": "BAJAJ-AUTO.NS"},
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy", "ticker": "RELIANCE.NS"},
    "ONGC.NS": {"name": "Oil & Natural Gas Corp", "sector": "Energy", "ticker": "ONGC.NS"},
    "IOC.NS": {"name": "Indian Oil Corporation", "sector": "Energy", "ticker": "IOC.NS"},
    "BPCL.NS": {"name": "Bharat Petroleum", "sector": "Energy", "ticker": "BPCL.NS"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel", "sector": "Telecom", "ticker": "BHARTIARTL.NS"},
    "IDEA.NS": {"name": "Vodafone Idea", "sector": "Telecom", "ticker": "IDEA.NS"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Pharma", "ticker": "SUNPHARMA.NS"},
    "DRREDDY.NS": {"name": "Dr. Reddy's Labs", "sector": "Pharma", "ticker": "DRREDDY.NS"},
    "CIPLA.NS": {"name": "Cipla Limited", "sector": "Pharma", "ticker": "CIPLA.NS"},
    "LUPIN.NS": {"name": "Lupin Limited", "sector": "Pharma", "ticker": "LUPIN.NS"},
    "LT.NS": {"name": "Larsen & Toubro", "sector": "Infrastructure", "ticker": "LT.NS"},
    "ULTRACEMCO.NS": {"name": "UltraTech Cement", "sector": "Cement", "ticker": "ULTRACEMCO.NS"},
    "ADANIPORTS.NS": {"name": "Adani Ports", "sector": "Infrastructure", "ticker": "ADANIPORTS.NS"},
    "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Steel", "ticker": "TATASTEEL.NS"},
    "HINDALCO.NS": {"name": "Hindalco Industries", "sector": "Metals", "ticker": "HINDALCO.NS"},
    "JSWSTEEL.NS": {"name": "JSW Steel", "sector": "Steel", "ticker": "JSWSTEEL.NS"}
}

def search_bse_stocks(query: str, max_results: int = 10):
    if not query or len(query) < 2:
        return []
    query = query.upper().strip()
    results = []
    for ticker, info in BSE_STOCKS.items():
        if query in ticker.upper():
            results.append({
                'ticker': ticker,
                'name': info['name'],
                'sector': info['sector'],
                'match_type': 'ticker',
                'score': 100
            })
        elif query in info['name'].upper():
            results.append({
                'ticker': ticker,
                'name': info['name'],
                'sector': info['sector'],
                'match_type': 'name',
                'score': 90
            })
    if len(results) < max_results:
        all_names = [info['name'] for info in BSE_STOCKS.values()]
        fuzzy_matches = difflib.get_close_matches(query, all_names, n=max_results-len(results), cutoff=0.6)
        for match in fuzzy_matches:
            for ticker, info in BSE_STOCKS.items():
                if info['name'] == match:
                    if not any(r['ticker'] == ticker for r in results):
                        results.append({
                            'ticker': ticker,
                            'name': info['name'],
                            'sector': info['sector'],
                            'match_type': 'fuzzy',
                            'score': 80
                        })
                    break
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def get_popular_stocks():
    popular = {
        'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS'],
        'Energy': ['RELIANCE.NS', 'ONGC.NS'],
        'Auto': ['MARUTI.NS', 'TATAMOTORS.NS'],
        'FMCG': ['HINDUNILVR.NS', 'ITC.NS']
    }
    return popular

def add_custom_stock_to_config(ticker: str, stock_info: dict):
    clean_key = ticker.replace('.NS', '').upper()
    existing_in_categories = False
    for category, indicators in CATEGORIES.items():
        if clean_key in indicators:
            existing_in_categories = True
            break
    if not existing_in_categories and clean_key not in INDIAN_INDICATORS:
        INDIAN_INDICATORS[clean_key] = {
            'name': stock_info['name'],
            'ticker': ticker,
            'color': '#4A90E2',
            'unit': '₹',
            'description': f"{stock_info['name']} - {stock_info['sector']} sector stock"
        }
        if 'Custom Stocks' not in CATEGORIES:
            CATEGORIES['Custom Stocks'] = []
        CATEGORIES['Custom Stocks'].append(clean_key)
        return True
    return False

def ensure_unique_indicators():
    seen_indicators = set()
    for category in CATEGORIES:
        unique_indicators = []
        for indicator_key in CATEGORIES[category]:
            if indicator_key not in seen_indicators:
                unique_indicators.append(indicator_key)
                seen_indicators.add(indicator_key)
        CATEGORIES[category] = unique_indicators

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
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: {COLORS['dark_bg']};
        color: {COLORS['dark_text_primary']};
    }}
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
    .stPlotlyChart {{
        background-color: {COLORS['dark_surface']} !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        border: 2px solid {COLORS['border_dark']} !important;
        margin: 1rem 0 !important;
        padding: 0.5rem !important;
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
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['saffron']} 0%, {COLORS['green']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}
</style>
""", unsafe_allow_html=True)

def calculate_moving_averages(data, windows):
    ma_data = pd.DataFrame(index=data.index)
    for window in windows:
        if len(data) >= window:
            ma_data[f'MA_{window}'] = data.rolling(window=window, min_periods=window).mean()
        else:
            ma_data[f'MA_{window}'] = None
    return ma_data

def calculate_technical_indicators(data, indicator_key):
    results = {}
    if len(data) > 1:
        daily_returns = data.pct_change()
        results['volatility'] = daily_returns.std() * 100 * np.sqrt(252)
        results['total_return'] = ((data.iloc[-1] / data.iloc[0]) - 1) * 100
        results['max_drawdown'] = ((data.cummax() - data) / data.cummax()).max() * 100
    return results

def create_professional_chart(data, title, color, ma_windows=None, chart_height=400, line_shape="linear", show_markers=False, quick_zoom=True, performance_mode=False):
    clean_data = data.dropna()
    if len(clean_data) == 0:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=COLORS['dark_surface'],
            paper_bgcolor=COLORS['dark_surface'],
            height=chart_height,
            annotations=[dict(
                text="No data available for selected period",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color=COLORS['dark_text_secondary'], size=14)
            )]
        )
        return fig
    if performance_mode:
        mode = 'lines'
        shape = 'linear'
        quick_zoom = False
    else:
        mode = 'lines+markers' if show_markers else 'lines'
        shape = line_shape
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
        mode=mode,
        line=dict(color=color, width=3, shape=shape),
        marker=dict(size=3, color=color, opacity=0.7) if show_markers else None,
        name=title,
        hovertemplate='<b>%{y:,.2f}</b><br>%{x|%d %b %Y}<br><extra></extra>',
        connectgaps=True
    ))
    y_min = float(np.nanmin(clean_data.values))
    y_max = float(np.nanmax(clean_data.values))
    if y_max > y_min:
        pad = 0.05 * (y_max - y_min)
    else:
        pad = 0.01 if y_min == 0 else abs(0.05 * y_min)
    y_lower = y_min - pad
    y_upper = y_max + pad
    if y_upper == y_lower:
        eps = 1e-6 if y_lower == 0 else abs(1e-4 * y_lower)
        y_lower -= eps
        y_upper += eps
    if ma_windows and len(ma_windows) > 0:
        ma_data = calculate_moving_averages(clean_data, ma_windows)
        ma_colors = ['#FFD700', '#FFA500', '#FF6347', '#9370DB']
        for i, window in enumerate(ma_windows):
            if f'MA_{window}' in ma_data.columns and not ma_data[f'MA_{window}'].isna().all():
                fig.add_trace(go.Scatter(
                    x=ma_data.index,
                    y=ma_data[f'MA_{window}'],
                    mode='lines',
                    line=dict(color=ma_colors[i % len(ma_colors)], width=2, dash='dash'),
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
            zeroline=False,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            ) if quick_zoom else None,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            gridcolor=COLORS['grid_dark'],
            linecolor=COLORS['border_dark'],
            tickfont=dict(color=COLORS['dark_text_secondary'], size=11),
            showgrid=True,
            gridwidth=1,
            zeroline=False,
            autorange=False,
            range=[y_lower, y_upper]
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

def export_chart_as_png(fig, chart_title, quality="High"):
    """Export with cloud-friendly fallback"""
    try:
        import os
        is_cloud = os.getenv('STREAMLIT_SHARING_MODE') or 'streamlit.io' in os.getenv('HOSTNAME', '')
        if is_cloud:
            return None, None
        if not get_kaleido():
            return None, None
        scale_map = {"Standard": 1, "High": 2, "Ultra": 3}
        scale = scale_map.get(quality, 2)
        png_bytes = fig.to_image(format="png", width=1200, height=600, scale=scale)
        safe_title = "".join(c for c in chart_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_').lower()
        filename = f"{safe_title}_chart.png"
        return png_bytes, filename
    except Exception as e:
        st.error(f"Export failed: {str(e)}")
        return None, None
def create_export_button(fig, chart_title, quality_setting):
    """Create export button with cloud detection"""
    import os
    is_cloud = os.getenv('STREAMLIT_SHARING_MODE') or 'streamlit.io' in os.getenv('HOSTNAME', '')
    if is_cloud:
        st.caption("📱 PNG export available in local deployment only")
        return
    png_bytes, filename = export_chart_as_png(fig, chart_title, quality_setting)
    if png_bytes:
        st.download_button(
            label="📥 Download Chart (PNG)",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
            help=f"Download as {quality_setting} quality PNG"
        )
    else:
        st.caption("💡 Install kaleido locally: `pip install kaleido`")

def create_correlation_heatmap(df, selected_indicators):
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
                font=dict(color=COLORS['dark_text_primary'], size=14, family="Inter")
            ),
            tickfont=dict(color=COLORS['dark_text_secondary'], size=12),
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

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
@monitor_performance("Parallel Data Fetch")
def get_optimized_market_data(selected_indicators: List[str], days_back: int = 365) -> Tuple[pd.DataFrame, List[str], Dict]:
    if not selected_indicators:
        selected_indicators = ["NIFTY_50", "INR_USD"]
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    data_dict = {}
    status_info = []
    performance_stats = {
        'total_indicators': len(selected_indicators),
        'successful_fetches': 0,
        'failed_fetches': 0,
        'data_quality_score': 0
    }
    def fetch_single_indicator(indicator_key: str) -> Tuple[str, Optional[pd.Series], str]:
        try:
            if indicator_key not in INDIAN_INDICATORS:
                return indicator_key, None, f"❌ Unknown indicator: {indicator_key}"
            indicator = INDIAN_INDICATORS[indicator_key]
            ticker = indicator["ticker"]
            for attempt in range(3):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    if not hist.empty and len(hist) >= 10:
                        clean_data = hist['Close'].fillna(method='ffill').fillna(method='bfill')
                        null_pct = (clean_data.isnull().sum() / len(clean_data)) * 100
                        quality_score = max(0, 100 - null_pct)
                        success_msg = f"✅ {indicator['name']}: {len(clean_data)} points (Quality: {quality_score:.1f}%)"
                        return indicator_key, clean_data, success_msg                
                except Exception as e:
                    if attempt == 2:
                        error_msg = f"❌ {indicator['name']}: {str(e)[:50]}..."
                        return indicator_key, None, error_msg
                    time.sleep(0.5)
        except Exception as e:
            error_msg = f"❌ {indicator_key}: Unexpected error - {str(e)[:50]}..."
            return indicator_key, None, error_msg   
        return indicator_key, None, f"⚠️ {indicator_key}: No data available"
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_indicator = {executor.submit(fetch_single_indicator, indicator_key): indicator_key for indicator_key in selected_indicators}
        for future in concurrent.futures.as_completed(future_to_indicator):
            indicator_key, data, status_msg = future.result()
            if data is not None:
                data_dict[indicator_key] = data
                performance_stats['successful_fetches'] += 1
            else:
                performance_stats['failed_fetches'] += 1
            status_info.append(status_msg)
    if data_dict:
        performance_stats['data_quality_score'] = (performance_stats['successful_fetches'] / performance_stats['total_indicators']) * 100
        df = pd.DataFrame(data_dict)
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df, status_info, performance_stats
    else:
        return pd.DataFrame(), status_info, performance_stats

@st.cache_data(ttl=3600)
@monitor_performance("Data Preprocessing")
def preprocess_market_data(df: pd.DataFrame) -> dict:
    preprocessed_data = {
        'raw_data': df,
        'returns_data': None,
        'correlation_matrix': None,
        'volatility_metrics': {},
        'trend_indicators': {},
        'statistical_summary': {}
    }
    if not df.empty:
        preprocessed_data['returns_data'] = df.pct_change().dropna()
    if preprocessed_data['returns_data'] is not None and len(df.columns) >= 2:
        preprocessed_data['correlation_matrix'] = preprocessed_data['returns_data'].corr()
    for col in df.columns:
        if col in INDIAN_INDICATORS:
            data_series = df[col].dropna()
            if len(data_series) > 30:
                returns = data_series.pct_change().dropna()
                daily_vol = returns.std()
                annualized_vol = daily_vol * np.sqrt(252) * 100
                preprocessed_data['volatility_metrics'][col] = {
                    'daily_volatility': daily_vol * 100,
                    'annualized_volatility': annualized_vol,
                    'volatility_category': 'High' if annualized_vol > 30 else 'Medium' if annualized_vol > 15 else 'Low'
                }
                ma_50 = data_series.rolling(50).mean().iloc[-1] if len(data_series) >= 50 else None
                ma_200 = data_series.rolling(200).mean().iloc[-1] if len(data_series) >= 200 else None
                current_price = data_series.iloc[-1]
                trend_direction = "Neutral"
                if ma_50 and ma_200:
                    if current_price > ma_50 > ma_200:
                        trend_direction = "Strong Uptrend"
                    elif current_price > ma_50:
                        trend_direction = "Uptrend"
                    elif current_price < ma_50 < ma_200:
                        trend_direction = "Strong Downtrend"
                    elif current_price < ma_50:
                        trend_direction = "Downtrend"
                preprocessed_data['trend_indicators'][col] = {
                    'ma_50': ma_50,
                    'ma_200': ma_200,
                    'trend_direction': trend_direction,
                    'trend_strength': abs(current_price - ma_50) / ma_50 * 100 if ma_50 else 0
                }
                preprocessed_data['statistical_summary'][col] = {
                    'mean': data_series.mean(),
                    'median': data_series.median(),
                    'std_dev': data_series.std(),
                    'skewness': data_series.skew(),
                    'kurtosis': data_series.kurtosis()
                }
    return preprocessed_data

def calculate_risk_metrics(data: pd.Series, confidence_level: float = 0.05) -> Dict:
    returns = data.pct_change().dropna()
    if len(returns) < 30:
        return {}
    var_95 = returns.quantile(confidence_level) * 100
    var_99 = returns.quantile(0.01) * 100
    cvar_95 = returns[returns <= returns.quantile(confidence_level)].mean() * 100
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    risk_free_rate = 0.06 / 252
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    returns_squared = returns ** 2
    volatility_clustering = returns_squared.autocorr(lag=1) if len(returns_squared) > 1 else 0
    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'volatility_clustering': volatility_clustering,
        'annualized_return': returns.mean() * 252 * 100,
        'annualized_volatility': returns.std() * np.sqrt(252) * 100
    }

def create_risk_analysis_chart(data: pd.Series, indicator_name: str) -> go.Figure:
    returns = data.pct_change().dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Daily Returns Distribution',
        marker=dict(color='rgba(58, 161, 105, 0.7)', line=dict(color='rgba(58, 161, 105, 1)', width=1)),
        hovertemplate='Return Range: %{x:.1f}%<br>Frequency: %{y}<extra></extra>'
    ))
    var_95 = returns.quantile(0.05) * 100
    var_99 = returns.quantile(0.01) * 100
    fig.add_vline(x=var_95, line_dash="dash", line_color="orange", annotation_text=f"VaR 95%: {var_95:.2f}%")
    fig.add_vline(x=var_99, line_dash="dash", line_color="red", annotation_text=f"VaR 99%: {var_99:.2f}%")
    fig.update_layout(
        title=f"<b>Risk Analysis: {indicator_name}</b>",
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        plot_bgcolor=COLORS['dark_surface'],
        paper_bgcolor=COLORS['dark_surface'],
        font=dict(color=COLORS['dark_text_primary']),
        height=400
    )
    return fig

@st.cache_data(ttl=CACHE_TTL)
def generate_market_insights(preprocessed_data: Dict) -> Dict[str, List[str]]:
    insights = {
        'trend_analysis': [],
        'risk_assessment': [],
        'correlation_insights': [],
        'performance_highlights': []
    }
    trend_data = preprocessed_data.get('trend_indicators', {})
    for indicator_key, trend_info in trend_data.items():
        indicator_name = INDIAN_INDICATORS[indicator_key]['name']
        direction = trend_info['trend_direction']
        strength = trend_info['trend_strength']
        if direction in ['Strong Uptrend', 'Uptrend']:
            insights['trend_analysis'].append(f"📈 **{indicator_name}** shows {direction.lower()} with {strength:.1f}% deviation from MA-50")
        elif direction in ['Strong Downtrend', 'Downtrend']:
            insights['trend_analysis'].append(f"📉 **{indicator_name}** shows {direction.lower()} with {strength:.1f}% deviation from MA-50")
    volatility_data = preprocessed_data.get('volatility_metrics', {})
    for indicator_key, vol_info in volatility_data.items():
        indicator_name = INDIAN_INDICATORS[indicator_key]['name']
        vol_category = vol_info['volatility_category']
        annualized_vol = vol_info['annualized_volatility']
        if vol_category == 'High':
            insights['risk_assessment'].append(f"⚠️ **{indicator_name}** shows high volatility ({annualized_vol:.1f}% annually) - Higher risk/reward potential")
        elif vol_category == 'Low':
            insights['risk_assessment'].append(f"✅ **{indicator_name}** shows low volatility ({annualized_vol:.1f}% annually) - More stable investment")
    if preprocessed_data.get('correlation_matrix') is not None:
        corr_matrix = preprocessed_data['correlation_matrix']
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    indicator1 = INDIAN_INDICATORS[corr_matrix.columns[i]]['name']
                    indicator2 = INDIAN_INDICATORS[corr_matrix.columns[j]]['name']
                    if corr_val > 0.7:
                        insights['correlation_insights'].append(f"🔗 **{indicator1}** and **{indicator2}** move together strongly (correlation: {corr_val:.2f})")
                    else:
                        insights['correlation_insights'].append(f"🔄 **{indicator1}** and **{indicator2}** move in opposite directions (correlation: {corr_val:.2f})")
    return insights

def display_data_quality_metrics(performance_stats: dict, status_messages: list):
    st.subheader("📊 Data Quality Dashboard")
    quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
    with quality_col1:
        success_rate = (performance_stats.get('successful_fetches', 0) / performance_stats.get('total_indicators', 1) * 100)
        st.metric("API Success Rate", f"{success_rate:.1f}%", delta=f"{performance_stats.get('successful_fetches', 0)} of {performance_stats.get('total_indicators', 0)}")
    with quality_col2:
        quality_score = performance_stats.get('data_quality_score', 0)
        st.metric("Data Quality Score", f"{quality_score:.1f}%")
    with quality_col3:
        load_time = st.session_state.performance_metrics.get("Parallel Data Fetch", 0)
        st.metric("Data Load Time", f"{load_time:.2f}s")
    with quality_col4:
        last_update = datetime.datetime.now().strftime('%H:%M:%S IST')
        st.metric("Last Updated", last_update)
    with st.expander("🔍 View Detailed Data Fetch Status"):
        for msg in status_messages:
            if "✅" in msg: st.success(msg)
            elif "⚠️" in msg: st.warning(msg)
            elif "❌" in msg: st.error(msg)

@st.cache_data
def create_realistic_sample_data(selected_indicators=None, days_back=365):
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

def perform_health_check():
    health_report = {
        'timestamp': datetime.datetime.now(),
        'overall_status': 'healthy',
        'checks': []
    }
    if 'performance_metrics' in st.session_state:
        avg_time = sum(st.session_state.performance_metrics.values()) / len(st.session_state.performance_metrics)
        if avg_time < 5:
            health_report['checks'].append({
                'component': 'Performance',
                'status': 'pass', 
                'message': f'Average load time: {avg_time:.2f}s'
            })
        else:
            health_report['checks'].append({
                'component': 'Performance',
                'status': 'warning',
                'message': f'Slow load time: {avg_time:.2f}s'
            })
            health_report['overall_status'] = 'degraded'
    return health_report

st.sidebar.header("📊 Dashboard Controls")
st.sidebar.subheader("📅 Time Period")
selected_range = st.sidebar.selectbox("Select time period:", options=list(DATE_RANGES.keys()), index=3, help="Choose how far back to fetch data")
days_back = DATE_RANGES[selected_range]

st.sidebar.subheader("📈 Select Indicators")
ensure_unique_indicators()
selected_indicators = []

for category, indicators in CATEGORIES.items():
    if not indicators:
        continue
    st.sidebar.markdown(f"**{category}:**")
    for idx, indicator_key in enumerate(indicators):
        if indicator_key not in INDIAN_INDICATORS:
            continue
        indicator = INDIAN_INDICATORS[indicator_key]
        default_selected = indicator_key in ['NIFTY_50', 'INR_USD']
        unique_key = f"indicator_{category}_{indicator_key}_{idx}"
        is_selected = st.sidebar.checkbox(
            f"{indicator['name']}",
            value=default_selected,
            key=unique_key,
            help=indicator['description']
        )
        if is_selected:
            selected_indicators.append(indicator_key)

st.sidebar.subheader("🔍 BSE Stock Search")
search_mode = st.sidebar.radio("Search Mode:", ["Quick Search", "Sector Browse"], horizontal=True)
if search_mode == "Quick Search":
    stock_query = st.sidebar.text_input(
        "Search BSE Stocks",
        placeholder="e.g. TCS, Reliance, HDFC...",
        help="Type stock name or ticker symbol"
    )
    if stock_query:
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("🔍 Searching...")
        search_results = search_bse_stocks(stock_query, max_results=8)
        status_placeholder.empty() 
        if search_results:
            st.sidebar.markdown("#### 📊 Search Results")
            if 'stock_message' not in st.session_state:
                st.session_state.stock_message = None
            for i, result in enumerate(search_results):
                stock_container = st.sidebar.container()
                with stock_container:
                    st.markdown(f"**{result['name']}**")
                    st.caption(f"`{result['ticker']}` • {result['sector']}")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("➕ Add", key=f"add_{i}", help=f"Add {result['name']}", use_container_width=True):
                            success = add_custom_stock_to_config(result['ticker'], {
                                'name': result['name'],
                                'sector': result['sector']
                            })
                            if success:
                                st.session_state.stock_message = ("success", f"✅ Added {result['name']}")
                                st.rerun()
                            else:
                                st.session_state.stock_message = ("warning", f"⚠️ {result['name']} already exists!")
                    st.markdown("---")
            if st.session_state.stock_message:
                msg_type, msg_text = st.session_state.stock_message
                if msg_type == "success":
                    st.sidebar.success(msg_text)
                else:
                    st.sidebar.warning(msg_text)
                if st.sidebar.button("✕ Clear", key="clear_msg"):
                    st.session_state.stock_message = None
                    st.rerun()
                else:
                    st.sidebar.info("🔍 No stocks found. Try different keywords.")
                    st.sidebar.markdown("#### 💡 Try searching for:")
                    suggestions = ["TCS", "Reliance", "HDFC Bank", "Infosys", "ICICI"]
                    for suggestion in suggestions:
                        if st.sidebar.button(suggestion, key=f"suggest_{suggestion}"):
                            st.sidebar.text_input("Search BSE Stocks", value=suggestion)

elif search_mode == "Sector Browse":
    popular_stocks = get_popular_stocks()
    selected_sector = st.sidebar.selectbox(
        "Browse by Sector:",
        list(popular_stocks.keys()),
        help="Select a sector to see popular stocks"
    )
    if selected_sector:
        st.sidebar.markdown(f"#### 📈 Popular {selected_sector} Stocks")
        sector_stocks = popular_stocks[selected_sector]
        for ticker in sector_stocks:
            if ticker in BSE_STOCKS:
                stock_info = BSE_STOCKS[ticker]
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.markdown(f"**{stock_info['name']}**")
                    st.caption(f"{ticker}")
                with col2:
                    if st.button("➕", key=f"sector_add_{ticker}", help=f"Add {stock_info['name']}"):
                        success = add_custom_stock_to_config(ticker, stock_info)
                        if success:
                            st.success(f"✅ Added {stock_info['name']}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {stock_info['name']} already exists!")


st.sidebar.markdown("---")
if st.sidebar.button("🚀 Add Top 5 BSE Stocks"):
    top_stocks = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    added_count = 0
    for ticker in top_stocks:
        if ticker in BSE_STOCKS:
            if add_custom_stock_to_config(ticker, BSE_STOCKS[ticker]):
                added_count += 1
    if added_count > 0:
        st.sidebar.success(f"✅ Added {added_count} stocks!")
        time.sleep(1)
        st.rerun()
    else:
        st.sidebar.info("All top 5 stocks already added!")

st.sidebar.subheader("🎛️ Display Mode")
display_mode = st.sidebar.radio("Choose display style:", ["Individual Charts", "Grid View"], help="How to display selected indicators")

st.sidebar.subheader("🎨 Chart Controls")
performance_mode = st.sidebar.checkbox("🚀 Performance Mode", value=False, help="Disable chart enhancements for faster rendering")
chart_quality = st.sidebar.selectbox("📊 Export Quality", ["Standard", "High", "Ultra"], index=1, help="Chart resolution for PNG exports")

if not performance_mode:
    smooth_lines = st.sidebar.checkbox("📈 Smooth Lines", value=False)
    show_markers = st.sidebar.checkbox("🔸 Show Data Points", value=False)
    enable_quick_zoom = st.sidebar.checkbox("⚡ Quick Zoom", value=True)
else:
    smooth_lines = False
    show_markers = False
    enable_quick_zoom = False
    st.sidebar.info("💨 Performance mode: Chart enhancements disabled")

st.sidebar.subheader("ℹ️ Selection Info")
st.sidebar.info(f"""**Selected Period:** {selected_range}
**Indicators:** {len(selected_indicators)} selected
**Display:** {display_mode}""")

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

st.sidebar.subheader("🔄 Data Refresh Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False, help="Automatically refresh data every few minutes. Useful during market hours.")
if auto_refresh:
    refresh_interval = st.sidebar.selectbox("Refresh Interval", options=[1, 2, 5, 10, 15], index=2, format_func=lambda x: f"{x} minute{'s' if x > 1 else ''}", help="How often to refresh the data")
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    current_time_unix = time.time()
    if (current_time_unix - st.session_state.last_refresh) / 60 >= refresh_interval:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()
if st.sidebar.button("🔄 Manual Refresh Now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("📊 Manage Custom Stocks", expanded=False):
    if 'Custom Stocks' in CATEGORIES and CATEGORIES['Custom Stocks']:
        st.markdown("#### Currently Added:")
        for i, stock_key in enumerate(CATEGORIES['Custom Stocks']):
            if stock_key in INDIAN_INDICATORS:
                stock_info = INDIAN_INDICATORS[stock_key]
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{stock_info['name']}**")
                    st.caption(f"{stock_info['ticker']}")
                with col2:
                    if st.button("🗑️", key=f"remove_{i}", help=f"Remove {stock_info['name']}"):
                        del INDIAN_INDICATORS[stock_key]
                        CATEGORIES['Custom Stocks'].remove(stock_key)
                        st.success(f"Removed {stock_info['name']}")
                        st.rerun()
        if st.button("🗑️ Clear All Custom Stocks"):
            if 'Custom Stocks' in CATEGORIES:
                for stock_key in CATEGORIES['Custom Stocks'].copy():
                    if stock_key in INDIAN_INDICATORS:
                        del INDIAN_INDICATORS[stock_key]
                CATEGORIES['Custom Stocks'] = []
            st.success("Cleared all custom stocks")
            st.rerun()
    else:
        st.info("No custom stocks added yet.")
        st.markdown("Use the **BSE Stock Search** above to add stocks!")

if 'Custom Stocks' in CATEGORIES:
    custom_count = len(CATEGORIES.get('Custom Stocks', []))
    total_indicators = sum(len(indicators) for indicators in CATEGORIES.values())
    st.sidebar.metric("📈 Total Indicators", total_indicators, f"+{custom_count} custom")

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ System Diagnostics", expanded=False):
    if st.button("🔍 Run Health Check"):
        health = perform_health_check()
        st.success(f"✅ Status: {health['overall_status']}")
        if 'performance_metrics' in st.session_state:
            st.json(st.session_state.performance_metrics)
    st.text(f"Session Items: {len(st.session_state.keys())}")
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

current_state = {
    'selected_indicators': selected_indicators,
    'selected_range': selected_range,
    'display_mode': display_mode,
    'smooth_lines': smooth_lines,
    'show_markers': show_markers
}

if 'last_state' not in st.session_state or st.session_state.last_state != current_state:
    st.session_state.last_state = current_state

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
        df, status_messages, performance_data = get_optimized_market_data(selected_indicators, days_back)
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

preprocessed_data = preprocess_market_data(df)

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
                st.metric(label=indicator['name'], value=f"{current_val:.2f}", delta=f"{pct_change:+.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 **Technical Analysis**", "🔗 **Correlation Matrix**", "📊 **Market Analytics**", "💾 **Data Export**"])
    
    with tab1:
        st.header("📈 Technical Analysis")
        st.markdown("**Analyze trends with price charts and moving averages**")
        col1, col2 = st.columns([1, 1])
        with col1:
            show_ma = st.checkbox("📊 Show Moving Averages", value=False)
        with col2:
            ma_windows = []
            if show_ma:
                ma_windows = st.multiselect('Select MA Periods:', options=[5, 10, 20, 50, 100, 200], default=[20, 50], help="Moving Average periods in days")
        
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
                            ma_windows if show_ma else None,
                            chart_height=400,
                            line_shape="spline" if smooth_lines else "linear",
                            show_markers=show_markers,
                            quick_zoom=enable_quick_zoom,
                            performance_mode=performance_mode
                        )
                        st.plotly_chart(chart, use_container_width=True)
                        create_export_button(chart, indicator['name'], chart_quality)
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
                                    chart_height=350,
                                    line_shape="spline" if smooth_lines else "linear",
                                    show_markers=show_markers,
                                    quick_zoom=enable_quick_zoom,
                                    performance_mode=performance_mode
                                )
                                st.plotly_chart(chart, use_container_width=True)
                                create_export_button(chart, indicator['name'], chart_quality)

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
            st.warning("⚠️ Please select at least 2 indicators to analyze correlations.")

    with tab3:
        st.header("📊 Advanced Market Analytics")
        st.markdown("**Professional risk analysis and market intelligence**")
        if 'performance_data' in locals():
            display_data_quality_metrics(performance_data, status_messages)
        st.markdown("---")
        analytics_mode = st.selectbox("Select Analysis Type:", ["Performance Comparison", "Risk Analysis", "Market Intelligence"], help="Choose which advanced analysis to display")
        
        if analytics_mode == "Performance Comparison":
            st.subheader("📈 Performance Comparison")
            if len(selected_indicators) > 1:
                rows = []
                for key in selected_indicators:
                    if key in df.columns:
                        series = df[key]
                        risk = calculate_risk_metrics(series)
                        rows.append({
                            "Indicator": INDIAN_INDICATORS[key]["name"],
                            "Total Return (%)": f"{((series.iloc[-1]/series.iloc[0])-1)*100:+.2f}",
                            "Ann. Volatility (%)": f"{risk.get('annualized_volatility',0):.2f}",
                            "Sharpe Ratio": f"{risk.get('sharpe_ratio',0):.2f}",
                            "Max Drawdown (%)": f"{risk.get('max_drawdown',0):.2f}",
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Select at least two indicators for comparison.")
        
        elif analytics_mode == "Risk Analysis":
            st.subheader("⚠️ Risk Analysis")
            sel_key = st.selectbox("Choose indicator:", selected_indicators, format_func=lambda k: INDIAN_INDICATORS[k]["name"])
            if sel_key in df.columns:
                series = df[sel_key]
                name = INDIAN_INDICATORS[sel_key]["name"]
                risk = calculate_risk_metrics(series)
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Ann. Volatility", f"{risk['annualized_volatility']:.2f}%")
                kpi2.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.2f}")
                kpi3.metric("VaR 95%", f"{risk['var_95']:.2f}%")
                kpi4.metric("Max Drawdown", f"{risk['max_drawdown']:.2f}%")
                chart = create_risk_analysis_chart(series, name)
                st.plotly_chart(chart, use_container_width=True)
        
        else:
            st.subheader("🧠 Market Intelligence Insights")
            insights = generate_market_insights(preprocessed_data)
            insight_tabs = st.tabs(["🔍 Trend", "⚠️ Risk", "🔗 Correlation"])
            with insight_tabs[0]:
                if insights['trend_analysis']:
                    for txt in insights['trend_analysis']:
                        st.info(txt)
                else:
                    st.info("No significant trend signals.")
            with insight_tabs[1]:
                if insights['risk_assessment']:
                    for txt in insights['risk_assessment']:
                        st.info(txt)
                else:
                    st.info("No unusual risk signals.")
            with insight_tabs[2]:
                if insights['correlation_insights']:
                    for txt in insights['correlation_insights']:
                        st.info(txt)
                else:
                    st.info("No strong correlations detected.")

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
            st.dataframe(export_df.head(5), use_container_width=True)
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
