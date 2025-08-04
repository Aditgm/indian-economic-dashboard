# config.py - Indian Economic Indicators Configuration
import datetime

# Indian Economic Indicators
INDIAN_INDICATORS = {
    "nifty50": {
        "name": "NIFTY 50 Index",
        "description": "India's benchmark stock market index",
        "unit": "Index Points",
        "source": "Yahoo Finance"
    },
    "sensex": {
        "name": "BSE SENSEX",
        "description": "Bombay Stock Exchange Sensitive Index",
        "unit": "Index Points", 
        "source": "Yahoo Finance"
    },
    "inr_usd": {
        "name": "INR/USD Exchange Rate",
        "description": "Indian Rupee to US Dollar exchange rate",
        "unit": "INR per USD",
        "source": "Yahoo Finance"
    },
    "india_10y": {
        "name": "India 10-Year Bond Yield",
        "description": "Government of India 10-year bond yield",
        "unit": "Percentage",
        "source": "Yahoo Finance"
    }
}

# Yahoo Finance tickers for Indian data
YAHOO_TICKERS = {
    "nifty50": "^NSEI",
    "sensex": "^BSESN", 
    "inr_usd": "INR=X",
    "india_10y": "^TNX"  # We'll use this as proxy, will be updated
}

# Date configuration
START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime.now()

# App configuration
APP_CONFIG = {
    "page_title": "Indian Economic Dashboard",
    "page_icon": "🇮🇳",
    "layout": "wide"
}
