import datetime
INDIAN_INDICATORS = {
    "NIFTY_50": {
        "name": "NIFTY 50",
        "ticker": "^NSEI",
        "description": "India's premier stock market index of top 50 companies",
        "category": "Market Indices",
        "unit": "Points",
        "color": "#E53E3E"
    },
    "INR_USD": {
        "name": "INR/USD Rate",
        "ticker": "INR=X",
        "description": "Indian Rupee to US Dollar exchange rate",
        "category": "Currency",
        "unit": "INR per USD",
        "color": "#38A169"
    },
    "BANK_NIFTY": {
        "name": "Bank NIFTY",
        "ticker": "^NSEBANK",
        "description": "Banking sector index of top 12 banking stocks",
        "category": "Market Indices",
        "unit": "Points",
        "color": "#3182CE"
    },
    "GOLD": {
        "name": "Gold Price",
        "ticker": "GC=F",
        "description": "International gold price per ounce",
        "category": "Commodities",
        "unit": "USD per ounce",
        "color": "#D69E2E"
    },
    "CRUDE_OIL": {
        "name": "Crude Oil",
        "ticker": "CL=F",
        "description": "WTI Crude Oil futures price",
        "category": "Commodities",
        "unit": "USD per barrel",
        "color": "#2D3748"
    },
    "RELIANCE": {
        "name": "Reliance Industries",
        "ticker": "RELIANCE.NS",
        "description": "India's largest company by market cap",
        "category": "Individual Stocks",
        "unit": "INR",
        "color": "#9F7AEA"
    },
    "TCS": {
        "name": "TCS",
        "ticker": "TCS.NS",
        "description": "Leading Indian IT services company",
        "category": "Individual Stocks",
        "unit": "INR",
        "color": "#4FD1C7"
    },
    "HDFC_BANK": {
        "name": "HDFC Bank",
        "ticker": "HDFCBANK.NS",
        "description": "Major Indian private sector bank",
        "category": "Individual Stocks",
        "unit": "INR",
        "color": "#ED8936"
    }
}
CATEGORIES = {
    "Market Indices": ["NIFTY_50", "BANK_NIFTY"],
    "Currency": ["INR_USD"],
    "Commodities": ["GOLD", "CRUDE_OIL"],
    "Individual Stocks": ["RELIANCE", "TCS", "HDFC_BANK"]
}
DATE_RANGES = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730
}
APP_CONFIG = {
    "page_title": "Indian Economic Dashboard",
    "page_icon": "🇮🇳",
    "layout": "wide"
}
