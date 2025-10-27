import re
from datetime import datetime

# ==============================================================
# Main Formatter Router
# ==============================================================

def format_telegram_output(task_name, company, raw_output, stock_data=None):
    """
    Ensures output matches the exact template format.
    Cleans up and reformats AI output to match expected structure.
    Optionally injects real market data (stock_data).
    """
    output = str(raw_output).strip()

    # Remove CrewAI artifacts
    output = re.sub(r'Thought:.*?(?=Action:|$)', '', output, flags=re.DOTALL | re.IGNORECASE)
    output = re.sub(r'Action:\s*', '', output, flags=re.IGNORECASE)
    output = output.strip()

    # Route based on task type
    if task_name == "short_term_price_prediction":
        return format_short_term_prediction(company, output, stock_data)
    elif task_name == "technical_analysis_summary":
        return format_technical_summary(company, output)
    elif task_name == "investment_summary":
        return format_investment_summary(company, output, stock_data)
    elif task_name == "ml_prediction_analysis":
        return format_ml_prediction(company, output, stock_data)

    # Default fallback
    return f"{output}\n\n⚠️ *Disclaimer:* This is for educational purposes only and not financial advice."


# ==============================================================
# Helper: Get Real Market Price
# ==============================================================

def get_current_price(stock_data):
    """Try to extract latest close price from yfinance data"""
    try:
        if stock_data is not None and not stock_data.empty:
            latest_close = float(stock_data['Close'].iloc[-1])
            if 1 < latest_close < 100000:
                return round(latest_close, 2)
        return None
    except Exception as e:
        print(f"[⚠️] Get current price failed: {e}")
        return None


# ==============================================================
# Formatter 1 — Short-Term Prediction
# ==============================================================

def format_short_term_prediction(company, raw_output, stock_data=None):
    ticker = extract_ticker(raw_output, company)

    # Real-time price if available
    real_price = get_current_price(stock_data)
    current_price = (
        real_price or
        extract_value(raw_output, r"current\s+price[:\s]*\$?([\d,]+\.?\d*)") or
        extract_value(raw_output, r"trading\s+at\s+\$?([\d,]+\.?\d*)") or
        extract_any_price(raw_output)
    )

    # Extract target range
    range_match = re.search(r"\$?([\d,]+\.?\d*)\s*[-–to]\s*\$?([\d,]+\.?\d*)", raw_output, re.IGNORECASE)
    if range_match:
        min_price = range_match.group(1).replace(',', '')
        max_price = range_match.group(2).replace(',', '')
    else:
        min_price = (
            extract_value(raw_output, r"(?:min|low|support|range)[:\s]*\$?([\d,]+\.?\d*)") or
            (str(float(current_price) * 0.95) if current_price else None)
        )
        max_price = (
            extract_value(raw_output, r"(?:max|high|resistance)[:\s]*\$?([\d,]+\.?\d*)") or
            (str(float(current_price) * 1.05) if current_price else None)
        )

    trend = extract_trend(raw_output)
    confidence = extract_confidence(raw_output)
    signal = extract_signal(raw_output)
    risk = extract_risk_level(raw_output)
    insight = extract_insight(raw_output)

    current_price_str = f"${float(current_price):.2f}" if current_price else "$N/A"
    min_price_str = f"${float(min_price):.2f}" if min_price else "$N/A"
    max_price_str = f"${float(max_price):.2f}" if max_price else "$N/A"

    return f"""📈 *Short-Term Price Prediction for {company} ({ticker})*

🔸 *Current Price:* {current_price_str}
🔸 *7-Day Price Target:* {min_price_str} – {max_price_str}
🔸 *Trend Direction:* {trend}
🔸 *Confidence Level:* {confidence}

🎯 *Trading Signal:* {signal}

💡 *Key Insights:* {insight}

⚠️ *Risk Level:* {risk}
⚠️ *Disclaimer:* This prediction is for educational purposes only and not financial advice."""


# ==============================================================
# Formatter 2 — Technical Summary
# ==============================================================

def format_technical_summary(company, raw_output):
    ticker = extract_ticker(raw_output, company)

    rsi = extract_value(raw_output, r"RSI[:\s]*(\d+\.?\d*)")
    rsi_status = extract_rsi_status(raw_output)
    macd = extract_value(raw_output, r"MACD[:\s]*([+-]?[\d.]+)")
    macd_signal = extract_macd_signal(raw_output)
    trend_strength = extract_trend_strength(raw_output)
    signal = extract_signal(raw_output)
    support = extract_value(raw_output, r"support[:\s]*\$?([\d,]+\.?\d*)")
    resistance = extract_value(raw_output, r"resistance[:\s]*\$?([\d,]+\.?\d*)")
    ma_signal = extract_ma_signal(raw_output)
    insight = extract_insight(raw_output)

    return f"""📊 *Technical Summary for {company} ({ticker})*

• *RSI (14):* {rsi or 'N/A'} - ({rsi_status})
• *MACD:* {macd or 'N/A'} - ({macd_signal})
• *Trend Strength:* {trend_strength}

🧭 *Technical Signal:* {signal}

📊 *Technical Indicators:*
• *Support Level:* ${support or 'N/A'}
• *Resistance Level:* ${resistance or 'N/A'}
• *Moving Average Signal:* {ma_signal}

🔍 *Insight:* {insight}

⚠️ *Disclaimer:* This is not financial advice. Please conduct your own research."""


# ==============================================================
# Formatter 3 — Investment Summary
# ==============================================================

def format_investment_summary(company, raw_output, stock_data=None):
    ticker = extract_ticker(raw_output, company)

    score = extract_value(raw_output, r"(?:score|rating)[:\s]*(\d+)")
    tech_rating = extract_rating(raw_output, "technical")
    fund_rating = extract_fundamental_rating(raw_output)
    sentiment = extract_sentiment(raw_output)
    recommendation = extract_recommendation(raw_output)
    reasoning = extract_reasoning(raw_output)

    # ✅ Insert real market data
    real_price = get_current_price(stock_data)
    price_text = f"${real_price:.2f}" if real_price else "N/A"

    return f"""💰 *AI Investment Suggestion for {company} ({ticker})*

📈 *Overall Score:* {score or 'N/A'}/100
💰 *Current Price:* {price_text}

🧩 *Technical Rating:* {tech_rating}
💼 *Fundamental Rating:* {fund_rating}
🗞️ *Sentiment Rating:* {sentiment}

🎯 *Final Recommendation:* {recommendation}

🧠 *Reasoning:*
{reasoning}

⚠️ *Disclaimer:* This is not financial advice. Please conduct your own research."""


# ==============================================================
# Formatter 4 — ML Prediction
# ==============================================================

def format_ml_prediction(company, raw_output, stock_data=None):
    ticker = extract_ticker(raw_output, company)
    current_date = datetime.now().strftime("%Y-%m-%d")

    real_price = get_current_price(stock_data)
    current_price = real_price or extract_value(raw_output, r"current\s+price[:\s]*\$?([\d,]+\.?\d*)")

    days_analyzed = extract_value(raw_output, r"(\d+)\s+days") or "90"

    lstm_1day = extract_value(raw_output, r"LSTM.*?1[-\s]?day[:\s]*\$?([\d.]+)")
    lstm_3day = extract_value(raw_output, r"LSTM.*?3[-\s]?day[:\s]*\$?([\d.]+)")
    lstm_7day = extract_value(raw_output, r"LSTM.*?7[-\s]?day[:\s]*\$?([\d.]+)")
    lstm_conf = extract_value(raw_output, r"LSTM\s+confidence[:\s]*([\d.]+)") or "75"

    xgb_1day = extract_value(raw_output, r"XGBoost.*?1[-\s]?day[:\s]*\$?([\d.]+)")
    xgb_3day = extract_value(raw_output, r"XGBoost.*?3[-\s]?day[:\s]*\$?([\d.]+)")
    xgb_7day = extract_value(raw_output, r"XGBoost.*?7[-\s]?day[:\s]*\$?([\d.]+)")
    xgb_conf = extract_value(raw_output, r"XGBoost\s+confidence[:\s]*([\d.]+)") or "75"

    ensemble_1day = extract_value(raw_output, r"(?:ensemble|consensus).*?1[-\s]?day[:\s]*\$?([\d.]+)")
    ensemble_7day = extract_value(raw_output, r"(?:ensemble|consensus).*?7[-\s]?day[:\s]*\$?([\d.]+)")
    model_agreement = extract_ml_agreement(raw_output)
    overall_conf = extract_value(raw_output, r"overall\s+confidence[:\s]*([\d.]+)") or "75"

    recommendation = extract_ml_recommendation(raw_output)
    risk_level = extract_risk_level(raw_output)
    position_size = extract_value(raw_output, r"position\s+size[:\s]*([\d.]+)") or "5"
    stop_loss = extract_value(raw_output, r"stop\s+loss[:\s]*\$?([\d.]+)") or "N/A"

    return f"""🤖 *ML Prediction Analysis*

📊 *Company:* {company} ({ticker})
🕰️ *Analysis Date:* {current_date}
💰 *Current Price:* ${current_price or 'N/A'}
📈 *Historical Data:* {days_analyzed} days analyzed

*LSTM Model Prediction:*
🎯 *1-Day:* ${lstm_1day or 'N/A'}
🎯 *3-Day:* ${lstm_3day or 'N/A'}
🎯 *7-Day:* ${lstm_7day or 'N/A'}
📊 *LSTM Confidence:* {lstm_conf}%

*XGBoost Model Prediction:*
🎯 *1-Day:* ${xgb_1day or 'N/A'}
🎯 *3-Day:* ${xgb_3day or 'N/A'}
🎯 *7-Day:* ${xgb_7day or 'N/A'}
📊 *XGBoost Confidence:* {xgb_conf}%

*Ensemble (Consensus) Prediction:*
🎯 *1-Day:* ${ensemble_1day or 'N/A'}
🎯 *7-Day:* ${ensemble_7day or 'N/A'}
📊 *Model Agreement:* {model_agreement}
📊 *Overall Confidence:* {overall_conf}%

*Trading Signals:*
💡 *Recommendation:* {recommendation}
📊 *Risk Level:* {risk_level}
💼 *Position Size:* {position_size}% of portfolio
🛑 *Stop Loss:* ${stop_loss}

⚠️ *Note:* ML predictions are based on historical patterns. Market conditions may vary."""


# ==============================================================
# Extraction Utilities
# ==============================================================

def extract_ticker(text, company):
    match = re.search(r'\(([A-Z]{1,5})\)', text)
    if match:
        return match.group(1)
    match = re.search(r'\b([A-Z]{2,5})\b', text)
    if match and match.group(1) not in ['RSI', 'MACD', 'LSTM', 'USD']:
        return match.group(1)
    return company.upper()[:5]

def extract_value(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    return None

def extract_any_price(text):
    prices = re.findall(r'\$?([\d,]+\.?\d{0,2})', text)
    if prices:
        cleaned = [float(p.replace(',', '')) for p in prices if float(p.replace(',', '')) > 1]
        if cleaned:
            return str(max(cleaned))
    return None

def extract_trend(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["uptrend", "bullish", "rising", "upward"]):
        return "Uptrend"
    elif any(w in text_lower for w in ["downtrend", "bearish", "falling", "downward"]):
        return "Downtrend"
    return "Neutral"

def extract_confidence(text):
    tl = text.lower()
    if "high confidence" in tl:
        return "High"
    elif "low confidence" in tl:
        return "Low"
    return "Medium"

def extract_signal(text):
    tl = text.lower()
    if "buy" in tl and "not buy" not in tl:
        return "Buy"
    elif "sell" in tl and "not sell" not in tl:
        return "Sell"
    return "Hold"

def extract_risk_level(text):
    tl = text.lower()
    if "high risk" in tl or "risky" in tl:
        return "High"
    elif "low risk" in tl or "safe" in tl:
        return "Low"
    return "Medium"

def extract_rsi_status(text):
    tl = text.lower()
    if "overbought" in tl:
        return "Overbought"
    elif "oversold" in tl:
        return "Oversold"
    return "Neutral"

def extract_macd_signal(text):
    tl = text.lower()
    if "bullish" in tl and "macd" in tl:
        return "Bullish"
    elif "bearish" in tl and "macd" in tl:
        return "Bearish"
    return "Neutral"

def extract_trend_strength(text):
    tl = text.lower()
    if "strong" in tl and "trend" in tl:
        return "Strong"
    elif "weak" in tl and "trend" in tl:
        return "Weak"
    return "Moderate"

def extract_ma_signal(text):
    tl = text.lower()
    if "golden cross" in tl:
        return "Bullish (Golden Cross)"
    elif "death cross" in tl:
        return "Bearish (Death Cross)"
    elif "above" in tl and "ma" in tl:
        return "Above Moving Average"
    elif "below" in tl and "ma" in tl:
        return "Below Moving Average"
    return "Neutral"

def extract_rating(text, rating_type):
    pattern = f"{rating_type}[\\s\\w]*?[:\\s]+(buy|hold|sell)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return extract_signal(text)

def extract_fundamental_rating(text):
    tl = text.lower()
    if "strong" in tl and "fundamental" in tl:
        return "Strong"
    elif "weak" in tl and "fundamental" in tl:
        return "Weak"
    return "Fair"

def extract_sentiment(text):
    tl = text.lower()
    if "positive" in tl:
        return "Positive"
    elif "negative" in tl:
        return "Negative"
    return "Neutral"

def extract_recommendation(text):
    for p in [r"final\s+recommendation[:\s]+\*?\*?(buy|hold|sell)", r"recommendation[:\s]+\*?\*?(buy|hold|sell)", r"suggest[:\s]+(buy|hold|sell)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return extract_signal(text).upper()

def extract_ml_recommendation(text):
    t = text.upper()
    if "BUY" in t:
        return "BUY"
    elif "SELL" in t:
        return "SELL"
    return "HOLD"

def extract_ml_agreement(text):
    tl = text.lower()
    if "high agreement" in tl or "strong agreement" in tl:
        return "High"
    elif "low agreement" in tl or "disagree" in tl:
        return "Low"
    return "Medium"

def extract_insight(text):
    for p in [
        r"insight[:\s]+(.*?)(?:\n\n|disclaimer|risk|⚠|$)",
        r"key insights?[:\s]+(.*?)(?:\n\n|disclaimer|risk|⚠|$)",
        r"reasoning[:\s]+(.*?)(?:\n\n|disclaimer|risk|⚠|$)",
        r"analysis[:\s]+(.*?)(?:\n\n|disclaimer|risk|⚠|$)"
    ]:
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = re.sub(r'\n+', ' ', m.group(1).strip())
            if len(val) > 300: val = val[:297] + "..."
            return val
    sents = re.split(r'[.!?]\s+', text)
    for s in sents:
        if len(s) > 30 and any(w in s.lower() for w in ['predict', 'expect', 'trend', 'momentum', 'suggest']):
            return s.strip()[:300]
    return "Based on market indicators, the model provides predictions for short-term movement."

def extract_reasoning(text):
    return extract_insight(text)
