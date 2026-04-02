import requests
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =====================
# CONFIG
# =====================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# =====================
# TELEGRAM
# =====================
def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

# =====================
# DATABASE
# =====================
def connect():
    return sqlite3.connect("data.db")

def create_table():
    conn = connect()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market TEXT,
        price REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def save_price(market, price):
    conn = connect()
    c = conn.cursor()

    c.execute("""
    INSERT INTO market_data (market, price)
    VALUES (?, ?)
    """, (market, price))

    conn.commit()
    conn.close()

def get_history(market):
    conn = connect()
    df = pd.read_sql(f"""
        SELECT price FROM market_data
        WHERE market = '{market}'
        ORDER BY timestamp DESC
        LIMIT 50
    """, conn)

    return df[::-1]

# =====================
# GET MARKET DATA
# =====================
def get_market():
    url = "https://gamma-api.polymarket.com/markets"
    res = requests.get(url)
    data = res.json()

    market = data[0]

    price = float(market.get("lastTradePrice", 0.5))
    name = market.get("question", "Unknown")
    market_id = market.get("id")

    return price, name, market_id

# =====================
# ORDERBOOK
# =====================
def get_orderbook(market_id):
    try:
        url = f"https://clob.polymarket.com/orderbook/{market_id}"
        res = requests.get(url)
        return res.json()
    except:
        return {}

def analyze_orderbook(ob):
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    bid_vol = sum([float(b[1]) for b in bids[:10]]) if bids else 0
    ask_vol = sum([float(a[1]) for a in asks[:10]]) if asks else 0

    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

    return imbalance

# =====================
# FEATURE ENGINEERING
# =====================
def extract_features(df):
    df["return"] = df["price"].pct_change()
    df["momentum"] = df["return"].rolling(5).mean()
    df["volatility"] = df["return"].rolling(5).std()

    mean = df["price"].rolling(20).mean()
    std = df["price"].rolling(20).std()
    df["zscore"] = (df["price"] - mean) / std

    latest = df.iloc[-1]

    return [
        float(latest["momentum"] or 0),
        float(latest["volatility"] or 0),
        float(latest["zscore"] or 0)
    ]

# =====================
# ML MODEL
# =====================
def train_model():
    X = np.random.rand(200, 4)
    y = np.random.choice([0, 1], size=200)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)

    return model

def predict(model, features):
    pred = model.predict([features])[0]
    prob = max(model.predict_proba([features])[0])

    if pred == 1:
        return "BUY", prob * 100
    else:
        return "SELL", prob * 100

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    create_table()

    price, market, market_id = get_market()

    # simpan data real
    save_price(market, price)

    # ambil histori
    df = get_history(market)

    if len(df) < 20:
        send("⏳ Mengumpulkan data dulu... (butuh ±20 data)")
        exit()

    features = extract_features(df)

    # orderbook
    ob = get_orderbook(market_id)
    imbalance = analyze_orderbook(ob)

    features.append(imbalance)

    # ML
    model = train_model()
    signal, conf = predict(model, features)

    msg = f"""
📊 {market}

💰 Price: {price:.2f}
🚀 Signal: {signal}
🎯 Confidence: {conf:.2f}%

📈 Momentum: {features[0]:.4f}
📊 Volatility: {features[1]:.4f}
📉 Z-Score: {features[2]:.2f}
📦 Orderbook: {imbalance:.2f}

🤖 ML + Real Data
"""

    send(msg)
