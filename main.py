import requests
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import subprocess
from flask import Flask, jsonify

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DB_FILE = "data.db"

# ================= TELEGRAM =================
def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

# ================= DATABASE =================
def connect():
    return sqlite3.connect(DB_FILE)

def create_tables():
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

    c.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market TEXT,
        signal TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def save_price(market, price):
    conn = connect()
    c = conn.cursor()
    c.execute("INSERT INTO market_data (market, price) VALUES (?, ?)", (market, price))
    conn.commit()
    conn.close()

def get_history(market):
    conn = connect()
    df = pd.read_sql(f"""
        SELECT price FROM market_data
        WHERE market = '{market}'
        ORDER BY timestamp DESC
        LIMIT 100
    """, conn)
    return df[::-1]

def save_trade(market, signal, conf):
    conn = connect()
    c = conn.cursor()
    c.execute("INSERT INTO trades (market, signal, confidence) VALUES (?, ?, ?)", (market, signal, conf))
    conn.commit()
    conn.close()

# ================= MARKET =================
def get_markets():
    url = "https://gamma-api.polymarket.com/markets"
    data = requests.get(url).json()

    markets = []
    for m in data[:10]:
        try:
            markets.append({
                "price": float(m.get("lastTradePrice", 0.5)),
                "name": m.get("question", "Unknown"),
                "id": m.get("id")
            })
        except:
            continue
    return markets

# ================= ORDERBOOK =================
def get_orderbook(market_id):
    try:
        return requests.get(f"https://clob.polymarket.com/orderbook/{market_id}").json()
    except:
        return {}

def analyze_orderbook(ob):
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    bid_vol = sum([float(b[1]) for b in bids[:10]]) if bids else 0
    ask_vol = sum([float(a[1]) for a in asks[:10]]) if asks else 0

    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

# ================= MACRO REAL API =================
def macro_bias():
    try:
        # contoh: pakai data USD index proxy (BTC sebagai risk proxy)
        btc = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json").json()
        price = float(btc["bpi"]["USD"]["rate"].replace(",", ""))

        if price > 30000:
            return "RISK_ON"
        else:
            return "RISK_OFF"
    except:
        return "NEUTRAL"

def macro_filter(signal, bias):
    if bias == "RISK_OFF" and signal == "BUY":
        return False
    return True

# ================= FEATURE =================
def prepare_ml_data(df):
    df = df.copy()

    df["return"] = df["price"].pct_change()
    df["momentum"] = df["return"].rolling(5).mean()
    df["volatility"] = df["return"].rolling(5).std()

    mean = df["price"].rolling(20).mean()
    std = df["price"].rolling(20).std()
    df["zscore"] = (df["price"] - mean) / std

    df["future"] = df["price"].shift(-3)
    df["target"] = (df["future"] > df["price"]).astype(int)

    df = df.dropna()

    X = df[["momentum", "volatility", "zscore"]]
    y = df["target"]

    return X, y

def train_model(df):
    X, y = prepare_ml_data(df)

    if len(X) < 20:
        return None

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model

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

def predict(model, features):
    if model is None:
        return "WAIT", 0

    pred = model.predict([features])[0]
    prob = max(model.predict_proba([features])[0])

    if prob < 0.6:
        return "NO TRADE", prob * 100

    return ("BUY", prob * 100) if pred == 1 else ("SELL", prob * 100)

# ================= RISK =================
def risk_management(price, signal):
    risk = 0.02

    if signal == "BUY":
        sl = price * (1 - risk)
        tp = price * (1 + risk * 2)
    else:
        sl = price * (1 + risk)
        tp = price * (1 - risk * 2)

    return sl, tp

# ================= SAVE DB =================
def save_db():
    try:
        subprocess.run(["git", "config", "--global", "user.email", "bot@bot.com"])
        subprocess.run(["git", "config", "--global", "user.name", "bot"])

        subprocess.run(["git", "add", DB_FILE])
        subprocess.run(["git", "commit", "-m", "update db"], check=False)
        subprocess.run(["git", "push"])
    except:
        pass

# ================= DASHBOARD =================
app = Flask(__name__)

@app.route("/")
def dashboard():
    conn = connect()
    df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
    return df.to_html()

@app.route("/api")
def api():
    conn = connect()
    df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
    return jsonify(df.to_dict(orient="records"))

# ================= MAIN =================
def run_bot():
    create_tables()

    markets = get_markets()
    bias = macro_bias()

    best = None
    best_score = 0

    for m in markets:
        price = m["price"]
        market = m["name"]
        market_id = m["id"]

        save_price(market, price)

        df = get_history(market)
        if len(df) < 20:
            continue

        features = extract_features(df)

        ob = get_orderbook(market_id)
        imbalance = analyze_orderbook(ob)
        features.append(imbalance)

        model = train_model(df)
        signal, conf = predict(model, features)

        if signal in ["NO TRADE", "WAIT"]:
            continue

        if not macro_filter(signal, bias):
            continue

        score = conf + abs(imbalance * 100)

        if score > best_score:
            best_score = score
            best = (market, price, signal, conf, imbalance)

    if best:
        market, price, signal, conf, imbalance = best
        sl, tp = risk_management(price, signal)

        msg = f"""
🔥 BEST SIGNAL

📊 {market}
💰 Entry: {price:.2f}
🚀 {signal}
🎯 {conf:.2f}%

🛑 SL: {sl:.2f}
🎯 TP: {tp:.2f}

🌍 Macro: {bias}
"""
        send(msg)
        save_trade(market, signal, conf)
    else:
        send("😴 Tidak ada peluang valid")

    save_db()

# run bot
if __name__ == "__main__":
    run_bot()
