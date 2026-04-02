import requests
import os
import random

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

# === SIMULASI DATA MARKET (nanti kita ganti real API)
def get_price():
    return random.uniform(0.4, 0.6)

# === STRATEGY SEDERHANA
def generate_signal(price):
    if price < 0.45:
        return "BUY"
    elif price > 0.55:
        return "SELL"
    else:
        return "NO TRADE"

if __name__ == "__main__":
    price = get_price()
    signal = generate_signal(price)

    if signal != "NO TRADE":
        msg = f"""
📊 Market: Polymarket (Simulasi)

💰 Price: {price:.2f}
🚀 Signal: {signal}
🎯 Confidence: 60%

🧠 Reason:
- Simple threshold strategy
"""
        send(msg)
