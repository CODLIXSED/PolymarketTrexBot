import requests, os, sqlite3, pandas as pd, numpy as np, subprocess, time
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, jsonify

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DB = "data.db"

# ================= TELEGRAM =================
def send(msg):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  json={"chat_id": CHAT_ID, "text": msg})

# ================= DATABASE =================
def db():
    return sqlite3.connect(DB)

def init_db():
    c = db().cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS market(market TEXT, price REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades(market TEXT, signal TEXT, conf REAL, result REAL DEFAULT 0, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    db().commit()

def save_price(m,p):
    c=db().cursor(); c.execute("INSERT INTO market VALUES(?,?,CURRENT_TIMESTAMP)",(m,p)); db().commit()

def get_hist(m):
    return pd.read_sql(f"SELECT price FROM market WHERE market='{m}' ORDER BY ts DESC LIMIT 120", db())[::-1]

def save_trade(m,s,c):
    db().cursor().execute("INSERT INTO trades(market,signal,conf) VALUES(?,?,?)",(m,s,c)); db().commit()

def stats():
    df=pd.read_sql("SELECT * FROM trades",db())
    if len(df)==0: return 0,0
    return len(df), round(df["result"].mean(),4)

# ================= MARKET =================
def markets():
    data=requests.get("https://gamma-api.polymarket.com/markets").json()
    out=[]
    for m in data[:12]:
        try:
            out.append((m["question"],float(m.get("lastTradePrice",0.5)),m["id"]))
        except: pass
    return out

# ================= ORDERBOOK =================
def imbalance(id):
    try:
        ob=requests.get(f"https://clob.polymarket.com/orderbook/{id}").json()
        b=sum(float(x[1]) for x in ob.get("bids",[])[:10])
        a=sum(float(x[1]) for x in ob.get("asks",[])[:10])
        return (b-a)/(b+a+1e-6)
    except: return 0

# ================= MACRO REAL =================
def macro():
    try:
        btc=float(requests.get("https://api.coindesk.com/v1/bpi/currentprice.json")
                  .json()["bpi"]["USD"]["rate"].replace(",",""))
        return "RISK_ON" if btc>30000 else "RISK_OFF"
    except: return "NEUTRAL"

# ================= FEATURE =================
def features(df):
    df["ret"]=df.price.pct_change()
    df["mom"]=df.ret.rolling(5).mean()
    df["vol"]=df.ret.rolling(5).std()
    df["ema9"]=df.price.ewm(span=9).mean()
    df["ema21"]=df.price.ewm(span=21).mean()

    delta=df.price.diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=-delta.clip(upper=0).rolling(14).mean()
    rs=gain/(loss+1e-6)
    df["rsi"]=100-(100/(1+rs))

    df=df.dropna()
    if len(df)<20: return None

    last=df.iloc[-1]
    return [
        last["mom"], last["vol"], last["rsi"],
        last["ema9"]-last["ema21"]
    ]

# ================= ML =================
def train(df):
    df=df.copy()
    df["future"]=df.price.shift(-3)
    df["y"]=(df.future>df.price).astype(int)
    df=df.dropna()

    if len(df)<30: return None

    X=np.column_stack([
        df.price.pct_change().fillna(0),
        df.price.rolling(5).std().fillna(0),
        df.price.rolling(10).mean().fillna(0)
    ])
    y=df["y"]

    m=RandomForestClassifier(n_estimators=120)
    m.fit(X,y)
    return m

def predict(m,feat):
    if m is None or feat is None: return "WAIT",0
    p=m.predict([feat])[0]
    prob=max(m.predict_proba([feat])[0])
    if prob<0.65: return "NO TRADE",prob*100
    return ("BUY",prob*100) if p==1 else ("SELL",prob*100)

# ================= RISK =================
def risk(price,s):
    r=0.02
    if s=="BUY":
        return price*(1-r), price*(1+r*2)
    else:
        return price*(1+r), price*(1-r*2)

# ================= FILTER =================
def valid(signal,conf,imb,macro):
    if signal in ["WAIT","NO TRADE"]: return False
    if conf<65: return False
    if abs(imb)<0.05: return False
    if macro=="RISK_OFF" and signal=="BUY": return False
    return True

# ================= SAVE DB =================
def save_repo():
    try:
        subprocess.run(["git","config","--global","user.email","bot@bot.com"])
        subprocess.run(["git","config","--global","user.name","bot"])
        subprocess.run(["git","add",DB])
        subprocess.run(["git","commit","-m","update"],check=False)
        subprocess.run(["git","push"])
    except: pass

# ================= DASHBOARD =================
app=Flask(__name__)
@app.route("/")
def dash():
    df=pd.read_sql("SELECT * FROM trades ORDER BY ts DESC LIMIT 30",db())
    return df.to_html()

@app.route("/api")
def api():
    df=pd.read_sql("SELECT * FROM trades ORDER BY ts DESC LIMIT 30",db())
    return jsonify(df.to_dict(orient="records"))

# ================= MAIN =================
def run():
    init_db()
    mkts=markets()
    bias=macro()

    best=[]

    for name,price,id in mkts:
        save_price(name,price)
        df=get_hist(name)
        if len(df)<30: continue

        feat=features(df)
        imb=imbalance(id)

        model=train(df)
        sig,conf=predict(model,feat)

        if not valid(sig,conf,imb,bias): continue

        score=conf+abs(imb*100)
        best.append((score,name,price,sig,conf,imb))

    best=sorted(best,reverse=True)[:3]

    if best:
        msg="🔥 TOP SIGNALS\n"
        for i,(s,n,p,sg,c,imb) in enumerate(best,1):
            sl,tp=risk(p,sg)
            msg+=f"""
#{i} {n}
{sg} | {c:.1f}%
Entry {p:.2f}
SL {sl:.2f} | TP {tp:.2f}
OB {imb:.2f}
"""
            save_trade(n,sg,c)
    else:
        msg="😴 No high-quality trades"

    t,wr=stats()
    msg+=f"\n📊 Trades: {t} | Avg: {wr}"

    send(msg)
    save_repo()

if __name__=="__main__":
    run()
