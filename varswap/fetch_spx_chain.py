"""Snapshot the SPX option chain from Yahoo Finance for every expiry within MAX_DAYS calendar days.

Run:  uvx --with yfinance --with pandas python fetch_spx_chain.py [out_dir]
Writes data/spx_chain_<YYYY-MM-DD>.csv (one row per option) and prints a per-expiry summary.
Yahoo serves only the current chain, so the file is the frozen record used by the notebook.
"""
import datetime as dt
import os
import sys

import pandas as pd
import yfinance as yf

MAX_DAYS = 10
out_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(out_dir, exist_ok=True)

tk = yf.Ticker("^SPX")
hist = tk.history(period="5d")
spot, spot_time = float(hist["Close"].iloc[-1]), hist.index[-1]
fetched = dt.datetime.now(dt.timezone.utc)
today = fetched.date()

rows = []
for exp in tk.options:
    d = dt.date.fromisoformat(exp)
    if (d - today).days > MAX_DAYS:
        break
    ch = tk.option_chain(exp)
    for kind, df in (("call", ch.calls), ("put", ch.puts)):
        df = df.copy()
        df["type"], df["expiry"] = kind, exp
        rows.append(df)

chain = pd.concat(rows, ignore_index=True)
keep = ["expiry", "type", "contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest",
        "impliedVolatility", "lastTradeDate"]
chain = chain[keep].rename(columns={"lastPrice": "last"})
chain["spot"], chain["spot_date"], chain["fetched_utc"] = spot, str(spot_time.date()), fetched.isoformat(timespec="seconds")
path = os.path.join(out_dir, f"spx_chain_{spot_time.date()}.csv")
chain.to_csv(path, index=False)

print("spot", spot, "as of", spot_time, "-> saved", path, len(chain), "rows")
for exp, g in chain.groupby("expiry"):
    two_sided = ((g.bid > 0) & (g.ask > 0)).sum()
    print(exp, "n", len(g), "two-sided", two_sided, "strikes", g.strike.min(), "-", g.strike.max(),
          "roots", sorted(set(g.contractSymbol.str.extract(r"^([A-Z]+)")[0])))
