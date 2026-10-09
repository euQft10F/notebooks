# Variance swap replication — discrete strikes, interactive

A [marimo](https://marimo.io) notebook with Plotly charts comparing how discrete option strips replicate a
variance swap, and how each method copes with large or uneven strike spacing:

- Derman's discrete replication (Demeterfi, Derman, Kamal & Zou 1999, *More than you ever wanted to know
  about volatility swaps*, Appendix A), which is also linear-in-strike price integration;
- the VIX / CBOE ΔK/K² rule;
- integrated-curvature cell weights;
- model-free sub/super-replication bounds by linear programming, with the optimal portfolios and their
  extremal measures;
- Derman with a market-implied convexity correction;
- smile-interpolated continuous replication (Le Floc'h 2018, *Variance Swap Replication: Discrete or
  Continuous?*).

Markets are synthetic: Black–Scholes, the DDKZ linear skew, and Heston with Le Floc'h's SPX calibration on
the listed January 2019 SPX strikes. A checks section reproduces published numbers from both papers.

## Run

With [uv](https://docs.astral.sh/uv/) (dependencies are read from the notebook's inline PEP 723 header):

```sh
uvx marimo edit varswap_replication.py --sandbox   # edit
uvx marimo run varswap_replication.py --sandbox    # app view
```

Without uv:

```sh
pip install -r requirements.txt
marimo edit varswap_replication.py
```
