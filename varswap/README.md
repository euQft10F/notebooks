# Variance swap replication — discrete strikes, interactive

Two [marimo](https://marimo.io) notebooks with Plotly charts on how a strip of listed options replicates a
variance swap when strikes are discrete.

## `varswap_practical.py` — the practical one (start here)

A focused, step-by-step notebook on short-dated SPX (1 week and 2 days to expiry):

- **Derman's chords** (Demeterfi, Derman, Kamal & Zou 1999, Appendix A), and Derman **plus the expected
  convexity correction**, derived step by step;
- the **VIX / CBOE** formula, and VIX with the **exact offset term** $\frac2T[(F/K_0-1)-\ln(F/K_0)]$ in place of
  the quadratic, with the full derivation;
- **model-free LP bounds**, with the LP-to-numpy translation spelled out;
- every method written twice: a plain loop and the numpy version, checked against each other.

Outputs: the log-contract payoff in the style of DDKZ's Appendix A figure (with the LP bounds), the fair
variance, and a side-by-side table of option weights (per unit of variance or in SPX contracts).

Data: a synthetic market (SVI fitted to the SPX smile, exact answer known) and a real SPX snapshot from
Yahoo Finance (close of 9 Oct 2026), saved in `data/spx_chain_2026-10-09.csv` by `fetch_spx_chain.py`.
To take a new snapshot (Yahoo only serves the current chain):

```sh
uvx --with yfinance --with pandas python fetch_spx_chain.py data
```

and point `SNAPSHOT` in the notebook's data cell at the new file (expiry dates in `load_spx` too).

## `varswap_replication.py` — the extended one

Six methods and many stress tests of large and uneven strike spacing:

- Derman's discrete replication, which is also linear-in-strike price integration;
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

With [uv](https://docs.astral.sh/uv/) (dependencies are read from each notebook's inline PEP 723 header):

```sh
uvx marimo edit varswap_practical.py --sandbox     # edit
uvx marimo run varswap_practical.py --sandbox      # app view
```

Without uv:

```sh
pip install -r requirements.txt
marimo edit varswap_practical.py
```
