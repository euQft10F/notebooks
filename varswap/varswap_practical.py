# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Replicating a variance swap with listed options: Derman vs VIX, with model-free bounds

    A **variance swap** pays, at expiry $T$, the realised variance of the index minus a fixed strike
    $K_{\text{var}}$ agreed today. The fair strike is the one that makes the swap free to enter, and it
    can be *built from options*: hold a static strip of out-of-the-money puts and calls (plus a
    forward), delta-hedge daily, and the profit equals realised variance. This notebook is about the
    practical step. **Real option chains only list a finite set of strikes**, so the textbook integral
    has to become a sum, and how you do that changes the price and the hedge.

    One complication shows up in every method. The forward $F$ is almost never a listed strike, so
    the strip is split at $K_0$, the listed strike just below $F$. That leaves a small **offset term**
    to correct, and the methods handle it differently.

    We compare four practical recipes and a pair of model-free bounds:

    | | Method | Option weights | Offset term |
    |---|---|---|---|
    | **A** | Derman's chords (Demeterfi, Derman, Kamal & Zou 1999, "DDKZ", Appendix A) | change in slope of straight lines joining the log payoff at the strikes | exact |
    | **A2** | Derman + expected convexity correction | same as A | exact, minus the expected over-payment of the chords |
    | **B** | VIX (CBOE VIX white paper) | $\frac{2}{T}\frac{\Delta K_i}{K_i^2}$ | quadratic approximation $\frac1T\big(\frac F{K_0}-1\big)^2$ |
    | **B2** | VIX with the exact offset | same as B | exact $\frac2T\big[\big(\frac F{K_0}-1\big)-\ln\frac F{K_0}\big]$ |
    | **LP** | Linear-programming bounds | cheapest portfolio above / dearest portfolio below the log payoff | built in |

    **What you will see.**

    1. *The log-contract payoff* each recipe actually delivers, drawn against the exact one in the
       style of DDKZ's Appendix A figure, with the LP upper and lower bounds around it.
    2. *The fair variance* each recipe gives, on a synthetic market where the exact answer is known
       and on a real SPX chain (Yahoo Finance close, 9 Oct 2026) with **1 week** and **2 trading days**
       to expiry.
    3. *The option weights*, i.e. what you would actually trade, strike by strike, in SPX contracts.

    Every method is written twice: a plain loop that mirrors the maths line by line, and the short
    numpy version used for the charts. An `assert` checks that the two agree.

    **Conventions and units.**

    * Option prices are *forward* prices: today's quoted price times $e^{rT}$, i.e. paid at expiry.
      Then $P(K)=\mathbb E[(K-S_T)^+]$ and $C(K)=\mathbb E[(S_T-K)^+]$ with no discount factor.
      $\mathbb E$ is the risk-neutral expectation, under which the forward has no drift. $F$ is the
      forward, and $T$ the time to expiry in years (calendar days / 365).
    * Variance is annualised and quoted as a volatility in % ("vol points", so $10.84$ means
      $\sigma=10.84\%$). Differences are in **vol basis points** (1 bp $=0.01$ vol pt). A **variance
      point** (used in the payoff charts) is (1 vol pt)$^2=10^{-4}$ in variance.
    * Two words used throughout: **truncation** is the variance hidden beyond the last listed strike,
      which no strip of listed options can see. The **support** $[S_{lo},S_{hi}]$ is the range of index
      levels at expiry that the LP bounds assume is possible.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Optional
    from scipy.integrate import simpson
    from scipy.optimize import least_squares, linprog
    from scipy.special import ndtr

    return (
        Optional,
        dataclass,
        field,
        go,
        least_squares,
        linprog,
        make_subplots,
        mo,
        ndtr,
        np,
        pd,
        simpson,
    )


@app.cell(hide_code=True)
def style(np):
    # One fixed colour + dash + marker per method (colour follows the method, never its rank).
    COLORS = {
        "Derman": "#2a78d6",
        "Derman + correction": "#7fb2ee",
        "VIX": "#eb6834",
        "VIX exact offset": "#f2a27c",
        "LP upper": "#4a3aa7",
        "LP lower": "#9085e9",
        "Exact": "#0b0b0b",
    }
    DASH = {"Derman": "solid", "Derman + correction": "dot", "VIX": "dash", "VIX exact offset": "dashdot",
            "LP upper": "solid", "LP lower": "dash", "Exact": "solid"}
    SYMBOL = {"Derman": "circle", "Derman + correction": "circle-open", "VIX": "square",
              "VIX exact offset": "square-open", "LP upper": "triangle-up", "LP lower": "triangle-down",
              "Exact": "diamond"}
    INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9",
               axis="#c3c2b7", surface="#fcfcfb", band="rgba(74,58,167,0.12)")


    def style_fig(fig, title=None, height=430, hover="x unified"):
        """Shared, recessive chart chrome; reserves room for the title, legend rows and subplot titles."""
        n_leg = sum(1 for t in fig.data if t.showlegend is not False and t.name)
        leg_rows = int(np.ceil(n_leg / 4)) if n_leg else 0
        sub_titles = any(a.yref == "paper" for a in fig.layout.annotations)
        top, bottom = 46 + 22 * leg_rows + (26 if sub_titles else 0), 56
        plot_h = max(height - top - bottom, 100)
        fig.update_layout(
            template="plotly_white", height=height,
            title=dict(text=title, x=0, xanchor="left", y=1 - 10 / height, yanchor="top",
                       font=dict(size=15, color=INK["primary"])),
            paper_bgcolor=INK["surface"], plot_bgcolor=INK["surface"],
            font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK["secondary"]),
            legend=dict(orientation="h", yanchor="bottom", y=1 + (28 if sub_titles else 6) / plot_h, x=0,
                        font=dict(size=11)),
            margin=dict(l=70, r=24, t=top, b=bottom), hovermode=hover,
        )
        fig.update_xaxes(gridcolor=INK["grid"], linecolor=INK["axis"], zerolinecolor=INK["axis"])
        fig.update_yaxes(gridcolor=INK["grid"], linecolor=INK["axis"], zerolinecolor=INK["axis"])
        return fig


    def vol_pts(var):
        """Annualised variance -> volatility in % points."""
        return 100.0 * np.sqrt(np.maximum(var, 0.0))


    def vol_bp(var, ref):
        """Difference between two variances, expressed in vol basis points (0.01 vol pt) around ref."""
        return 100.0 * (vol_pts(var) - vol_pts(ref))

    return COLORS, DASH, INK, SYMBOL, style_fig, vol_bp, vol_pts


@app.cell(hide_code=True)
def theory(mo):
    mo.md(r"""
    ## 1 · From variance to the log contract (the one derivation everything rests on)

    **Step 1: variance is a log contract plus a delta hedge.** Let the forward follow
    $dF_t/F_t=\sigma_t\,dW_t$. The volatility $\sigma_t$ may be random and may depend on the path;
    nothing below needs a model for it. Assume no jumps and continuous hedging; with jumps or daily
    sampling the result holds only approximately. At expiry the forward equals the index, $F_T=S_T$.
    Itô's lemma for $\ln F_t$ gives $d\ln F_t=\frac{dF_t}{F_t}-\tfrac12\sigma_t^2dt$. Integrate from
    $0$ to $T$:

    $$
    \int_0^T d\ln F_t\;=\;\int_0^T\frac{dF_t}{F_t}\;-\;\frac12\int_0^T\sigma_t^2\,dt .
    $$

    **Why the left side becomes a log contract.** $\int_0^T d\ln F_t$ just adds up all the small
    changes in $\ln F_t$, so it collapses to *last value minus first value*. Chop $[0,T]$ into days
    $t_0=0<t_1<\dots<t_n=T$; the daily log returns telescope:

    $$
    \sum_{i=1}^{n}\big(\ln F_{t_i}-\ln F_{t_{i-1}}\big)=\ln F_T-\ln F_0=\ln\frac{F_T}{F_0}=\ln\frac{S_T}{F_0},
    $$

    using $F_T=S_T$ in the last step. Every intermediate price cancels, so this term depends **only on
    where the index ends**, not on the path it took. A payoff that depends only on $S_T$ is an ordinary
    European payoff: you can buy it today and hold it to expiry without trading. That is what *static*
    means here. The other term, $\int_0^T dF_t/F_t$, cannot be simplified this way, because each day's
    return is weighted by $1/F_t$ at that day's price; it needs trading along the path.

    Now rearrange (move the $\sigma^2$ term to the left and $\ln\frac{S_T}{F_0}$ to the right) and
    multiply by $\frac2T$:

    $$
    \frac1T\int_0^T\sigma_t^2\,dt\;=\;\underbrace{\frac2T\int_0^T\frac{dF_t}{F_t}}_{\text{hold }2/(TF_t)\text{ forwards, rebalanced daily}}\;\;\underbrace{-\;\frac2T\ln\frac{S_T}{F_0}}_{\text{static "log contract"}} .
    $$

    The minus sign means you are **short** the log contract (equivalently, long the payoff
    $-\frac2T\ln\frac{S_T}{F_0}$).

    The daily-rebalanced forward position costs nothing to enter and has zero expected value, because
    the forward has no drift. So the fair variance strike is the price of the log contract. Writing $F$
    for today's forward $F_0$:

    $$
    K_{\text{var}}=\mathbb E\big[L(S_T)\big],\qquad L(S)=-\frac2T\ln\frac SF .
    $$

    **Step 2: any smooth payoff is a strip of options (Carr–Madan).** For a twice-differentiable
    $g$ and any split point $\kappa$:

    $$
    g(S)=g(\kappa)+g'(\kappa)(S-\kappa)+\int_0^{\kappa}g''(K)\,(K-S)^+\,dK+\int_{\kappa}^{\infty}g''(K)\,(S-K)^+\,dK .
    $$

    *Proof in two lines.* For $S>\kappa$ only the call integral is non-zero; integrate by parts:
    $\int_\kappa^S g''(K)(S-K)\,dK=\big[g'(K)(S-K)\big]_\kappa^S+\int_\kappa^S g'(K)\,dK=-g'(\kappa)(S-\kappa)+g(S)-g(\kappa)$.
    Rearranging gives the formula; $S<\kappa$ is the mirror image with puts. Each option is a kink,
    and **the amount of options at strike $K$ is the curvature $g''(K)$ there**.

    **Step 3: apply it to the log payoff, split at a listed strike $K_0$.** We cannot split at $F$,
    because it is not a strike. So take $K_0$ = the highest listed strike $\le F$. For $L$:

    $$
    L''(K)=\frac{2}{TK^2},\qquad L'(K_0)=-\frac{2}{TK_0},\qquad L(K_0)=-\frac2T\ln\frac{K_0}{F}=\frac2T\ln\frac{F}{K_0}.
    $$

    Take expectations of Carr–Madan, using $\mathbb E[S_T-K_0]=F-K_0$,
    $\mathbb E[(K-S_T)^+]=P(K)$ and $\mathbb E[(S_T-K)^+]=C(K)$:

    $$
    \mathbb E[L]=\underbrace{L(K_0)+L'(K_0)(F-K_0)}_{=\;\frac2T\ln\frac F{K_0}\;-\;\frac2T\left(\frac F{K_0}-1\right)}+\frac2T\left[\int_0^{K_0}\frac{P(K)}{K^2}dK+\int_{K_0}^{\infty}\frac{C(K)}{K^2}dK\right],
    $$

    which is

    $$
    \boxed{K_{\text{var}}=\frac2T\left[\int_0^{K_0}\frac{P(K)}{K^2}dK+\int_{K_0}^{\infty}\frac{C(K)}{K^2}dK\right]\;-\;\frac2T\Big[\Big(\frac F{K_0}-1\Big)-\ln\frac F{K_0}\Big]}
    $$

    * **Why $1/K^2$:** the curvature of $\ln$ is $1/K^2$, so low strikes carry more weight. A
      1-point move matters more, in return terms, when the index is lower.
    * **What the offset term is.** Ideally we would use out-of-the-money options everywhere: puts
      below $F$, calls above. The split at $K_0<F$ means the call integral also covers $[K_0,F]$,
      where puts would have been the out-of-the-money choice. By put–call parity, $C(K)=P(K)+(F-K)$,
      so each of those calls carries an extra intrinsic value $F-K$. Weighted like everything else,
      that extra is exactly the offset term:
      $\frac2T\int_{K_0}^{F}\frac{F-K}{K^2}dK=\frac2T\big[(\frac F{K_0}-1)-\ln\frac F{K_0}\big]$.

      *Step by step.* Split the fraction into two simple powers of $K$:
      $\frac{F-K}{K^2}=\frac{F}{K^2}-\frac1K$. Each has a standard antiderivative:
      $\int\frac{F}{K^2}dK=-\frac FK$ (since $F$ is a constant) and $\int\frac1K\,dK=\ln K$. So

      $\displaystyle\int_{K_0}^{F}\Big(\frac{F}{K^2}-\frac1K\Big)dK=\Big[-\frac FK-\ln K\Big]_{K_0}^{F}=\Big(-\frac FF-\ln F\Big)-\Big(-\frac F{K_0}-\ln K_0\Big)=\Big(\frac F{K_0}-1\Big)-\ln\frac F{K_0},$

      using $\frac FF=1$ and $\ln F-\ln K_0=\ln\frac F{K_0}$. Multiplying by $\frac2T$ gives the
      offset. Writing $x=\frac F{K_0}-1$, it is $\frac2T\big[x-\ln(1+x)\big]$, which is $\ge0$ because
      $\ln(1+x)\le x$.

      The term removes value that is not volatility, and it is zero when $F$ sits on a strike.
    * **How big?** With $K_0=7800$, $F=7803.5$, so $x=F/K_0-1=4.5\times10^{-4}$, and one week to expiry
      ($T=7/365$), the offset is about $x^2/T\approx1.06\times10^{-5}$ in variance, roughly $0.005$ vol
      points at an 11% vol. It is small, but it is a systematic bias, and it grows quickly with the gap.

    Every method below is a different way to turn the two integrals into a finite sum over the listed
    strikes $K_1<\dots<K_N$, and a different way to treat the offset.
    """)
    return


@app.cell(hide_code=True)
def data_md(mo):
    mo.vstack([mo.md(r"""
    ## 2 · Market data

    ### 2.1 A real chain: SPX weeklies from Yahoo Finance

    `fetch_spx_chain.py` (in this folder) saved the SPX chain from Yahoo Finance after the close on
    **Friday 9 October 2026** to `data/spx_chain_2026-10-09.csv`. Yahoo only serves the current chain,
    so this file is the frozen record. We use the PM-settled weekly `SPXW` options:

    * **1 week:** expiry Friday 16 Oct, $T=7/365$.
    * **2 trading days:** expiry Tuesday 13 Oct, $T=4/365$. That is two trading days but four calendar
      days, because of the weekend. Measured in calendar time, $T$ includes two quiet weekend days, so
      the implied vols come out **low**: about 6.7% at the money, against 9.4% for the 1-week options.
      Measured in trading days ($T=2/252$), the same prices give about 7.9%.

    From the quotes we need, for each strike, one option price, plus the forward $F$ and the split
    strike $K_0$. The steps follow the CBOE VIX white paper, plus two clean-up steps that the LP of
    section 8 needs. Open the box below for the details. They matter for reproducing the numbers, not
    for understanding the methods.

    *A caveat about the real chain.* The exchange lists SPXW strikes every 5 points near the money,
    but Yahoo's snapshot leaves some of them out (a strike shows up only if it was quoted or traded).
    The real chain therefore has the occasional 10–55-point hole near the money and 25–100-point
    spacing in the wings. That makes it a realistic test of uneven spacing. The synthetic market of
    section 2.2 supplies the clean 5-point case.
    """),
    mo.accordion({"Data plumbing: from Yahoo quotes to one price per strike (click to open)": mo.md(r"""
    1. **Price = mid of bid and ask.** Two kinds of quote are dropped:
       * quotes with a zero bid;
       * *stale* quotes, from contracts that did not trade on 9 October. Yahoo keeps showing the bid/ask
         they had on an earlier day. For example, the 2-day put at 7,815 showed 51.8/52.2 when its
         neighbours traded near 20.
    2. **Forward from put–call parity.** For forward prices, $C(K)-P(K)=F-K$. Find the strike $K^*$
       where $|C-P|$ is smallest and set $F=K^*+e^{rT}\,(C-P)$, with the quoted (discounted) prices. We
       take $r=4\%$; over a week the discount factor is $0.9992$, so $r$ barely matters.
    3. **$K_0$** = the highest strike $\le F$ that has both a put and a call quote. Use puts at and
       below $K_0$, calls at and above it.
    4. **Wings:** walk away from $K_0$ and stop after two consecutive zero bids (CBOE's rule).
    5. **Remove bid–ask noise.** Two facts make this necessary:
       * Absence of arbitrage requires call prices to be **convex** in the strike: their slope must rise
         with $K$. A dip in the price curve is a butterfly spread with negative cost.
       * The slope of the call price is $C'(K)=-\text{Prob}(S_T>K)$, a probability with a minus sign, so
         it must lie between $-1$ and $0$.

       Mid prices are noisy at the tick level, and about one strike in four breaks convexity by a
       fraction of a tick. The four methods barely notice. The LP of section 8, however, can hold any
       long or short position, and it would exploit that noise without limit.

       The fix works on a single curve:
       * Use parity ($C=P+F-K$) to write every quote as a call price.
       * Compute the slopes between neighbouring strikes. Wherever a slope is *lower* than the one to
         its left, replace both by their (width-weighted) average, and repeat until the slopes increase.
         This is the *pool-adjacent-violators* algorithm, about 10 lines of code.
       * Clip the slopes to $[-1,0]$, rebuild the prices from them, and convert back to puts below $K_0$.

       On the 1-week chain no price moves by more than half its bid–ask spread; on the 2-day chain one
       price of 142 does.
    6. **Trim flat wings.** Far out-of-the-money options stuck at the minimum tick form a flat price
       curve. A flat curve means zero probability beyond that strike, which contradicts a positive
       price, so those quotes are dropped.
    """)})])
    return


@app.cell(hide_code=True)
def pricing(ndtr, np):
    def bs_price(F, K, T, vol, cp):
        """Undiscounted Black-76 price; cp = +1 call, -1 put."""
        K = np.asarray(K, float)
        vol = np.broadcast_to(np.asarray(vol, float), K.shape)
        sd = vol * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
        return cp * (F * ndtr(cp * d1) - K * ndtr(cp * (d1 - sd)))


    def implied_vol(F, K, T, price, cp, lo=1e-4, hi=5.0, iters=80):
        """Vectorised bisection implied vol (slow-ish but cannot fail)."""
        K = np.atleast_1d(np.asarray(K, float))
        price = np.broadcast_to(np.asarray(price, float), K.shape)
        a, b = np.full(K.shape, lo), np.full(K.shape, hi)
        for _ in range(iters):
            mid = 0.5 * (a + b)
            high = bs_price(F, K, T, mid, cp) > price
            b, a = np.where(high, mid, b), np.where(high, a, mid)
        return 0.5 * (a + b)

    return bs_price, implied_vol


@app.cell(hide_code=True)
def data(Optional, dataclass, mo, np, pd):
    R_ASSUMED = 0.04
    SNAPSHOT = mo.notebook_dir() / "data" / "spx_chain_2026-10-09.csv"
    chain_raw = pd.read_csv(SNAPSHOT)


    @dataclass
    class Quotes:
        """What every method needs: forward F, expiry T, strikes K, and at each strike the price of the
        option we use there: put P(K) for K <= K0, call C(K) for K >= K0 (both at K0)."""
        name: str
        F: float
        T: float
        K: np.ndarray
        P: np.ndarray            # NaN above K0
        C: np.ndarray            # NaN below K0
        K0: float
        truth: Optional[float] = None           # exact fair variance (synthetic markets only)
        truth_in_range: Optional[float] = None  # exact integral over [K_1, K_N] only

        @property
        def Kp(self):
            return self.K[self.K <= self.K0]

        @property
        def Kc(self):
            return self.K[self.K >= self.K0]

        @property
        def Pp(self):
            return self.P[self.K <= self.K0]

        @property
        def Cc(self):
            return self.C[self.K >= self.K0]

        def held_price(self):
            """Price of the option held at each strike once a call at K0 is rewritten as a put: P at and below K0, C above."""
            return np.where(self.K <= self.K0, self.P, self.C)


    def pava(y, w):
        """Closest non-decreasing sequence to y in weighted least squares (pool-adjacent-violators):
        whenever two neighbours are in the wrong order, replace both by their weighted average."""
        vals, wts, cnt = [], [], []
        for yi, wi in zip(y, w):
            vals.append(yi), wts.append(wi), cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
                vals[-2:], wts[-2:], cnt[-2:] = [v], [wts[-2] + wts[-1]], [cnt[-2] + cnt[-1]]
        return np.repeat(vals, cnt)


    def remove_butterfly_noise(K, V, slope_lo, slope_hi):
        """Make a price curve convex with slopes in [slope_lo, slope_hi] (no static arbitrage), moving the
        prices as little as possible: fix the slopes with PAVA, rebuild prices, keep the average level."""
        h = np.diff(K)
        slopes = np.clip(pava(np.diff(V) / h, h), slope_lo, slope_hi)
        V_new = np.r_[0.0, np.cumsum(slopes * h)]
        return V_new + np.mean(V - V_new)


    def load_spx(expiry, days, r=R_ASSUMED):
        """CBOE-style selection from the Yahoo snapshot. Returns (Quotes, table of the quotes used)."""
        every = chain_raw[(chain_raw.expiry == expiry) & chain_raw.contractSymbol.str.startswith("SPXW")]
        traded = pd.to_datetime(every.lastTradeDate).dt.tz_convert("America/New_York").dt.date.astype(str)
        g = every[(every.bid > 0) & (every.ask > 0) & (traded == every.spot_date)]   # live two-sided quotes only
        T = days / 365.0
        growth = np.exp(r * T)                                         # forward price = e^{rT} x quoted price
        calls = g[g.type == "call"].set_index("strike").sort_index()
        puts = g[g.type == "put"].set_index("strike").sort_index()
        mid_c, mid_p = 0.5 * (calls.bid + calls.ask) * growth, 0.5 * (puts.bid + puts.ask) * growth

        # 1. forward from put-call parity at the strike where |C - P| is smallest
        both = mid_c.index.intersection(mid_p.index)
        K_star = (mid_c[both] - mid_p[both]).abs().idxmin()
        F = K_star + (mid_c[K_star] - mid_p[K_star])
        # 2. K0 = highest strike <= F that has both a put and a call quote
        K0 = both[both <= F].max()

        # 3. walk outwards from K0 through the listed strikes; stop after two zero bids in a row.
        #    Stale quotes (no trade on the snapshot day) are skipped: their bid/ask is left over from an earlier day.
        def walk(kind, prices, strikes):
            bids = every[every.type == kind].set_index("strike").bid
            keep, zeros = [], 0
            for k in strikes:
                if k in prices.index:
                    keep.append(k)
                    zeros = 0
                elif bids.get(k, 0.0) == 0.0:
                    zeros += 1
                    if zeros == 2:
                        break
            return keep

        listed_p = np.unique(every[every.type == "put"].strike)
        listed_c = np.unique(every[every.type == "call"].strike)
        kp = np.array(walk("put", mid_p, listed_p[listed_p <= K0][::-1])[::-1], float)
        kc = np.array(walk("call", mid_c, listed_c[listed_c >= K0]), float)

        # 4. remove bid-ask noise that would look like arbitrage: write every quote as a call price via parity
        #    (C = P + F - K), make that one curve convex with slopes in [-1, 0], then convert back
        K = np.union1d(kp, kc)
        as_call = pd.concat([mid_p[kp] + (F - kp), mid_c[kc]]).groupby(level=0).mean().reindex(K).values
        C_all = remove_butterfly_noise(K, as_call, -1.0, 0.0)

        # 5. drop flat wing quotes (a far-OTM option stuck at the minimum tick has slope 0 = zero probability,
        #    yet a positive price; no distribution can produce both)
        slope = np.diff(C_all) / np.diff(K)
        lo, hi = 0, K.size - 1
        while slope[lo] <= -1.0 + 1e-9:
            lo += 1
        while slope[hi - 1] >= -1e-9:
            hi -= 1
        K, C_all = K[lo:hi + 1], C_all[lo:hi + 1]

        P = np.where(K <= K0, C_all - (F - K), np.nan)
        C = np.where(K >= K0, C_all, np.nan)
        q = Quotes(f"SPX {expiry} ({days} cal. days)", float(F), T, K, P, C, float(K0))
        side = np.where(K < K0, "put", np.where(K > K0, "call", "put + call"))
        bid = np.where(K <= K0, puts.bid.reindex(K).values, calls.bid.reindex(K).values) * growth
        ask = np.where(K <= K0, puts.ask.reindex(K).values, calls.ask.reindex(K).values) * growth
        raw = np.where(K <= K0, mid_p.reindex(K).values, mid_c.reindex(K).values)
        used = np.where(K <= K0, P, C)
        table = pd.DataFrame({"strike": K, "side": side, "bid": bid, "ask": ask, "mid": raw,
                              "used (after noise fix)": used, "change / half-spread": (used - raw) / (0.5 * (ask - bid))})
        return q, table


    spx_1w, spx_1w_table = load_spx("2026-10-16", 7)
    spx_2d, spx_2d_table = load_spx("2026-10-13", 4)
    return Quotes, spx_1w, spx_1w_table, spx_2d, spx_2d_table


@app.cell(hide_code=True)
def data_view(
    COLORS,
    INK,
    go,
    mo,
    np,
    spx_1w,
    spx_1w_table,
    spx_2d,
    spx_2d_table,
    style_fig,
):
    def chain_summary(q, t):
        gaps = np.diff(q.K)
        near = np.abs(q.K[1:] / q.F - 1) < 0.01
        outside = int((t["change / half-spread"].abs() > 1).sum())
        facts = mo.md(
            f"**{q.name}** · forward $F$ = {q.F:,.2f} · $K_0$ = {q.K0:,.0f} · "
            f"$F/K_0-1$ = {q.F / q.K0 - 1:.5f} · $T$ = {q.T * 365:.0f}/365 · "
            f"{len(q.Kp)} puts ({q.Kp.min():,.0f}–{q.K0:,.0f}), {len(q.Kc)} calls ({q.K0:,.0f}–{q.Kc.max():,.0f}) · "
            f"median gap within ±1% of $F$: {np.median(gaps[near]):.0f} pts, largest: {gaps[near].max():.0f} pts; "
            f"in the wings up to {gaps.max():.0f} pts · noise fix moved {outside} of {len(t)} prices by more than half the spread"
        )
        fig = go.Figure(go.Scatter(x=q.K[1:], y=gaps, mode="markers", marker=dict(color=COLORS["Derman"], size=7),
                                   hovertemplate="gap ending at %{x:,.0f}: %{y:.0f} pts<extra></extra>", name="gap"))
        fig.add_vline(x=q.F, line=dict(color=INK["muted"], dash="dot"), annotation_text="F")
        fig.update_yaxes(type="log", title_text="gap to previous strike (pts)")
        fig.update_xaxes(title_text="strike")
        style_fig(fig, "Strike spacing of the quotes actually used", height=300, hover="closest")
        return mo.vstack([facts, fig, mo.ui.table(t.round(4), selection=None, page_size=10, label="Quotes used (forward prices)")])


    mo.ui.tabs({"1 week (16-Oct)": chain_summary(spx_1w, spx_1w_table),
                "2 trading days (13-Oct)": chain_summary(spx_2d, spx_2d_table)})
    return


@app.cell(hide_code=True)
def svi_md(mo):
    mo.vstack([mo.md(r"""
    ### 2.2 A synthetic market with a known answer

    With real quotes there is no "right answer" to compare against. So we also build a **synthetic
    market fitted to the same SPX smile**: a smooth implied-vol curve through the 1-week quotes, with
    every option priced by Black's formula on that curve. This market has a price at *any* strike. We
    can list strikes exactly **5 points apart**, slide $F$ anywhere inside a gap, and compute the exact
    answer by integrating the formula of section 1 numerically on a very fine grid. We report two
    versions of it:

    * **in-range exact**: the integral over the listed strike range $[K_1,K_N]$ only. This is the right
      yardstick for *discretisation* error, the part each method controls, because no method can see
      beyond the last strike.
    * **full exact**: the integral from 5% to 270% of $F$, which in practice means all strikes. Beyond
      the quotes it uses the fitted curve's extrapolation. The gap between the two is the
      **truncation** error: about 2 vol bp at one week. It is the same for every method, and it depends
      on how you believe the smile continues beyond the last quote.
    """),
    mo.accordion({"How the smile is fitted and moved to 2 days (click to open)": mo.md(r"""
    1. Turn the 1-week OTM prices into implied vols, keeping quotes worth at least 0.5 index points.
       Cheaper quotes sit near the 0.05 minimum tick, where implied vol is meaningless.
    2. Fit the standard **SVI** ("stochastic volatility inspired") curve for the total implied
       variance $\theta(k)=\sigma_{\text{imp}}^2T$ as a function of log-moneyness $k=\ln(K/F)$:
       $$\theta(k)=\alpha+\beta\big[\rho(k-\mu)+\sqrt{(k-\mu)^2+\varsigma^2}\big].$$
       This is a hyperbola: two straight wings joined by a smooth bottom. It has five parameters,
       with $\rho$ setting the skew. A least-squares fit gets within about 0.04 vol points of the
       quotes. An `assert` checks that the fitted curve implies a positive probability density
       everywhere (no butterfly arbitrage).
    3. **2 days to expiry:** keep the same smile *in standard-deviation units*,
       $\theta_{T_2}(k)=\tfrac{T_2}{T_1}\,\theta_{T_1}\!\big(k\sqrt{T_1/T_2}\big)$, and the same strike
       range measured in standard deviations.

    Because SVI's wings are straight lines in total variance, implied vol keeps rising far from the
    money. That puts noticeable value beyond the last quoted strike, which is why the truncation is
    larger than the discretisation errors.
    """),
    "About SVI: where it comes from and why this shape (click to open)": mo.md(r"""
    **Origin.** Jim Gatheral devised SVI at Merrill Lynch in 1999 and presented it in 2004 in *A
    parsimonious arbitrage-free implied volatility parameterization with application to the valuation
    of volatility derivatives* (Global Derivatives, Madrid). It is also covered in his book *The
    Volatility Surface* (2006). The main follow-ups:

    * Gatheral & Jacquier, *Convergence of Heston to SVI* (Quantitative Finance, 2011): SVI is the exact
      large-maturity limit of the Heston smile, hence "stochastic volatility inspired".
    * Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces* (Quantitative Finance, 2014): the
      no-butterfly condition $g(k)\ge0$ that our `assert` checks, and the surface version SSVI.
    * Zeliade Systems, *Quasi-explicit calibration of Gatheral's SVI model* (2009): the standard way to
      make the fit fast and stable.

    **Why this shape.**

    * **Straight wings.** For large $|k|$, $\theta$ grows linearly, with slopes $\beta(1+\rho)$ on the
      right and $\beta(1-\rho)$ on the left. Roger Lee's moment formula (Mathematical Finance, 2004) says
      total implied variance can grow *at most* linearly in log-strike, with slope at most 2. So SVI
      has exactly the allowed kind of wing, and the bound is a simple parameter check.
    * **One job per parameter.** $\alpha$ sets the level, $\beta$ the steepness of the wings, $\rho$ the
      tilt (equity skew, $\rho<0$), $\mu$ shifts the smile sideways, and $\varsigma$ rounds the bottom
      (the at-the-money curvature; $\varsigma\to0$ gives a V).
    * **Few parameters.** Five parameters fit a single equity-index expiry well and are easy to
      sanity-check.

    **Caveats.** SVI is not arbitrage-free on its own: a fit to tick-floored wing quotes created
    butterfly arbitrage, which is why we fit only quotes of at least 0.5 points and check $g(k)>0$.
    Very short expiries have sharper, more V-shaped smiles, which SVI fits less well. Our 2-day market
    is the 1-week smile rescaled, not a separate fit. Finally, the "full exact" value depends on the
    straight-line wings beyond the last quote: the truncation estimate is only as good as that
    extrapolation.
    """)})])
    return


@app.cell(hide_code=True)
def svi(
    Quotes,
    bs_price,
    dataclass,
    implied_vol,
    least_squares,
    np,
    simpson,
    spx_1w,
):
    def svi_theta(params, k):
        """SVI total implied variance theta(k) = alpha + beta[rho(k - mu) + sqrt((k - mu)^2 + sig^2)]."""
        alpha, beta, rho, mu, sig = params
        return alpha + beta * (rho * (k - mu) + np.sqrt((k - mu) ** 2 + sig**2))


    def svi_butterfly(params, k):
        """Gatheral's g(k): the smile implies a positive density (no butterfly arbitrage) wherever g(k) >= 0."""
        alpha, beta, rho, mu, sig = params
        th, r = svi_theta(params, k), np.sqrt((k - mu) ** 2 + sig**2)
        d1, d2 = beta * (rho + (k - mu) / r), beta * sig**2 / r**3
        return (1 - k * d1 / (2 * th)) ** 2 - d1**2 / 4 * (1 / th + 0.25) + d2 / 2


    def fit_svi(q, min_price=0.5):
        """Least-squares SVI fit to the OTM implied vols. Quotes below `min_price` are left out: near the
        minimum tick (0.05) their implied vols are meaningless. Returns params, all implied vols, fitted mask."""
        otm = np.where(q.K < q.F, q.P, q.C)
        iv = implied_vol(q.F, q.K, q.T, otm, np.where(q.K < q.F, -1, 1))
        used = otm >= min_price
        k, theta = np.log(q.K[used] / q.F), iv[used] ** 2 * q.T
        fit = least_squares(lambda p: (svi_theta(p, k) - theta) / theta.mean(), [0.5 * theta.min(), 0.1, -0.5, 0.0, 0.05],
                            bounds=([-1, 1e-6, -0.999, -0.5, 1e-4], [1, 10, 0.999, 0.5, 2]))
        return fit.x, iv, used


    @dataclass
    class SviMarket:
        """Black prices on an SVI smile: a synthetic market whose exact fair variance we can compute."""
        F: float
        T: float
        params: np.ndarray

        def vol(self, K):
            return np.sqrt(svi_theta(self.params, np.log(np.asarray(K, float) / self.F)) / self.T)

        def put(self, K):
            return bs_price(self.F, K, self.T, self.vol(K), -1)

        def call(self, K):
            return bs_price(self.F, K, self.T, self.vol(K), 1)

        def at_horizon(self, T2):
            """Same smile in standard-deviation units at another expiry."""
            alpha, beta, rho, mu, sig = self.params
            c = np.sqrt(self.T / T2)
            return SviMarket(self.F, T2, np.array([alpha * T2 / self.T, beta * T2 / self.T * c, rho, mu / c, sig / c]))

        def exact_kvar(self, K_lo=None, K_hi=None, n=40001):
            """(2/T)[int P/K^2 dK below F + int C/K^2 dK above F], in y = ln(K/F) where dK/K^2 = dy/K.
            Without limits it runs from K = 5% to 270% of F, far beyond any measurable contribution."""
            y_lo = -3.0 if K_lo is None else np.log(K_lo / self.F)
            y_hi = 1.0 if K_hi is None else np.log(K_hi / self.F)
            y = np.linspace(y_lo, 0.0, n)
            below = simpson(self.put(self.F * np.exp(y)) / (self.F * np.exp(y)), x=y)
            y = np.linspace(0.0, y_hi, n)
            above = simpson(self.call(self.F * np.exp(y)) / (self.F * np.exp(y)), x=y)
            return 2.0 / self.T * float(below + above)


    svi_params, spx_1w_iv, svi_used = fit_svi(spx_1w)
    svi_1w = SviMarket(spx_1w.F, spx_1w.T, svi_params)
    assert svi_butterfly(svi_params, np.linspace(-1.0, 0.5, 600)).min() > 0, "fitted smile admits butterfly arbitrage"


    def synthetic_quotes(base, T, step, gap_pos):
        """Round strikes every `step` points over the real 1-week strike range (in sd units at horizon T).
        The forward sits a fraction `gap_pos` of the way from K0 to the next strike."""
        K0 = np.floor(base.F / step) * step
        F = K0 + gap_pos * step
        mk = SviMarket(F, base.T, base.params).at_horizon(T)
        c = np.sqrt(T / base.T)
        lo, hi = F * (spx_1w.K.min() / base.F) ** c, F * (spx_1w.K.max() / base.F) ** c
        K = step * np.arange(np.ceil(lo / step), np.floor(hi / step) + 1)
        P, C = np.where(K <= K0, mk.put(K), np.nan), np.where(K >= K0, mk.call(K), np.nan)
        q = Quotes(f"Synthetic SVI, {T * 365:.0f} days, {step:g}-pt strikes", F, T, K, P, C, float(K0))
        q.truth, q.truth_in_range = mk.exact_kvar(), mk.exact_kvar(K.min(), K.max())
        return q


    demo = synthetic_quotes(svi_1w, 7 / 365, 5.0, 0.5)   # fixed example used by the loop-vs-numpy checks
    return (
        SviMarket,
        demo,
        spx_1w_iv,
        svi_1w,
        svi_params,
        svi_used,
        synthetic_quotes,
    )


@app.cell(hide_code=True)
def svi_view(
    COLORS,
    INK,
    go,
    mo,
    np,
    spx_1w,
    spx_1w_iv,
    style_fig,
    svi_1w,
    svi_params,
    svi_used,
):
    def chart_smile():
        fig = go.Figure()
        Kd = np.linspace(spx_1w.K.min(), spx_1w.K.max(), 600)
        for mask, name, marker in ((~svi_used, "quotes below 0.5 pts (not fitted)", dict(color=INK["muted"], size=5, symbol="circle-open")),
                                   (svi_used, "SPX 1 week, fitted quotes", dict(color=COLORS["Derman"], size=6))):
            fig.add_trace(go.Scatter(x=spx_1w.K[mask], y=100 * spx_1w_iv[mask], mode="markers", name=name, marker=marker,
                                     hovertemplate="K=%{x:,.0f}<br>%{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=Kd, y=100 * svi_1w.vol(Kd), mode="lines", name="SVI fit (synthetic 1-week market)",
                                 line=dict(color=INK["primary"], width=2),
                                 hovertemplate="K=%{x:,.0f}<br>%{y:.2f}%<extra>SVI</extra>"))
        rms = 100 * np.sqrt(np.mean((svi_1w.vol(spx_1w.K[svi_used]) - spx_1w_iv[svi_used]) ** 2))
        fig.add_vline(x=spx_1w.F, line=dict(color=INK["muted"], dash="dot"), annotation_text="F")
        fig.update_xaxes(title_text="strike")
        fig.update_yaxes(title_text="implied vol (%)", range=[0, 100 * float(svi_1w.vol(Kd).max()) * 1.15])
        style_fig(fig, f"1-week SPX smile and its SVI fit (RMS error {rms:.2f} vol pts)", height=360, hover="closest")
        alpha, beta, rho, mu, sig = svi_params
        note = mo.md(f"SVI parameters: $\\alpha$={alpha:.5f}, $\\beta$={beta:.4f}, $\\rho$={rho:.3f}, $\\mu$={mu:.4f}, "
                     f"$\\varsigma$={sig:.4f}. ATM vol {100 * float(svi_1w.vol(spx_1w.F)):.2f}%.")
        return mo.vstack([fig, note])


    chart_smile()
    return


@app.cell(hide_code=True)
def portfolio_md(mo):
    mo.md(r"""
    ## 3 · A replicating portfolio, in code

    Every method below produces the same kind of object: a static portfolio of

    * a **bond** paying $c$ at expiry (cash),
    * a **forward** position $a\,(S_T-F)$, which costs nothing today,
    * **puts** with weights $w^p_i$ at strikes $K_i\le K_0$ and **calls** with weights $w^c_j$ at
      $K_j\ge K_0$,

    so its payoff and price are

    $$
    \Pi(S)=c+a\,(S-F)+\sum_i w^p_i\,(K_i-S)^+ +\sum_j w^c_j\,(S-K_j)^+,
    \qquad
    \text{price}=c+\sum_i w^p_iP(K_i)+\sum_j w^c_jC(K_j).
    $$

    For the log contract, read the legs off Carr–Madan in section 1, after rewriting
    $S-K_0=(S-F)+(F-K_0)$:

    $$
    L(S)=\underbrace{L(K_0)+L'(K_0)(F-K_0)}_{c\;=\;-\text{offset}}+\underbrace{L'(K_0)}_{a\;=\;-\frac{2}{TK_0}}(S-F)+\text{options}.
    $$

    So every method uses the same forward leg $a=-\frac{2}{TK_0}$, and the bond $c$ is minus that
    method's offset term (exact or quadratic). With these, $\Pi(S)$ is directly comparable with
    $L(S)$, and **its price is the method's fair variance**. The methods differ only in the option
    weights and in the offset.

    *Notation used from here on:* $Q(K)$ is the **price** of the option used at strike $K$ (put
    below $K_0$, call above). Probabilities are written $\text{Prob}(\cdot)$.
    """)
    return


@app.cell(hide_code=True)
def portfolio(dataclass, field, np):
    @dataclass
    class Portfolio:
        """bond c + forward a(S - F) + puts wp at Kp + calls wc at Kc. `kvar` is its price."""
        name: str
        Kp: np.ndarray
        wp: np.ndarray
        Kc: np.ndarray
        wc: np.ndarray
        bond: float
        fwd: float
        kvar: float
        F: float
        K0: float
        extra: dict = field(default_factory=dict)

        def payoff(self, S):
            S = np.atleast_1d(np.asarray(S, float))
            puts = np.maximum(self.Kp[None, :] - S[:, None], 0.0) @ self.wp
            calls = np.maximum(S[:, None] - self.Kc[None, :], 0.0) @ self.wc
            return self.bond + self.fwd * (S - self.F) + puts + calls


    def offset_exact(F, K0, T):
        """(2/T)[(F/K0 - 1) - ln(F/K0)]: the weighted intrinsic value of the in-the-money calls on [K0, F]."""
        x = F / K0 - 1.0
        return 2.0 / T * (x - np.log1p(x))


    def offset_vix(F, K0, T):
        """CBOE's term (1/T)(F/K0 - 1)^2."""
        return (F / K0 - 1.0) ** 2 / T


    def log_payoff(S, F, T):
        """The target L(S) = -(2/T) ln(S/F)."""
        return -2.0 / T * np.log(np.asarray(S, float) / F)


    def f_ddkz(S, K0, T):
        """DDKZ's shifted log payoff f(S) = (2/T)[(S - K0)/K0 - ln(S/K0)]: zero value and slope at K0."""
        S = np.asarray(S, float)
        return 2.0 / T * ((S - K0) / K0 - np.log(S / K0))


    def make_portfolio(name, q, wp, wc, offset, **extra):
        """Wrap option weights into a portfolio priced off the quotes q; `offset` is subtracted as cash."""
        price = float(wp @ q.Pp + wc @ q.Cc) - offset
        return Portfolio(name, q.Kp, wp, q.Kc, wc, bond=-offset, fwd=-2.0 / (q.T * q.K0), kvar=price,
                         F=q.F, K0=q.K0, extra=extra)

    return (
        Portfolio,
        f_ddkz,
        log_payoff,
        make_portfolio,
        offset_exact,
        offset_vix,
    )


@app.cell(hide_code=True)
def derman_md(mo):
    mo.md(r"""
    ## 4 · Method A: Derman's chords (DDKZ 1999, Appendix A)

    **Idea.** Options can only make *straight lines with kinks at the strikes*. So replace the
    curved payoff by the closest such shape that touches it at every strike: the **chords** joining
    the points $\big(K_i,f(K_i)\big)$. DDKZ work with the log payoff shifted by a straight line,

    $$
    f(S)=\frac2T\left[\frac{S-K_0}{K_0}-\ln\frac{S}{K_0}\right],
    $$

    which is $L(S)$ plus a straight line: $f(S)=L(S)+\frac{2}{TK_0}(S-F)+\text{offset}$ (expand both
    logarithms to check). A straight line is just a forward and a bond, so $f$ needs exactly the same
    options as $L$. It has $f(K_0)=0$ and $f'(K_0)=0$: it sits on the axis at $K_0$ and curves up on
    both sides.

    **Deriving the weights (calls; puts are the mirror image).** Walk up from $K_0$ through the
    strikes $K_0<K_1<K_2<\dots$

    * Between $K_0$ and $K_1$ the payoff must rise along the first chord, with slope
      $s_0=\frac{f(K_1)-f(K_0)}{K_1-K_0}$. A call struck at $K_0$ has slope 1 above $K_0$, so buy
      $w_c(K_0)=s_0$ calls.
    * Between $K_1$ and $K_2$ the slope must become $s_1=\frac{f(K_2)-f(K_1)}{K_2-K_1}$. The calls
      already held contribute slope $w_c(K_0)$, so add $w_c(K_1)=s_1-w_c(K_0)$ calls at $K_1$.
    * In general (DDKZ eq. A7):
      $\;w_c(K_n)=s_n-\sum_{i<n}w_c(K_i)$, i.e. **each weight is the change in chord slope at that
      strike.**
    * The outermost strike has no chord beyond it, so its weight is $0$.

    Puts work the same way walking down from $K_0$, using the slopes' absolute values. (DDKZ's eq. A8
    has a typo: its sum should run over the *put* weights $w_p$.)

    **Why the weights look like $\frac{2}{T}\frac{\Delta K}{K^2}$.** The slope of a chord is roughly
    the slope of $f$ at the middle of its gap. The weight at $K_n$, the change between the chord to its
    left and the chord to its right, is therefore about curvature × the distance between those two
    midpoints: $f''(K_n)\,\Delta K_n=\frac{2\Delta K_n}{TK_n^2}$, with
    $\Delta K_n=\frac{K_{n+1}-K_{n-1}}{2}$. That is the continuous weight $\frac{2}{TK^2}dK$ of section 1
    with $dK$ replaced by the local strike spacing. At the outermost strike the rule breaks down:
    Derman's weight there is $0$.

    **Why it over-prices, a little.** $f$ is convex, and a convex curve lies *below* each of its
    chords. So the chord portfolio pays at least $f(S)$ for every $S$ between the first and the last
    strike: equal at the strikes, more in between. Outside the strike range the chords continue as
    straight lines while $f$ keeps curving up, so there the portfolio pays *less*: that is truncation.
    Derman's price is therefore an upper estimate **of the in-range part** of the log contract, but not
    of the whole thing. Section 5 sizes the in-range excess.
    """)
    return


@app.cell(hide_code=True)
def derman_loop(f_ddkz, np, offset_exact):
    def derman_loop(q):
        """Derman's weights written exactly as derived: walk outwards from K0, weight = chord slope - slope already held."""
        def f(S):
            return f_ddkz(S, q.K0, q.T)

        # calls: K0, K1, K2, ... upwards
        Kc = list(q.Kc)
        wc, slope_held = [], 0.0
        for n in range(len(Kc) - 1):
            chord_slope = (f(Kc[n + 1]) - f(Kc[n])) / (Kc[n + 1] - Kc[n])
            w = chord_slope - slope_held
            wc.append(w)
            slope_held += w
        wc.append(0.0)                       # outermost call: nothing beyond it to match

        # puts: K0, K-1, K-2, ... downwards (slope measured as the rise per point moving down)
        Kp = list(q.Kp)[::-1]
        wp, slope_held = [], 0.0
        for n in range(len(Kp) - 1):
            chord_slope = (f(Kp[n + 1]) - f(Kp[n])) / (Kp[n] - Kp[n + 1])
            w = chord_slope - slope_held
            wp.append(w)
            slope_held += w
        wp.append(0.0)                       # outermost put
        wp = wp[::-1]                        # back to ascending strike order

        price = sum(w * P for w, P in zip(wp, q.Pp)) + sum(w * C for w, C in zip(wc, q.Cc))
        return np.array(wp), np.array(wc), price - offset_exact(q.F, q.K0, q.T)

    return (derman_loop,)


@app.cell(hide_code=True)
def derman(
    demo,
    derman_loop,
    f_ddkz,
    make_portfolio,
    mo,
    np,
    offset_exact,
    vol_pts,
):
    def chord_weights(nodes, K0, T):
        """nodes run outwards from K0. Weight = change in |chord slope|; the outermost node gets 0."""
        slopes = np.abs(np.diff(f_ddkz(nodes, K0, T)) / np.diff(nodes))
        w = np.zeros(nodes.size)
        w[:-1] = np.diff(np.r_[0.0, slopes])
        return w


    def derman(q):
        wc = chord_weights(q.Kc, q.K0, q.T)
        wp = chord_weights(q.Kp[::-1], q.K0, q.T)[::-1]
        return make_portfolio("Derman", q, wp, wc, offset_exact(q.F, q.K0, q.T))


    # the loop and the numpy version must agree
    _wp, _wc, _price = derman_loop(demo)
    _d = derman(demo)
    assert np.allclose(_wp, _d.wp) and np.allclose(_wc, _d.wc) and np.isclose(_price, _d.kvar)
    mo.md(f"✅ Loop and numpy agree on the 5-pt synthetic 1-week chain: Derman fair variance "
          f"**{vol_pts(_d.kvar):.4f}** vol pts, vs in-range exact {vol_pts(demo.truth_in_range):.4f}.")
    return (derman,)


@app.cell(hide_code=True)
def corr_md(mo):
    mo.md(r"""
    ## 5 · Method A2: Derman + the expected convexity correction

    Derman's portfolio pays *more* than $f$ between strikes, so its price is too high. The logic in one
    line: the portfolio's price is $\mathbb E[\text{chord}(S_T)]$, the target is $\mathbb E[f(S_T)]$, so

    $$
    \mathbb E[f(S_T)]=\underbrace{\mathbb E[\text{chord}(S_T)]}_{\text{Derman's price}}-\underbrace{\mathbb E\big[\text{chord}(S_T)-f(S_T)\big]}_{\text{expected over-payment}} .
    $$

    Method A2 estimates the expected over-payment from the same option quotes and subtracts it. Here is
    the derivation, step by step.

    **(a) How far a chord sits above a convex curve.** On one gap $[a,b]$ of width $h=b-a$, the
    straight line through $\big(a,f(a)\big)$ and $\big(b,f(b)\big)$ exceeds $f$ by

    $$
    \text{chord}(S)-f(S)=\tfrac12 f''(\xi)\,(S-a)(b-S)\qquad\text{for some }\xi\in[a,b].
    $$

    This is the standard linear-interpolation error. For a parabola it holds exactly with constant
    $f''$: the difference of a line and a parabola is a parabola that vanishes at $a$ and $b$, so it must be
    $\text{const}\times(S-a)(b-S)$, and comparing second derivatives gives the constant $\tfrac12f''$.
    Over a small gap $f''$ hardly changes, so use $f''(\bar K)=\frac{2}{T\bar K^2}$ at the mid-gap $\bar K$:
    the error is a little arch, zero at the strikes and largest in the middle, where it equals
    $\frac{h^2}{4T\bar K^2}$.

    **(b) The average error over the gap.** If $S_T$ lands in the gap, it is roughly equally likely
    to be anywhere in it (the gap is much narrower than the distribution). "Equally likely anywhere in
    $[a,b]$" means a uniform density $\frac1h$ on the gap, so the expected value of $(S-a)(b-S)$ given
    that $S_T$ is in the gap is its plain average over $[a,b]$:

    $$
    \text{average}=\frac1h\int_a^b(S-a)(b-S)\,dS .
    $$

    To compute it, measure distance from the left strike: $u=S-a$, so $b-S=h-u$, $dS=du$, and $u$
    runs from $0$ to $h$:

    $$
    \int_a^b(S-a)(b-S)\,dS=\int_0^h u\,(h-u)\,du=\int_0^h\big(hu-u^2\big)\,du
    =\Big[\tfrac{h u^2}{2}-\tfrac{u^3}{3}\Big]_0^h=\frac{h^3}{2}-\frac{h^3}{3}=\frac{h^3}{6}.
    $$

    Dividing by the width $h$ gives $\frac1h\cdot\frac{h^3}6=\frac{h^2}6$.

    *Sanity check:* the arch peaks at the midpoint, $u=\frac h2$, at $\frac h2\cdot\frac h2=\frac{h^2}4$,
    and the average is $\frac{h^2/6}{h^2/4}=\frac23$ of that peak. This is Archimedes' rule: a
    parabolic arch fills $\frac23$ of the rectangle drawn around it. Combining with (a):

    $$
    \text{mean error on gap }i\;\approx\;\tfrac12\cdot\frac{2}{T\bar K_i^2}\cdot\frac{h_i^2}{6}=\frac{h_i^2}{6T\bar K_i^2}.
    $$

    The code uses the *exact* average of $\text{chord}-f$ (we can integrate $f$ in closed form), so
    nothing is lost to the Taylor step.

    **(c) Weight each gap by the chance of landing in it.** With $p_i=\text{Prob}(K_i<S_T<K_{i+1})$:

    $$
    \text{correction}=\sum_i p_i\cdot\text{mean error}_i\;\approx\;\sum_i p_i\,\frac{h_i^2}{6T\bar K_i^2}
    \;\xrightarrow{\;\text{equal gaps }h\;}\;\frac{h^2}{6TF^2}.
    $$

    The last step uses $\bar K_i\approx F$ for the gaps that matter, since most of the probability sits
    near the forward, and $\sum_i p_i\approx1$.

    **(d) The probabilities come from the quotes.** Differentiate $P(K)=\mathbb E[(K-S_T)^+]$ in
    $K$: $P'(K)=\text{Prob}(S_T<K)$. A put spread per point of width is a digital option. So the slope of
    the quoted put prices across a gap is the probability of finishing below the middle of that gap.
    On the call side $1+C'(K)$ gives the same thing. Interpolating these mid-gap probabilities to the
    strikes gives the cumulative distribution at every strike, and $p_i$ is the difference between
    neighbours. No model is needed.

    **(e) What changes in the portfolio: nothing.** The correction is a *number*, not a position.
    A2 holds exactly Derman's options and just quotes a lower price (a smaller bond leg). The cost:
    Derman's price over-states the in-range part, while A2's price is a best estimate that can land on
    either side. A2 relies on $S_T$ being spread roughly evenly within each gap. Could the correction
    be built into the positions instead? Appendix A tries it.

    *Size, for intuition:* 5-point SPX strikes, $F\approx7{,}800$, one week:
    $\frac{h^2}{6TF^2}=\frac{25}{6\cdot0.0192\cdot7800^2}\approx3.6\times10^{-6}$ in variance, about
    $0.002$ vol points at an 11% vol. The correction is small on a clean 5-point grid. It grows with
    $h^2$, so it matters where the spacing is wide. It also grows with $1/T$; part of that is just
    annualisation, since the same price error is divided by a smaller $T$.
    """)
    return


@app.cell
def corr_loop(f_ddkz, np):
    def convexity_correction_loop(q, n_points=2001):
        """The correction exactly as derived: per gap, (probability of the gap) x (average of chord - f)."""
        K, N = q.K, len(q.K)
        f = lambda S: f_ddkz(S, q.K0, q.T)

        # (d) digital = price slope across each gap: P' on the put side, 1 + C' on the call side
        mid, cdf_mid = [], []
        for i in range(N - 1):
            a, b = K[i], K[i + 1]
            if b <= q.K0:
                slope = (q.P[i + 1] - q.P[i]) / (b - a)
                cdf_mid.append(slope)
            else:
                slope = (q.C[i + 1] - q.C[i]) / (b - a)
                cdf_mid.append(1.0 + slope)
            mid.append(0.5 * (a + b))

        # CDF at each strike: straight-line interpolation between the neighbouring mid-gap values
        cdf = []
        for i in range(N):
            j = min(max(i, 1), N - 2)        # use mid-gaps j-1 and j (extrapolate at the two ends)
            t = (K[i] - mid[j - 1]) / (mid[j] - mid[j - 1])
            cdf.append(cdf_mid[j - 1] + t * (cdf_mid[j] - cdf_mid[j - 1]))
        cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))

        total = 0.0
        for i in range(N - 1):
            a, b = K[i], K[i + 1]
            p_i = cdf[i + 1] - cdf[i]
            S = a + (np.arange(n_points) + 0.5) * (b - a) / n_points   # midpoints of n_points equal slices
            chord = f(a) + (f(b) - f(a)) * (S - a) / (b - a)
            mean_error = np.mean(chord - f(S))   # (a)+(b) by brute force: average over the gap
            total += p_i * mean_error
        return total

    return (convexity_correction_loop,)


@app.cell
def corr(Portfolio, convexity_correction_loop, demo, derman, f_ddkz, mo, np):
    def chord_mean_error(a, b, K0, T):
        """Exact average of chord - f over [a, b], using the antiderivative of f. Close to h^2/(6 T Kbar^2)."""
        def f_integral(x):
            return 2.0 / T * (x**2 / (2 * K0) - x - (x * np.log(x / K0) - x))
        return 0.5 * (f_ddkz(a, K0, T) + f_ddkz(b, K0, T)) - (f_integral(b) - f_integral(a)) / (b - a)


    def quote_cdf(q):
        """Market-implied probability Prob(S_T < K) at every strike, from price slopes (puts below K0, calls from K0 up)."""
        K, h = q.K, np.diff(q.K)
        D = np.where(K[1:] <= q.K0, np.diff(q.P) / h, 1.0 + np.diff(q.C) / h)      # at mid-gaps
        m = 0.5 * (K[1:] + K[:-1])
        j = np.clip(np.arange(K.size), 1, K.size - 2)
        cdf = D[j - 1] + (K - m[j - 1]) / (m[j] - m[j - 1]) * (D[j] - D[j - 1])
        return np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))


    def derman_corrected(q):
        base = derman(q)
        p = np.diff(quote_cdf(q))                                         # probability of each gap
        corr = float(np.sum(p * chord_mean_error(q.K[:-1], q.K[1:], q.K0, q.T)))
        return Portfolio("Derman + correction", base.Kp, base.wp, base.Kc, base.wc, base.bond - corr, base.fwd,
                         base.kvar - corr, q.F, q.K0, extra={"correction": corr})


    _c_loop = convexity_correction_loop(demo)
    _c = derman_corrected(demo).extra["correction"]
    assert np.isclose(_c_loop, _c, rtol=1e-5), (_c_loop, _c)
    _h = 5.0
    mo.md(f"✅ Loop and numpy agree: correction = **{_c:.3e}** in variance "
          f"(rule of thumb $h^2/(6TF^2)$ = {_h**2 / (6 * demo.T * demo.F**2):.3e}).")
    return chord_mean_error, derman_corrected, quote_cdf


@app.cell(hide_code=True)
def vix_md(mo):
    mo.md(r"""
    ## 6 · Method B: the VIX / CBOE formula

    The CBOE computes the VIX from

    $$
    \sigma^2_{\text{VIX}}=\frac2T\sum_i\frac{\Delta K_i}{K_i^2}\,Q(K_i)\;-\;\frac1T\Big(\frac F{K_0}-1\Big)^2,
    \qquad \Delta K_i=\frac{K_{i+1}-K_{i-1}}2 ,
    $$

    with $Q(K_i)$ the put below $K_0$, the call above, and $\tfrac12(P+C)$ at $K_0$. At the two
    outermost strikes $\Delta K$ is simply the distance to the single neighbour. (We use undiscounted
    prices, so the CBOE's $e^{rT}$ factor is already in $Q$.) Each piece follows from the boxed
    formula of section 1.

    **(a) The sum is a midpoint rule.** Give each strike the "cell" of strikes nearer to it than to
    its neighbours: from halfway to the previous strike to halfway to the next one. The cell has
    width $\Delta K_i$. Approximate the integrand $Q(K)/K^2$ on that cell by its value at the strike:

    $$
    \int_{\text{cell }i}\frac{Q(K)}{K^2}\,dK\;\approx\;\frac{Q(K_i)}{K_i^2}\,\Delta K_i .
    $$

    The option weights are therefore $\frac2T\frac{\Delta K_i}{K_i^2}$. Inside the chain they are very
    close to Derman's slope changes, which are also about $\frac{2\Delta K_i}{TK_i^2}$.

    *Worked example.* One week ($\frac2T=\frac{2}{7/365}=104.3$), a put at $7{,}790$ whose neighbours are
    $7{,}785$ and $7{,}800$, so $\Delta K=\frac{7800-7785}{2}=7.5$. Its weight is
    $104.3\times\frac{7.5}{7790^2}=1.29\times10^{-5}$ per unit of variance. If the put costs 10 index
    points, it contributes $1.29\times10^{-4}$ to the variance, i.e. 1.29 variance points.

    **(b) Why $\tfrac12(P+C)$ at $K_0$.** The cell around $K_0$ straddles the split point of the
    formula. Its left half belongs to the put integral and its right half to the call integral, so
    the honest midpoint value is half of each: $\tfrac{\Delta K_0}{K_0^2}\cdot\tfrac12\big(P(K_0)+C(K_0)\big)$.
    As a portfolio, that is half a put and half a call at $K_0$. This is only the quadrature of the
    integrals. The integral being approximated, $\int_{K_0}^\infty C/K^2$, contains the intrinsic value of
    the in-the-money calls on $[K_0,F]$ (section 1), and that is what the second term removes.

    **(c) Where $\frac1T(F/K_0-1)^2$ comes from.** Section 1 gave the exact amount to subtract,
    which is the weighted intrinsic value of the in-the-money calls on $[K_0,F]$. Write
    $x=\frac{F}{K_0}-1$ (tiny: $F$ is less than one strike gap above $K_0$) and expand the logarithm,
    $\ln(1+x)=x-\frac{x^2}2+\frac{x^3}3-\dots$:

    $$
    \frac2T\Big[x-\ln(1+x)\Big]=\frac2T\Big[\frac{x^2}2-\frac{x^3}3+\frac{x^4}4-\dots\Big]
    =\underbrace{\frac{x^2}{T}}_{\text{CBOE}}\;-\;\frac{2x^3}{3T}+\frac{x^4}{2T}-\dots
    $$

    **The CBOE term is the first term of this series.** Sanity check, using the intrinsic-value
    form: $\frac2T\int_{K_0}^F\frac{F-K}{K^2}dK\approx\frac2T\cdot\frac{1}{K_0^2}\cdot\frac{(F-K_0)^2}2=\frac{x^2}T$,
    the area of a triangle of height and width $F-K_0$, weighted by $\frac{2}{TK_0^2}$.

    **(d) How big is the neglected part?** The CBOE subtracts slightly *too much*, so its variance is
    low by

    $$
    \frac{x^2}T-\frac2T\big[x-\ln(1+x)\big]\;=\;\frac{2x^3}{3T}-\frac{x^4}{2T}+\dots\;\approx\;\frac{2x^3}{3T}.
    $$

    Since $F$ lies between $K_0$ and the next strike, $x<h/K_0$. For SPX with 5-point strikes,
    $x<0.00064$. One week ($T=7/365$) gives at most $\frac{2(0.00064)^3}{3\cdot0.0192}\approx 9\times10^{-9}$
    in variance, i.e. about $4\times10^{-6}$ vol points. Even at 2 days, or 0DTE, it stays far below
    anything that matters. The *offset itself*, $x^2/T$, is at most about $2\times10^{-5}$ in variance
    (≈ 0.01 vol pt) at one week. That is small but not invisible, and the CBOE does capture it. The cubic
    remainder reaches 1 vol bp only when strike gaps approach 1% of the forward, as for many single
    stocks and low-priced underlyings. Section 10 gives the exact thresholds.
    """)
    return


@app.function
def vix_loop(q):
    """The CBOE recipe, strike by strike."""
    K, N = q.K, len(q.K)
    total = 0.0
    for i in range(N):
        if i == 0:
            dK = K[1] - K[0]                       # outermost strikes: distance to the one neighbour
        elif i == N - 1:
            dK = K[-1] - K[-2]
        else:
            dK = (K[i + 1] - K[i - 1]) / 2.0
        if K[i] < q.K0:
            Q = q.P[i]
        elif K[i] > q.K0:
            Q = q.C[i]
        else:
            Q = 0.5 * (q.P[i] + q.C[i])            # the K0 cell straddles puts and calls
        total += dK / K[i] ** 2 * Q
    x = q.F / q.K0 - 1.0
    return 2.0 / q.T * total - x**2 / q.T


@app.cell
def vix(demo, make_portfolio, mo, np, offset_vix, vol_pts):
    def textbook_weights(q):
        """(2/T) dK/K^2 at every strike, dK = half the distance between the neighbours (one-sided at the ends)."""
        K = q.K
        dK = np.empty(K.size)
        dK[1:-1] = 0.5 * (K[2:] - K[:-2])
        dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
        return 2.0 / q.T * dK / K**2


    def vix_weights(q):
        """Textbook weights split into the put strip (K <= K0) and the call strip (K >= K0); half each at K0."""
        w = textbook_weights(q)
        wp, wc = w[q.K <= q.K0].copy(), w[q.K >= q.K0].copy()
        wp[-1] *= 0.5                                      # half a put at K0 ...
        wc[0] *= 0.5                                       # ... and half a call
        return wp, wc


    def vix(q):
        wp, wc = vix_weights(q)
        return make_portfolio("VIX", q, wp, wc, offset_vix(q.F, q.K0, q.T))


    _v = vix(demo)
    assert np.isclose(vix_loop(demo), _v.kvar)
    mo.md(f"✅ Loop and numpy agree: VIX fair variance **{vol_pts(_v.kvar):.4f}** vol pts on the same chain.")
    return textbook_weights, vix, vix_weights


@app.cell(hide_code=True)
def vix_exact_md(mo):
    mo.md(r"""
    ## 7 · Method B2: VIX with the exact offset

    Same weights, same options, one line different: subtract
    $\frac2T\big[(F/K_0-1)-\ln(F/K_0)\big]$ instead of $\frac1T(F/K_0-1)^2$. The two prices differ by a
    pure cash amount, which section 6(d) sizes as $\approx\frac{2x^3}{3T}$. The check below confirms the
    algebra on the synthetic chain with the forward placed almost a full gap above $K_0$ (the
    worst case).
    """)
    return


@app.cell
def vix_exact(
    make_portfolio,
    mo,
    np,
    offset_exact,
    svi_1w,
    synthetic_quotes,
    vix,
    vix_weights,
    vol_bp,
):
    def vix_exact(q):
        wp, wc = vix_weights(q)
        return make_portfolio("VIX exact offset", q, wp, wc, offset_exact(q.F, q.K0, q.T))


    _worst = synthetic_quotes(svi_1w, 7 / 365, 5.0, 0.99)
    _x = _worst.F / _worst.K0 - 1
    _diff = vix_exact(_worst).kvar - vix(_worst).kvar
    assert np.isclose(_diff, 2 * _x**3 / (3 * _worst.T) - _x**4 / (2 * _worst.T), rtol=1e-3)
    mo.md(f"✅ With $x = F/K_0-1$ = {_x:.2e}: exact − CBOE = **{_diff:.2e}** in variance "
          f"= {vol_bp(vix_exact(_worst).kvar, vix(_worst).kvar):.5f} vol bp; "
          f"$2x^3/(3T)$ = {2 * _x**3 / (3 * _worst.T):.2e}.")
    return (vix_exact,)


@app.cell(hide_code=True)
def lp_md(mo):
    mo.vstack([mo.md(r"""
    ## 8 · Model-free bounds by linear programming

    **The question in plain words.** Using only the listed options, a bond and a forward:

    * **Upper bound:** what is the *cheapest* portfolio whose payoff is **at least** $L(S)$ for every
      possible final price $S$? Anyone selling the variance swap above that price could be arbitraged.
      You would sell the swap, buy this portfolio, and never lose.
    * **Lower bound:** what is the *most expensive* portfolio whose payoff is **at most** $L(S)$
      everywhere?

    The two prices bound the log contract's price, i.e. the fair variance strike. Every arbitrage-free
    model that matches the quotes, *and in which the index ends inside the assumed range
    $[S_{lo},S_{hi}]$*, must price it between the two. Each answer is an actual portfolio too, so we
    also get its weights.

    **From words to an LP.** The unknowns are the holdings, collected in a vector
    $x=(c,\;a,\;w^p_1,\dots,w^p_n,\;w^c_1,\dots,w^c_m)$ (bond, forward, puts at $K\le K_0$, calls at
    $K>K_0$; the call at $K_0$ is left out because, by parity, it equals put + forward + bond). We
    cannot check "every $S$", so we check a long list of test prices $S_1,\dots,S_M$:

    $$
    \text{upper}=\min_x\;\pi^\top x\quad\text{s.t.}\quad A\,x\ \ge\ L(S_j)\;\;\text{for all } j,
    \qquad
    \text{lower}=\max_x\;\pi^\top x\quad\text{s.t.}\quad A\,x\ \le\ L(S_j)\;\;\text{for all } j .
    $$

    **From the LP to numpy, symbol by symbol:**

    | LP | numpy | meaning |
    |---|---|---|
    | $x$ | `res.x` (what `linprog` returns) | holdings: bond, forward, puts, calls |
    | $\pi$ | `price = np.r_[1, 0, P, C]` | today's (forward) price of each instrument; the forward costs 0 |
    | $S_j$ | `S` (1-D array, length $M$) | test prices |
    | row $j$ of $A$ | `[1, S_j-F, max(K-S_j,0) for each put, max(S_j-K,0) for each call]` | payoff of each instrument if $S_T=S_j$ |
    | $A$ | `np.column_stack([ones, S-F, np.maximum(Kp[None,:]-S[:,None],0), np.maximum(S[:,None]-Kc[None,:],0)])` | all rows at once, by broadcasting |
    | $L(S_j)$ | `target = -2/T*np.log(S/F)` | what we must beat / stay under |
    | "$\ge$" | `A_ub=-A, b_ub=-target` | `linprog` only accepts $\le$, so multiply both sides by $-1$ |
    | "max" | minimise `-price`, then flip the sign | `linprog` only minimises |
    | short positions allowed | `bounds=[(None, None)] * n` | default bounds would force $x\ge0$ |

    The broadcasting line is the only non-obvious step. `Kp[None,:]` is a row of strikes and
    `S[:,None]` a column of test prices, so `Kp[None,:] - S[:,None]` is the $M\times n$ table of
    $K_i-S_j$, and `np.maximum(…, 0)` turns it into put payoffs. The small example below builds $A$ both
    ways.

    **Two practical points.**

    * *Which test prices?* For the upper bound the strikes plus the two ends are enough: between
      two kinks the portfolio is a straight line, and a straight line that is above a convex curve at
      both ends is above it in between. The lower bound needs a dense grid, because a convex curve can
      dip below a line between two points. We use the strikes plus about 2,000 log-spaced points.
    * *A support range $[S_{lo},S_{hi}]$ is required.* $-\ln S\to\infty$ as $S\to0$, and no finite set
      of options can stay above that. So we assume the index ends between $S_{lo}=0.5F$ and
      $S_{hi}=1.5F$, which is generous for one week. The choice matters in both directions:
      * a wider range makes the upper bound larger, without limit as $S_{lo}\to0$;
      * a range that is too narrow contradicts the quotes. Far out-of-the-money options have positive
        prices, so they imply some probability beyond $S_{lo}$ and $S_{hi}$. `linprog` then reports
        "unbounded", which is an arbitrage relative to the assumption, and the notebook stops with an
        error.

      On the 1-week chains, $[0.75F, 1.1F]$ already fails.

    **What the optimiser finds** (you will see it in the payoff and weights views):

    * **Upper portfolio = Derman's chords**, plus two large extra options at the extreme strikes. These
      act as tail insurance: they bend the payoff up so it stays above $L$ out to $S_{lo}$ and $S_{hi}$.
    * **Lower portfolio = tangent lines** that touch $L$ inside each gap and meet their neighbours at the
      strikes. Near the ends it takes large positions in the outermost options, so that beyond the last
      strikes it follows $L$ much more closely than Derman's portfolio does, while staying below
      it. Because neighbouring tangents must meet exactly at
      a strike, the touch points alternate left and right, and the weights zig-zag. On an uneven chain
      the zig-zag becomes large long/short pairs.
    """),
    mo.accordion({"Why is this still an LP when the target is a logarithm? (click to open)": mo.md(r"""
    "Linear" in *linear programming* refers to the **unknowns**, not to $S$. The unknowns are the
    holdings $x$. The log curve $L(S)$ and the hockey-stick option payoffs only ever enter as **numbers**,
    evaluated at fixed test prices. Written out, the upper-bound constraint at test price $S_j$ is

    $$
    c\cdot1\;+\;a\,(S_j-F)\;+\;\sum_i w^p_i\,(K_i-S_j)^+\;+\;\sum_k w^c_k\,(S_j-K_k)^+\;\ge\;-\frac2T\ln\frac{S_j}F .
    $$

    * The coefficients $1$, $S_j-F$, $(K_i-S_j)^+$, $(S_j-K_k)^+$ are fixed numbers once $S_j$ is chosen.
      They fill row $j$ of $A$.
    * The right-hand side $L(S_j)$ is also a fixed number. It becomes entry $j$ of `target`. The
      logarithm is "baked into" this constant.
    * So each row says $A_j\,x\ge L(S_j)$, which is linear in $x$, and the cost $\pi^\top x$ is linear in $x$ too.

    Linear objective plus linear inequalities is exactly an LP, however nonlinear the payoffs and the
    target are in $S$. You can see this in the code: `A` and `target` are built with `np.maximum` and
    `np.log` *before* `linprog` is called, and the solver only ever sees a matrix and a vector of plain
    numbers.

    **What we give up.** The true requirement is "portfolio $\ge L$ for *every* $S$ in
    $[S_{lo},S_{hi}]$": infinitely many constraints, a so-called *semi-infinite* LP. Replacing them by a
    finite list of test prices is the only approximation:

    * **Upper bound: nothing.** The test prices include every strike and both ends of the support.
      Between neighbouring kinks the portfolio is a straight line and $L$ is convex, so a line that is
      above $L$ at both ends is above it in between. The grid enforces the constraint everywhere.
    * **Lower bound: a tiny violation is possible.** There the portfolio must stay *below* a convex
      curve, and a line can poke slightly above $L$ between two test prices. With about 2,000
      log-spaced points (spacing about 0.05% of $F$, or 4 index points), any such violation, and its effect on the
      bound, is negligible.
    """)})])
    return


@app.cell
def lp_toy(log_payoff, mo, np, pd):
    # A tiny example: 3 puts, 2 calls, 7 test prices. Build the payoff matrix A row by row ...
    toy_F, toy_T = 7821.0, 7 / 365
    toy_Kp = np.array([7700.0, 7750.0, 7800.0])            # puts at and below K0 = 7800
    toy_Kc = np.array([7850.0, 7900.0])                    # calls above K0
    toy_S = np.array([7600.0, 7700.0, 7750.0, 7800.0, 7850.0, 7900.0, 8000.0])

    A_loop = []
    for S_j in toy_S:
        row = [1.0, S_j - toy_F]                           # bond pays 1, forward pays S - F
        row += [max(K - S_j, 0.0) for K in toy_Kp]         # each put pays (K - S)^+
        row += [max(S_j - K, 0.0) for K in toy_Kc]         # each call pays (S - K)^+
        A_loop.append(row)
    A_loop = np.array(A_loop)

    # ... and with broadcasting, as used in lp_bounds below
    A_numpy = np.column_stack([np.ones_like(toy_S), toy_S - toy_F,
                               np.maximum(toy_Kp[None, :] - toy_S[:, None], 0.0),
                               np.maximum(toy_S[:, None] - toy_Kc[None, :], 0.0)])
    assert np.array_equal(A_loop, A_numpy)

    mo.vstack([
        mo.md("✅ Both constructions give the same $A$. Rows = test prices $S_j$, columns = instruments; "
              "each entry is that instrument's payoff if $S_T = S_j$. The last column is the target $L(S_j)$."),
        mo.ui.table(pd.DataFrame(np.column_stack([A_numpy, log_payoff(toy_S, toy_F, toy_T)]),
                                 index=[f"S = {s:,.0f}" for s in toy_S],
                                 columns=["bond", "forward"] + [f"put {k:,.0f}" for k in toy_Kp]
                                         + [f"call {k:,.0f}" for k in toy_Kc] + ["target L(S)"]).round(3),
                    selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def lp(Portfolio, demo, derman, linprog, log_payoff, mo, np, vol_pts):
    LP_SUPPORT = (0.5, 1.5)    # the index is assumed to end between 50% and 150% of the forward


    def lp_bounds(q, S_lo, S_hi, n_grid=2000):
        """Cheapest super-replicating and dearest sub-replicating portfolio of the log payoff on [S_lo, S_hi].
        Returns (upper, lower) Portfolios."""
        Kp, Kc = q.Kp, q.Kc[1:]                         # calls strictly above K0 (call at K0 = put + fwd + bond)
        P, C = q.Pp, q.Cc[1:]

        # test prices: all strikes, the two support ends, and a dense log-spaced grid in between
        S = np.unique(np.r_[np.geomspace(S_lo, S_hi, n_grid), q.K, S_lo, S_hi])
        S = S[(S >= S_lo) & (S <= S_hi)]
        target = log_payoff(S, q.F, q.T)

        # payoff matrix: one row per test price, one column per instrument
        A = np.column_stack([np.ones_like(S), S - q.F,
                             np.maximum(Kp[None, :] - S[:, None], 0.0),
                             np.maximum(S[:, None] - Kc[None, :], 0.0)])
        price = np.r_[1.0, 0.0, P, C]
        free = [(None, None)] * A.shape[1]

        up = linprog(price, A_ub=-A, b_ub=-target, bounds=free, method="highs")      # min cost, A x >= L
        lo = linprog(-price, A_ub=A, b_ub=target, bounds=free, method="highs")       # max cost, A x <= L

        def portfolio(res, name):
            # "unbounded" means the quotes imply probability outside [S_lo, S_hi]: widen the support
            assert res.status == 0, f"{name} LP failed on {q.name}: {res.message}"
            x, n = res.x, Kp.size
            return Portfolio(name, Kp, x[2:2 + n], Kc, x[2 + n:], bond=x[0], fwd=x[1], kvar=float(price @ x),
                             F=q.F, K0=q.K0, extra={"S_lo": S_lo, "S_hi": S_hi})

        return portfolio(up, "LP upper"), portfolio(lo, "LP lower")


    _up, _lo = lp_bounds(demo, LP_SUPPORT[0] * demo.F, LP_SUPPORT[1] * demo.F)
    assert _lo.kvar <= demo.truth <= _up.kvar                                         # brackets the full exact value
    _d = derman(demo)
    assert np.allclose(_up.wp[1:-1], _d.wp[1:-1], rtol=5e-3)        # interior puts = Derman's chords (to solver tolerance)
    assert np.allclose(_up.wc[:-1], _d.wc[1:-1], rtol=5e-3)         # interior calls
    assert np.isclose(_up.wp[-1], _d.wp[-1] + _d.wc[0], rtol=5e-3)  # at K0: LP's put = Derman's put + call (parity)
    mo.md(f"✅ On the 5-pt synthetic chain with support [0.5F, 1.5F] the bounds are "
          f"[{vol_pts(_lo.kvar):.4f}, {vol_pts(_up.kvar):.4f}] vol pts and contain the exact "
          f"{vol_pts(demo.truth):.4f}. Inside the strike range the upper portfolio holds exactly Derman's options. "
          f"At $K_0$ it holds one put in place of Derman's put + call, which is the same thing by parity.")
    return LP_SUPPORT, lp_bounds


@app.cell(hide_code=True)
def controls_md(mo):
    mo.md(r"""
    ## 9 · Results

    Pick a market below. All three result sections (payoff, fair variance, weights) follow the
    selection.

    * **Synthetic** markets have a known exact answer. You can change the strike spacing and slide
      the forward through a strike gap.
    * **SPX** markets are the real Yahoo chains from section 2.
    * The LP bounds always assume the index ends between $0.5F$ and $1.5F$ (section 8).
    """)
    return


@app.cell(hide_code=True)
def controls(mo, spx_1w, spx_2d):
    DATASETS = {
        "Synthetic · 1 week (exact answer known)": ("synthetic", 7 / 365),
        "Synthetic · 2 days (exact answer known)": ("synthetic", 2 / 365),
        "SPX · 1 week (16-Oct)": ("spx", spx_1w),
        "SPX · 2 trading days (13-Oct)": ("spx", spx_2d),
    }
    dataset_ui = mo.ui.dropdown(list(DATASETS), value="Synthetic · 1 week (exact answer known)", label="Market")
    step_ui = mo.ui.dropdown(["5", "10", "25", "50", "100"], value="5", label="Synthetic strike step (pts)")
    gap_ui = mo.ui.slider(0.0, 0.95, step=0.05, value=0.5, show_value=True, label="Synthetic: F position in the gap above K₀")
    mo.hstack([dataset_ui, step_ui, gap_ui], justify="start", gap=2, wrap=True)
    return DATASETS, dataset_ui, gap_ui, step_ui


@app.cell(hide_code=True)
def markets(
    DATASETS,
    LP_SUPPORT,
    derman,
    derman_corrected,
    lp_bounds,
    svi_1w,
    synthetic_quotes,
    vix,
    vix_exact,
):
    METHOD_NAMES = ["Derman", "Derman + correction", "VIX", "VIX exact offset"]


    def build_quotes(label, step=5.0, gap_pos=0.5):
        kind, arg = DATASETS[label]
        return synthetic_quotes(svi_1w, arg, step, gap_pos) if kind == "synthetic" else arg


    def all_portfolios(q):
        """The four methods + both LP bounds, in a fixed order."""
        ports = {p.name: p for p in (derman(q), derman_corrected(q), vix(q), vix_exact(q))}
        ports["LP upper"], ports["LP lower"] = lp_bounds(q, LP_SUPPORT[0] * q.F, LP_SUPPORT[1] * q.F)
        return ports


    # the four markets at their default settings (5-pt synthetic strikes, F mid-gap), computed once
    MARKETS = {label: (q, all_portfolios(q)) for label, q in ((lbl, build_quotes(lbl)) for lbl in DATASETS)}
    return MARKETS, METHOD_NAMES, all_portfolios, build_quotes


@app.cell(hide_code=True)
def scenario(all_portfolios, build_quotes, dataset_ui, gap_ui, mo, step_ui):
    q_sel = build_quotes(dataset_ui.value, float(step_ui.value), gap_ui.value)
    ports_sel = all_portfolios(q_sel)
    mo.md(f"**Selected:** {q_sel.name} · $F$ = {q_sel.F:,.2f}, $K_0$ = {q_sel.K0:,.0f}, "
          f"$F/K_0-1$ = {q_sel.F / q_sel.K0 - 1:.2e} · {q_sel.K.size} strikes from {q_sel.K.min():,.0f} to {q_sel.K.max():,.0f}")
    return ports_sel, q_sel


@app.cell(hide_code=True)
def payoff_md(mo):
    mo.md(r"""
    ### 9.1 · The payoff: how well does each portfolio reproduce the log contract?

    Drawn as in DDKZ's Appendix A: the target is $f(S)=\frac2T\big[\frac{S-K_0}{K_0}-\ln\frac S{K_0}\big]$,
    the log payoff shifted by a straight line so that it touches zero at $K_0$. Every portfolio is
    shifted by the *same* line, so the vertical gaps between curves, which are the replication
    errors, are unchanged. The bottom panel shows those errors directly: portfolio payoff minus
    $L(S)$, in variance points (vol%²).

    **What to look for.**

    * **Derman** touches $f$ at every strike and arches above it in between. It lies above $f$
      inside the strike range and below it beyond the last strikes (truncation).
    * **Derman + correction** is the same zig-zag moved down by a constant, so that its *average*
      in-range error, weighted by the probability of each gap, is zero.
    * **VIX** has nearly the same kinks but does not pass through the strike points. Its weights come
      from quadrature, not interpolation. Near $K_0$ and at the outermost strikes it differs most.
    * **VIX exact offset** is the VIX curve moved by $\approx\frac{2x^3}{3T}$. Even at the worst $F$
      position it is invisible on a 5-point grid.
    * **LP upper** is Derman's chords inside the strike range, plus two tail options that keep it
      above $f$ all the way to $S_{lo}$ and $S_{hi}$. **LP lower** is made of tangents and stays below
      $f$ everywhere on the support. The shaded band is the region between the two LP payoffs. The
      exact payoff lies inside it by construction, and the band's width shows how much room the
      listed strikes leave.

    Try a 25- or 50-point synthetic step to make the chords visible, and zoom out to see the tails.
    """)
    return


@app.cell(hide_code=True)
def payoff_ctl(mo):
    zoom_ui = mo.ui.dropdown(["±6 strikes around K₀", "±3 standard deviations", "Whole strike range", "LP support"],
                             value="±6 strikes around K₀", label="Zoom")
    zoom_ui
    return (zoom_ui,)


@app.cell(hide_code=True)
def payoff_chart(
    COLORS,
    DASH,
    INK,
    METHOD_NAMES,
    f_ddkz,
    go,
    log_payoff,
    make_subplots,
    np,
    offset_exact,
    ports_sel,
    q_sel,
    style_fig,
    zoom_ui,
):
    def payoff_range(q, ports, zoom):
        i0 = int(np.searchsorted(q.K, q.K0))
        if zoom.startswith("±6"):
            return q.K[max(i0 - 6, 0)], q.K[min(i0 + 7, q.K.size - 1)]
        if zoom.startswith("±3"):
            sd = q.F * np.sqrt(ports["Derman"].kvar * q.T)
            return q.F - 3 * sd, q.F + 3 * sd
        if zoom.startswith("Whole"):
            pad = 0.03 * (q.K[-1] - q.K[0])
            return q.K[0] - pad, q.K[-1] + pad
        return ports["LP upper"].extra["S_lo"], ports["LP upper"].extra["S_hi"]


    def chart_payoff(q, ports, zoom):
        a, b = payoff_range(q, ports, zoom)
        S = np.unique(np.r_[np.linspace(a, b, 1500), q.K[(q.K >= a) & (q.K <= b)]])
        shift = 2.0 / (q.T * q.K0) * (S - q.F) + offset_exact(q.F, q.K0, q.T)     # L(S) + shift = f(S)
        L = log_payoff(S, q.F, q.T)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38],
                            subplot_titles=("Payoff, shifted to DDKZ's f(S)  (variance points, vol%²)",
                                            "Error: portfolio payoff − L(S)  (variance points)"))
        fig.add_trace(go.Scatter(x=S, y=1e4 * (L + shift), name="Exact f(S)", mode="lines",
                                 line=dict(color=COLORS["Exact"], width=2.5)), 1, 1)
        for nm in ["LP lower", "LP upper"] + METHOD_NAMES:     # LP upper fills down to LP lower: the band
            p, lp = ports[nm], nm.startswith("LP")
            style = dict(mode="lines", legendgroup=nm, line=dict(color=COLORS[nm], width=1.5 if lp else 2, dash=DASH[nm]),
                         fill="tonexty" if nm == "LP upper" else None, fillcolor=INK["band"])
            fig.add_trace(go.Scatter(x=S, y=1e4 * (p.payoff(S) + shift), name=nm, **style), 1, 1)
            fig.add_trace(go.Scatter(x=S, y=1e4 * (p.payoff(S) - L), name=nm, showlegend=False, **style), 2, 1)
        Ks = q.K[(q.K >= a) & (q.K <= b)]
        fig.add_trace(go.Scatter(x=Ks, y=1e4 * f_ddkz(Ks, q.K0, q.T), mode="markers", name="listed strikes",
                                 marker=dict(color=INK["primary"], size=7, symbol="line-ns-open", line=dict(width=2)),
                                 hovertemplate="K=%{x:,.0f}<extra>strike</extra>"), 1, 1)
        fig.add_hline(y=0, line=dict(color=INK["axis"], width=1), row=2, col=1)
        for x in (q.F, q.K0):
            fig.add_vline(x=x, line=dict(color=INK["muted"], width=1, dash="dot"))
        fig.update_xaxes(title_text="index level at expiry S", row=2, col=1)
        # scale the error panel to the four methods (the LP band may run off it in the tails)
        errs = np.concatenate([1e4 * (ports[nm].payoff(S) - L) for nm in METHOD_NAMES])
        span = max(np.abs(np.percentile(errs, [1, 99])).max(), 1e-9)
        fig.update_yaxes(range=[-1.6 * span, 1.6 * span], row=2, col=1)
        return style_fig(fig, f"Log-contract payoff vs replicating portfolios: {q.name}", height=640, hover="x unified")


    chart_payoff(q_sel, ports_sel, zoom_ui.value)
    return


@app.cell(hide_code=True)
def fair_md(mo):
    mo.md(r"""
    ### 9.2 · The fair variance

    The price of each portfolio is that method's fair variance. On the synthetic markets we can split
    any method's error into two parts:

    * **Truncation** = full exact − in-range exact. This is the variance hidden beyond the last
      listed strikes. It is the same for every method, and it depends on how you believe the smile
      continues.
    * **Discretisation** = method − in-range exact. This is what the choice of method controls.

    The chart shows each method's distance from the reference, in vol basis points
    (1 bp = 0.01 vol pt). The reference is the in-range exact value on synthetic markets and Derman on
    SPX, where there is no exact answer. The shaded band is the LP's model-free range.

    **How to read it.**

    * On a clean 5-point grid the four methods agree to within a fraction of a basis point. Derman
      and VIX sit slightly above the in-range exact value (the chord arches), and the correction
      removes almost all of that.
    * **The two VIX variants coincide.** The exact-vs-quadratic offset difference is
      $\sim10^{-4}$ bp here (last-but-one table column).
    * **All four methods sit *below* the LP lower bound on the synthetic markets.** This is not a
      contradiction. The methods' portfolios stop curving at the last strike, while the LP lower
      portfolio uses large positions in the outermost options to keep following $L$ beyond them,
      without ever paying more than $L$. So, provided the index ends inside the support
      $[0.5F,1.5F]$, *every* model consistent with the quotes has more variance than a strip that
      simply stops. That missing part is the truncation. It costs about 2 bp here, ten times the
      discretisation error.
    * On the **real SPX chains** the methods spread out more. The 1-week Yahoo chain has 10–55-point
      holes near the money, so the convexity correction is about 6 bp. On the 2-day chain, Derman and
      VIX differ by about 4 bp because of the coarse 25-point wing spacing and their different
      treatment of the outermost strikes.
    """)
    return


@app.cell(hide_code=True)
def fair_chart(
    COLORS,
    DASH,
    INK,
    MARKETS,
    METHOD_NAMES,
    SYMBOL,
    go,
    mo,
    pd,
    ports_sel,
    q_sel,
    style_fig,
    vol_bp,
    vol_pts,
):
    def chart_fair(q, ports):
        synthetic = q.truth is not None
        ref, ref_name = (q.truth_in_range, "in-range exact") if synthetic else (ports["Derman"].kvar, "Derman")
        rows = ([("Exact (full)", q.truth), ("Exact (in-range)", q.truth_in_range)] if synthetic else []) + \
               [(nm, ports[nm].kvar) for nm in METHOD_NAMES]
        fig = go.Figure()
        up, lo = ports["LP upper"], ports["LP lower"]
        fig.add_vrect(x0=vol_bp(lo.kvar, ref), x1=vol_bp(up.kvar, ref), fillcolor=INK["band"], line_width=0,
                      annotation_text="LP model-free range", annotation_position="top left")
        for p, pos in ((lo, "bottom left"), (up, "bottom right")):
            fig.add_vline(x=vol_bp(p.kvar, ref), line=dict(color=COLORS[p.name], width=1.5, dash=DASH[p.name]),
                          annotation_text=f"{p.name} {vol_pts(p.kvar):.4f}", annotation_position=pos,
                          annotation_font=dict(color=INK["secondary"], size=11))
        for nm, v in rows:
            fig.add_trace(go.Scatter(x=[vol_bp(v, ref)], y=[nm], mode="markers+text", name=nm, showlegend=False,
                                     marker=dict(color=COLORS.get(nm, INK["primary"]), size=13, symbol=SYMBOL.get(nm, "diamond"),
                                                 line=dict(color=INK["surface"], width=2)),
                                     text=[f"{vol_pts(v):.4f}"], textposition="middle right",
                                     textfont=dict(color=INK["secondary"]),
                                     hovertemplate=f"{nm}<br>{vol_pts(v):.5f} vol pts<br>%{{x:+.3f}} bp vs {ref_name}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=INK["axis"], width=1))
        fig.update_xaxes(title_text=f"vol basis points vs {ref_name} ({vol_pts(ref):.4f})")
        fig.update_yaxes(autorange="reversed")
        return style_fig(fig, f"Fair variance by method: {q.name}", height=360, hover="closest")


    def fair_row(label, q, ports):
        row = {"market": label,
               "exact (full)": vol_pts(q.truth) if q.truth else None,
               "exact (in-range)": vol_pts(q.truth_in_range) if q.truth else None}
        row.update({nm: vol_pts(ports[nm].kvar) for nm in METHOD_NAMES + ["LP lower", "LP upper"]})
        row.update({"exact − CBOE offset (bp)": vol_bp(ports["VIX exact offset"].kvar, ports["VIX"].kvar),
                    "correction (bp)": vol_bp(ports["Derman"].kvar, ports["Derman + correction"].kvar)})
        return row


    fair_table = pd.DataFrame([fair_row(lbl, q, ports) for lbl, (q, ports) in MARKETS.items()]).set_index("market")
    mo.vstack([chart_fair(q_sel, ports_sel),
               mo.md("**All four markets** (5-pt synthetic strikes, $F$ mid-gap), in vol points:"),
               mo.ui.table(fair_table.round(4).reset_index(), selection=None, show_column_summaries=False)])
    return


@app.cell(hide_code=True)
def weights_md(mo):
    mo.md(r"""
    ### 9.3 · The option weights: what you would actually trade

    The portfolios are compared in one common form: **bond, forward, puts at and below $K_0$,
    calls above $K_0$**. Derman and VIX hold a call *at* $K_0$, and parity rewrites it as
    $C(K_0)=P(K_0)+(S-F)+(F-K_0)$: one more put at $K_0$, one more forward, and $F-K_0$ of cash. Without
    this step the methods would *look* different at $K_0$ while holding economically identical
    positions.

    * The **chart** shows each weight divided by the "textbook" weight $\frac2T\frac{\Delta K}{K^2}$
      (so 1 = textbook), and below it the cost of each position, $w\cdot Q$. It shows Derman, VIX and
      LP upper. The corrected versions hold the same options, and LP lower is in the table only (see
      below).
    * The **table** lists every position side by side for all six portfolios, either per unit of
      variance or as **SPX option contracts** (multiplier \$100) for a chosen vega notional.

    **From vega notional to contracts.** A vega notional $N_{\text{vega}}$ (\$ per vol point) is a
    variance notional $N_{\text{var}}=N_{\text{vega}}/(2K_{\text{vol}})$ (\$ per variance point), where
    $K_{\text{vol}}$ is the strike in vol points. A weight $w$ per unit of annualised variance then
    needs $w\times N_{\text{var}}\times10^4/100$ contracts: $\times10^4$ converts to variance points, and
    $/100$ is the contract multiplier.

    *Example:* \$100k vega at a 10.84 strike gives $N_{\text{var}}=100{,}000/21.68\approx\$4{,}613$
    per variance point. The put in the VIX example of section 6 ($w=1.29\times10^{-5}$) then needs
    $1.29\times10^{-5}\times4{,}613\times10^4/100\approx6$ contracts.

    **What to look for.**

    * **A2 = A and B2 = B, position for position.** The corrections change only the cash (bond) line.
    * **Inside the chain**, Derman, VIX and LP upper hold almost the same options (ratio ≈ 1). This holds
      even where the spacing jumps: next to a 95-point hole in the real chain, VIX's
      $\Delta K=\frac{K_{i+1}-K_{i-1}}2$ and Derman's slope change agree to within 1%.
    * **At the edges they really disagree.** Derman holds *nothing* at the outermost strikes. VIX holds a
      full $\Delta K$ there. LP upper holds a *large* position: tail insurance that keeps its payoff
      above $L$ out to $S_{lo}$ and $S_{hi}$. Together with $K_0$ (cash and forward), these are the only
      places where what you trade differs. The table's default view picks those rows out, and cells
      more than 5% away from Derman are highlighted.
    * **LP lower zig-zags** (table only): each tangent must meet its neighbour exactly at a strike,
      which pins down where it touches $L$. On the uneven real chain this forces large long/short pairs.
      *Its price is a valid bound, but its portfolio is not a usable hedge.*
    * **The forward leg** is large and negative: $-\frac{2}{TK_0}$, the log contract's delta at $K_0$.
      Two other deltas offset it at inception:
      * the options' own delta brings the static portfolio's delta to $L'(F)=-\frac{2}{TF}$;
      * the day-one dynamic hedge of section 1 adds $+\frac{2}{TF}$.

      So the whole book starts delta-neutral, as a variance swap should.
    """)
    return


@app.cell(hide_code=True)
def weights_ctl(mo):
    units_ui = mo.ui.radio(["SPX contracts", "per unit of variance (×10⁴)"], value="SPX contracts", label="Units",
                           inline=True)
    vega_ui = mo.ui.number(start=10_000, stop=10_000_000, step=10_000, value=100_000, label="Vega notional ($ per vol pt)")
    rows_ui = mo.ui.radio(["where methods differ most", "all strikes"], value="where methods differ most", label="Rows",
                          inline=True)
    mo.hstack([units_ui, vega_ui, rows_ui], justify="start", gap=2, wrap=True)
    return rows_ui, units_ui, vega_ui


@app.cell(hide_code=True)
def weights(
    COLORS,
    DASH,
    INK,
    SYMBOL,
    go,
    make_subplots,
    mo,
    np,
    pd,
    ports_sel,
    q_sel,
    rows_ui,
    style_fig,
    textbook_weights,
    units_ui,
    vega_ui,
    vol_pts,
):
    PORT_ORDER = ["Derman", "Derman + correction", "VIX", "VIX exact offset", "LP upper", "LP lower"]


    def common_basis(p, q):
        """Weights on every strike of q (puts <= K0, calls > K0), plus bond and forward, after rewriting any
        call at K0 as put + forward + cash (put-call parity)."""
        w = np.zeros(q.K.size)
        np.add.at(w, np.searchsorted(q.K, p.Kp), p.wp)
        np.add.at(w, np.searchsorted(q.K, p.Kc), p.wc)
        wc0 = p.wc[p.Kc == q.K0].sum()                 # call at K0 -> already added to the K0 slot as a put
        return w, p.bond + wc0 * (q.F - q.K0), p.fwd + wc0


    def weights_frame(q, ports, units, vega):
        """One row per instrument (bond, forward, each strike), one column per portfolio, in the chosen units."""
        vol_strike = vol_pts(ports["Derman"].kvar)
        contracts = units == "SPX contracts"
        scale = vega / (2 * vol_strike) * 1e4 / 100 if contracts else 1e4      # weight per unit variance -> display
        kind = np.where(q.K <= q.K0, "put", "call")
        df = pd.DataFrame({"instrument": ["bond (cash)", "forward"] + [f"{k} {K:,.0f}" for k, K in zip(kind, q.K)],
                           "strike": [np.nan, np.nan] + list(q.K)})
        for nm in PORT_ORDER:
            w, bond, fwd = common_basis(ports[nm], q)
            df[nm] = np.r_[bond * scale * (100 if contracts else 1), fwd * scale, w * scale]   # bond in $
        for nm in ("VIX", "LP upper", "LP lower"):
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"{nm} vs Derman (%)"] = np.where(np.abs(df["Derman"]) > 1e-12, 100 * (df[nm] / df["Derman"] - 1), np.nan)
        summary = pd.DataFrame([
            {"instrument": "TOTAL price = fair variance (vol pts)", **{nm: vol_pts(ports[nm].kvar) for nm in PORT_ORDER}},
            {"instrument": "number of strikes held",
             **{nm: float((np.abs(common_basis(ports[nm], q)[0]) > 1e-12).sum()) for nm in PORT_ORDER}}])
        return df, summary


    def pick_rows(df, how):
        """Bond, forward, and either every strike or the 20 where VIX or LP upper differ most from Derman."""
        body = df.iloc[2:]
        if how.startswith("where"):
            score = body[["VIX vs Derman (%)", "LP upper vs Derman (%)"]].abs().max(axis=1).fillna(np.inf)
            body = body.loc[score.sort_values(ascending=False).index[:20]].sort_values("strike")
        return pd.concat([df.iloc[:2], body])


    def chart_weights(q, ports):
        """Weight / textbook weight (top) and cost w*Q (bottom) for Derman, VIX and LP upper. The off-scale
        tail options of LP upper are pinned to the top edge with their value printed."""
        base, Q = textbook_weights(q), q.held_price()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Weight ÷ textbook weight (2/T)·ΔK/K²   (▲ = off the scale, value printed)",
                                            "Cost of each position w·Q   (variance points, vol%²)"))
        shown = {"Derman": "Derman (= Derman + correction)", "VIX": "VIX (= VIX exact offset)", "LP upper": "LP upper"}
        weights = {nm: common_basis(ports[nm], q)[0] for nm in shown}
        top = max(2.0, 1.2 * max((weights[nm] / base)[1:-1].max() for nm in shown))
        for nm, label in shown.items():
            y = weights[nm] / base
            off = y > top
            fig.add_trace(go.Scatter(x=q.K, y=np.minimum(y, top), mode="lines+markers", name=label, legendgroup=nm,
                                     line=dict(color=COLORS[nm], width=1.5, dash=DASH[nm]),
                                     marker=dict(color=COLORS[nm], size=np.where(off, 11, 5),
                                                 symbol=np.where(off, "triangle-up", SYMBOL[nm])),
                                     customdata=y, hovertemplate="K=%{x:,.0f}<br>ratio %{customdata:.3f}<extra>" + label + "</extra>"),
                          1, 1)
            for k, v in zip(q.K[off], y[off]):
                fig.add_annotation(x=k, y=top, text=f"{v:,.0f}×", showarrow=False, yshift=12, row=1, col=1,
                                   font=dict(color=INK["primary"], size=11))
            fig.add_trace(go.Scatter(x=q.K, y=1e4 * weights[nm] * Q, mode="lines", name=label, legendgroup=nm,
                                     showlegend=False, line=dict(color=COLORS[nm], width=1.5, dash=DASH[nm])), 2, 1)
        fig.add_hline(y=1, line=dict(color=INK["axis"], width=1), row=1, col=1)
        fig.add_vline(x=q.F, line=dict(color=INK["muted"], width=1, dash="dot"))
        fig.update_yaxes(range=[-0.2, top * 1.2], row=1, col=1)
        fig.update_xaxes(title_text="strike", row=2, col=1)
        return style_fig(fig, f"Option weights by strike: {q.name}", height=600, hover="closest")


    def weights_view(q, ports):
        df, summary = weights_frame(q, ports, units_ui.value, vega_ui.value)
        shown = pd.concat([pick_rows(df, rows_ui.value), summary], ignore_index=True).drop(columns="strike")
        pct_cols = [c for c in shown if c.endswith("(%)")]

        def style_cell(row_id, col, value):
            if col in pct_cols and isinstance(value, (int, float)) and np.isfinite(value) and abs(value) > 5:
                return {"backgroundColor": "rgba(235,104,52,0.18)"}
            if str(shown.loc[int(row_id), "instrument"]).startswith(("TOTAL", "number")):
                return {"fontWeight": "bold"}
            return {}

        k_vol = vol_pts(ports["Derman"].kvar)
        unit_note = (f"SPX contracts for ${vega_ui.value:,.0f} vega (strike {k_vol:.2f} vol, variance notional "
                     f"${vega_ui.value / (2 * k_vol):,.0f} per vol pt²); bond in $; forward in SPX-contract equivalents"
                     if units_ui.value == "SPX contracts" else "per unit of annualised variance, ×10⁴")
        table = mo.ui.table(shown, selection=None, page_size=30, show_column_summaries=False, style_cell=style_cell,
                            freeze_columns_left=["instrument"],
                            format_mapping={c: "{:,.2f}" if units_ui.value == "SPX contracts" else "{:,.4f}" for c in PORT_ORDER}
                                           | {c: "{:+.1f}" for c in pct_cols},
                            label=f"Positions: {unit_note}")
        return mo.vstack([chart_weights(q, ports), table])


    weights_view(q_sel, ports_sel)
    return


@app.cell(hide_code=True)
def sweep_md(mo):
    mo.md(r"""
    ## 10 · Where do the corrections start to matter?

    On SPX at 5-point strikes both corrections are tiny. To see when they matter, scale the strike
    gap $h$ up, as a fraction of the forward, on the synthetic market (same smile, $F$ in the middle
    of a gap), for 2 days and 1 week to expiry.

    The chart shows the **chord over-pricing** that Derman's correction removes, and what is left
    after the correction, both measured against the in-range exact value. VIX is shown too. The error
    grows like $h^2$ and like $1/T$.

    The **VIX offset approximation** needs no chart: it costs
    $\frac{x^2}T-\frac2T[x-\ln(1+x)]\approx\frac{2x^3}{3T}$, which is worst when $F$ sits just below the
    next strike ($x\approx h/K_0$). It grows like $h^3$. The note below the chart gives the gap at which
    it reaches 1 bp.

    Reference gaps are marked:

    * SPX 5-point spacing near the money (0.06% of $F$);
    * SPX 25-point and 100-point wing spacing;
    * two single-stock cases: \$1 strikes on a \$50 stock (2%, "stock 1 on 50") and \$5 strikes on a
      \$100 stock (5%, "stock 5 on 100").
    """)
    return


@app.cell(hide_code=True)
def sweep(
    COLORS,
    INK,
    SYMBOL,
    derman,
    derman_corrected,
    go,
    mo,
    np,
    pd,
    style_fig,
    svi_1w,
    synthetic_quotes,
    vix,
    vol_bp,
):
    SWEEP_T = {"2 days": 2 / 365, "1 week": 7 / 365}
    REF_GAPS = {"SPX 5": 5 / 7800, "SPX 25": 25 / 7800, "SPX 100": 100 / 7800, "stock 1 on 50": 0.02, "stock 5 on 100": 0.05}


    @mo.cache
    def sweep_errors():
        """Error vs the in-range exact value (vol bp) for strike gaps from 0.04% to 5% of F, F mid-gap."""
        rows = []
        for label, T in SWEEP_T.items():
            for rel in np.geomspace(4e-4, 0.05, 22):
                q = synthetic_quotes(svi_1w, T, rel * svi_1w.F, 0.5)
                if q.K.size >= 4:
                    rows.append({"T": label, "gap / F": rel,
                                 **{p.name: vol_bp(p.kvar, q.truth_in_range) for p in (derman(q), derman_corrected(q), vix(q))}})
        return pd.DataFrame(rows)


    def chart_sweep(df):
        fig = go.Figure()
        dashes = {"2 days": "solid", "1 week": "dash"}
        for label in SWEEP_T:
            d = df[df["T"] == label]
            for nm in ("Derman", "Derman + correction", "VIX"):
                fig.add_trace(go.Scatter(x=d["gap / F"], y=d[nm].abs(), mode="lines+markers", name=f"{nm}, {label}",
                                         line=dict(color=COLORS[nm], dash=dashes[label], width=2), marker=dict(size=6, symbol=SYMBOL[nm]),
                                         customdata=d[nm], hovertemplate="gap %{x:.3%}<br>%{customdata:+.3g} bp<extra></extra>"))
        for txt, g in REF_GAPS.items():
            fig.add_vline(x=g, line=dict(color=INK["muted"], width=1, dash="dot"))
            fig.add_annotation(x=np.log10(g), y=0.0, yref="paper", text=txt, showarrow=False, textangle=-90,
                               xanchor="right", yanchor="bottom", font=dict(size=10, color=INK["secondary"]))
        fig.add_hline(y=1.0, line=dict(color=INK["axis"], width=1))
        fig.update_xaxes(type="log", tickvals=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                         ticktext=["0.05%", "0.1%", "0.2%", "0.5%", "1%", "2%", "5%"], title_text="strike gap h / F")
        fig.update_yaxes(type="log", title_text="|error| vs in-range exact, vol bp (log)", exponentformat="power")
        return style_fig(fig, "When does the convexity correction matter? (1 bp = 0.01 vol pt; grey line = 1 bp)",
                         height=480, hover="closest")


    chart_sweep(sweep_errors())
    return SWEEP_T, sweep_errors


@app.cell(hide_code=True)
def sweep_note(
    SWEEP_T,
    demo,
    mo,
    np,
    offset_exact,
    offset_vix,
    sweep_errors,
    vol_bp,
):
    _d = sweep_errors()
    _sigma = np.sqrt(demo.truth)


    def _first_above(col, label, level=1.0):
        d = _d[(_d["T"] == label) & (_d[col].abs() >= level)]
        return f"{d['gap / F'].min():.2%}" if len(d) else "never in this range"


    def _offset_threshold(T):
        """Smallest gap h/F at which the CBOE quadratic costs 1 vol bp, with F just below the next strike (x = h/K0)."""
        x = np.geomspace(1e-4, 0.1, 4000)
        cost = offset_vix(1 + x, 1.0, T) - offset_exact(1 + x, 1.0, T)
        return x[np.argmax(vol_bp(_sigma**2 + cost, _sigma**2) >= 1.0)]


    def _sd(label):
        return _sigma * np.sqrt(SWEEP_T[label])


    mo.md(rf"""
    **Reading the sweep.**

    * **Chord over-pricing.** Derman's (and VIX's) error reaches 1 bp at gaps of about
      **{_first_above('Derman', '2 days')}** of the forward at 2 days and **{_first_above('Derman', '1 week')}** at 1 week.
      The SPX 25-point wing spacing already exceeds both. The **convexity correction cuts the error by a factor of
      10–100**, as long as the gap is smaller than about one standard deviation of $S_T$ (here {_sd('2 days'):.1%} of $F$ at
      2 days, {_sd('1 week'):.1%} at 1 week). Beyond that, the assumption that $S_T$ is spread evenly within a gap fails
      and the correction stops working.
    * **The VIX offset approximation**, in closed form $\approx\frac{{2x^3}}{{3T}}$, only reaches 1 bp once the gap exceeds
      **{_offset_threshold(SWEEP_T['2 days']):.2%}** of the forward at 2 days and **{_offset_threshold(SWEEP_T['1 week']):.2%}**
      at 1 week, even with $F$ in the worst spot. That is roughly ten times the 5-point SPX gap, but well inside the 2–5%
      gaps of many single stocks, where the exact term is worth using.
    * Shortening the expiry from 1 week to 2 days scales both errors by about $7/2$, because both carry a $1/T$. Part of
      that is just annualisation: the same dollar error is divided by a smaller $T$.
    """)
    return


@app.cell(hide_code=True)
def checks(
    MARKETS,
    convexity_correction_loop,
    derman_loop,
    mo,
    np,
    pd,
    spx_1w_table,
    spx_2d_table,
    vol_pts,
):
    def run_checks():
        out = []

        def add(what, market, ok, detail=""):
            out.append({"check": what, "market": market, "pass": "✅" if ok else "❌", "detail": detail})

        for label, (q, ps) in MARKETS.items():
            short = label.split(" (")[0]
            wp, wc, price = derman_loop(q)
            add("Derman: loop = numpy", short, np.allclose(wp, ps["Derman"].wp) and np.isclose(price, ps["Derman"].kvar))
            c_loop, c = convexity_correction_loop(q), ps["Derman + correction"].extra["correction"]
            add("Correction: loop = numpy", short, np.isclose(c_loop, c, rtol=1e-4), f"{c_loop:.3e} vs {c:.3e}")
            add("VIX: loop = numpy", short, np.isclose(vix_loop(q), ps["VIX"].kvar))
            x = q.F / q.K0 - 1
            diff = ps["VIX exact offset"].kvar - ps["VIX"].kvar
            add("VIX exact − VIX ≈ 2x³/(3T)", short, np.isclose(diff, 2 * x**3 / (3 * q.T), rtol=0.01, atol=1e-13),
                f"{diff:.2e} vs {2 * x**3 / (3 * q.T):.2e}")
            up, d = ps["LP upper"], ps["Derman"]
            add("LP upper holds Derman's options inside the strike range", short,
                np.allclose(up.wp[1:-1], d.wp[1:-1], rtol=5e-3) and np.allclose(up.wc[:-1], d.wc[1:-1], rtol=5e-3))
            if q.truth is not None:
                lo = ps["LP lower"]
                add("LP lower ≤ exact ≤ LP upper", short, lo.kvar <= q.truth <= up.kvar,
                    f"{vol_pts(lo.kvar):.4f} ≤ {vol_pts(q.truth):.4f} ≤ {vol_pts(up.kvar):.4f}")
                add("Derman ≥ in-range exact (chords lie above)", short, d.kvar >= q.truth_in_range)
                add("correction moves Derman closer to in-range exact", short,
                    abs(ps["Derman + correction"].kvar - q.truth_in_range) < abs(d.kvar - q.truth_in_range))
        for t, nm in ((spx_1w_table, "SPX · 1 week"), (spx_2d_table, "SPX · 2 trading days")):
            n_out = int((t["change / half-spread"].abs() > 1).sum())
            add("noise fix keeps prices within the bid–ask", nm, n_out <= 0.05 * len(t),
                f"{n_out} of {len(t)} prices moved by more than half the spread")
        return pd.DataFrame(out)


    checks = run_checks()
    mo.vstack([mo.md(f"## 11 · Checks\n\n{(checks['pass'] == '✅').sum()} of {len(checks)} checks pass."),
               mo.ui.table(checks, selection=None, page_size=50, show_column_summaries=False)])
    return


@app.cell(hide_code=True)
def conclusions(LP_SUPPORT, MARKETS, mo, vol_bp):
    _r = {label.split(" (")[0]: v for label, v in MARKETS.items()}
    _q1, _p1 = _r["Synthetic · 1 week"]
    _q2, _p2 = _r["Synthetic · 2 days"]
    _s1, _ps1 = _r["SPX · 1 week"]
    _s2, _ps2 = _r["SPX · 2 trading days"]


    def _bp(a, b):
        return f"{vol_bp(a, b):+.2f} bp"


    mo.md(rf"""
    ## 12 · Conclusions

    **1. On a clean 5-point SPX grid all four methods are good to a fraction of a vol basis point.**
    At 1 week, Derman is {_bp(_p1['Derman'].kvar, _q1.truth_in_range)} and VIX
    {_bp(_p1['VIX'].kvar, _q1.truth_in_range)} from the in-range exact value. At 2 days the errors
    grow (roughly ×7/2, the $1/T$) to {_bp(_p2['Derman'].kvar, _q2.truth_in_range)} and
    {_bp(_p2['VIX'].kvar, _q2.truth_in_range)}. The convexity correction brings Derman to
    {_bp(_p1['Derman + correction'].kvar, _q1.truth_in_range)} and
    {_bp(_p2['Derman + correction'].kvar, _q2.truth_in_range)}.

    **2. The exact offset term never matters for SPX.** The VIX quadratic is off by
    $\approx\frac{{2x^3}}{{3T}}$: {vol_bp(_ps1['VIX exact offset'].kvar, _ps1['VIX'].kvar):.1e} bp on the real
    1-week chain and {vol_bp(_ps2['VIX exact offset'].kvar, _ps2['VIX'].kvar):.1e} bp on the 2-day chain. It
    reaches 1 bp only for strike gaps of roughly 0.6–0.9% of the forward or more (section 10), as on many single
    stocks. Using the exact term costs nothing, so there is no reason not to.

    **3. Truncation dominates.** The variance beyond the listed strikes is
    {_bp(_q1.truth, _q1.truth_in_range)} at 1 week, ten times the discretisation error at 5-point spacing.
    The LP lower bound shows this without a model: if the index ends inside the LP support
    $[{LP_SUPPORT[0]:g}F,{LP_SUPPORT[1]:g}F]$, every model that fits the quotes gives more variance than a strip
    that simply stops at the last strike.

    **4. The real chain has holes, and that is where the correction earns its keep.** On the 1-week
    Yahoo chain (gaps of up to 55 points near the money) the correction is
    {vol_bp(_ps1['Derman'].kvar, _ps1['Derman + correction'].kvar):.1f} bp. On the 2-day chain,
    Derman and VIX differ by {vol_bp(_ps2['VIX'].kvar, _ps2['Derman'].kvar):.1f} bp, mostly from how
    they treat the outermost strikes.

    **5. What you trade is almost the same across methods, and the LP is for prices, not hedges.**
    Inside the chain, Derman, VIX and the LP upper portfolio hold the same options to within 1%. They differ
    only at the two outermost strikes (Derman nothing, VIX a full $\Delta K$, LP upper a large tail
    position) and at $K_0$ (cash and forward). The corrections A2 and B2 never change a position, only the
    price. The LP lower portfolio's weights zig-zag wildly on an uneven chain: its price is a valid bound,
    but its weights are not a usable hedge.

    **Practical recipe for short-dated index variance:** use Derman's chords (or VIX weights, which are
    nearly identical) with the exact offset term. Add the convexity correction whenever the chain has
    holes or coarse wings. Treat the wing extrapolation, not the quadrature rule, as the real source of
    model risk.
    """)
    return


@app.cell(hide_code=True)
def appx_md(mo):
    mo.md(r"""
    ## Appendix A · Can the convexity correction be traded?

    Method A2 turns the convexity correction into a *number*: it keeps Derman's options and lowers the
    price. A natural question is whether the same correction can be built into the **positions**
    instead, so that the portfolio itself stops over-paying.

    **The idea.** A portfolio of bond, forward and options pays a piecewise-linear amount with kinks only
    at the listed strikes, so it is fixed entirely by its values $g_i$ at the strikes. Derman sets
    $g_i=f(K_i)$. Lower each one by an amount $\delta_i$:

    $$
    g_i=f(K_i)-\delta_i,\qquad w_i=\text{change of slope of }g\text{ at }K_i .
    $$

    The new portfolio pays Derman's payoff minus the straight-line interpolation of the $\delta_i$, so
    its price falls by the market price of that interpolation, roughly $\mathbb E[\delta(S_T)]$. The option
    prices do the probability weighting.

    **What should $\delta$ be?** On gap $g=[K_i,K_{i+1}]$ the chord over-pays $e_g$ on average (section 5),
    and the lowering averages $\frac{\delta_i+\delta_{i+1}}2$ there, so we want
    $\frac{\delta_i+\delta_{i+1}}2=e_g$ on every gap.

    * **On an even grid this is just A2.** $\delta(K)\approx\frac{h^2}{6TK^2}$ varies slowly, and a slowly
      varying shift is almost all bond and forward. The options only pick up its curvature, which scales
      each weight by about $1-\frac{h^2}{2K^2}$, i.e. $1-2\times10^{-7}$ for 5-point SPX strikes. Same price,
      same trades.
    * **Walking outward gap by gap fails.** Solving the equations one at a time,
      $\delta_{i+1}=2e_g-\delta_i$, makes each node absorb what the previous gap missed. The recursion
      alternates in sign and nothing damps it: wherever the gap size jumps (a hole, or 5 → 25 points in
      the wings), $\delta$ flips sign and grows. The weights zig-zag, for the same reason as the LP lower
      portfolio: neighbouring gaps share the value at the strike between them.
    * **The root cause.** There are $N$ strikes but only $N-1$ gaps, and inside a hole you cannot lower
      the middle of the gap without also lowering its ends. No choice of weights removes the arch; it
      can only be moved around.

    **A sounder version: choose all the $\delta_i$ together.** Solve a regularised least-squares problem:

    $$
    \min_\delta\;\sum_g p_g\Big(\tfrac{\delta_i+\delta_{i+1}}2-e_g\Big)^2\;+\;\lambda\sum_i\big(\Delta s_i\,\bar h_i\big)^2 .
    $$

    * The first term asks each gap's average lowering to match its over-payment, weighted by the
      probability $p_g$ of landing in the gap (from the quotes, as in A2). Without these weights the
      wide wing gaps, with huge over-payment but almost no probability, dominate the fit.
    * $\Delta s_i$ is the change of slope of $\delta$ at $K_i$, which is exactly **how much the option
      weight at $K_i$ moves away from Derman's**. Scaling by the local spacing $\bar h_i$ puts it in
      variance units, so $\lambda$ is a plain number. Large $\lambda$: positions barely move. Small $\lambda$:
      a closer fit, but more trading.

    In numpy this is one stacked linear system, $\begin{bmatrix}\sqrt{p}\,A\\\sqrt\lambda\,D\end{bmatrix}\delta\approx\begin{bmatrix}\sqrt{p}\,e\\0\end{bmatrix}$,
    solved by `np.linalg.lstsq`. Each row of $A$ has two entries of $\frac12$ (one gap's average), and
    each row of $D$ has three entries (a change of slope).

    **How we test it.** A real chain has no exact answer, and an even synthetic grid has no holes. So we
    use the **SVI market on the real SPX strike layout**: every listed strike of the Yahoo chain, holes
    and wide wings included, but priced from the fitted smile, so the exact answer is known.
    """)
    return


@app.cell
def appx_code(
    chord_mean_error,
    demo,
    derman,
    derman_corrected,
    f_ddkz,
    np,
    offset_exact,
    quote_cdf,
    vol_bp,
):
    def node_portfolio(q, delta):
        """Derman's chord portfolio with its payoff at each strike lowered by delta.
        Returns (fair variance it prices, option weight at each strike)."""
        K = q.K
        g = f_ddkz(K, q.K0, q.T) - delta              # payoff value at each strike
        s = np.diff(g) / np.diff(K)                    # slope on each gap
        w = np.zeros_like(K)
        w[1:-1] = s[1:] - s[:-1]                       # weight = change of slope (outermost strikes: 0, as in Derman)
        i0 = np.searchsorted(K, q.K0)
        # payoff = g(K0) + slope right of K0 * (S - K0) + puts at K <= K0 + calls above K0
        price = g[i0] + s[i0] * (q.F - q.K0) + np.sum(w * q.held_price())
        return price - offset_exact(q.F, q.K0, q.T), w


    def gap_overpay(q):
        """e_g: average over-payment of the chord on each gap (section 5)."""
        return chord_mean_error(q.K[:-1], q.K[1:], q.K0, q.T)


    def naive_node_shifts(q):
        """Walk outward from K0; each node absorbs what the previous gap missed: delta_i + delta_(i+1) = 2 e_g."""
        e, K = gap_overpay(q), q.K
        i0 = np.searchsorted(K, q.K0)
        d = np.zeros_like(K)
        d[i0] = e[i0]
        for i in range(i0, K.size - 1):
            d[i + 1] = 2 * e[i] - d[i]
        for i in range(i0, 0, -1):
            d[i - 1] = 2 * e[i - 1] - d[i]
        return d


    def ls_node_shifts(q, lam, prob_weights=True):
        """All delta_i together: min sum_g p_g (avg lowering - e_g)^2 + lam * sum_i (weight change_i * spacing_i)^2."""
        K, e = q.K, gap_overpay(q)
        n, h = K.size, np.diff(K)
        A = np.zeros((n - 1, n))                       # row g: average of delta over gap g
        A[np.arange(n - 1), np.arange(n - 1)] = 0.5
        A[np.arange(n - 1), np.arange(1, n)] = 0.5
        D = np.zeros((n - 2, n))                       # row i: change of slope of delta at K_i, times local spacing
        for i in range(1, n - 1):
            span = 0.5 * (h[i - 1] + h[i])
            D[i - 1, i - 1:i + 2] = np.array([1 / h[i - 1], -1 / h[i - 1] - 1 / h[i], 1 / h[i]]) * span
        p = np.diff(quote_cdf(q)) if prob_weights else np.ones(n - 1)
        sp = np.sqrt(np.maximum(p, 1e-12))[:, None]
        M = np.vstack([sp * A, np.sqrt(lam) * D])
        b = np.r_[sp[:, 0] * e, np.zeros(n - 2)]
        return np.linalg.lstsq(M, b, rcond=None)[0]


    # no shift reproduces Derman exactly; on an even 5-point grid every variant lands on A2's price
    assert np.isclose(node_portfolio(demo, 0 * demo.K)[0], derman(demo).kvar, rtol=0, atol=1e-14)
    for _d in (naive_node_shifts(demo), ls_node_shifts(demo, 0.1)):
        assert abs(vol_bp(node_portfolio(demo, _d)[0], derman_corrected(demo).kvar)) < 1e-3
    return ls_node_shifts, naive_node_shifts, node_portfolio


@app.cell(hide_code=True)
def appx_setup(
    Quotes,
    SviMarket,
    f_ddkz,
    mo,
    node_portfolio,
    np,
    simpson,
    spx_1w,
    spx_2d,
    svi_1w,
    vol_bp,
):
    APPX_LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]


    @mo.cache
    def spx_layout_reference(which):
        """SVI market on a real chain's strikes, plus what the comparison needs: the true density on the
        strike range, the true in-range over-payment of the chords, and Liu's least-expected-squares nodes."""
        chain = {"1 week": spx_1w, "2 days": spx_2d}[which]
        mk = SviMarket(chain.F, svi_1w.T, svi_1w.params)
        if not np.isclose(chain.T, svi_1w.T):
            mk = mk.at_horizon(chain.T)
        K = chain.K.copy()
        K0 = float(K[K <= chain.F].max())
        q = Quotes(f"SVI on the SPX {which} strikes", chain.F, chain.T, K,
                   np.where(K <= K0, mk.put(K), np.nan), np.where(K >= K0, mk.call(K), np.nan), K0)
        q.truth, q.truth_in_range = mk.exact_kvar(), mk.exact_kvar(K.min(), K.max())
        S = np.linspace(K.min(), K.max(), 40001)
        dens_raw = np.maximum(np.gradient(np.gradient(mk.put(S), S), S), 0.0)   # density of S_T = P''(K)
        dens = dens_raw / simpson(dens_raw, x=S)       # conditional on ending inside the strike range
        f_S = f_ddkz(S, K0, q.T)
        overpay = simpson(dens_raw * (np.interp(S, K, f_ddkz(K, K0, q.T)) - f_S), x=S)
        hats = np.stack([np.interp(S, K, np.eye(K.size)[i]) for i in range(K.size)], axis=1)
        sw = np.sqrt(dens * np.gradient(S))
        liu = f_ddkz(K, K0, q.T) - np.linalg.lstsq(hats * sw[:, None], f_S * sw, rcond=None)[0]
        return dict(q=q, S=S, dens=dens, f_S=f_S, overpay=overpay, liu=liu)


    def appx_row(ref, name, delta):
        """Price error, path-by-path payoff mismatch and how far the weights move from Derman's."""
        q = ref["q"]
        der_price, w_der = node_portfolio(q, 0 * q.K)
        price, w = node_portfolio(q, delta)
        payoff = np.interp(ref["S"], q.K, f_ddkz(q.K, q.K0, q.T) - delta)
        rms = np.sqrt(simpson(ref["dens"] * (payoff - ref["f_S"]) ** 2, x=ref["S"]))
        m = np.abs(w_der) > 0
        change = np.abs(w[m] - w_der[m]) / np.abs(w_der[m])
        return {"method": name,
                "correction, bp": vol_bp(der_price, price),
                "left after correcting, bp": vol_bp(price, der_price - ref["overpay"]),
                "payoff mismatch RMS, bp": vol_bp(q.truth_in_range + rms, q.truth_in_range),
                "largest weight change": change.max(),
                "95th pct weight change": np.quantile(change, 0.95),
                "short options": int(np.sum(w[1:-1] < 0))}


    appx_market_ui = mo.ui.dropdown(["1 week", "2 days"], value="1 week", label="SPX strike layout")
    appx_lam_ui = mo.ui.slider(steps=APPX_LAMBDAS, value=0.1, label="Penalty λ on weight changes", show_value=True)
    mo.hstack([appx_market_ui, appx_lam_ui], justify="start", gap=2, wrap=True)
    return (
        APPX_LAMBDAS,
        appx_lam_ui,
        appx_market_ui,
        appx_row,
        spx_layout_reference,
    )


@app.cell(hide_code=True)
def appx_view(
    APPX_LAMBDAS,
    INK,
    appx_lam_ui,
    appx_market_ui,
    appx_row,
    derman_corrected,
    go,
    ls_node_shifts,
    make_subplots,
    mo,
    naive_node_shifts,
    node_portfolio,
    np,
    pd,
    spx_layout_reference,
    style_fig,
    vol_bp,
):
    _ref = spx_layout_reference(appx_market_ui.value)
    _q, _lam = _ref["q"], appx_lam_ui.value
    _corr = derman_corrected(_q).extra["correction"]
    _rows = [
        appx_row(_ref, "Derman (no correction)", 0 * _q.K),
        appx_row(_ref, "A2: correction as cash", np.full_like(_q.K, _corr)),
        appx_row(_ref, "Walk outward gap by gap", naive_node_shifts(_q)),
        appx_row(_ref, f"Least squares, equal gap weights, λ={_lam:g}", ls_node_shifts(_q, _lam, prob_weights=False)),
        appx_row(_ref, f"Least squares, probability weights, λ={_lam:g}", ls_node_shifts(_q, _lam)),
        appx_row(_ref, "Best possible with true density (Liu 2010)", _ref["liu"]),
    ]
    _fmt = {"correction, bp": "{:+.3f}", "left after correcting, bp": "{:+.3f}", "payoff mismatch RMS, bp": "{:.2f}",
            "largest weight change": "{:.0%}", "95th pct weight change": "{:.1%}"}
    _table = mo.ui.table(pd.DataFrame(_rows), format_mapping=_fmt, selection=None, pagination=False,
                         show_column_summaries=False)

    _sweep = pd.DataFrame([{**appx_row(_ref, "", ls_node_shifts(_q, lam)), "λ": lam} for lam in APPX_LAMBDAS])
    _sweep = _sweep[["λ", "left after correcting, bp", "payoff mismatch RMS, bp", "largest weight change",
                     "95th pct weight change", "short options"]]
    _sweep_table = mo.ui.table(_sweep, format_mapping=_fmt, selection=None, pagination=False, show_column_summaries=False)

    # where do the positions move? weight change vs Derman at each strike, for the selected lambda
    _, _w_der = node_portfolio(_q, 0 * _q.K)
    _, _w_ls = node_portfolio(_q, ls_node_shifts(_q, _lam))
    _m = np.abs(_w_der) > 0
    _chg = (_w_ls[_m] - _w_der[_m]) / np.abs(_w_der[_m])
    _gap = np.maximum(np.r_[np.diff(_q.K), 0][_m], np.r_[0, np.diff(_q.K)][_m])
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                         subplot_titles=("Option weight change vs Derman (probability-weighted least squares)",
                                         "Strike spacing: widest neighbouring gap"))
    _fig.add_trace(go.Scatter(x=_q.K[_m], y=_chg, mode="markers", name=f"λ = {_lam:g}", showlegend=False,
                              marker=dict(size=8, color="#1a9e77", line=dict(color=INK["surface"], width=2)),
                              customdata=_gap, hovertemplate="K %{x:.0f}<br>change %{y:+.1%}<br>widest gap %{customdata:.0f} pts<extra></extra>"),
                   row=1, col=1)
    _fig.add_trace(go.Bar(x=_q.K[_m], y=_gap, name="widest neighbouring gap", marker_color=INK["axis"], showlegend=False,
                          hovertemplate="K %{x:.0f}<br>gap %{y:.0f} pts<extra></extra>"), row=2, col=1)
    _fig.add_vline(x=_q.F, line=dict(color=INK["muted"], width=1, dash="dot"))
    _fig.update_yaxes(tickformat="+.0%", title_text="change", row=1, col=1)
    _fig.update_yaxes(title_text="points", row=2, col=1)
    _fig.update_xaxes(title_text="strike K (dotted line: forward)", row=2, col=1)

    mo.vstack([
        mo.md(f"**{_q.name}** · {_q.K.size} strikes, gaps from {np.diff(_q.K).min():.0f} to {np.diff(_q.K).max():.0f} "
              f"points · true in-range over-payment of the chords: "
              f"**{vol_bp(node_portfolio(_q, 0 * _q.K)[0], node_portfolio(_q, 0 * _q.K)[0] - _ref['overpay']):.2f} bp**"),
        _table,
        mo.md("**The trade-off as λ varies** (probability-weighted least squares):"),
        _sweep_table,
        style_fig(_fig, f"Where the positions move, λ = {_lam:g}", height=480, hover="closest"),
    ])
    return


@app.cell(hide_code=True)
def appx_verdict(
    appx_row,
    derman_corrected,
    ls_node_shifts,
    mo,
    naive_node_shifts,
    np,
    spx_layout_reference,
):
    def _summary(which, lam=0.1):
        ref = spx_layout_reference(which)
        q = ref["q"]
        a2 = appx_row(ref, "", np.full_like(q.K, derman_corrected(q).extra["correction"]))
        ls = appx_row(ref, "", ls_node_shifts(q, lam))
        eq = appx_row(ref, "", ls_node_shifts(q, lam, prob_weights=False))
        walk = appx_row(ref, "", naive_node_shifts(q))
        liu = appx_row(ref, "", ref["liu"])
        return a2, ls, eq, walk, liu


    _a1, _l1, _e1, _n1, _o1 = _summary("1 week")
    _a2, _l2, _e2, _n2, _o2 = _summary("2 days")
    _k = "left after correcting, bp"
    _r = "payoff mismatch RMS, bp"

    mo.md(rf"""
    **How to read the table.**

    * *correction*: how much each method lowers Derman's price. *Left after correcting*: the true
      over-payment of the chords inside the strike range (computed from the SVI density) minus that
      correction. Zero is perfect; positive means the price is still too high, negative means
      over-corrected.
    * *payoff mismatch RMS*: the typical size of (portfolio payoff − $f$) if the index ends inside the
      range, weighted by the true density and shown in vol bp. A2 has the same payoff shape as Derman, just
      shifted down, so this measures how well each portfolio **hedges**, not how well it **prices**.
    * *weight change*: $|w-w_{{\text{{Derman}}}}|/|w_{{\text{{Derman}}}}|$ across the strikes Derman trades.

    **What the numbers say** (λ = 0.1 unless stated):

    1. **Walking outward gap by gap is unusable.** The weights move by up to
       {_n1['largest weight change']:,.0%} on the 1-week layout ({_n1['short options']} short options) and
       {_n2['largest weight change']:,.0%} on the 2-day one.
    2. **Probability weights are essential.** With equal gap weights the wide wing gaps dominate the fit,
       and {_e1[_k]:.2f} bp of over-payment is left uncorrected at 1 week.
    3. **For the price, least squares gains nothing over A2.** What is left after correcting is
       {_l1[_k]:+.3f} bp vs A2's {_a1[_k]:+.3f} bp at 1 week, and {_l2[_k]:+.3f} vs {_a2[_k]:+.3f} bp at 2 days. Both are
       far below the noise in real quotes: in a side test that perturbed every quote by up to ±0.1 points
       (200 random draws), either answer moved by about 1.3–1.5 bp (standard deviation).
    4. **For the hedge, it helps a little.** The payoff mismatch falls from {_a1[_r]:.1f} to {_l1[_r]:.1f} bp
       at 1 week and from {_a2[_r]:.1f} to {_l2[_r]:.1f} bp at 2 days. The best any weights can do, knowing the
       true density, is {_o1[_r]:.1f} and {_o2[_r]:.1f} bp, but those weights change by up to
       {_o1['largest weight change']:,.0%} with {_o1['short options']} short options.
    5. **The cost is in the positions.** At λ = 0.1 the largest change is
       {_l1['largest weight change']:.0%}, and 95% of strikes move less than {_l1['95th pct weight change']:.0%}. The big
       changes sit **near the money**, around places where a strike is missing from the 5-point grid (10- or
       15-point gaps), not at the wide wing holes: the probability weights make the fit care most where
       $S_T$ is likely to end. These positions also inherit the noise in the quote-implied probabilities.

    **Verdict.** A position-based correction is easy to build and, with probability weights and a
    moderate λ, gives sensible all-long weights. But for the fair-variance number it is no better than
    A2, which leaves the positions alone. It only pays off if the path-by-path error of the static leg
    matters to you, and even then it competes with replication errors not modelled here (daily rather
    than continuous monitoring, jumps) and with the bid–ask cost of the extra trades.

    **Further reading.** Liu, *Optimal approximations of nonlinear payoffs in static replication*,
    Journal of Futures Markets 30(11), 2010: its "least expected squares" method picks the weights on
    fixed listed strikes to minimise the expected squared payoff error, the last row of the table. Leung &
    Lorig, *Optimal static quadratic hedging*, Quantitative Finance 16(9), 2016: a model-free version
    with a cost constraint. Both need a density or a model. The regularised least squares above is a
    cheap, model-light approximation, and λ controls how far it moves toward their zig-zagging optimum.
    """)
    return


if __name__ == "__main__":
    app.run()
