# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Variance swap replication")


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Replicating a variance swap with a *discrete* strip of options

    A variance swap pays realised variance minus a fixed strike. Its fair strike is the price of a
    **log contract**, and the log contract can be built statically from out-of-the-money options
    (Carr–Madan; Demeterfi, Derman, Kamal & Zou 1999 — "DDKZ"). The textbook formula needs a
    **continuum** of strikes from $0$ to $\infty$. Real option chains have a finite set of strikes,
    often with uneven spacing and sparse wings. This notebook compares how the common discretisations
    cope with that, and where they break down.

    | # | Method | Idea in one line |
    |---|---|---|
    | 1 | **Derman discrete (DDKZ Appendix A)** | Chords between strikes approximate the log payoff; weights are the slope changes. Equivalently: assume $Q(K)$ is linear between strikes and integrate $Q/K^2$ exactly. |
    | 2 | **VIX / CBOE** | Rectangle rule $\sum \frac{\Delta K_i}{K_i^2}Q_i$ plus the $-\frac1T(\frac F{K_0}-1)^2$ term. |
    | 3 | **Integrated-curvature cells** | Weights $\int_{\text{cell}} dK/K^2 = \frac1{L_i}-\frac1{U_i}$ instead of $\Delta K/K^2$. |
    | 4 | **LP super/sub-replication** | Cheapest dominating and dearest dominated portfolios give model-free bounds. |
    | 5 | Derman + convexity correction | Subtracts the chord bias $\sum_i p_i\,h_i^2/(6T\bar K_i^2)$, with $p_i$ taken from the quotes. |
    | 6 | Smile-interpolated continuous | Spline the implied vols, then integrate continuously (Le Floc'h 2018). |

    Every method is checked against published numbers (DDKZ Table 1; Le Floc'h 2018, Tables 1–3) in
    the **Checks** section. All prices are *forward* (undiscounted) option prices, and the variance is
    annualised. Variance is quoted in "vol²" units ($\times 10^4$, so $20\%$ vol $= 400$), and errors
    in vol points.
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dataclasses import dataclass, field
    from typing import Callable, Optional
    from scipy.optimize import linprog
    from scipy.special import ndtr
    from scipy.interpolate import CubicSpline
    from scipy.integrate import simpson

    return (
        Callable,
        CubicSpline,
        Optional,
        dataclass,
        field,
        go,
        linprog,
        make_subplots,
        mo,
        ndtr,
        np,
        simpson,
    )


@app.cell(hide_code=True)
def theory(mo):
    mo.md(r"""
    ## 1 · The common framework (read this first)

    **Why options can replicate the log payoff.** Any twice-differentiable payoff $g(S)$ decomposes
    exactly into a bond, a forward, and a strip of options weighted by its curvature:

    $$
    g(S)=g(\kappa)+g'(\kappa)(S-\kappa)+\int_0^{\kappa} g''(K)\,(K-S)^+dK+\int_{\kappa}^{\infty} g''(K)\,(S-K)^+dK .
    $$

    This is a second-order Taylor expansion with integral remainder. Each option $(K-S)^+$ or
    $(S-K)^+$ is a "kink" at $K$, and convexity is made by stacking kinks. For variance we need
    $L(S)=-\tfrac{2}{T}\ln(S/F)$, because $\mathbb E[L(S_T)]=\mathbb E[\tfrac1T\int_0^T\sigma_t^2dt]$
    when the price path is continuous. Its curvature is $g''(K)=\tfrac{2}{T K^2}$, so **the option
    density is $2/(TK^2)$**. Low strikes get more weight because a 1-point move matters more, in
    return terms, at a low price.

    **Splitting at a quoted strike $K_0$, not at $F$.** The forward rarely sits on a strike. Take
    $K_0$ as the highest strike $\le F$, use puts at and below $K_0$ and calls at and above it. The
    mismatch between $K_0$ and $F$ is then an *exact* constant (DDKZ eq. 27):

    $$
    K_{\text{var}}=\frac{2}{T}\left[\int_0^{K_0}\frac{P(K)}{K^2}dK+\int_{K_0}^{\infty}\frac{C(K)}{K^2}dK\right]-\frac{2}{T}\Big[\big(\tfrac{F}{K_0}-1\big)-\ln\tfrac{F}{K_0}\Big].
    $$

    The last term comes from put–call parity on $[K_0,F]$: those calls are in the money.
    **The VIX term $-\tfrac1T(F/K_0-1)^2$ is just the 2nd-order Taylor expansion** of this exact
    term ($x-\ln(1+x)\approx x^2/2$). The difference is about $\tfrac{2}{3T}x^3$, which is
    negligible for 30-day SPX but not for weekly options with coarse strikes.

    **Every discrete method is a quadrature rule** for the two integrals: it picks weights $w_i$ and
    prices $K_{\text{var}}\approx\sum_i w_iQ_i-\text{(offset term)}$. Each such rule is also a static
    portfolio (forward + puts + calls), so we can look at it in *payoff space* (does it track $L(S)$?)
    and in *integrand space* (does it integrate $Q/K^2$?). Both views are plotted below.

    **Three identities that organise the comparison** (all checked numerically below):

    1. **Derman = linear-in-strike price integration.** Derman's slope-jump weights equal
       $\tfrac2T\int\varphi_i(K)\,K^{-2}dK$ for the piecewise-linear "hat" functions $\varphi_i$, which
       is exactly the rule you get by assuming $Q(K)$ is linear between strikes. The two differ
       **only at the outermost strikes**: Derman gives them weight $0$, the linear-price rule a
       half-hat $\tfrac2T\big[\tfrac1{K_N}-\tfrac{\ln(K_N/K_{N-1})}{h}\big]$ (mirrored on the put
       side). That changes $K_{\text{var}}$ by about the price of the cheapest wing option times a
       small weight, which is far below the truncation error at the same strike. So the notebook
       treats linear-price integration as a second derivation of Derman's weights, not a separate
       method.
    2. **VIX $\Delta K/K^2$ = trapezoid on $Q/K^2$** in the interior. Integrated-curvature cells
       differ from it only by $O(h^3/K^3)$ kernel terms.
    3. Hence **all piecewise-linear rules share one dominant bias**. Chords over a convex payoff
       overstate it by $\approx h^2/(8K^2)$ at mid-gap, so

    $$
    \text{bias}\;\approx\;\sum_i p_i\,\frac{h_i^2}{6T\bar K_i^2}\qquad\xrightarrow{\text{uniform }h}\qquad\frac{h^2}{6T}\,\mathbb E\big[S_T^{-2}\big]=\frac{h^2}{6TF^2}\,\mathbb E\Big[\big(\tfrac{F}{S_T}\big)^2\Big],
    $$

    where $p_i$ is the risk-neutral probability of gap $i$. The factor $\mathbb E[(F/S_T)^2]$ is
    $e^{3\sigma^2T}\approx1$ under Black–Scholes, so in variance terms the bias is **almost
    independent of volatility**, and the relative error is $\approx(h/F)^2/(6\sigma^2T)$. The meaningful
    measure of "coarse" is $h/(F\sigma\sqrt T)$: the spacing in units of the distribution's width.
    The same \$5 grid is fine for a 1-year index option and terrible for a 1-week single-stock
    option.

    The other error is **truncation**. Outside $[K_{\min},K_{\max}]$ the portfolio is linear while
    $-\ln S$ keeps curving, so missing wings always *understate* the variance (and $-\ln S\to\infty$
    as $S\to0$). Interior gaps push the estimate up and missing wings push it down. Results can look
    accurate only because the two errors cancel.
    """)
    return


@app.cell(hide_code=True)
def style(np):
    # Fixed categorical order (validated palette): colour follows the method, never its rank.
    COLORS = {
        "Derman (DDKZ App. A)": "#2a78d6",
        "VIX / CBOE ΔK/K²": "#eb6834",
        "Integrated-curvature cells": "#eda100",
        "Derman + convexity correction": "#e87ba4",
        "Smile-interpolated continuous": "#008300",
        "LP bounds": "#4a3aa7",
    }
    DASH = {  # secondary encoding so identity never rests on colour alone
        "Derman (DDKZ App. A)": "solid",
        "VIX / CBOE ΔK/K²": "dash",
        "Integrated-curvature cells": "dashdot",
        "Derman + convexity correction": "longdash",
        "Smile-interpolated continuous": "longdashdot",
        "LP bounds": "solid",
    }
    SYMBOL = {
        "Derman (DDKZ App. A)": "circle",
        "VIX / CBOE ΔK/K²": "square",
        "Integrated-curvature cells": "triangle-up",
        "Derman + convexity correction": "x",
        "Smile-interpolated continuous": "star",
        "LP bounds": "hexagon",
    }
    INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9",
               axis="#c3c2b7", surface="#fcfcfb", density="rgba(137,135,129,0.25)")


    def style_fig(fig, title=None, height=430, hover="x unified"):
        """Shared, recessive chart chrome; reserves room for title, legend rows and subplot titles."""
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
            font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12,
                      color=INK["secondary"]),
            legend=dict(orientation="h", yanchor="bottom", y=1 + (28 if sub_titles else 6) / plot_h, x=0,
                        font=dict(size=11)),
            margin=dict(l=64, r=24, t=top, b=bottom), hovermode=hover,
        )
        fig.update_xaxes(gridcolor=INK["grid"], linecolor=INK["axis"], zerolinecolor=INK["axis"])
        fig.update_yaxes(gridcolor=INK["grid"], linecolor=INK["axis"], zerolinecolor=INK["axis"])
        return fig


    def vol_pts(kvar):
        """Annualised variance -> volatility in % points."""
        return 100.0 * np.sqrt(np.maximum(kvar, 0.0))

    return COLORS, DASH, INK, SYMBOL, style_fig, vol_pts


@app.cell(hide_code=True)
def models_md(mo):
    mo.md(r"""
    ## 2 · Market models (the "truth" generators)

    To measure errors we need option prices whose exact fair variance is known.

    * **Black–Scholes, flat vol**: the truth is $\sigma^2$.
    * **DDKZ linear skew** $\Sigma(K)=\Sigma_0-b\,(K-S_F)/S_F$ (the paper's own example, floored at 1%).
      The truth is a dense numerical Carr–Madan integral of the same smile.
    * **Heston** with Le Floc'h's (2018) SPX calibration: a realistic equity skew with a fat left tail.
      The truth is the closed form $\theta+(v_0-\theta)\frac{1-e^{-\kappa T}}{\kappa T}$. Prices come from
      Lewis' Fourier formula $C=F-\frac{\sqrt{FK}}{\pi}\int_0^\infty\mathrm{Re}\big[e^{iu\ln(F/K)}\varphi(u-\tfrac i2)\big]\frac{du}{u^2+1/4}$
      with the "little Heston trap" characteristic function.

    `continuous_kvar(m, Kmin, Kmax)` integrates in $y=\ln(K/F)$, where $dK/K^2=dy/K$. With limits it
    gives the **truncated truth**: exact prices, but only over the quoted range. This splits every
    error into *truncation* (truth − truncated) and *discretisation* (method − truncated).
    """)
    return


@app.cell
def models(Callable, Optional, dataclass, ndtr, np, simpson):
    def bs_price(F, K, T, vol, cp):
        """Undiscounted Black-76 price; cp=+1 call, -1 put."""
        K = np.asarray(K, float)
        vol = np.broadcast_to(np.asarray(vol, float), K.shape)
        sd = vol * np.sqrt(T)
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(F / K) + 0.5 * sd**2) / sd
            d2 = d1 - sd
            px = cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2))
        return np.where(sd > 0, px, np.maximum(cp * (F - K), 0.0))


    def implied_vol(F, K, T, price, cp, lo=1e-4, hi=5.0, iters=80):
        """Vectorised bisection implied vol (robust, derivative-free)."""
        K = np.atleast_1d(np.asarray(K, float))
        price = np.broadcast_to(np.asarray(price, float), K.shape)
        a, b = np.full(K.shape, lo), np.full(K.shape, hi)
        for _ in range(iters):
            mid = 0.5 * (a + b)
            high = bs_price(F, K, T, mid, cp) > price
            b, a = np.where(high, mid, b), np.where(high, a, mid)
        return 0.5 * (a + b)


    @dataclass
    class Market:
        """Option market under the forward measure (undiscounted prices)."""
        name: str
        F: float
        T: float
        r: float
        S0: float
        call: Callable
        put: Callable
        vol: Callable
        truth: Optional[float] = None
        atm_vol: float = 0.2

        def otm(self, K, K0=None):
            K = np.asarray(K, float)
            b = self.F if K0 is None else K0
            return np.where(K < b, self.put(K), self.call(K))

        @property
        def sd(self):
            return self.atm_vol * np.sqrt(self.T)


    def black_scholes_market(S0=100.0, T=1.0, r=0.0, q=0.0, sigma=0.2):
        F = S0 * np.exp((r - q) * T)
        return Market(f"Black–Scholes σ={sigma:.0%}, T={T:.3g}", F, T, r, S0,
                      call=lambda K: bs_price(F, K, T, sigma, 1),
                      put=lambda K: bs_price(F, K, T, sigma, -1),
                      vol=lambda K: np.full(np.shape(K), sigma), truth=sigma**2, atm_vol=sigma)


    def linear_skew_market(S0=100.0, T=0.25, r=0.0, q=0.0, sigma0=0.2, b=0.2, center=None, floor=0.01):
        """DDKZ eq. 30: Σ(K) = Σ0 − b (K − S_c)/S_c (floored)."""
        F = S0 * np.exp((r - q) * T)
        c = F if center is None else center

        def vol(K):
            return np.maximum(sigma0 - b * (np.asarray(K, float) - c) / c, floor)

        mk = Market(f"Linear skew Σ0={sigma0:.0%}, b={b:g}, T={T:.3g}", F, T, r, S0,
                    call=lambda K: bs_price(F, K, T, vol(K), 1),
                    put=lambda K: bs_price(F, K, T, vol(K), -1),
                    vol=vol, atm_vol=float(vol(F)))
        mk.truth = continuous_kvar(mk)
        return mk


    GL_X, GL_W = np.polynomial.legendre.leggauss(400)


    def heston_cf(u, T, v0, kappa, theta, sigma, rho):
        """Characteristic function of ln(S_T/F), 'little Heston trap' form (Albrecher et al.)."""
        iu = 1j * u
        a = kappa - rho * sigma * iu
        d = np.sqrt(a**2 + sigma**2 * (iu + u**2))
        g = (a - d) / (a + d)
        e = np.exp(-d * T)
        C = kappa * theta / sigma**2 * ((a - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
        D = (a - d) / sigma**2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0)


    def heston_market(T=0.986301, S0=2839.19, r=0.0223, q=None,
                      v0=0.001006, kappa=2.4056, theta=0.04264, sigma=0.8121, rho=-0.7588):
        """Default parameters: Le Floc'h (2018) Table 5 (SPX, 23-Jan-2018); q implied by F=2858.41 at T=0.986301."""
        if q is None:
            q = r - np.log(2858.41 / 2839.19) / 0.986301
        F = S0 * np.exp((r - q) * T)
        vbar = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)
        umax = max(60.0, 14.0 / np.sqrt(vbar * T))
        u = 0.5 * umax * (GL_X + 1.0)
        kern = 0.5 * umax * GL_W / (u**2 + 0.25)
        phi = heston_cf(u - 0.5j, T, v0, kappa, theta, sigma, rho)

        def lewis(K):
            K = np.atleast_1d(np.asarray(K, float))
            integ = np.real(np.exp(1j * np.outer(np.log(F / K), u)) * phi) @ kern
            return np.sqrt(F * K) / np.pi * integ

        def call(K):
            Ka = np.asarray(K, float)
            return np.maximum(np.reshape(F - lewis(Ka), Ka.shape), np.maximum(F - Ka, 0.0))

        def put(K):
            Ka = np.asarray(K, float)
            return np.maximum(np.reshape(Ka - lewis(Ka), Ka.shape), np.maximum(Ka - F, 0.0))

        def vol(K):
            Ka = np.atleast_1d(np.asarray(K, float))
            return implied_vol(F, Ka, T, np.where(Ka < F, put(Ka), call(Ka)), np.where(Ka < F, -1, 1))

        return Market(f"Heston (Le Floc'h SPX calibration), T={T:.3g}", F, T, r, S0,
                      call=call, put=put, vol=vol, truth=vbar, atm_vol=float(np.sqrt(vbar)))


    def continuous_kvar(m, Kmin=None, Kmax=None, n=8001, nsd=20.0, price_fn=None):
        """(2/T)[∫_Kmin^F P/K² dK + ∫_F^Kmax C/K² dK] in y = ln(K/F) (dK/K² = dy/K).
        nsd=20 standard deviations is needed for Heston's fat left tail."""
        ylo = -nsd * m.sd - 0.5 * m.sd**2 if Kmin is None else np.log(Kmin / m.F)
        yhi = nsd * m.sd if Kmax is None else np.log(Kmax / m.F)
        put = m.put if price_fn is None else (lambda K: price_fn(K, -1))
        call = m.call if price_fn is None else (lambda K: price_fn(K, 1))
        total = 0.0
        if ylo < 0:
            y = np.linspace(ylo, min(0.0, yhi), n)
            total += simpson(put(m.F * np.exp(y)) / (m.F * np.exp(y)), x=y)
        if yhi > 0:
            y = np.linspace(max(0.0, ylo), yhi, n)
            total += simpson(call(m.F * np.exp(y)) / (m.F * np.exp(y)), x=y)
        return 2.0 / m.T * float(total)


    def inv_sq_moment(m, n=8001, nsd=20.0):
        """E[(F/S_T)²] = 1 + 6F²∫Q(K)/K⁴ dK (Carr–Madan for g = 1/S²), integrated in y = ln(K/F)."""
        total = 0.0
        for a, b, pf in ((-nsd * m.sd, 0.0, m.put), (0.0, nsd * m.sd, m.call)):
            y = np.linspace(a, b, n)
            K = m.F * np.exp(y)
            total += simpson(pf(K) / K**3, x=y)
        return 1.0 + 6.0 * m.F**2 * float(total)


    def density(m, S):
        """Risk-neutral density q(S) = ∂²C/∂K² by central differences."""
        S = np.asarray(S, float)
        h = 1e-3 * m.F
        return np.maximum((m.call(S + h) - 2 * m.call(S) + m.call(S - h)) / h**2, 0.0)

    return (
        Market,
        black_scholes_market,
        bs_price,
        continuous_kvar,
        density,
        heston_market,
        implied_vol,
        inv_sq_moment,
        linear_skew_market,
    )


@app.cell
def grids(np):
    SPX_2019_STRIKES = np.r_[np.arange(1275, 3001, 25), np.arange(3050, 3201, 50),
                             np.arange(3300, 3601, 100)].astype(float)
    SPX_2019_F = 2858.41   # forward of that chain (Le Floc'h 2018, Appendix A)


    def uniform_grid(F, h, lo, hi, offset=0.0):
        """Strikes h apart covering [lo, hi]; one strike sits at F − offset·h (offset∈[0,1) ⇒ F inside a gap)."""
        anchor = F - offset * h
        j = np.arange(int(np.floor((lo - anchor) / h)), int(np.ceil((hi - anchor) / h)) + 1)
        K = anchor + h * j
        return K[(K > 0) & (K >= lo - 1e-9) & (K <= hi + 1e-9)]


    def tiered_grid(F, h_inner, ratio, inner_band, lo, hi, offset=0.0):
        """Exchange-style grid: step h inside |K/F − 1| ≤ inner_band, step ratio·h outside (ratio integer)."""
        K = uniform_grid(F, h_inner, lo, hi, offset)
        j = np.rint((K - (F - offset * h_inner)) / h_inner).astype(int)
        keep = (np.abs(K / F - 1) <= inner_band) | (j % int(ratio) == 0)
        return K[keep]


    def drop_strikes(K, F, prob, seed):
        """Randomly delete strikes (illiquid / unquoted), always keeping the two strikes around F."""
        rng = np.random.default_rng(int(seed))
        K = np.sort(K)
        keep = rng.random(K.size) >= prob
        i0 = np.searchsorted(K, F, side="right") - 1
        keep[max(i0, 0)] = True
        keep[min(i0 + 1, K.size - 1)] = True
        return K[keep]


    def apply_quote_cutoff(m, K, min_price):
        """CBOE-style truncation: walking outwards from F, stop at the first OTM quote below min_price."""
        K = np.sort(np.asarray(K, float))
        if min_price <= 0:
            return K
        q = m.otm(K)
        i0 = np.searchsorted(K, m.F, side="right") - 1
        keep = np.zeros(K.size, bool)
        for idx in (range(i0, -1, -1), range(i0 + 1, K.size)):
            for i in idx:
                if q[i] < min_price and abs(i - i0) > 1:
                    break
                keep[i] = True
        return K[keep]

    return (
        SPX_2019_F,
        SPX_2019_STRIKES,
        apply_quote_cutoff,
        drop_strikes,
        tiered_grid,
        uniform_grid,
    )


@app.cell(hide_code=True)
def methods_md(mo):
    mo.md(r"""
    ## 3 · The replication methods

    Each method returns a `Rep`, a static portfolio
    $\;\Pi(S)=c+a\,(S-F)+\sum_i w^p_i(K_i-S)^+ +\sum_j w^c_j(S-K_j)^+$, whose forward price is the
    variance estimate. The forward leg is $a=-2/(TK_0)$ and the constant $c$ carries the offset term,
    so $\Pi(S)$ can be compared directly with $L(S)=-\tfrac2T\ln(S/F)$.

    **Derman (DDKZ Appendix A).** Replace $f(S)=\tfrac2T[\tfrac{S-K_0}{K_0}-\ln\tfrac S{K_0}]$ by the
    chords through $(K_i,f(K_i))$. A call at $K_0$ creates the first segment's slope, and each further
    strike adds just enough options to change the slope to the next chord's (eq. A7):
    $w_c(K_n)=\frac{f(K_{n+1})-f(K_n)}{K_{n+1}-K_n}-\sum_{i<n}w_c(K_i)$. Puts mirror this; the paper's
    eq. A8 has a typo ($\sum w_c$ should read $\sum w_p$). The outermost strike only closes the last
    chord, so its weight is $0$. *Why it works:* a convex function lies below its chords, so the
    portfolio matches $f$ at every strike and **over-replicates** in between. DDKZ call it a cost that
    "will always exceed or match" the log contract, but that holds *only inside the strike range*.

    **VIX / CBOE.** $\sigma^2=\frac2T\sum_i\frac{\Delta K_i}{K_i^2}Q(K_i)-\frac1T(\frac F{K_0}-1)^2$ with
    $\Delta K_i=\frac{K_{i+1}-K_{i-1}}2$, a full one-sided $\Delta K$ at the two ends, and
    $Q(K_0)=\frac{P+C}2$. This is a midpoint rule that freezes the kernel $1/K^2$ at the node.

    *The same weights, derived from prices (linear-in-strike integration).* If $Q$ is linear on
    $[a,b]$, then
    $\int_a^b\frac{Q}{K^2}dK=Q_a\big[\frac1a-\frac{\ln(b/a)}{b-a}\big]+Q_b\big[\frac{\ln(b/a)}{b-a}-\frac1b\big]$,
    so strike $K_i$ gets $\tfrac2T\big[\frac{\ln(K_i/K_{i-1})}{h_-}-\frac{\ln(K_{i+1}/K_i)}{h_+}\big]$, which is
    exactly Derman's slope jump. Both brackets are positive (the log-mean $\frac{b-a}{\ln(b/a)}$ lies
    between $a$ and $b$), so the weights are positive for any spacing. *Why they coincide:* the price
    of a piecewise-linear payoff $g$ is $\int Q(K)\,g''(K)dK$, and interpolating $Q$ linearly is the
    same integral seen from the other side. Only the outermost strikes differ (Derman: $0$; linear
    price: a half-hat), so this is not carried as a separate method.

    **Integrated-curvature cells.** Give strike $i$ the cell $[L_i,U_i]$ between midpoints (cut at
    $K_0$) and the exact curvature mass $\tfrac2T\int_{L_i}^{U_i}K^{-2}dK=\tfrac2T(\frac1{L_i}-\frac1{U_i})$.
    This fixes the *kernel* error of $\Delta K/K^2$, which is an $O(h^3)$ effect. Watch how little it
    changes the result: the dominant error is in the *price* curve, not the kernel.

    **Derman + convexity correction.** Subtract each gap's expected chord excess,
    $p_i\cdot\text{mean}_{[K_i,K_{i+1}]}(\text{chord}-f)\approx p_i\,h_i^2/(6T\bar K_i^2)$. Here
    $p_i$ is the market-implied probability of the gap, read from **price slopes**: undiscounted
    $P'(K)=\mathbb Q(S_T<K)$ and $1+C'(K)$ on the call side. No model is needed, but it assumes the
    density is roughly flat inside each gap.

    **Smile-interpolated continuous.** Invert implied vols at the quoted strikes, fit a natural
    cubic spline of total variance against $\ln(K/F)$, extrapolate flat vol, and integrate the
    continuous formula. *What it assumes:* the smile is smooth between and beyond the quotes. When that
    holds it removes both the chord bias and most of the truncation; when it fails, nothing warns you.
    """)
    return


@app.cell
def methods(
    CubicSpline,
    Market,
    bs_price,
    continuous_kvar,
    dataclass,
    field,
    implied_vol,
    np,
):
    @dataclass
    class Rep:
        """Static portfolio const + fwd·(S−F) + Σ wp (Kp−S)⁺ + Σ wc (S−Kc)⁺ ; its forward price is kvar."""
        name: str
        Kp: np.ndarray
        wp: np.ndarray
        Kc: np.ndarray
        wc: np.ndarray
        const: float
        fwd: float
        kvar: float
        extra: dict = field(default_factory=dict)

        def payoff(self, S):
            S = np.atleast_1d(np.asarray(S, float))
            out = self.const + self.fwd * (S - self.extra["F"])
            out = out + np.maximum(self.Kp[None, :] - S[:, None], 0.0) @ self.wp
            return out + np.maximum(S[:, None] - self.Kc[None, :], 0.0) @ self.wc


    def split_strikes(K, F):
        """K0 = highest strike ≤ F; puts on strikes ≤ K0, calls on strikes ≥ K0 (K0 in both)."""
        K = np.unique(np.asarray(K, float))
        below = K[K <= F * (1 + 1e-12)]
        if below.size == 0 or K.size < 3 or (K > below.max()).sum() < 1 or below.size < 2:
            raise ValueError("need ≥2 strikes at/below F, ≥1 above, ≥3 in total")
        K0 = below.max()
        return K0, K[K <= K0], K[K >= K0]


    def offset_term(F, K0, T):
        """Exact forward-offset term (2/T)[(F/K0 − 1) − ln(F/K0)]  (DDKZ eq. 27 with S* = K0)."""
        x = F / K0 - 1.0
        return 2.0 / T * (x - np.log1p(x))


    def make_rep(name, m, K0, Kp, wp, Kc, wc, corr, **extra):
        kvar = float(wp @ m.put(Kp) + wc @ m.call(Kc)) - corr
        return Rep(name, Kp, wp, Kc, wc, const=-corr, fwd=-2.0 / (m.T * K0), kvar=kvar,
                   extra={"F": m.F, "K0": K0, **extra})


    def f_log(S, K0, T):
        """Derman's payoff f(S) = (2/T)[(S − S*)/S* − ln(S/S*)] with S* = K0 (zero value & slope at K0)."""
        S = np.asarray(S, float)
        return 2.0 / T * ((S - K0) / K0 - np.log(S / K0))


    def derman_weights(nodes, K0, T):
        """DDKZ eqs. A5–A8 for nodes running outward from K0.
        w_n = |slope of chord n| − Σ_{i<n} w_i ; outermost node closes the last chord ⇒ weight 0."""
        slopes = np.abs(np.diff(f_log(nodes, K0, T)) / np.diff(nodes))
        w = np.zeros(nodes.size)
        w[:-1] = np.diff(np.r_[0.0, slopes])
        return w


    def method_derman(m, K):
        """Derman piecewise-linear (chord) replication, DDKZ Appendix A."""
        K0, Kp, Kc = split_strikes(K, m.F)
        wc = derman_weights(Kc, K0, m.T)
        wp = derman_weights(Kp[::-1], K0, m.T)[::-1]
        return make_rep("Derman (DDKZ App. A)", m, K0, Kp, wp, Kc, wc, offset_term(m.F, K0, m.T))


    def hat_weights(K):
        """∫ φ_i(K)/K² dK for hat functions on ascending K: A = 1/a − ln(b/a)/h, B = ln(b/a)/h − 1/b per gap."""
        h = np.diff(K)
        lr = np.log(K[1:] / K[:-1])
        w = np.zeros(K.size)
        w[:-1] += 1.0 / K[:-1] - lr / h
        w[1:] += lr / h - 1.0 / K[1:]
        return w


    def method_vix(m, K):
        """CBOE: σ² = (2/T) Σ ΔK_i/K_i² Q(K_i) − (1/T)(F/K0 − 1)², full ΔK at the ends, Q(K0) = (P+C)/2."""
        K = np.unique(np.asarray(K, float))
        K0, Kp, Kc = split_strikes(K, m.F)
        dK = np.empty(K.size)
        dK[1:-1] = 0.5 * (K[2:] - K[:-2])
        dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
        w = 2.0 / m.T * dK / K**2
        wp, wc = w[K <= K0].copy(), w[K >= K0].copy()
        wp[-1] *= 0.5
        wc[0] *= 0.5
        return make_rep("VIX / CBOE ΔK/K²", m, K0, Kp, wp, Kc, wc, (m.F / K0 - 1.0) ** 2 / m.T)


    def method_trapezoid(m, K):
        """Trapezoid on Q/K² per side (Le Floc'h 2018 eqs. 17–19). Used for validation only."""
        K0, Kp, Kc = split_strikes(K, m.F)

        def tw(G):
            h = np.diff(G)
            w = np.zeros(G.size)
            w[:-1] += 0.5 * h
            w[1:] += 0.5 * h
            return 2.0 / m.T * w / G**2

        return make_rep("Trapezoid on Q/K²", m, K0, Kp, tw(Kp), Kc, tw(Kc), offset_term(m.F, K0, m.T))


    def cell_bounds(K0, Kp, Kc):
        """Cells between midpoints, cut at K0; the outermost cells extend half a step."""
        mid_p, mid_c = 0.5 * (Kp[1:] + Kp[:-1]), 0.5 * (Kc[1:] + Kc[:-1])
        Lp = np.r_[max(Kp[0] - 0.5 * (Kp[1] - Kp[0]), 1e-9 * K0), mid_p]
        Up = np.r_[mid_p, K0]
        Lc = np.r_[K0, mid_c]
        Uc = np.r_[mid_c, Kc[-1] + 0.5 * (Kc[-1] - Kc[-2])]
        return Lp, Up, Lc, Uc


    def method_cells(m, K):
        """Integrated curvature: w_i = (2/T)∫_{L_i}^{U_i} dK/K² = (2/T)(1/L_i − 1/U_i)."""
        K0, Kp, Kc = split_strikes(K, m.F)
        Lp, Up, Lc, Uc = cell_bounds(K0, Kp, Kc)
        s = 2.0 / m.T
        return make_rep("Integrated-curvature cells", m, K0, Kp, s * (1 / Lp - 1 / Up), Kc,
                        s * (1 / Lc - 1 / Uc), offset_term(m.F, K0, m.T))


    def chord_mean_error(a, b, K0, T):
        """Exact mean over [a, b] of chord(f) − f (uniform weighting); ≈ h²/(6T K̄²)."""
        def F_int(x):  # antiderivative of f
            return 2.0 / T * (x**2 / (2 * K0) - x - (x * np.log(x / K0) - x))
        return 0.5 * (f_log(a, K0, T) + f_log(b, K0, T)) - (F_int(b) - F_int(a)) / (b - a)


    def quote_cdf(m, K, K0):
        """Market-implied CDF at the strikes from price slopes: P'(K) (puts), 1 + C'(K) (calls)."""
        K = np.unique(K)
        h = np.diff(K)
        # each gap uses one instrument at both ends: puts left of K0, calls from K0 upwards
        D = np.where(K[:-1] >= K0, 1.0 + np.diff(m.call(K)) / h, np.diff(m.put(K)) / h)  # ≈ CDF mid-gap
        cdf = np.empty(K.size)
        cdf[1:-1] = (h[1:] * D[:-1] + h[:-1] * D[1:]) / (h[:-1] + h[1:])
        cdf[0] = D[0] - (D[1] - D[0]) * h[0] / (h[0] + h[1])
        cdf[-1] = D[-1] + (D[-1] - D[-2]) * h[-1] / (h[-1] + h[-2])
        return K, np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))


    def method_derman_corrected(m, K):
        """Derman − Σ_i p_i · mean_{gap i}(chord − f), with p_i from quoted price slopes."""
        base = method_derman(m, K)
        K0 = base.extra["K0"]
        Ks, cdf = quote_cdf(m, K, K0)
        p = np.diff(cdf)
        bias_i = p * chord_mean_error(Ks[:-1], Ks[1:], K0, m.T)
        return Rep("Derman + convexity correction", base.Kp, base.wp, base.Kc, base.wc,
                   base.const - bias_i.sum(), base.fwd, base.kvar - bias_i.sum(),
                   extra={**base.extra, "bias_i": bias_i, "p_i": p})


    def method_smile_continuous(m, K):
        """Implied vols at quotes → natural cubic spline of total variance in ln(K/F), flat-vol
        extrapolation → continuous Carr–Madan integral."""
        K = np.unique(np.asarray(K, float))
        iv = implied_vol(m.F, K, m.T, m.otm(K), np.where(K < m.F, -1, 1))
        k = np.log(K / m.F)
        spl = CubicSpline(k, iv**2 * m.T, bc_type="natural")

        def volf(Kx):
            kx = np.clip(np.log(np.asarray(Kx, float) / m.F), k[0], k[-1])
            return np.sqrt(np.maximum(spl(kx), 1e-10) / m.T)

        atm = float(volf(m.F))
        proxy = Market(m.name, m.F, m.T, m.r, m.S0, m.call, m.put, volf, atm_vol=atm)
        kvar = continuous_kvar(proxy, price_fn=lambda Kx, cp: bs_price(m.F, Kx, m.T, volf(Kx), cp))
        K0, Kp, Kc = split_strikes(K, m.F)
        return Rep("Smile-interpolated continuous", Kp, np.zeros(Kp.size), Kc, np.zeros(Kc.size),
                   const=kvar, fwd=0.0, kvar=kvar,
                   extra={"F": m.F, "K0": K0, "volf": volf, "iv": iv, "no_payoff": True})


    METHODS = {
        "Derman (DDKZ App. A)": method_derman,
        "VIX / CBOE ΔK/K²": method_vix,
        "Integrated-curvature cells": method_cells,
        "Derman + convexity correction": method_derman_corrected,
        "Smile-interpolated continuous": method_smile_continuous,
    }
    return (
        METHODS,
        Rep,
        cell_bounds,
        hat_weights,
        method_derman,
        method_trapezoid,
        offset_term,
        split_strikes,
    )


@app.cell(hide_code=True)
def lp_md(mo):
    mo.md(r"""
    ## 4 · Model-free bounds by linear programming

    The instruments are a bond (payoff 1), a forward ($S-F$), and OTM puts and calls at the quoted
    strikes, and the target is $L(S)=-\tfrac2T\ln(S/F)$. Then

    $$
    \overline V=\min_{x}\;c^\top x\;\;\text{s.t.}\;\;\Pi_x(S_j)\ge L(S_j)\;\forall j,
    \qquad
    \underline V=\max_{x}\;c^\top x\;\;\text{s.t.}\;\;\Pi_x(S_j)\le L(S_j)\;\forall j,
    $$

    on a dense grid $S_j\in[S_{lo},S_{hi}]$ that includes every strike. The weights are free in sign.

    *Why it works.* Any portfolio that dominates the payoff must cost at least the payoff's price,
    under **every** model consistent with the quoted prices. By LP duality, $\overline V$ equals the
    largest $\mathbb E^{\mathbb Q}[L]$ over all risk-neutral distributions that reprice the quotes. The
    dual variables *are* that extremal distribution (point masses), which is plotted below.

    *Two facts to look for:*
    1. Inside the strike range the cheapest dominating payoff is **exactly Derman's chord
       interpolant**. For convex $L$, matching at the strikes is enough: a line above $L$ at both ends
       of a gap is above $L$ in between. Derman is the *upper* edge of the model-free interval, not a
       middle estimate.
    2. **The upper bound needs a support floor $S_{lo}$**, because $-\ln S\to\infty$ as $S\to0$ and no
       finite set of options covers that. Push $S_{lo}$ toward $0$ and $\overline V$ grows without
       bound. A variance swap has no finite model-free super-hedge, which is one reason dealers cap
       single-stock variance swaps.

    The gap $\overline V-\underline V$ is the true price uncertainty given only these quotes. Any
    single-number method chooses a point inside it through an implicit smoothness assumption.
    """)
    return


@app.cell
def lp(Rep, linprog, np, split_strikes):
    def lp_bounds(m, K, S_lo=None, S_hi=None, n_grid=1500):
        """Model-free [lower, upper] for E[−(2/T)ln(S/F)] from bond, forward, OTM puts (≤K0) and calls (>K0).
        Solved in moneyness x = S/F with HiGHS. Returns (upper Rep, lower Rep); None if infeasible."""
        F, T = m.F, m.T
        K0, Kp, Kc = split_strikes(K, F)
        Kc = Kc[1:]                                   # put at K0 + calls above: no redundant columns
        S_lo = 0.25 * Kp.min() if S_lo is None else S_lo
        S_hi = 2.0 * Kc.max() if S_hi is None else S_hi
        xs = np.unique(np.r_[np.geomspace(S_lo / F, S_hi / F, n_grid), Kp / F, Kc / F])
        xs = xs[(xs >= S_lo / F * (1 - 1e-12)) & (xs <= S_hi / F * (1 + 1e-12))]
        target = -2.0 / T * np.log(xs)
        A = np.column_stack([np.ones_like(xs), xs - 1.0,
                             np.maximum(Kp[None, :] / F - xs[:, None], 0.0),
                             np.maximum(xs[:, None] - Kc[None, :] / F, 0.0)])
        cost = np.r_[1.0, 0.0, m.put(Kp) / F, m.call(Kc) / F]
        free = [(None, None)] * A.shape[1]
        up = linprog(cost, A_ub=-A, b_ub=-target, bounds=free, method="highs")
        lo = linprog(-cost, A_ub=A, b_ub=target, bounds=free, method="highs")

        def to_rep(res, name):
            if res.status != 0:
                return None
            x, npu = res.x, Kp.size
            return Rep(name, Kp, x[2:2 + npu] / F, Kc, x[2 + npu:] / F, const=x[0], fwd=x[1] / F,
                       kvar=float(cost @ x),
                       extra={"F": F, "K0": K0, "S_grid": xs * F, "dual": np.abs(res.ineqlin.marginals),
                              "S_lo": S_lo, "S_hi": S_hi})

        return to_rep(up, "LP super-replication (upper)"), to_rep(lo, "LP sub-replication (lower)")

    return (lp_bounds,)


@app.cell(hide_code=True)
def checks_md(mo):
    mo.md(r"""
    ## 5 · Checks against the literature

    Before trusting any chart, the implementation must reproduce the published numbers.

    * **Le Floc'h (2018), Tables 1–3**: Black–Scholes, $F=100$, $T=1$, strikes 60–140 step 10.
    * **DDKZ (1999), Table 1**: linear skew, $S_0=100$, $r=5\%$, three months, strikes every 5.
      It reproduces to four decimals with $T=90/365$ and the put strip running down to 45 (the
      table's weights imply both).
    * **Algebraic identities** from section 1.
    * **LP sanity checks.**
    """)
    return


@app.cell
def checks(
    SPX_2019_STRIKES,
    black_scholes_market,
    continuous_kvar,
    hat_weights,
    heston_market,
    inv_sq_moment,
    linear_skew_market,
    lp_bounds,
    method_derman,
    method_trapezoid,
    mo,
    np,
    offset_term,
    split_strikes,
    uniform_grid,
    vol_pts,
):
    def run_checks():
        rows = []

        def add(name, expected, got, ok, note=""):
            ok = None if ok is None else bool(ok)
            rows.append({"check": name, "expected": expected, "got": got,
                         "status": "✅ pass" if ok is True else ("❌ FAIL" if ok is False else "ℹ️ info"),
                         "note": note})

        bs = black_scholes_market(100, 1.0, 0.0, 0.0, 0.2)
        Kt = np.array([60.0, 100.0, 150.0])
        par = np.max(np.abs(bs.call(Kt) - bs.put(Kt) - (bs.F - Kt)))
        add("BS put–call parity", "0", f"{par:.1e}", par < 1e-10)

        hb = heston_market(T=1.0, S0=100, r=0.0, q=0.0, v0=0.04, kappa=1.0, theta=0.04, sigma=1e-3, rho=0.0)
        dev = np.max(np.abs(hb.call(Kt) - bs.call(Kt)))
        add("Heston → BS as vol-of-vol → 0", "≈ 0", f"{dev:.1e}", dev < 1e-4)

        hm = heston_market()
        hc = continuous_kvar(hm)
        add("Heston: continuous integral = closed form (vol²)", f"{hm.truth*1e4:.3f}", f"{hc*1e4:.3f}",
            abs(hc - hm.truth) * 1e4 < 0.01,
            f"discounted {hm.truth*1e4*np.exp(-hm.r*hm.T):.2f} vs Le Floc'h Table 6: 261.44")

        K_lf = np.arange(60.0, 141.0, 10.0)
        m10 = black_scholes_market(100, 1.0, 0.0, 0.0, 0.10)
        d10, t10 = method_derman(m10, K_lf), method_trapezoid(m10, K_lf)
        exp_d = np.array([0, 41.24, 31.50, 24.85, 10.72, 9.38, 16.60, 13.94, 11.87, 0])
        exp_t = np.array([27.78, 40.82, 31.25, 24.69, 10.0, 10.0, 16.53, 13.89, 11.83, 5.10])
        got_d = np.round(np.r_[d10.wp, d10.wc] * 1e4, 2)
        got_t = np.round(np.r_[t10.wp, t10.wc] * 1e4, 2)
        add("Le Floc'h T1: Derman weights ×10⁴", "0, 41.24 … 11.87, 0", ", ".join(f"{v:g}" for v in got_d),
            bool(np.allclose(got_d, exp_d, atol=0.011)))
        add("Le Floc'h T1: trapezoid weights ×10⁴", "27.78, 40.82 … 5.10", ", ".join(f"{v:g}" for v in got_t),
            bool(np.allclose(got_t, exp_t, atol=0.011)))
        add("Le Floc'h T2 (σ=10%): Derman vol", "10.8264", f"{vol_pts(d10.kvar):.4f}", abs(vol_pts(d10.kvar) - 10.8264) < 2e-3)
        add("Le Floc'h T2 (σ=10%): trapezoid vol", "10.7986", f"{vol_pts(t10.kvar):.4f}", abs(vol_pts(t10.kvar) - 10.7986) < 2e-3)
        m40 = black_scholes_market(100, 1.0, 0.0, 0.0, 0.40)
        add("Le Floc'h T3 (σ=40%): Derman / trapezoid / truncated", "36.51 / 37.32 / 37.18",
            f"{vol_pts(method_derman(m40, K_lf).kvar):.2f} / {vol_pts(method_trapezoid(m40, K_lf).kvar):.2f}"
            f" / {vol_pts(continuous_kvar(m40, 60, 140)):.2f}",
            abs(vol_pts(method_derman(m40, K_lf).kvar) - 36.51) < 0.006
            and abs(vol_pts(method_trapezoid(m40, K_lf).kvar) - 37.32) < 0.006
            and abs(vol_pts(continuous_kvar(m40, 60, 140)) - 37.18) < 0.006)

        md = linear_skew_market(S0=100, T=90 / 365, r=0.05, sigma0=0.2, b=0.2, center=100.0)
        dd = method_derman(md, np.arange(45.0, 150.1, 5.0))
        pv = np.exp(-md.r * md.T)
        pi_cp = float(dd.wp @ md.put(dd.Kp) + dd.wc @ md.call(dd.Kc)) * pv * 1e4
        add("DDKZ T1: option portfolio cost Π_CP (vol²)", "419.8671", f"{pi_cp:.4f}", abs(pi_cp - 419.8671) < 0.01)
        add("DDKZ T1: K_var = (20.467)²", "20.467", f"{vol_pts(dd.kvar):.4f}", abs(vol_pts(dd.kvar) - 20.467) < 1e-3,
            f"theoretical fair vol {vol_pts(md.truth):.3f} — DDKZ Fig. 5: ≈ √402")

        rng = np.random.default_rng(7)
        Ku = np.sort(np.r_[100.0, 100 + np.cumsum(rng.uniform(1, 12, 12)), 100 - np.cumsum(rng.uniform(1, 12, 12))])
        mu = black_scholes_market(100, 0.5, 0.0, 0.0, 0.25)
        dU = method_derman(mu, Ku)
        _, Kpu, Kcu = split_strikes(Ku, mu.F)
        hp, hc = 2.0 / mu.T * hat_weights(Kpu), 2.0 / mu.T * hat_weights(Kcu)
        inner = np.allclose(dU.wp[1:], hp[1:]) and np.allclose(dU.wc[:-1], hc[:-1])
        add("Derman ≡ linear-price hats (uneven grid, all but outermost)", "identical",
            f"max |Δw| = {max(np.abs(dU.wp[1:]-hp[1:]).max(), np.abs(dU.wc[:-1]-hc[:-1]).max()):.1e}", bool(inner))
        hw = hat_weights(Ku)
        add("Hat weights: Σw = 1/K₀ − 1/K_N ; Σ wK = ln(K_N/K₀)", "exact",
            f"{abs(hw.sum() - (1/Ku[0] - 1/Ku[-1])):.1e} ; {abs(hw @ Ku - np.log(Ku[-1]/Ku[0])):.1e}",
            abs(hw.sum() - (1/Ku[0] - 1/Ku[-1])) < 1e-12 and abs(hw @ Ku - np.log(Ku[-1]/Ku[0])) < 1e-12)

        x, T_ = 0.05, 1 / 52
        diff_ = offset_term(1 + x, 1.0, T_) - x**2 / T_
        add("Exact offset − VIX term ≈ −(2/3T)x³ (x=5%, T=1w)", f"{-2/(3*T_)*x**3*1e4:.2f} vol²",
            f"{diff_*1e4:.2f} vol²", abs(diff_ / (-2 / (3 * T_) * x**3) - 1) < 0.15)

        up, lo = lp_bounds(m10, K_lf, S_lo=10.0, S_hi=400.0)
        add("LP: lower ≤ truth ≤ upper (BS 10%)", f"{vol_pts(lo.kvar):.3f} ≤ 10 ≤ {vol_pts(up.kvar):.3f}", "",
            lo.kvar <= m10.truth <= up.kvar)
        add("LP upper ≈ Derman when wings are irrelevant", f"{vol_pts(d10.kvar):.4f}", f"{vol_pts(up.kvar):.4f}",
            abs(vol_pts(up.kvar) - vol_pts(d10.kvar)) < 0.01)
        ups = [lp_bounds(m40, K_lf, S_lo=s, S_hi=400.0)[0].kvar for s in (20.0, 5.0, 1.0)]
        add("LP upper rises as S_lo → 0 (σ=40%, S_lo 20 → 5 → 1)", "increasing",
            " → ".join(f"{vol_pts(v):.2f}" for v in ups), ups[0] < ups[1] < ups[2])

        mb = black_scholes_market(100, 1.0, 0.0, 0.0, 0.20)
        Kb = uniform_grid(100.0, 1.0, 100 * np.exp(-8 * 0.2), 100 * np.exp(8 * 0.2))
        bias = method_derman(mb, Kb).kvar - mb.truth
        thumb = 1.0 / 6e4 * inv_sq_moment(mb)
        add("Rule of thumb: bias ≈ h²/(6TF²)·E[(F/S)²] (h=1%·F, σ=20%)", f"{thumb*1e4:.4f} vol²", f"{bias*1e4:.4f} vol²",
            abs(bias / thumb - 1) < 0.02, f"E[(F/S)²] = {inv_sq_moment(mb):.4f} (= e^(3σ²T) = {np.exp(0.12):.4f})")

        sp = method_trapezoid(hm, SPX_2019_STRIKES).kvar * np.exp(-hm.r * hm.T) * 1e4
        add("Le Floc'h T6: SPX discrete trapezoid (discounted vol²)", "246.52", f"{sp:.2f}", None,
            f"Not reproduced. The exact truncated integral is "
            f"{continuous_kvar(hm, SPX_2019_STRIKES.min(), SPX_2019_STRIKES.max())*np.exp(-hm.r*hm.T)*1e4:.2f}, "
            "so a trapezoid on these strikes cannot fall to 246.5; the paper's uneven-grid convention is unstated.")
        return rows


    check_rows = run_checks()
    n_fail = sum(r["status"].startswith("❌") for r in check_rows)
    mo.vstack([
        mo.md(f"**{len(check_rows) - n_fail - sum(r['status'].startswith('ℹ️') for r in check_rows)} passed, "
              f"{n_fail} failed** (ℹ️ = informational)."),
        mo.ui.table(check_rows, selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def explorer_md(mo):
    mo.md(r"""
    ## 6 · Explorer

    Choose a **scenario preset** to reproduce a paper's setting, or **Custom** to build your own
    market and strike grid. Every chart below reacts to these controls.

    * **Spacing $h$** is a % of $F$. The diagnostic that matters is $h/(F\sigma\sqrt T)$, shown in the
      summary line: above roughly $0.5$ the chord bias becomes visible, and above about $1.5$ the
      methods start to disagree in sign.
    * **F offset** puts the forward inside a strike gap, which exercises the $K_0$ correction.
    * **Min quote** drops wing options priced below a tick, CBOE-style. It is the usual way
      truncation enters in practice.
    * **Grid types.** *Tiered* = fine strikes within $\pm1\sigma\sqrt T$ and coarse outside, like a
      listed chain. *Listed SPX* = the real Jan-2019 chain (25/50/100 steps), rescaled to your $F$.
      *Missing* = uniform with random gaps.
    """)
    return


@app.cell(hide_code=True)
def controls(METHODS, mo):
    scenario_ui = mo.ui.dropdown(
        options=["Custom (controls below)", "DDKZ 1999 · Table 1 (skew, 3m, strikes every 5)",
                 "Le Floc'h 2018 · BS σ=10%, strikes 60–140 step 10",
                 "Le Floc'h 2018 · BS σ=40%, strikes 60–140 step 10",
                 "Le Floc'h 2018 · Heston SPX, real listed strikes"],
        value="Custom (controls below)", label="Scenario preset")
    model_ui = mo.ui.dropdown(options=["Black–Scholes (flat)", "DDKZ linear skew", "Heston (SPX calibration)"],
                              value="DDKZ linear skew", label="Model")
    T_ui = mo.ui.dropdown(options={"1 week": 1 / 52, "2 weeks": 2 / 52, "1 month": 1 / 12, "3 months": 0.25,
                                   "6 months": 0.5, "1 year": 1.0, "2 years": 2.0}, value="3 months", label="Maturity")
    vol_ui = mo.ui.slider(5, 80, step=1, value=20, label="ATM vol (%)", show_value=True)
    skew_ui = mo.ui.slider(0.0, 0.6, step=0.05, value=0.2, label="Skew b", show_value=True)
    grid_ui = mo.ui.dropdown(options=["Uniform", "Tiered (exchange-style)", "Listed SPX Jan-2019 (rescaled)",
                                      "Uniform with missing strikes"], value="Uniform", label="Strike grid")
    h_ui = mo.ui.slider(steps=[0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25], value=5,
                        label="Spacing h (% of F)", show_value=True)
    range_ui = mo.ui.slider(1.0, 10.0, step=0.5, value=4.0, label="Strike range (± σ√T, log)", show_value=True)
    offset_ui = mo.ui.slider(0.0, 0.95, step=0.05, value=0.4, label="F offset inside gap", show_value=True)
    ratio_ui = mo.ui.slider(steps=[1, 2, 3, 4, 5, 10], value=4, label="Tiered: outer/inner step", show_value=True)
    drop_ui = mo.ui.slider(0.0, 0.7, step=0.05, value=0.3, label="Missing: drop prob.", show_value=True)
    seed_ui = mo.ui.number(start=0, stop=10_000, step=1, value=1, label="Seed")
    cutoff_ui = mo.ui.dropdown(options={"none": 0.0, "0.1 bp of F": 1e-5, "0.5 bp": 5e-5, "1 bp": 1e-4,
                                        "5 bp": 5e-4, "10 bp": 1e-3}, value="none", label="Min quote")
    methods_ui = mo.ui.multiselect(options=list(METHODS), value=list(METHODS), label="Methods shown")
    slo_ui = mo.ui.dropdown(options={"0.5% of F": 0.005, "1%": 0.01, "2%": 0.02, "5%": 0.05, "10%": 0.10,
                                     "20%": 0.20, "30%": 0.30}, value="5%", label="LP support floor S_lo")
    shi_ui = mo.ui.dropdown(options={"2×F": 2.0, "3×F": 3.0, "5×F": 5.0, "10×F": 10.0}, value="3×F",
                            label="LP support cap S_hi")
    mo.vstack([
        scenario_ui,
        mo.hstack([model_ui, T_ui, vol_ui, skew_ui], justify="start", wrap=True),
        mo.hstack([grid_ui, h_ui, range_ui, offset_ui], justify="start", wrap=True),
        mo.hstack([ratio_ui, drop_ui, seed_ui, cutoff_ui], justify="start", wrap=True),
        mo.hstack([methods_ui, slo_ui, shi_ui], justify="start", wrap=True),
    ])
    return (
        T_ui,
        cutoff_ui,
        drop_ui,
        grid_ui,
        h_ui,
        methods_ui,
        model_ui,
        offset_ui,
        range_ui,
        ratio_ui,
        scenario_ui,
        seed_ui,
        shi_ui,
        skew_ui,
        slo_ui,
        vol_ui,
    )


@app.cell(hide_code=True)
def scenario(
    METHODS,
    SPX_2019_F,
    SPX_2019_STRIKES,
    T_ui,
    apply_quote_cutoff,
    black_scholes_market,
    continuous_kvar,
    cutoff_ui,
    drop_strikes,
    drop_ui,
    grid_ui,
    h_ui,
    heston_market,
    linear_skew_market,
    lp_bounds,
    mo,
    model_ui,
    np,
    offset_ui,
    range_ui,
    ratio_ui,
    scenario_ui,
    seed_ui,
    shi_ui,
    skew_ui,
    slo_ui,
    tiered_grid,
    uniform_grid,
    vol_pts,
    vol_ui,
):
    def build_market(model, T, vol, skew):
        if model == "Black–Scholes (flat)":
            return black_scholes_market(100.0, T, 0.0, 0.0, vol)
        if model == "DDKZ linear skew":
            return linear_skew_market(100.0, T, 0.0, 0.0, sigma0=vol, b=skew)
        return heston_market(T=T)


    def build_grid(m, kind, h_rel, nsd, offset, ratio, drop, seed):
        F, sd = m.F, m.sd
        lo, hi = F * np.exp(-nsd * sd), F * np.exp(nsd * sd)
        h = h_rel * F
        if kind == "Tiered (exchange-style)":
            return tiered_grid(F, h, ratio, sd, lo, hi, offset)
        if kind == "Listed SPX Jan-2019 (rescaled)":
            return SPX_2019_STRIKES * F / SPX_2019_F
        K = uniform_grid(F, h, lo, hi, offset)
        if kind == "Uniform with missing strikes":
            K = drop_strikes(K, F, drop, seed)
        return K


    preset = scenario_ui.value
    if preset.startswith("DDKZ"):
        mkt = linear_skew_market(S0=100, T=90 / 365, r=0.05, sigma0=0.2, b=0.2, center=100.0)
        strikes = np.arange(45.0, 150.1, 5.0)
    elif "σ=10%" in preset:
        mkt, strikes = black_scholes_market(100, 1.0, 0.0, 0.0, 0.10), np.arange(60.0, 141.0, 10.0)
    elif "σ=40%" in preset:
        mkt, strikes = black_scholes_market(100, 1.0, 0.0, 0.0, 0.40), np.arange(60.0, 141.0, 10.0)
    elif "Heston" in preset:
        mkt, strikes = heston_market(), SPX_2019_STRIKES.copy()
    else:
        mkt = build_market(model_ui.value, T_ui.value, vol_ui.value / 100, skew_ui.value)
        strikes = build_grid(mkt, grid_ui.value, h_ui.value / 100, range_ui.value, offset_ui.value,
                             ratio_ui.value, drop_ui.value, seed_ui.value)
    strikes = apply_quote_cutoff(mkt, strikes, cutoff_ui.value * mkt.F)

    reps = {name: fn(mkt, strikes) for name, fn in METHODS.items()}
    kv_truth = mkt.truth
    kv_trunc = continuous_kvar(mkt, strikes.min(), strikes.max())
    lp_up, lp_lo = lp_bounds(mkt, strikes, S_lo=slo_ui.value * mkt.F, S_hi=shi_ui.value * mkt.F)
    K0_s = reps["Derman (DDKZ App. A)"].extra["K0"]
    gaps = np.diff(np.sort(strikes))
    near = gaps[np.abs(np.sort(strikes)[:-1] / mkt.F - 1) < mkt.sd] if gaps.size else gaps
    h_atm = float(np.median(near)) if near.size else float(np.median(gaps))
    spacing_ratio = h_atm / (mkt.F * mkt.sd)

    mo.callout(mo.md(
        f"**{mkt.name}** · F = {mkt.F:,.2f}, K₀ = {K0_s:,.2f} · {strikes.size} strikes in "
        f"[{strikes.min():,.1f}, {strikes.max():,.1f}] · σ√T = {mkt.sd:.3f} · "
        f"ATM spacing h = {h_atm:,.3g} ⇒ **h/(Fσ√T) = {spacing_ratio:.2f}** · "
        f"truth = **{vol_pts(kv_truth):.3f}** vol pts · truncated-to-range = {vol_pts(kv_trunc):.3f}"
    ), kind="info")
    return (
        K0_s,
        kv_trunc,
        kv_truth,
        lp_lo,
        lp_up,
        mkt,
        reps,
        spacing_ratio,
        strikes,
    )


@app.cell(hide_code=True)
def market_md(mo):
    mo.md(r"""
    ### 6.1 · Where is the probability, and where are the strikes?

    The variance integral is $\frac2T\int Q(K)/K^2\,dK$. Its integrand peaks at the forward,
    where OTM option values are largest, and decays into the wings. Low strikes keep weight because
    of the $1/K^2$ kernel. **Discretisation error comes from gaps where the integrand is curved and
    the probability is concentrated** (near ATM). **Truncation error comes from integrand mass outside
    the quoted range**, and it is mostly in the put wing under a skew. Compare the strike rug with
    the integrand and density panels.
    """)
    return


@app.cell(hide_code=True)
def chart_market(
    COLORS,
    INK,
    density,
    go,
    make_subplots,
    mkt,
    np,
    reps,
    strikes,
    style_fig,
):
    def chart_market():
        F, sd = mkt.F, mkt.sd
        lo = min(strikes.min() * 0.85, F * np.exp(-4.5 * sd))
        hi = max(strikes.max() * 1.15, F * np.exp(4.0 * sd))
        Kd = np.geomspace(lo, hi, 500)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.36, 0.38, 0.26],
                            subplot_titles=("Implied volatility (%)", "Variance integrand (2/T)·Q(K)/K²  (vol² per unit strike)",
                                            "Risk-neutral density q(S)"))
        fig.add_trace(go.Scatter(x=Kd, y=100 * mkt.vol(Kd), name="model smile", line=dict(color=INK["primary"], width=2)), 1, 1)
        smile = reps["Smile-interpolated continuous"].extra
        fig.add_trace(go.Scatter(x=Kd, y=100 * smile["volf"](Kd), name="spline through quotes (method 7)",
                                 line=dict(color=COLORS["Smile-interpolated continuous"], width=2, dash="longdashdot")), 1, 1)
        fig.add_trace(go.Scatter(x=strikes, y=100 * smile["iv"], mode="markers", name="quoted strikes",
                                 marker=dict(color=INK["secondary"], size=8, symbol="line-ns-open", line=dict(width=2))), 1, 1)
        integ = 2 / mkt.T * mkt.otm(Kd) / Kd**2 * 1e4
        fig.add_trace(go.Scatter(x=Kd, y=integ, name="integrand", line=dict(color=INK["primary"], width=2),
                                 fill="tozeroy", fillcolor=INK["density"], showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=strikes, y=2 / mkt.T * mkt.otm(strikes) / strikes**2 * 1e4, mode="markers",
                                 marker=dict(color=INK["secondary"], size=7), name="at strikes", showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=Kd, y=density(mkt, Kd), line=dict(color=INK["secondary"], width=2),
                                 fill="tozeroy", fillcolor=INK["density"], name="density", showlegend=False), 3, 1)
        for r in (1, 2, 3):
            fig.add_vline(x=mkt.F, line=dict(color=INK["muted"], width=1, dash="dot"), row=r, col=1)
            fig.add_vrect(x0=lo, x1=strikes.min(), fillcolor="rgba(227,73,72,0.06)", line_width=0, row=r, col=1)
            fig.add_vrect(x0=strikes.max(), x1=hi, fillcolor="rgba(227,73,72,0.06)", line_width=0, row=r, col=1)
        fig.update_xaxes(title_text="strike K / terminal price S", row=3, col=1)
        return style_fig(fig, "Market view — shaded red = outside the quoted strike range (truncated)", height=720)


    chart_market()
    return


@app.cell(hide_code=True)
def payoff_md(mo):
    mo.md(r"""
    ### 6.2 · Payoff space: how well does each portfolio track $L(S)=-\frac2T\ln\frac SF$?

    Each method's weights define a piecewise-linear payoff, with kinks only at the strikes. The top
    panel overlays them on the target and the middle panel shows the error $\Pi(S)-L(S)$ in vol².
    **What to look for:**
    * *Derman*: the error is $\ge0$ humps between strikes (chords above a convex
      curve) and goes negative beyond the outermost strikes, where the portfolio runs linear but
      $-\ln S$ keeps bending.
    * *VIX / cells*: these don't pass through the strike nodes, so the error oscillates around zero
      inside the range.
    * Error only costs money where there is probability. The bottom panel is the density, so a big
      error in a region with no mass is harmless.

    The corrected-Derman portfolio is Derman's shifted down by a constant (the bias estimate). The
    smile-interpolated method has no finite portfolio, so it isn't shown here.
    """)
    return


@app.cell(hide_code=True)
def chart_payoff(
    COLORS,
    DASH,
    INK,
    density,
    go,
    make_subplots,
    methods_ui,
    mkt,
    np,
    reps,
    strikes,
    style_fig,
):
    def chart_payoff():
        kmin, kmax = strikes.min(), strikes.max()
        pad = 0.08 * (kmax - kmin)
        Si = np.linspace(max(kmin - pad, 1e-6 * kmin), kmax + pad, 1500)
        Sw = np.geomspace(kmin * 0.35, kmax * 1.8, 1500)
        Li, Lw = -2.0 / mkt.T * np.log(Si / mkt.F), -2.0 / mkt.T * np.log(Sw / mkt.F)
        fig = make_subplots(rows=4, cols=1, vertical_spacing=0.07, row_heights=[0.3, 0.26, 0.26, 0.18],
                            subplot_titles=("Payoff over the strike range (vol²)",
                                            "Error Π(S) − L(S) inside the strike range (vol²): chord humps",
                                            "Error Π(S) − L(S) including the wings (vol²): truncation",
                                            "density q(S)"))
        fig.add_trace(go.Scatter(x=Si, y=Li * 1e4, name="target L(S)", line=dict(color=INK["primary"], width=3)), 1, 1)
        inner = (Si >= kmin) & (Si <= kmax)
        zoom = 1e-12
        for name in methods_ui.value:
            rp = reps[name]
            if rp.extra.get("no_payoff"):
                continue
            Pi, Pw = rp.payoff(Si), rp.payoff(Sw)
            style = dict(color=COLORS[name], width=2, dash=DASH[name])
            fig.add_trace(go.Scatter(x=Si, y=Pi * 1e4, name=name, line=style, legendgroup=name), 1, 1)
            fig.add_trace(go.Scatter(x=Si, y=(Pi - Li) * 1e4, name=name, line=style, legendgroup=name,
                                     showlegend=False), 2, 1)
            fig.add_trace(go.Scatter(x=Sw, y=(Pw - Lw) * 1e4, name=name, line=style, legendgroup=name,
                                     showlegend=False), 3, 1)
            zoom = max(zoom, np.abs((Pi - Li)[inner]).max() * 1e4)
        fig.add_trace(go.Scatter(x=Sw, y=density(mkt, Sw), line=dict(color=INK["secondary"], width=1.5),
                                 fill="tozeroy", fillcolor=INK["density"], showlegend=False, name="density"), 4, 1)
        for r in (2, 3):
            fig.add_hline(y=0, line=dict(color=INK["axis"], width=1), row=r, col=1)
        for k in strikes:
            fig.add_vline(x=k, line=dict(color=INK["grid"], width=1), row=2, col=1)
        for r in (3, 4):
            fig.add_vrect(x0=Sw[0], x1=kmin, fillcolor="rgba(227,73,72,0.06)", line_width=0, row=r, col=1)
            fig.add_vrect(x0=kmax, x1=Sw[-1], fillcolor="rgba(227,73,72,0.06)", line_width=0, row=r, col=1)
        fig.update_yaxes(range=[-1.15 * zoom, 1.15 * zoom], row=2, col=1)
        for r in (1, 2):
            fig.update_xaxes(range=[Si[0], Si[-1]], row=r, col=1)
        fig.update_xaxes(range=[Sw[0], Sw[-1]], row=3, col=1)
        fig.update_xaxes(range=[Sw[0], Sw[-1]], title_text="terminal price S (shaded = beyond the quoted strikes)",
                         row=4, col=1)
        return style_fig(fig, "Replicating payoffs vs the log target (grey verticals = strikes)", height=940,
                         hover="closest")


    chart_payoff()
    return


@app.cell(hide_code=True)
def integrand_md(mo):
    mo.md(r"""
    ### 6.3 · Integrand space: what each rule assumes about $Q(K)$ between strikes

    The same weights, seen as a quadrature, imply a reconstruction of the integrand
    $\frac2T Q(K)/K^2$. Each rule's estimate is the area under its reconstruction, minus the offset
    term. *Derman* draws $Q$ as straight lines between strikes with the exact kernel (that is the
    linear-in-strike reading of its weights), except that on the outermost gap on each side it lets
    $Q$ fall to zero at the last strike (its zero "terminator" weight). *VIX* uses flat steps at node values times $1/K_i^2$. *Cells* uses flat $Q$
    times the true $1/K^2$. The shaded area between the truth (black) and the selected rule is that
    rule's discretisation error. With wide gaps the largest pieces sit at the ATM peak, where the
    straight lines cut across the curved top.

    (The black curve uses the $K_0$ split: puts at and below $K_0$, calls above. It has a small jump
    of $(F-K_0)/K_0^2$ at $K_0$, which the offset term accounts for.)
    """)
    return


@app.cell(hide_code=True)
def integrand_pick(mo):
    integrand_ui = mo.ui.dropdown(options=["Derman (DDKZ App. A)", "VIX / CBOE ΔK/K²",
                                            "Integrated-curvature cells"], value="Derman (DDKZ App. A)",
                                   label="Shade error for")
    integrand_ui
    return (integrand_ui,)


@app.cell(hide_code=True)
def chart_integrand(
    COLORS,
    DASH,
    INK,
    K0_s,
    cell_bounds,
    go,
    integrand_ui,
    mkt,
    np,
    strikes,
    style_fig,
):
    def reconstruct(name, Kx):
        """Integrand (2/T)Q(K)/K² implied by each quadrature rule, on points Kx."""
        K0 = K0_s
        s = 2.0 / mkt.T
        Ks = np.unique(strikes)
        side_put = Kx <= K0
        out = np.zeros_like(Kx)
        Kp, Kc = Ks[Ks <= K0], Ks[Ks >= K0]
        Qp, Qc = mkt.put(Kp), mkt.call(Kc)
        if name == "Derman (DDKZ App. A)":
            Qp_, Qc_ = Qp.copy(), Qc.copy()
            Qp_[0], Qc_[-1] = 0.0, 0.0   # zero terminator weight ⇔ Q falls to 0 at the outermost strike
            inp = side_put & (Kx >= Kp[0])
            inc = ~side_put & (Kx <= Kc[-1])
            out[inp] = np.interp(Kx[inp], Kp, Qp_) / Kx[inp] ** 2
            out[inc] = np.interp(Kx[inc], Kc, Qc_) / Kx[inc] ** 2
            return s * out
        if name == "Integrated-curvature cells":
            Lp, Up, Lc, Uc = cell_bounds(K0, Kp, Kc)
            for Lb, Ub, Q in ((Lp, Up, Qp), (Lc, Uc, Qc)):
                for a, b, q in zip(Lb, Ub, Q):
                    sel = (Kx >= a) & (Kx < b)
                    out[sel] = q / Kx[sel] ** 2
            return s * out
        # VIX: flat cells between midpoints at Q_i/K_i² (full-width end cells, (P+C)/2 at K0)
        Q = np.where(Ks < K0, mkt.put(Ks), mkt.call(Ks))
        Q[Ks == K0] = 0.5 * (mkt.put(K0) + mkt.call(K0))
        mids = 0.5 * (Ks[1:] + Ks[:-1])
        edges = np.r_[Ks[0] - 0.5 * (Ks[1] - Ks[0]), mids, Ks[-1] + 0.5 * (Ks[-1] - Ks[-2])]
        for a, b, q, k in zip(edges[:-1], edges[1:], Q, Ks):
            sel = (Kx >= a) & (Kx < b)
            out[sel] = q / k**2
        return s * out


    def chart_integrand():
        lo, hi = strikes.min() * 0.9, strikes.max() * 1.1
        Kx = np.unique(np.r_[np.geomspace(lo, hi, 2500), strikes, K0_s - 1e-9 * K0_s])
        truth_y = 2 / mkt.T * mkt.otm(Kx, K0=K0_s + 1e-12 * K0_s) / Kx**2 * 1e4
        fig = go.Figure()
        sel = integrand_ui.value
        for name in ["Derman (DDKZ App. A)", "VIX / CBOE ΔK/K²", "Integrated-curvature cells"]:
            if name == sel:
                continue
            fig.add_trace(go.Scatter(x=Kx, y=reconstruct(name, Kx) * 1e4, name=name, line_shape="linear",
                                     line=dict(color=COLORS[name], width=1.2, dash=DASH[name]), opacity=0.7))
        fig.add_trace(go.Scatter(x=Kx, y=truth_y, name="true integrand", line=dict(color=INK["primary"], width=2.5)))
        fig.add_trace(go.Scatter(x=Kx, y=reconstruct(sel, Kx) * 1e4, name=f"{sel} (shaded)", fill="tonexty",
                                 fillcolor="rgba(74,58,167,0.18)", line=dict(color=COLORS[sel], width=2.5)))
        fig.add_trace(go.Scatter(x=strikes, y=2 / mkt.T * np.where(strikes <= K0_s, mkt.put(strikes), mkt.call(strikes)) / strikes**2 * 1e4,
                                 mode="markers", name="strikes", marker=dict(color=INK["secondary"], size=7)))
        fig.add_vline(x=mkt.F, line=dict(color=INK["muted"], width=1, dash="dot"), annotation_text="F")
        fig.update_xaxes(title_text="strike K")
        fig.update_yaxes(title_text="(2/T)·Q(K)/K²  (vol² per unit strike)")
        return style_fig(fig, "Integrand reconstructions — shaded = discretisation error of the selected rule", height=520)


    chart_integrand()
    return


@app.cell(hide_code=True)
def weights_md(mo):
    mo.md(r"""
    ### 6.4 · The weights themselves

    The top panel shows each method's total weight at each strike (put + call at $K_0$), divided by
    the naive density $\frac2T\Delta K_i/K_i^2$. A value of $1$ means "same as VIX". The interior is
    close to $1$ for all rules, and the methods differ at the **edges** (Derman: $0$; VIX: $1$ with
    a full one-sided $\Delta K$) and wherever the **spacing changes**,
    because $\Delta K_i$ is no longer the right width there. The bottom panel shows each strike's
    **contribution** $w_iQ_i$ to the price (DDKZ Table 1's last column). The cost is concentrated
    near the money, even though the deep wings carry large *weights*.
    """)
    return


@app.cell(hide_code=True)
def chart_weights(
    COLORS,
    DASH,
    INK,
    SYMBOL,
    go,
    make_subplots,
    methods_ui,
    mkt,
    np,
    reps,
    strikes,
    style_fig,
):
    def chart_weights():
        Ks = np.unique(strikes)
        dK = np.empty(Ks.size)
        dK[1:-1] = 0.5 * (Ks[2:] - Ks[:-2])
        dK[0], dK[-1] = Ks[1] - Ks[0], Ks[-1] - Ks[-2]
        naive = 2.0 / mkt.T * dK / Ks**2
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Weight ÷ (2/T)·ΔK/K²", "Contribution w·Q  (vol²)"))
        for name in methods_ui.value:
            rp = reps[name]
            if rp.extra.get("no_payoff") or name.startswith("Derman +"):
                continue
            w = np.zeros(Ks.size)
            c = np.zeros(Ks.size)
            for Kset, wset, pf in ((rp.Kp, rp.wp, mkt.put), (rp.Kc, rp.wc, mkt.call)):
                idx = np.searchsorted(Ks, Kset)
                np.add.at(w, idx, wset)
                np.add.at(c, idx, wset * pf(Kset))
            mk = dict(color=COLORS[name], size=8, symbol=SYMBOL[name])
            ln = dict(color=COLORS[name], width=1.5, dash=DASH[name])
            fig.add_trace(go.Scatter(x=Ks, y=w / naive, mode="lines+markers", name=name, marker=mk, line=ln,
                                     legendgroup=name), 1, 1)
            fig.add_trace(go.Scatter(x=Ks, y=c * 1e4, mode="lines+markers", name=name, marker=mk, line=ln,
                                     legendgroup=name, showlegend=False), 2, 1)
        fig.add_hline(y=1, line=dict(color=INK["axis"], width=1), row=1, col=1)
        fig.add_vline(x=mkt.F, line=dict(color=INK["muted"], width=1, dash="dot"))
        fig.update_xaxes(title_text="strike K", row=2, col=1)
        return style_fig(fig, "Replication weights and price contributions", height=640)


    chart_weights()
    return


@app.cell(hide_code=True)
def results_md(mo):
    mo.md(r"""
    ### 6.5 · Prices, errors and their decomposition

    On the chart, the black line is the truth. The dotted grey line is the **truncated truth**:
    exact prices, integrated only over the quoted range. The violet band is the **LP model-free
    interval**. Any method's error splits as

    $$\underbrace{(\text{method}-\text{truncated})}_{\text{discretisation}}+\underbrace{(\text{truncated}-\text{truth})}_{\text{truncation}\;\le 0}.$$

    Watch for a method that looks accurate only because a positive chord bias cancels a negative
    truncation. A method that lands **outside the LP interval** gives a price that no distribution on
    the support could produce while also repricing every quote. This is truncation bias showing; the
    Le Floc'h σ = 40% and Heston SPX presets both do it.
    """)
    return


@app.cell(hide_code=True)
def chart_results(
    COLORS,
    INK,
    SYMBOL,
    go,
    kv_trunc,
    kv_truth,
    lp_lo,
    lp_up,
    methods_ui,
    mo,
    reps,
    style_fig,
    vol_pts,
):
    def results_rows():
        rows = []
        for name in methods_ui.value:
            rp = reps[name]
            rows.append({"method": name, "fair vol (%)": round(vol_pts(rp.kvar), 4),
                         "K_var (vol²)": round(rp.kvar * 1e4, 3),
                         "error (vol pts)": round(vol_pts(rp.kvar) - vol_pts(kv_truth), 4),
                         "discretisation (vol²)": round((rp.kvar - kv_trunc) * 1e4, 3),
                         "truncation (vol²)": round((kv_trunc - kv_truth) * 1e4, 3)})
        return rows


    def chart_results():
        names = list(methods_ui.value)[::-1]
        fig = go.Figure()
        if lp_up is not None and lp_lo is not None:
            lo_v, up_v = vol_pts(lp_lo.kvar), vol_pts(lp_up.kvar)
            fig.add_trace(go.Scatter(x=[lo_v, up_v], y=["LP model-free interval"] * 2, mode="lines+markers+text",
                                     text=[f"{lo_v:.3f}", f"{up_v:.3f}"], textposition=["middle left", "middle right"],
                                     line=dict(color=COLORS["LP bounds"], width=6),
                                     marker=dict(color=COLORS["LP bounds"], size=12, symbol="line-ns", line=dict(width=3)),
                                     showlegend=False, hovertemplate="LP bound %{x:.4f}<extra></extra>"))
        fig.add_vline(x=vol_pts(kv_truth), line=dict(color=INK["primary"], width=2), annotation_text="truth",
                      annotation_position="bottom right")
        fig.add_vline(x=vol_pts(kv_trunc), line=dict(color=INK["muted"], width=2, dash="dot"),
                      annotation_text="truncated", annotation_position="top right")
        for n in names:
            v = vol_pts(reps[n].kvar)
            fig.add_trace(go.Scatter(x=[v], y=[n], mode="markers+text", text=[f"{v:.3f}"], textposition="middle right",
                                     marker=dict(color=COLORS[n], size=14, symbol=SYMBOL[n],
                                                 line=dict(color=INK["surface"], width=2)),
                                     name=n, showlegend=False, hovertemplate=f"{n}: %{{x:.4f}}<extra></extra>"))
        fig.update_xaxes(title_text="fair volatility (%)")
        style_fig(fig, "Estimated fair volatility by method", height=130 + 46 * (len(names) + 1), hover="closest")
        fig.update_layout(margin=dict(l=230))
        return fig


    lp_txt = (f"LP bounds: [{vol_pts(lp_lo.kvar):.3f}, {vol_pts(lp_up.kvar):.3f}] vol pts"
              if (lp_up is not None and lp_lo is not None) else "LP infeasible for this support")
    mo.vstack([chart_results(), mo.md(lp_txt), mo.ui.table(results_rows(), selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def bias_md(mo):
    mo.md(r"""
    ### 6.6 · Which gaps cost you? Bias attribution for Derman

    Derman's error is exactly $\mathbb E[\Pi(S_T)-L(S_T)]=\int(\Pi-L)\,q\,dS$, so it splits by region:
    one term per strike gap and one per tail. The bars give the **exact** per-gap contribution from
    the model density. The markers give the **market-only estimate**
    $p_i\cdot\overline{(\text{chord}-f)}_i\approx p_i h_i^2/(6T\bar K_i^2)$ that the convexity
    correction subtracts. *Why this matters for uneven grids:* a gap contributes in proportion to
    its squared width **times the probability it carries**. A coarse wing gap with little mass
    costs almost nothing in discretisation, while a single missing near-the-money strike can
    dominate. The two tail bars are negative: that is truncation. When gaps are a large fraction of
    $F\sigma\sqrt T$, the density is no longer flat inside a gap, and the estimate (markers) starts to
    diverge from the exact value (bars).
    """)
    return


@app.cell(hide_code=True)
def chart_bias(
    COLORS,
    INK,
    density,
    go,
    kv_truth,
    mkt,
    np,
    reps,
    strikes,
    style_fig,
):
    def bias_attribution():
        rp = reps["Derman (DDKZ App. A)"]
        Ks = np.unique(strikes)
        gx, gw = np.polynomial.legendre.leggauss(16)

        def integral(a, b):
            S = 0.5 * (b - a)[:, None] * gx[None, :] + 0.5 * (a + b)[:, None]
            val = (rp.payoff(S.ravel()) + 2 / mkt.T * np.log(S.ravel() / mkt.F)) * density(mkt, S.ravel())
            return (val.reshape(S.shape) * gw[None, :]).sum(1) * 0.5 * (b - a)

        exact = integral(Ks[:-1], Ks[1:])
        lo_edges = np.geomspace(mkt.F * np.exp(-20 * mkt.sd), Ks[0], 60)
        hi_edges = np.geomspace(Ks[-1], mkt.F * np.exp(20 * mkt.sd), 60)
        left = integral(lo_edges[:-1], lo_edges[1:]).sum()
        right = integral(hi_edges[:-1], hi_edges[1:]).sum()
        est = reps["Derman + convexity correction"].extra["bias_i"]
        return Ks, exact, est, left, right


    def chart_bias():
        Ks, exact, est, left, right = bias_attribution()
        mids, widths = 0.5 * (Ks[1:] + Ks[:-1]), np.diff(Ks)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mids, y=exact * 1e4, width=widths * 0.92, name="exact per-gap contribution",
                             marker=dict(color=COLORS["Derman (DDKZ App. A)"], line=dict(width=0))))
        fig.add_trace(go.Scatter(x=mids, y=est * 1e4, mode="markers", name="market estimate p·mean(chord−f)",
                                 marker=dict(color=COLORS["Derman + convexity correction"], size=9, symbol="x")))
        span = Ks[-1] - Ks[0]
        fig.add_trace(go.Bar(x=[Ks[0] - 0.06 * span, Ks[-1] + 0.06 * span], y=[left * 1e4, right * 1e4],
                             width=[0.05 * span] * 2, name="tails (truncation)",
                             marker=dict(color=INK["muted"], line=dict(width=0)),
                             text=[f"{left*1e4:.2f}", f"{right*1e4:.2f}"], textposition="outside"))
        fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
        total = exact.sum() + left + right
        fig.update_xaxes(title_text="strike gap")
        fig.update_yaxes(title_text="contribution to K_var error (vol²)")
        style_fig(fig, f"Derman error by region — gaps {exact.sum()*1e4:+.3f}, tails {(left+right)*1e4:+.3f}, "
                       f"total {total*1e4:+.3f} vol² (Derman − truth = {(reps['Derman (DDKZ App. A)'].kvar - kv_truth)*1e4:+.3f})",
                  height=470, hover="closest")
        return fig


    chart_bias()
    return


@app.cell(hide_code=True)
def lp_view_md(mo):
    mo.md(r"""
    ### 6.7 · LP bounds: the cheapest super-hedge and the dearest sub-hedge

    **Top:** $\Pi-L$ for the two optimal portfolios inside the strike range. The super-hedge
    (solid) touches zero at every strike and bulges in between: it **is** Derman's chord
    interpolant, drawn over Derman's dotted line. The sub-hedge (dashed) stays below by following
    tangent lines, which touch $L$ *between* strikes.

    **Middle:** the same over the whole support $[S_{lo},S_{hi}]$. Beyond the outermost put, the
    super-hedge has to buy extra optionality to stay above $-\ln S$ all the way down to $S_{lo}$.
    This tail insurance is the expensive part of the upper bound.

    **Bottom:** the **extremal risk-neutral distributions** (LP duals), aggregated by gap and compared
    with the model's probabilities. Both reprice every quoted option exactly. The upper bound's
    measure puts **all its mass on the strikes**, where chords and curve coincide, plus a tiny sliver
    at the support floor $S_{lo}$ (typically $\sim10^{-4}$, invisible here) that is worth many vol²
    because $-\ln S_{lo}$ is huge. The lower bound's measure has **no mass on the strikes**: it sits
    inside the gaps, at the tangency points.

    **Last chart:** the upper bound as a function of the support floor. It has no finite limit as
    $S_{lo}\to0$.
    """)
    return


@app.cell(hide_code=True)
def chart_lp(
    COLORS,
    INK,
    density,
    go,
    kv_truth,
    lp_bounds,
    lp_lo,
    lp_up,
    make_subplots,
    mkt,
    mo,
    np,
    reps,
    shi_ui,
    strikes,
    style_fig,
    vol_pts,
):
    def chart_lp():
        if lp_up is None or lp_lo is None:
            return mo.md("LP infeasible for this support.")
        S = lp_up.extra["S_grid"]
        L = -2 / mkt.T * np.log(S / mkt.F)
        Ks = np.unique(strikes)
        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09, row_heights=[0.34, 0.33, 0.33],
                            subplot_titles=("Π(S) − L(S) inside the strike range (vol²): super sits on Derman's chords",
                                            "Π(S) − L(S) over the whole support (vol²): the super-hedge buys the tails",
                                            "Extremal measures vs model: probability per region"))
        der = reps["Derman (DDKZ App. A)"]
        inner = (S >= Ks[0]) & (S <= Ks[-1])
        zoom = 1e-12
        for rp, nm, dash in ((lp_up, "LP super (upper)", "solid"), (lp_lo, "LP sub (lower)", "dash"), (der, "Derman", "dot")):
            col = COLORS["Derman (DDKZ App. A)"] if nm == "Derman" else COLORS["LP bounds"]
            err = (rp.payoff(S) - L) * 1e4
            zoom = max(zoom, np.abs(err[inner]).max())
            for r in (1, 2):
                fig.add_trace(go.Scatter(x=S, y=err, name=nm, legendgroup=nm, showlegend=(r == 1),
                                         line=dict(color=col, width=2 if nm != "Derman" else 1.5, dash=dash)), r, 1)
        for r in (1, 2):
            fig.add_hline(y=0, line=dict(color=INK["axis"], width=1), row=r, col=1)
        fig.update_yaxes(range=[-1.15 * zoom, 1.15 * zoom], row=1, col=1)
        fig.update_xaxes(range=[Ks[0], Ks[-1]], row=1, col=1)
        edges = np.unique(np.r_[lp_up.extra["S_lo"], Ks[:-1] + 0.5 * np.diff(Ks), lp_up.extra["S_hi"], Ks[0], Ks[-1]])
        cen = 0.5 * (edges[1:] + edges[:-1])

        def agg(masses):
            return np.histogram(S, bins=edges, weights=masses)[0]

        model_p = np.array([np.trapezoid(density(mkt, g), g) for g in (np.linspace(a, b, 40) for a, b in zip(edges[:-1], edges[1:]))])
        fig.add_trace(go.Bar(x=cen, y=model_p, width=np.diff(edges) * 0.9, name="model", marker_color=INK["axis"]), 3, 1)
        fig.add_trace(go.Bar(x=cen, y=agg(lp_up.extra["dual"]), width=np.diff(edges) * 0.45, name="upper-bound measure",
                             marker_color=COLORS["LP bounds"]), 3, 1)
        fig.add_trace(go.Bar(x=cen, y=agg(lp_lo.extra["dual"]), width=np.diff(edges) * 0.25, name="lower-bound measure",
                             marker_color="#9085e9"), 3, 1)
        wide = [lp_up.extra["S_lo"], min(lp_up.extra["S_hi"], Ks[-1] * 1.6)]
        fig.update_xaxes(range=wide, row=2, col=1)
        fig.update_xaxes(range=wide, title_text="terminal price S", row=3, col=1)
        fig.update_layout(barmode="overlay")
        return style_fig(fig, f"LP bounds [{vol_pts(lp_lo.kvar):.3f}, {vol_pts(lp_up.kvar):.3f}] vs truth {vol_pts(kv_truth):.3f} (vol pts)",
                         height=900, hover="closest")


    def chart_lp_floor():
        floors = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45])
        floors = floors[floors * mkt.F < strikes.min()]
        ups = [lp_bounds(mkt, strikes, S_lo=f * mkt.F, S_hi=shi_ui.value * mkt.F, n_grid=600)[0] for f in floors]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=floors * 100, y=[vol_pts(u.kvar) if u else np.nan for u in ups], mode="lines+markers",
                                 name="LP upper bound", line=dict(color=COLORS["LP bounds"], width=2), marker=dict(size=8)))
        fig.add_hline(y=vol_pts(kv_truth), line=dict(color=INK["primary"], width=1.5), annotation_text="truth")
        fig.add_hline(y=vol_pts(reps["Derman (DDKZ App. A)"].kvar), line=dict(color=COLORS["Derman (DDKZ App. A)"], dash="dot"),
                      annotation_text="Derman", annotation_position="bottom right")
        fig.update_xaxes(type="log", title_text="support floor S_lo (% of F)", autorange="reversed")
        fig.update_yaxes(title_text="upper bound (vol pts)")
        return style_fig(fig, "Upper bound vs support floor (no finite limit as S_lo → 0)", height=380, hover="closest")


    mo.vstack([chart_lp(), chart_lp_floor()])
    return


@app.cell(hide_code=True)
def lp_weights_md(mo):
    mo.md(r"""
    #### The LP portfolios as option weights

    Each bound is the price of a portfolio you can hold: the LP's solution vector *is* the bond,
    forward, put and call positions. The chart and table compare them with Derman's weights, strike by
    strike. **What to look for:**

    * **Super-hedge = Derman + two tail positions.** Inside the strike range the weights are exactly
      Derman's. All three portfolios are shown in one basis (bond, forward, puts at and below $K_0$, calls
      above), so Derman's call at $K_0$ is rewritten by put–call parity as a put plus forward and bond.
      In that basis Derman and the super-hedge also hold the same forward.
      Only the two outermost strikes differ. Derman gives them weight $0$, so beyond them its payoff
      just continues the last chord, and that line falls below $-\ln S$ further out. The LP instead extends the outermost chords to the support
      edges:
      $$w_p(K_{\min})=\text{slope}[K_{\min},K_{1}]-\text{slope}[S_{lo},K_{\min}],\qquad
      w_c(K_{\max})=\text{slope}[K_{\max},S_{hi}]-\text{slope}[K_{N-1},K_{\max}],$$
      with chord slopes of $L$. As $S_{lo}\to0$ the put weight explodes, which is why the upper bound
      needs a floor.
    * **Sub-hedge = tangent lines.** Between neighbouring strikes the payoff is a line tangent to $L$,
      and adjacent tangents must meet exactly at a strike. That ties the tangency points together,
      so the weights zig-zag around Derman's. No tail positions are needed, because tangents to a
      convex payoff stay below it everywhere.
    * **Neither portfolio depends on the option prices.** The weights come from the strike grid and
      the support alone; the prices only set the cost (bottom panel).

    *As hedges:* the portfolio plus the usual daily forward position $\frac{2}{T S_t}$ super-replicates
    (upper) or sub-replicates (lower) the variance swap, provided paths are continuous and $S_T$ stays
    inside $[S_{lo},S_{hi}]$.
    """)
    return


@app.cell(hide_code=True)
def chart_lp_weights(
    COLORS,
    INK,
    SYMBOL,
    go,
    lp_lo,
    lp_up,
    make_subplots,
    mkt,
    mo,
    np,
    reps,
    strikes,
    style_fig,
    vol_pts,
):
    def lp_weight_rows():
        """Per-strike weights of Derman and both LP portfolios in one basis: bond, forward, puts at and
        below K0, calls above K0. A call at K0 is rewritten by parity, C(K0) = P(K0) + (S − F) + (F − K0)."""
        Ks = np.unique(strikes)
        K0 = lp_up.extra["K0"]
        ports = {"Derman": reps["Derman (DDKZ App. A)"], "LP upper": lp_up, "LP lower": lp_lo}
        W, C, legs = {}, {}, {}
        for nm, rp in ports.items():
            w = np.zeros(Ks.size)
            for Kset, wset in ((rp.Kp, rp.wp), (rp.Kc, rp.wc)):
                np.add.at(w, np.searchsorted(Ks, Kset), wset)
            wc0 = rp.wc[rp.Kc == K0].sum()
            legs[nm] = (rp.const + wc0 * (mkt.F - K0), rp.fwd + wc0)
            W[nm], C[nm] = w, w * mkt.otm(Ks, K0=K0)
        return Ks, ports, W, C, legs


    def chart_lp_weights():
        if lp_up is None or lp_lo is None:
            return mo.md("LP infeasible for this support.")
        Ks, ports, W, C, legs = lp_weight_rows()
        dK = np.empty(Ks.size)
        dK[1:-1] = 0.5 * (Ks[2:] - Ks[:-2])
        dK[0], dK[-1] = Ks[1] - Ks[0], Ks[-1] - Ks[-2]
        naive = 2.0 / mkt.T * dK / Ks**2
        style = {"Derman": (COLORS["Derman (DDKZ App. A)"], "dot", SYMBOL["Derman (DDKZ App. A)"]),
                 "LP upper": (COLORS["LP bounds"], "solid", "circle"),
                 "LP lower": ("#9085e9", "dash", "diamond")}
        ratio = {nm: W[nm] / naive for nm in W}
        inner = np.r_[ratio["Derman"], ratio["LP lower"], ratio["LP upper"][1:-1]]
        top = 1.3 * inner.max()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Weight ÷ (2/T)·ΔK/K²  (▲ = off-scale, value printed)",
                                            "Cost of each position w·Q  (vol²)"))
        for nm in W:
            col, dash, sym = style[nm]
            y = ratio[nm]
            off = y > top
            fig.add_trace(go.Scatter(x=Ks, y=np.minimum(y, top), mode="lines+markers", name=nm, legendgroup=nm,
                                     line=dict(color=col, width=2, dash=dash),
                                     marker=dict(color=col, size=np.where(off, 13, 9), symbol=np.where(off, "triangle-up", sym)),
                                     customdata=y, hovertemplate="K=%{x}<br>ratio %{customdata:.3f}<extra>" + nm + "</extra>"), 1, 1)
            for k, v in zip(Ks[off], y[off]):
                fig.add_annotation(x=k, y=top, text=f"{v:.1f}×", showarrow=False, yshift=16, row=1, col=1,
                                   font=dict(color=INK["primary"], size=12))
            fig.add_trace(go.Scatter(x=Ks, y=C[nm] * 1e4, mode="lines+markers", name=nm, legendgroup=nm, showlegend=False,
                                     line=dict(color=col, width=2, dash=dash), marker=dict(color=col, size=9, symbol=sym)), 2, 1)
        fig.add_hline(y=1, line=dict(color=INK["axis"], width=1), row=1, col=1)
        fig.add_vline(x=mkt.F, line=dict(color=INK["muted"], width=1, dash="dot"))
        fig.update_yaxes(range=[min(0.0, inner.min() * 1.1), top * 1.12], row=1, col=1)
        fig.update_xaxes(title_text="strike K", row=2, col=1)
        fig = style_fig(fig, f"Replicating portfolios: Derman vs LP super/sub-hedge (support [{lp_up.extra['S_lo']:.3g}, {lp_up.extra['S_hi']:.3g}])",
                        height=620, hover="closest")

        rows = [{"position": "bond (×1e4)", **{nm: round(float(legs[nm][0]) * 1e4, 3) for nm in ports}},
                {"position": "forward (×1e4)", **{nm: round(float(legs[nm][1]) * 1e4, 3) for nm in ports}}]
        K0 = lp_up.extra["K0"]
        for i, k in enumerate(Ks):
            kind = "put" if k <= K0 else "call"
            rows.append({"position": f"{kind} {k:g} (×1e4)", **{nm: round(float(W[nm][i]) * 1e4, 3) for nm in W}})
        rows.append({"position": "price (vol pts)", **{nm: round(float(vol_pts(rp.kvar)), 4) for nm, rp in ports.items()}})
        return mo.vstack([fig, mo.ui.table(rows, selection=None, page_size=40, label="Weights per unit notional of the variance contract")])


    chart_lp_weights()
    return


@app.cell(hide_code=True)
def sweep_md(mo):
    mo.md(r"""
    ### 6.8 · Spacing sweep (DDKZ Fig. 5, extended to coarse grids)

    This holds the market fixed and varies a uniform grid's spacing, with the x-axis in units of the
    distribution width $h/(F\sigma\sqrt T)$. The range is $\pm$ the slider value in $\sigma\sqrt T$,
    widened when needed so that at least two strikes sit on each side of $F$. DDKZ's Figure 5 covers only $\Delta K\le5$ (about $1$
    on this axis); here the sweep runs to $3$. **What to look for:**

    * All piecewise-linear rules follow the rule of thumb $\sigma^2+\frac{h^2}{6TF^2}\mathbb E[(F/S)^2]$
      (black dashed) while $h\lesssim F\sigma\sqrt T$.
    * Beyond that the strip resolves only a few strikes inside $\pm1\sigma$. The bias saturates and
      then jumps around as $K_0$ and the edges move relative to the mass, and the rules separate.
    * The convexity-corrected and smile-interpolated methods stay near zero until gaps exceed about
      $1\sigma$.
    * The LP band widens like $h^2$: that is the price uncertainty that is *really* there.
    * At very fine spacing the error flattens onto the *truncation* floor (grey dotted line). Only a
      wider strike range removes it.

    The bottom panel shows the absolute error in log–log axes, where the slope 2 of the $h^2$ law is
    visible.
    """)
    return


@app.cell(hide_code=True)
def sweep_controls(mo):
    sweep_range_ui = mo.ui.slider(2.0, 12.0, step=0.5, value=8.0, label="Sweep strike range (± σ√T)", show_value=True)
    sweep_lp_ui = mo.ui.checkbox(value=True, label="include LP band")
    mo.hstack([sweep_range_ui, sweep_lp_ui], justify="start")
    return sweep_lp_ui, sweep_range_ui


@app.cell(hide_code=True)
def chart_sweep(
    COLORS,
    DASH,
    INK,
    METHODS,
    SYMBOL,
    continuous_kvar,
    go,
    inv_sq_moment,
    lp_bounds,
    make_subplots,
    methods_ui,
    mkt,
    np,
    offset_ui,
    shi_ui,
    slo_ui,
    style_fig,
    sweep_lp_ui,
    sweep_range_ui,
    uniform_grid,
    vol_pts,
):
    def grid_errors(m, K, with_lp=False, names=None):
        """Signed vol-point errors of every method on strike set K (NaN if the grid is unusable)."""
        out = {}
        for name in (names or METHODS):
            try:
                out[name] = vol_pts(METHODS[name](m, K).kvar) - vol_pts(m.truth)
            except ValueError:
                out[name] = np.nan
        if with_lp:
            try:
                u, l = lp_bounds(m, K, S_lo=slo_ui.value * m.F, S_hi=shi_ui.value * m.F, n_grid=600)
                out["LP upper"] = vol_pts(u.kvar) - vol_pts(m.truth) if u else np.nan
                out["LP lower"] = vol_pts(l.kvar) - vol_pts(m.truth) if l else np.nan
            except ValueError:
                out["LP upper"] = out["LP lower"] = np.nan
        try:
            out["truncated"] = vol_pts(continuous_kvar(m, K.min(), K.max(), n=2001)) - vol_pts(m.truth)
        except ValueError:
            out["truncated"] = np.nan
        return out


    def run_sweep():
        F, sd = mkt.F, mkt.sd
        ratios = np.geomspace(0.05, 3.0, 22)
        lo, hi = F * np.exp(-sweep_range_ui.value * sd), F * np.exp(sweep_range_ui.value * sd)
        res = [grid_errors(mkt, uniform_grid(F, r * F * sd, min(lo, F - 2.5 * r * F * sd), max(hi, F + 2.5 * r * F * sd),
                                             offset_ui.value), sweep_lp_ui.value) for r in ratios]
        return ratios, res


    def chart_sweep():
        ratios, res = run_sweep()
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                            subplot_titles=("Signed error (vol pts)", "|variance error| (vol²), log–log"))
        if sweep_lp_ui.value:
            fig.add_trace(go.Scatter(x=ratios, y=[r["LP upper"] for r in res], line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"), 1, 1)
            fig.add_trace(go.Scatter(x=ratios, y=[r["LP lower"] for r in res], fill="tonexty", line=dict(width=0),
                                     fillcolor="rgba(74,58,167,0.15)", name="LP model-free band"), 1, 1)
        ism = inv_sq_moment(mkt)
        thumb = vol_pts(mkt.truth + (ratios * mkt.F * mkt.sd) ** 2 / (6 * mkt.T * mkt.F**2) * ism) - vol_pts(mkt.truth)
        fig.add_trace(go.Scatter(x=ratios, y=thumb, name="rule of thumb h²/(6TF²)·E[(F/S)²]", line=dict(color=INK["primary"], dash="dash", width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=ratios, y=[r["truncated"] for r in res], name="truncation only",
                                 line=dict(color=INK["muted"], dash="dot", width=2)), 1, 1)
        tv = mkt.truth
        for name in methods_ui.value:
            y = np.array([r[name] for r in res])
            style = dict(color=COLORS[name], width=2, dash=DASH[name])
            mk = dict(symbol=SYMBOL[name], size=7, color=COLORS[name])
            fig.add_trace(go.Scatter(x=ratios, y=y, name=name, line=style, marker=mk, mode="lines+markers", legendgroup=name), 1, 1)
            verr = np.abs((y / 100 + np.sqrt(tv)) ** 2 - tv) * 1e4
            fig.add_trace(go.Scatter(x=ratios, y=np.maximum(verr, 1e-6), line=style, marker=mk, mode="lines+markers",
                                     legendgroup=name, showlegend=False, name=name), 2, 1)
        trunc_v = np.abs((np.array([r["truncated"] for r in res]) / 100 + np.sqrt(tv)) ** 2 - tv) * 1e4
        fig.add_trace(go.Scatter(x=ratios, y=np.maximum(trunc_v, 1e-6), showlegend=False, name="truncation only",
                                 line=dict(color=INK["muted"], dash="dot", width=2)), 2, 1)
        fig.add_trace(go.Scatter(x=ratios, y=(ratios * mkt.F * mkt.sd) ** 2 / (6 * mkt.T * mkt.F**2) * ism * 1e4, showlegend=False,
                                 line=dict(color=INK["primary"], dash="dash", width=2), name="h²/(6TF²)"), 2, 1)
        fig.add_hline(y=0, line=dict(color=INK["axis"], width=1), row=1, col=1)
        ticks = dict(tickvals=[0.05, 0.1, 0.2, 0.5, 1, 2, 3], ticktext=["0.05", "0.1", "0.2", "0.5", "1", "2", "3"])
        fig.update_xaxes(type="log", row=1, col=1, **ticks)
        fig.update_xaxes(type="log", title_text="strike spacing h / (F σ√T)", row=2, col=1, **ticks)
        fig.update_yaxes(type="log", dtick=1, row=2, col=1)
        return style_fig(fig, f"Error vs spacing — {mkt.name}", height=780, hover="closest")


    chart_sweep()
    return (grid_errors,)


@app.cell(hide_code=True)
def stress_md(mo):
    mo.md(r"""
    ### 6.9 · Uneven-spacing stress tests

    These use the current market and the controls *h*, range, offset and drop probability.

    1. **Tiered grid.** Strikes every $h$ within $\pm1\sigma\sqrt T$ and every $m\cdot h$ outside.
       Because the wings carry little probability, coarse wings barely move the discretisation
       error. The error grows only once the outer step is large enough to move the outermost
       strikes inwards (truncation) or to misplace mass near $\pm1\sigma$.
    2. **Random missing strikes.** 60 random patterns at the chosen drop probability. The spread
       shows how sensitive each rule is to *where* the holes fall. A hole next to $F$ is far more
       damaging than one in the wing. Rules that use the *actual* gap widths (Derman, cells)
       cope better than ones whose implicit cell no longer matches the gap.
    3. **Forward position inside a gap.** This moves $F$ across one gap. The exact offset term keeps
       the piecewise-linear rules flat; VIX's quadratic approximation drifts by $\approx-\frac2{3T}x^3$,
       which shows up for short maturities with coarse strikes (try *1 week* with $h=5\%$).
    4. **Strike range.** Truncation error against the range half-width, in $\sigma\sqrt T$ units.
       Discrete rules underprice once the range is narrower than about $\pm4\sigma$. Skewed and
       fat-tailed (Heston) markets need much wider put wings.
    """)
    return


@app.cell(hide_code=True)
def chart_stress(
    COLORS,
    DASH,
    INK,
    METHODS,
    SYMBOL,
    drop_strikes,
    drop_ui,
    go,
    grid_errors,
    h_ui,
    methods_ui,
    mkt,
    mo,
    np,
    offset_term,
    offset_ui,
    range_ui,
    style_fig,
    tiered_grid,
    uniform_grid,
    vol_pts,
):
    STRESS_NAMES = [n for n in METHODS if n != "Smile-interpolated continuous"] + ["Smile-interpolated continuous"]


    def lines_chart(x, res, title, xlab, logx=False, extra=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[r["truncated"] for r in res], name="truncation only",
                                 line=dict(color=INK["muted"], dash="dot", width=2)))
        for name in methods_ui.value:
            fig.add_trace(go.Scatter(x=x, y=[r[name] for r in res], name=name, mode="lines+markers",
                                     line=dict(color=COLORS[name], width=2, dash=DASH[name]),
                                     marker=dict(symbol=SYMBOL[name], size=7, color=COLORS[name])))
        if extra is not None:
            fig.add_trace(extra)
        fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
        fig.update_xaxes(title_text=xlab, type="log" if logx else "linear")
        fig.update_yaxes(title_text="error (vol pts)")
        return style_fig(fig, title, height=420, hover="closest")


    def stress_figs():
        F, sd, h = mkt.F, mkt.sd, h_ui.value / 100 * mkt.F
        lo, hi = F * np.exp(-range_ui.value * sd), F * np.exp(range_ui.value * sd)
        ratios = [1, 2, 3, 4, 5, 6, 8, 10]
        t1 = lines_chart(ratios, [grid_errors(mkt, tiered_grid(F, h, r, sd, lo, hi, offset_ui.value)) for r in ratios],
                         f"1 · Tiered grid: inner step {h_ui.value:g}% of F, outer step × m", "outer / inner step ratio m")
        base = uniform_grid(F, h, lo, hi, offset_ui.value)
        draws = [grid_errors(mkt, drop_strikes(base, F, drop_ui.value, s)) for s in range(60)]
        t2 = go.Figure()
        for name in methods_ui.value:
            t2.add_trace(go.Box(y=[d[name] for d in draws], name=name, marker_color=COLORS[name], boxmean=True,
                                line=dict(width=1.5)))
        t2.add_hline(y=0, line=dict(color=INK["axis"], width=1))
        t2.update_yaxes(title_text="error (vol pts)")
        style_fig(t2, f"2 · Random missing strikes: drop prob. {drop_ui.value:.0%}, 60 draws (base h = {h_ui.value:g}% of F)",
                  height=440, hover="closest")
        t2.update_layout(showlegend=False)
        offs = np.linspace(0.0, 0.95, 20)
        r3 = [grid_errors(mkt, uniform_grid(F, h, lo, hi, o)) for o in offs]
        xg = offs * h / (F - offs * h)
        gap = go.Scatter(x=offs, y=[vol_pts(mkt.truth + (offset_term(1 + x, 1.0, mkt.T) - x**2 / mkt.T)) - vol_pts(mkt.truth)
                                    for x in xg], name="exact − VIX offset term", line=dict(color=INK["primary"], dash="dash"))
        t3 = lines_chart(offs, r3, f"3 · Forward position inside a gap (h = {h_ui.value:g}% of F, T = {mkt.T:.3g}y)",
                         "F offset from K₀ (fraction of a gap)", extra=gap)
        nsds = np.array([1.0, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10])
        t4 = lines_chart(nsds, [grid_errors(mkt, uniform_grid(F, h, F * np.exp(-n * sd), F * np.exp(n * sd), offset_ui.value))
                                for n in nsds], f"4 · Strike range (h = {h_ui.value:g}% of F)", "range half-width (σ√T units, log-moneyness)")
        return [t1, t2, t3, t4]


    mo.vstack(stress_figs())
    return


@app.cell(hide_code=True)
def conclusions(kv_truth, mkt, mo, reps, spacing_ratio, vol_pts):
    def ranking_md():
        rows = sorted(((abs(vol_pts(r.kvar) - vol_pts(kv_truth)), n, vol_pts(r.kvar) - vol_pts(kv_truth))
                       for n, r in reps.items()))
        body = "\n".join(f"| {i+1} | {n} | {e:+.4f} |" for i, (_, n, e) in enumerate(rows))
        return f"| rank | method | error (vol pts) |\n|---|---|---|\n{body}"


    mo.md(rf"""
    ## 7 · Conclusions

    **This scenario** ({mkt.name}; h/(Fσ√T) = {spacing_ratio:.2f}):

    {ranking_md()}

    **General lessons** (they hold across scenarios; check them with the controls):

    1. **Derman's discrete appendix and the "linear-in-strike" price integration are the same
       method** in the interior: the weights are integrated hat functions. They differ only at the
       two outermost strikes, by much less than the truncation error there, so only Derman is
       carried through the comparison.
    2. **VIX $\Delta K/K^2$, the trapezoid and integrated-curvature cells are all close relatives.**
       Fixing the $1/K^2$ kernel (cells) is a third-order refinement. The error that matters is
       second order and comes from treating the *price* curve as piecewise linear.
    3. **The dominant interior bias is $\approx\sum_i p_i h_i^2/(6T\bar K_i^2)$**, and
       $\approx\frac{{h^2}}{{6TF^2}}\mathbb E[(F/S)^2]$ for uniform grids. It is always positive
       (over-replication), it barely depends on the vol level, and it is small only when
       $h\ll F\sigma\sqrt T$. Short maturities and low vols
       make it worse for the same strike grid.
    4. **Uneven spacing is handled correctly by any rule that uses the actual gap widths**
       (Derman, cells). What matters is the spacing *where the probability is*. Coarse
       wings mainly hurt through truncation, which is negative, and it can mask the positive chord
       bias.
    5. **The model-free LP interval is the honest answer** for sparse strikes. Derman sits on its
       upper edge inside the range, and the upper bound is unbounded as the support floor goes to 0.
       A single-number estimate needs a smoothness assumption: either the market-implied convexity
       correction, or a smile interpolation (Le Floc'h's recommendation). Both work well until gaps
       approach $1\sigma$, and then nothing can recover information that is not quoted.
    6. Use the **exact forward-offset term** $\frac2T[(F/K_0-1)-\ln(F/K_0)]$ rather than the VIX
       quadratic. They are equally simple, and the exact one is right for weeklies with coarse
       strikes.

    *References:* Demeterfi, Derman, Kamal & Zou (1999), *More Than You Ever Wanted to Know About
    Volatility Swaps*, GS QS Research Notes. Le Floc'h (2018), *Variance Swap Replication: Discrete
    or Continuous?*, JRFM 11(1):11. Carr & Madan (2001). CBOE VIX White Paper.
    """)
    return


if __name__ == "__main__":
    app.run()
