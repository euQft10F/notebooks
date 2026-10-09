# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "plotly==7.1.0",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import math
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, math, mo, np


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # The conceptual visualization of TPE

    Watanabe, *Tree-Structured Parzen Estimator: Understanding Its
    Algorithm Components and Their Roles for Better Empirical Performance* ([arXiv:2304.11127](https://arxiv.org/abs/2304.11127)).

    TPE splits the observations $\mathcal{D}$ at the top-$\gamma$ quantile $y^\gamma$ into a **better group**
    $\mathcal{D}^{(l)}$ and a **worse group** $\mathcal{D}^{(g)}$, builds a KDE for each (Eq. 5), and scores
    candidates with the density ratio

    $$r(x \mid \mathcal{D}) = \frac{p(x \mid \mathcal{D}^{(l)})}{p(x \mid \mathcal{D}^{(g)})}.$$

    Candidates $\mathcal{S} = \{x_s\}_{s=1}^{N_s}$ are sampled from $p(x \mid \mathcal{D}^{(l)})$ and the one with the
    largest $r$ is evaluated next (Algorithm 1, lines 12–13). The data below is synthetic.
    """)
    return


@app.cell
def controls(mo):
    gamma = mo.ui.slider(0.05, 0.5, step=0.01, value=0.15, label="Top quantile γ", show_value=True)
    n_obs = mo.ui.slider(10, 80, step=1, value=30, label="Observations N", show_value=True)
    bw_factor = mo.ui.slider(0.25, 3.0, step=0.05, value=1.0, label="Bandwidth scale", show_value=True)
    n_candidates = mo.ui.slider(1, 50, step=1, value=24, label="Candidates Nₛ", show_value=True)
    seed = mo.ui.number(start=0, stop=10_000, step=1, value=0, label="Seed")
    mo.hstack([gamma, n_obs, bw_factor, n_candidates, seed], wrap=True, justify="start", gap=1.5)
    return bw_factor, gamma, n_candidates, n_obs, seed


@app.cell(hide_code=True)
def observations(n_obs, np, seed):
    def objective(x):
        """Toy 1-D objective shaped like the paper's: a hump on the left, minimum near x ≈ 0.88."""
        return np.cos(4.0 * x - 0.4)


    def make_observations(n, rng):
        """Mimic a TPE history: random start-up trials, then trials concentrated near the optimum."""
        _n_init = max(3, int(0.6 * n))
        _x = np.concatenate([
            rng.uniform(0.0, 1.0, _n_init),
            np.clip(rng.normal(0.87, 0.05, n - _n_init), 0.0, 1.0),
        ])
        _y = objective(_x) + rng.normal(0.0, 0.02, n)
        _order = np.argsort(_y)  # sort so that y_1 <= y_2 <= ... <= y_N
        return _x[_order], _y[_order]


    x_obs, y_obs = make_observations(n_obs.value, np.random.default_rng(seed.value))
    return objective, x_obs, y_obs


@app.cell(hide_code=True)
def tpe(bw_factor, gamma, math, n_candidates, np, seed, x_obs, y_obs):
    def _std_normal_cdf(z):
        return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))


    def bandwidth(x, scale):
        """Scott-style rule of thumb, with a floor so a tiny group still gets a usable kernel."""
        _sd = max(np.std(x), 0.05) if len(x) > 1 else 0.1
        return scale * _sd * len(x) ** (-1 / 5)


    def kde_pdf(grid, centers, b):
        """Eq. (5): uniform weights, a uniform prior p0 on [0, 1], Gaussian kernels truncated to [0, 1]."""
        _w = 1.0 / (len(centers) + 1)
        _z = (np.asarray(grid)[:, None] - centers[None, :]) / b
        _mass = _std_normal_cdf((1 - centers) / b) - _std_normal_cdf(-centers / b)
        _kernels = np.exp(-0.5 * _z**2) / (b * math.sqrt(2 * math.pi)) / _mass
        return _w * 1.0 + _w * _kernels.sum(axis=1)


    def sample_kde(centers, b, n, rng):
        """Draw from the mixture: pick a component (prior or kernel), then sample it inside [0, 1]."""
        _out = []
        for _k in rng.integers(0, len(centers) + 1, n):
            if _k == len(centers):
                _out.append(rng.uniform())
                continue
            while True:
                _s = rng.normal(centers[_k], b)
                if 0.0 <= _s <= 1.0:
                    _out.append(_s)
                    break
        return np.array(_out)


    n_better = math.ceil(gamma.value * len(y_obs))
    y_gamma = y_obs[n_better - 1]
    x_better, y_better = x_obs[:n_better], y_obs[:n_better]
    x_worse, y_worse = x_obs[n_better:], y_obs[n_better:]
    b_better = bandwidth(x_better, bw_factor.value)
    b_worse = bandwidth(x_worse, bw_factor.value)

    x_grid = np.linspace(0.0, 1.0, 501)
    p_better = kde_pdf(x_grid, x_better, b_better)
    p_worse = kde_pdf(x_grid, x_worse, b_worse)
    acq = p_better / p_worse

    x_samples = sample_kde(x_better, b_better, n_candidates.value, np.random.default_rng(seed.value + 1))
    acq_samples = kde_pdf(x_samples, x_better, b_better) / kde_pdf(x_samples, x_worse, b_worse)
    x_next = x_samples[np.argmax(acq_samples)]
    acq_next = acq_samples.max()
    return (
        acq,
        acq_next,
        acq_samples,
        b_better,
        b_worse,
        n_better,
        p_better,
        p_worse,
        x_better,
        x_grid,
        x_next,
        x_samples,
        x_worse,
        y_better,
        y_gamma,
        y_worse,
    )


@app.cell(hide_code=True)
def figure(
    acq,
    acq_next,
    acq_samples,
    gamma,
    go,
    make_subplots,
    n_better,
    np,
    objective,
    p_better,
    p_worse,
    x_better,
    x_grid,
    x_next,
    x_samples,
    x_worse,
    y_better,
    y_gamma,
    y_worse,
):
    RED, BLUE, MAGENTA, GREEN, INK, MUTED = "#d62728", "#1f6fd1", "#c0399b", "#2ca02c", "#1f1f1f", "#8a8a8a"

    fig1 = make_subplots(
        rows=2, cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        row_heights=[0.74, 0.26], column_widths=[0.5, 0.5],
        horizontal_spacing=0.05, vertical_spacing=0.04,
        subplot_titles=("y = f(x)", "Kernel density estimators", ""),
    )

    # --- Left: objective, observations, split line -------------------------------
    _hover_obs = "x = %{x:.3f}<br>y = %{y:.3f}<extra>%{fullData.name}</extra>"
    fig1.add_trace(go.Scatter(x=x_grid, y=objective(x_grid), mode="lines", name="f(x)",
                              line=dict(color=INK, width=1.5, dash="dash"), hoverinfo="skip", showlegend=False), 1, 1)
    fig1.add_trace(go.Scatter(x=[0, 1], y=[y_gamma, y_gamma], mode="lines", showlegend=False,
                              line=dict(color=GREEN, width=2, dash="dot"),
                              hovertemplate=f"y<sup>γ</sup> = {y_gamma:.3f}<extra></extra>"), 1, 1)
    fig1.add_trace(go.Scatter(x=x_better, y=y_better, mode="markers", name="Better group D<sup>(l)</sup>", legendgroup="better",
                              marker=dict(color=RED, symbol="square", size=8, line=dict(color="white", width=1)),
                              hovertemplate=_hover_obs), 1, 1)
    fig1.add_trace(go.Scatter(x=x_worse, y=y_worse, mode="markers", name="Worse group D<sup>(g)</sup>", legendgroup="worse",
                              marker=dict(color=BLUE, symbol="square", size=8, line=dict(color="white", width=1)),
                              hovertemplate=_hover_obs), 1, 1)
    fig1.add_annotation(x=0.04, y=y_gamma, text="y = y<sup>γ</sup>", showarrow=False, yshift=11,
                        xanchor="left", font=dict(color=INK, size=13), row=1, col=1)

    # --- Left inset: magnified view around the split -----------------------------
    _x_lo = max(0.0, x_better.min() - 0.12)
    _x_hi = min(1.0, x_better.max() + 0.08)
    _y_lo = y_better.min() - 0.03
    _y_hi = y_gamma + 1.6 * (y_gamma - y_better.min() + 0.03)
    _xi = np.linspace(_x_lo, _x_hi, 200)
    _near = (x_worse >= _x_lo) & (x_worse <= _x_hi) & (y_worse <= _y_hi)
    for _trace in [
        go.Scatter(x=_xi, y=objective(_xi), mode="lines", line=dict(color=INK, width=1.5, dash="dash"), hoverinfo="skip"),
        go.Scatter(x=[_x_lo, _x_hi], y=[y_gamma, y_gamma], mode="lines", line=dict(color=GREEN, width=2, dash="dot"), hoverinfo="skip"),
        go.Scatter(x=x_better, y=y_better, mode="markers", name="Better group D<sup>(l)</sup>",
                   marker=dict(color=RED, symbol="square", size=9, line=dict(color="white", width=1)), hovertemplate=_hover_obs),
        go.Scatter(x=x_worse[_near], y=y_worse[_near], mode="markers", name="Worse group D<sup>(g)</sup>",
                   marker=dict(color=BLUE, symbol="square", size=9, line=dict(color="white", width=1)), hovertemplate=_hover_obs),
    ]:
        _trace.update(xaxis="x4", yaxis="y4", showlegend=False)
        fig1.add_trace(_trace)
    fig1.add_annotation(text=f"Split at y = y<sup>γ</sup>  (γ = {gamma.value:.2f}, N<sup>(l)</sup> = {n_better})",
                        xref="x4 domain", yref="y4 domain", x=0.03, y=0.97, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.85)",
                        showarrow=False, font=dict(size=11, color=INK))
    # Outline of the magnified region on the main panel
    fig1.add_shape(type="rect", x0=_x_lo, x1=_x_hi, y0=_y_lo, y1=_y_hi, xref="x", yref="y",
                   line=dict(color=MUTED, width=1, dash="dot"))

    # --- Top right: KDEs with rug ticks for each group ---------------------------
    fig1.add_trace(go.Scatter(x=x_grid, y=p_better, mode="lines", name="KDE for better group p(x|D<sup>(l)</sup>)",
                              line=dict(color=RED, width=2),
                              hovertemplate="x = %{x:.3f}<br>p(x|D<sup>(l)</sup>) = %{y:.3f}<extra></extra>"), 1, 2)
    fig1.add_trace(go.Scatter(x=x_grid, y=p_worse, mode="lines", name="KDE for worse group p(x|D<sup>(g)</sup>)",
                              line=dict(color=BLUE, width=2),
                              hovertemplate="x = %{x:.3f}<br>p(x|D<sup>(g)</sup>) = %{y:.3f}<extra></extra>"), 1, 2)
    _rug_h = 0.08 * max(p_better.max(), p_worse.max())
    for _xs, _color, _group in [(x_better, RED, "better"), (x_worse, BLUE, "worse")]:
        fig1.add_trace(go.Scatter(
            x=np.repeat(_xs, 3), y=np.tile([0.0, _rug_h, np.nan], len(_xs)), mode="lines",
            line=dict(color=_color, width=1), legendgroup=_group, showlegend=False, hoverinfo="skip"), 1, 2)

    # --- Bottom right: acquisition function, candidates, chosen point ------------
    fig1.add_trace(go.Scatter(x=x_grid, y=acq, mode="lines", name="Acquisition function r(x|D)",
                              line=dict(color=MAGENTA, width=2, dash="dot"),
                              hovertemplate="x = %{x:.3f}<br>r(x|D) = %{y:.3f}<extra></extra>"), 2, 2)
    fig1.add_trace(go.Scatter(x=x_samples, y=acq_samples, mode="markers", name="Samples S = {x<sub>s</sub>}",
                              marker=dict(color=INK, symbol="triangle-up", size=9, line=dict(color="white", width=1)),
                              hovertemplate="candidate x = %{x:.3f}<br>r = %{y:.3f}<extra></extra>"), 2, 2)
    fig1.add_trace(go.Scatter(x=[x_next], y=[acq_next], mode="markers", name="Next configuration x<sup>*</sup>",
                              marker=dict(color=GREEN, symbol="star", size=16, line=dict(color="white", width=1)),
                              hovertemplate="x<sup>*</sup> = %{x:.3f}<br>r = %{y:.3f}<extra>argmax over S</extra>"), 2, 2)

    # --- Layout ------------------------------------------------------------------
    _axis = dict(showgrid=False, zeroline=False, showline=True, linecolor=MUTED, mirror=True, ticks="")
    fig1.update_xaxes(range=[0, 1], showticklabels=False, **_axis)
    fig1.update_yaxes(showticklabels=False, **_axis)
    fig1.update_xaxes(matches="x2", row=2, col=2)
    fig1.update_yaxes(rangemode="tozero", row=1, col=2)
    fig1.update_yaxes(range=[0, acq.max() * 1.15], row=2, col=2)
    fig1.update_layout(
        xaxis4=dict(domain=[0.22, 0.45], anchor="y4", range=[_x_lo, _x_hi], showticklabels=False, **_axis),
        yaxis4=dict(domain=[0.6, 0.95], anchor="x4", range=[_y_lo, _y_hi], showticklabels=False, **_axis),
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Times New Roman, serif", size=14, color=INK),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
                    bordercolor=MUTED, borderwidth=1, font=dict(size=13)),
    )
    fig1
    return


@app.cell(hide_code=True)
def caption(b_better, b_worse, mo, n_better, x_next, x_samples):
    mo.md(rf"""
    **Left:** the objective $y = f(x)$ (black dashed) and its observations $\mathcal{{D}}$; the inset magnifies the boundary
    $y = y^\gamma$ (green dotted) between $\mathcal{{D}}^{{(l)}}$ (red squares, $N^{{(l)}} = \lceil \gamma N \rceil = {n_better}$) and
    $\mathcal{{D}}^{{(g)}}$ (blue squares). **Top right:** KDEs built from each group (bandwidths
    $b^{{(l)}} = {b_better:.3f}$, $b^{{(g)}} = {b_worse:.3f}$). **Bottom right:** the density ratio $r(x\mid\mathcal{{D}})$ (magenta dotted);
    of the {len(x_samples)} candidates (black triangles) sampled from $p(x\mid\mathcal{{D}}^{{(l)}})$, the green star at
    $x^\star = {x_next:.3f}$ has the largest $r$ and would be evaluated next.

    *Things to try:* raise γ and watch $\mathcal{{D}}^{{(l)}}$ absorb worse points, flattening $r$; shrink the bandwidth scale to make
    the search more exploitative; use a single candidate ($N_s = 1$) to see pure sampling from $p(x\mid\mathcal{{D}}^{{(l)}})$.
    """)
    return


if __name__ == "__main__":
    app.run()
