# TPE — Figure 1, interactive

Interactive recreation of Figure 1 from Watanabe, *Tree-Structured Parzen Estimator: Understanding Its
Algorithm Components and Their Roles for Better Empirical Performance*
([arXiv:2304.11127](https://arxiv.org/abs/2304.11127)), built as a [marimo](https://marimo.io) notebook with Plotly.
Sliders control the top quantile γ, the number of observations, the KDE bandwidth scale, the number of
candidates, and the random seed. All data is synthetic.

## Run

With [uv](https://docs.astral.sh/uv/) (dependencies are read from the notebook's inline PEP 723 header, Python ≥ 3.14):

```sh
uvx marimo edit tpe_figure1.py --sandbox   # edit
uvx marimo run tpe_figure1.py --sandbox    # app view
```

Without uv:

```sh
pip install -r requirements.txt
marimo edit tpe_figure1.py
```
