# tgn-stock

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Temporal Graph Networks application to stock market

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tgn_stock and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tgn_stock   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tgn_stock a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Graph Builder
The following are the steps to compute the daily influence network. The approach is based on [GCNET: graph-based prediction of stock price movement using graph convolutional network](https://arxiv.org/pdf/2203.11091v1):
- Select reference date. At start point, it will be the latest available date. In the following iterations, we compute the previous five business days
- Select the set of stocks with available data. Compute the number of stock pairs
- For each stock, split the training $X^T_{s{i}}, Y^T_{s{i}}$ and validation $X^V_{s{i}}, Y^V_{s{i}}$ parts of data
- For each pair of stock $(s_{i}, s_{j})$:
    - Check if data length respects the minimum required amount of data (e.g. 100 data points)
    - Average the two feature vector: $X_{processed} \gets 1/2(X_{s_{i}, X_{s{j}}})$
    - Split data: $X^T_{processed}$ and $X^V_{processed}$
    - Normalize data
    - Compute: \
        $\text{F-1} score_{i} \gets Classifier_{i}.fit(X^T_{s{i}}, Y^T_{s{i}}).score(X^V_{s{i}}, Y^V_{s{i}})$ \
        $\text{F-1} score_{j} \gets Classifier_{j}.fit(X^T_{s{j}}, Y^T_{s{j}}).score(X^V_{s{j}}, Y^V_{s{j}})$
    - Compute: \
        $\text{F-1} score_{i, j} \gets Classifier_{p_{i,j}}.fit(X^T_{processed}, Y^T_{s{i}}).score(X^V_{processed}, Y^V_{s{i}})$ \
        $\text{F-1} score_{j, i} \gets Classifier_{p_{j,i}}.fit(X^T_{processed}, Y^T_{s{j}}).score(X^V_{processed}, Y^V_{s{j}})$
    - Compute $Influence_{i,j} = 1/2((\text{F-1} score_{i, j}-\text{F-1} score_{i})+ (\text{F-1} score_{j, i}-\text{F-1} score_{j}))$
    - If $Influence_{i,j} > 0$: \
        Compute linear correlation $Corr_{i,j}(price_{i}, price_{j})$
        Compute edge weight $w_{i,j} = w_{j,i} = \lambda * Influence_{i,j} + (1-\lambda) * Corr_{i,j}$
        Add edge to graph's edge set
- Once the graph is computed, prune the edge with the small weights. That is, sorted by edge weights and remove edge with the smallest weight. Repeat until the graph becomes disconnected
- Normalize edge weights
- Save the computed graph in a structure mapping reference date to graph

## Resources
- [DeepNeet-Code](https://github.com/alireza-jafari/DeepNet-Code/blob/main/main.py)
- [GCNET-Code](https://github.com/alireza-jafari/GCNET-Code)
- [pandas-ta](https://github.com/twopirllc/pandas-ta?tab=readme-ov-file#momentum-41)
- [Top-Chinese-Companies](https://www.financecharts.com/screener/biggest-country-cn#:~:text=The%20most%20valuable%20company%20in%20China%20is%20Tencent,Commercial%20Bank%20of%20China%20%28IDCBY%29%20and%20Meituan%20%28MPNGY%29.)
- [TNG](https://github.com/twitter-research/tgn)
- [GNN-Finance](https://github.com/kyawlin/GNN-finance)
- [GAT](https://github.com/PetarV-/GAT)


- https://stackoverflow.com/questions/69950509/tensorflow-install-error-windows-longpath-support-not-enabled
- https://github.com/twitter-research/tgn/blob/master/utils/data_processing.py

