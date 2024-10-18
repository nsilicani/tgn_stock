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

## Resources
- [DeepNeet-Code](https://github.com/alireza-jafari/DeepNet-Code/blob/main/main.py)
- [GCNET-Code](https://github.com/alireza-jafari/GCNET-Code)
- [pandas-ta](https://github.com/twopirllc/pandas-ta?tab=readme-ov-file#momentum-41)
- [Top-Chinese-Companies](https://www.financecharts.com/screener/biggest-country-cn#:~:text=The%20most%20valuable%20company%20in%20China%20is%20Tencent,Commercial%20Bank%20of%20China%20%28IDCBY%29%20and%20Meituan%20%28MPNGY%29.)
- [TNG](https://github.com/twitter-research/tgn)
- [GNN-Finance](https://github.com/kyawlin/GNN-finance)
- [GAT](https://github.com/PetarV-/GAT)

