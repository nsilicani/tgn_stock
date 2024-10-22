from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import itertools

from sklearn.svm import SVC
from tqdm import tqdm
from loguru import logger

from typing import Any, Callable, List, Tuple, Dict


class StockGraph:
    def __init__(
        self,
        stock_data: pd.DataFrame,
        features: List[str],
        features_to_norm: List[str],
        cls_name: str,
        cls_params: Dict[str, Any],
        lambda_weight: float = 0.7,
        min_data_points: int = 125,
        train_size: float = 0.8,
    ):
        """
        Initialize the stock graph computation object.
        :param stock_data: DataFrame with stocks data, datetime index, columns include features, prices, etc.
        :param lambda_weight: Weight for influence vs correlation in edge weight calculation.
        :param min_data_points: Minimum required number of data points for stock pairs.
        """
        self.stock_data = stock_data
        self.features = features
        self.features_to_norm = features_to_norm
        self.lambda_weight = lambda_weight
        self.min_data_points = min_data_points
        self.train_size = train_size
        self.cls_name = cls_name
        self.cls_params = cls_params
        self.graph = nx.Graph()
        self.avg_score = []

    def get_classifier(self) -> Callable:
        """Return a classifier object based on the config."""
        if self.cls_name == "SVC":
            return SVC(**self.cls_params)
        elif self.cls_name == "RandomForestClassifier":
            return RandomForestClassifier(**self.cls_params)
        elif self.cls_name == "XGBClassifier":
            return XGBClassifier(**self.cls_params)
        else:
            raise ValueError(f"Unsupported classifier: {self.cls_name}")

    def get_historic_data(self, reference_date) -> pd.DataFrame:
        """Get data for the trading days before the reference date."""
        return self.stock_data[self.stock_data.index <= reference_date]

    def compute_pairs(self, stock_list: List[str]) -> List[Tuple[str, str]]:
        """Compute all pairs of stocks."""
        return list(itertools.combinations(stock_list, 2))

    def normalize_data(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize training and validation data."""
        scaler = MinMaxScaler()
        X_train[self.features_to_norm] = scaler.fit_transform(
            X_train[self.features_to_norm]
        )
        X_val[self.features_to_norm] = scaler.transform(X_val[self.features_to_norm])
        return X_train, X_val

    def compute_score(
        self,
        cls: Any,
        X_train: np.ndarray,
        Y_train: pd.Series,
        X_val: np.ndarray,
        Y_val: pd.Series,
    ) -> float:
        """Train classifier and compute F1 score."""
        cls.fit(X_train, Y_train)
        Y_pred = cls.predict(X_val)
        return f1_score(Y_val, Y_pred, average="weighted")

    def process_stock_pairs(
        self, stock1: str, stock2: str, data: pd.DataFrame, reference_date: pd.Timestamp
    ) -> None:
        """Process each stock pair to compute influence and add edges to the graph."""
        stock1_data = data.loc[data["Ticker"] == stock1]
        stock2_data = data.loc[data["Ticker"] == stock2]

        # Align data by date to keep only common dates between both stocks
        common_dates = stock1_data.index.intersection(stock2_data.index)
        stock1_data = stock1_data.loc[common_dates]
        stock2_data = stock2_data.loc[common_dates]

        # Ensure sufficient data
        if (
            len(stock1_data) < self.min_data_points
            or len(stock2_data) < self.min_data_points
        ):
            return

        # Split into training and validation sets
        X_s1 = stock1_data[self.features]
        X_s2 = stock2_data[self.features]
        Y_s1 = stock1_data["target"]
        Y_s2 = stock2_data["target"]

        # Train-Test Split for both stocks
        X_train_s1, X_val_s1, Y_train_s1, Y_val_s1 = train_test_split(
            X_s1, Y_s1, test_size=1 - self.train_size, shuffle=False
        )
        X_train_s2, X_val_s2, Y_train_s2, Y_val_s2 = train_test_split(
            X_s2, Y_s2, test_size=1 - self.train_size, shuffle=False
        )

        # Processed data by averaging feature vectors
        X_processed = (X_s1 + X_s2) / 2
        X_train_proc, X_val_proc, _, _ = train_test_split(
            X_processed, Y_s1, test_size=1 - self.train_size, shuffle=False
        )

        # Normalize data
        X_train_s1, X_val_s1 = self.normalize_data(X_train_s1, X_val_s1)
        X_train_s2, X_val_s2 = self.normalize_data(X_train_s2, X_val_s2)
        X_train_proc, X_val_proc = self.normalize_data(X_train_proc, X_val_proc)

        # Initialize classifiers
        cls_s1 = self.get_classifier()
        cls_s2 = self.get_classifier()
        cls_s1_s2 = self.get_classifier()
        cls_s2_s1 = self.get_classifier()

        # Compute scores for both stocks
        score_s1 = self.compute_score(
            cls_s1, X_train_s1, Y_train_s1, X_val_s1, Y_val_s1
        )
        score_s2 = self.compute_score(
            cls_s2, X_train_s2, Y_train_s2, X_val_s2, Y_val_s2
        )

        # Compute scores for processed data
        score_s1_s2 = self.compute_score(
            cls_s1_s2, X_train_proc, Y_train_s1, X_val_proc, Y_val_s1
        )
        score_s2_s1 = self.compute_score(
            cls_s2_s1, X_train_proc, Y_train_s2, X_val_proc, Y_val_s2
        )

        # Store scores
        self.avg_score.extend([score_s1, score_s2, score_s1_s2, score_s2_s1])

        # Influence Calculation
        influence = 0.5 * ((score_s1_s2 - score_s1) + (score_s2_s1 - score_s2))

        if influence > 0:
            # Compute linear correlation between the prices of both stocks
            corr = np.corrcoef(stock1_data["Adj Close"], stock2_data["Adj Close"])[0, 1]

            # Compute edge weight
            weight = self.lambda_weight * influence + (1 - self.lambda_weight) * corr

            # Add edge to the graph
            self.graph.add_edge(stock1, stock2, weight=weight)

    def prune_graph(self) -> None:
        """Prune the graph by removing edges until the removal of the next edge would disconnect the graph."""
        # Sort edges by their weight in ascending order
        sorted_edges = sorted(self.graph.edges(data=True), key=lambda x: x[2]["weight"])

        for edge in sorted_edges:
            # Create a copy of the graph to test edge removal
            temp_graph = self.graph.copy()
            temp_graph.remove_edge(*edge[:2])

            # If the graph is still connected after removing the edge, proceed with removal
            if nx.is_connected(temp_graph):
                self.graph.remove_edge(*edge[:2])
            else:
                # Stop when removing the next edge would disconnect the graph
                break

    def normalize_edge_weights(self) -> None:
        """Normalize edge weights to a [0, 1] range."""
        max_weight = max(nx.get_edge_attributes(self.graph, "weight").values())
        for u, v, data in self.graph.edges(data=True):
            data["weight"] /= max_weight

    def build_graph(self, stock_list: List[str], reference_date: pd.Timestamp) -> None:
        """Build the stock influence graph."""

        df_historic = self.get_historic_data(reference_date)

        # Get stock pairs
        stock_pairs = self.compute_pairs(stock_list)

        # Process each pair of stocks
        logger.info("Processing stock pairs ...")
        for stock1, stock2 in tqdm(stock_pairs, total=len(stock_pairs)):
            self.process_stock_pairs(stock1, stock2, df_historic, reference_date)

        # Prune the graph
        logger.info("Pruning graph ...")
        self.prune_graph()

        # Normalize edge weights
        logger.info("Normalizing edge weights ...")
        self.normalize_edge_weights()

        # Assing graph attributes
        self.graph.graph["ref_date"] = reference_date
        self.graph.graph["avg_score"] = np.mean(self.avg_score)

    def save_graph(self, date: str, output_path: Path) -> None:
        """Save the graph for a particular date."""
        with open(output_path / f"stock_graph_{date}.pickle", "wb") as fb:
            pickle.dump(self.graph, fb)


class GraphManager:
    def __init__(self, stock_data: pd.DataFrame, output_path: Path):
        self.output_path = output_path
        self.stock_data = stock_data
        self.graphs = {}

    def build_graph_for_reference_date(
        self,
        reference_date: pd.Timestamp,
        stock_list: List[str],
        features: List[str],
        features_to_norm: List[str],
        cls_name: str,
        cls_params: Dict[str, Any],
        lambda_weight: float = 0.7,
        min_data_points: int = 125,
        train_size: float = 0.8,
    ) -> None:
        """Build graph for a particular reference date and store it."""
        stock_graph = StockGraph(
            stock_data=self.stock_data,
            features=features,
            features_to_norm=features_to_norm,
            cls_name=cls_name,
            cls_params=cls_params,
            lambda_weight=lambda_weight,
            min_data_points=min_data_points,
            train_size=train_size,
        )
        stock_graph.build_graph(stock_list, reference_date)
        self.graphs[reference_date] = stock_graph.graph
        stock_graph.save_graph(
            date=reference_date.strftime("%Y-%m-%d"), output_path=self.output_path
        )
