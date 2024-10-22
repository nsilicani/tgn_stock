import os
from pathlib import Path
import pickle
from typing import Optional

import typer
from typing_extensions import Annotated

from loguru import logger
from tqdm import tqdm

import pandas as pd
import h5py

from tgn_stock.config import PROCESSED_DATA_DIR, graph_build_config

app = typer.Typer()


@app.command()
def main(
    dataset_path: Annotated[
        Path, typer.Option(help="The dataset path")
    ] = PROCESSED_DATA_DIR / "stock_data_2.0.0.parquet",
    output_path: Annotated[
        Path, typer.Option(help="The output path where dataframe of graphs are saved")
    ] = PROCESSED_DATA_DIR / "graph_dataframes",
    df_graph_version: Annotated[
        Optional[str], typer.Option(help="The dataframe version")
    ] = "1.0.0",
    feature_path: Annotated[
        Path, typer.Option(help="The path where node features are saved")
    ] = PROCESSED_DATA_DIR / "feat",
):
    """
    Load stored graphs and prepare data for analysis along with node feature vectors.
    """
    # Paths
    assert dataset_path.exist(), f"Path {dataset_path} does not exist"
    assert output_path.exist(), f"Path {output_path} does not exist"
    assert feature_path.exist(), f"Path {feature_path} does not exist"
    # Load Data
    df = pd.read_parquet(dataset_path)
    node_2_ids = {v: k for k, v in enumerate(df["Ticker"].unique())}
    logger.info(f"# nodes: {len(node_2_ids)}")
    timestamp_2_ids = {v: k for k, v in enumerate(df.index)}
    logger.info(f"# timestamps: {len(timestamp_2_ids)}")

    graphs = []
    for filename in os.listdir(PROCESSED_DATA_DIR):
        if "stock_graph" in filename and filename.endswith(".pickle"):
            file_path = os.path.join(PROCESSED_DATA_DIR, filename)
            with open(file_path, "rb") as f:
                graph = pickle.load(f)
                graphs.append(graph)

    features_cols = graph_build_config["features"]

    source = []
    dest = []
    edge_feat = []
    ts_list = []
    source_label = []
    dest_label = []
    idx_list = []
    date_list = []

    feat_dict = {n: {} for n in node_2_ids.values()}

    idx_counter = 0

    for G in tqdm(graphs):
        ref_date = G.graph["ref_date"]
        df_ref = df.loc[ref_date]
        time_id = timestamp_2_ids[ref_date]
        for u, v, wt in G.edges.data("weight"):
            # Setting idxs for source and destination
            source_node_id = node_2_ids[u]
            dest_node_id = node_2_ids[v]

            idx_list.append(idx_counter)
            edge_feat.append(wt)
            ts_list.append(time_id)
            date_list.append(ref_date)

            # Source features
            source.append(source_node_id)
            s_feat = df_ref.loc[df_ref["Ticker"] == u]
            assert not s_feat.empty
            source_label.append(s_feat["target"].item())
            # Store feature vectors
            if time_id not in feat_dict[source_node_id]:
                feat_dict[source_node_id][time_id] = s_feat[features_cols].values

            # Destination features
            dest.append(dest_node_id)
            d_feat = df_ref.loc[df_ref["Ticker"] == v]
            assert not d_feat.empty
            dest_label.append(d_feat["target"].item())
            # Store feature vectors
            if time_id not in feat_dict[dest_node_id]:
                feat_dict[dest_node_id][time_id] = d_feat[features_cols].values

            idx_counter += 1

    df_graphs = pd.DataFrame(
        {
            "idx": idx_list,
            "date": date_list,
            "ts": ts_list,
            "u": source,
            "v": dest,
            "wt": edge_feat,
            "u_label": source_label,
            "v_label": dest_label,
        }
    )
    df_graphs.to_parquet(output_path / f"df_graph_{df_graph_version}.parquet")

    with h5py.File(feature_path / "node_feature_vectors.h5", "w") as hf:
        for node in tqdm(feat_dict):
            for time_id in feat_dict[node]:
                hf.create_dataset(
                    f"node_{node}/ts_{time_id}", data=feat_dict[node][time_id]
                )
