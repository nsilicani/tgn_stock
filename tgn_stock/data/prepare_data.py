import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from typing_extensions import Annotated

from loguru import logger
from tqdm import tqdm

import pandas as pd

import h5py
import json
import pickle

from tgn_stock.config import PROCESSED_DATA_DIR, graph_build_config

app = typer.Typer()


@app.command()
def main(
    dataset_path: Annotated[
        Path, typer.Option(help="The dataset path")
    ] = PROCESSED_DATA_DIR / "stock_data_2.0.0.parquet",
    influence_graph_path: Annotated[
        Path, typer.Option(help="Folder where influence graphs are stored")
    ] = PROCESSED_DATA_DIR / "influence_graphs",
    output_path: Annotated[
        Path, typer.Option(help="The output path where dataframe of graphs are saved")
    ] = PROCESSED_DATA_DIR / "graph_dataframes",
    df_graph_version: Annotated[
        Optional[str], typer.Option(help="The graph dataframe version")
    ] = "1.0.0",
    features_path: Annotated[
        Path, typer.Option(help="The path where node features are saved")
    ] = PROCESSED_DATA_DIR / "feat",
    idxs_path: Annotated[
        Path, typer.Option(help="The path where idxs mappings are saved")
    ] = PROCESSED_DATA_DIR / "idxs",
):
    """
    Load stored graphs and prepare data for analysis along with node feature vectors.
    """
    # Paths
    assert dataset_path.exists(), f"Path {dataset_path} does not exist"
    assert influence_graph_path.exists(), f"Path {influence_graph_path} does not exist"
    assert output_path.exists(), f"Path {output_path} does not exist"
    assert features_path.exists(), f"Path {features_path} does not exist"
    influence_graph_path /= df_graph_version
    influence_graph_path.mkdir(exist_ok=True, parents=True)
    output_path /= df_graph_version
    output_path.mkdir(exist_ok=True, parents=True)
    idxs_path /= df_graph_version
    idxs_path.mkdir(exist_ok=True, parents=True)
    features_path /= df_graph_version
    features_path.mkdir(exist_ok=True, parents=True)

    # Load Data
    df = pd.read_parquet(dataset_path)
    node_2_ids = {v: k for k, v in enumerate(df["Ticker"].unique())}
    ids_2_node = {v: k for k, v in node_2_ids.items()}
    logger.info(f"# nodes: {len(node_2_ids)}")

    graphs = []
    for filename in os.listdir(influence_graph_path):
        if "stock_graph" in filename and filename.endswith(".pickle"):
            file_path = os.path.join(influence_graph_path, filename)
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

    # Edge idx
    idx_counter = 0

    # Timestamp idx
    timestamp_2_ids = dict()

    for time_id, G in tqdm(enumerate(graphs), total=len(graphs)):
        ref_date = G.graph["ref_date"]
        # Filter dataframe by timestamp
        df_ref = df.loc[ref_date]
        timestamp_2_ids[ref_date] = time_id
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
            if time_id not in feat_dict[source_node_id]:
                feat_dict[source_node_id][time_id] = s_feat[
                    features_cols
                ].values.reshape(-1)

            dest.append(dest_node_id)
            d_feat = df_ref.loc[df_ref["Ticker"] == v]
            assert not d_feat.empty
            dest_label.append(d_feat["target"].item())
            if time_id not in feat_dict[dest_node_id]:
                feat_dict[dest_node_id][time_id] = d_feat[features_cols].values.reshape(
                    -1
                )

            idx_counter += 1
    ids_2_timestamp = {v: k for k, v in timestamp_2_ids.items()}

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
    assert df_graphs.loc[df_graphs["u"] == df_graphs["v"]].empty
    logger.info(f"Graph df total rows: {len(df_graphs)}")
    logger.info(
        f"Class distribution:\n{df_graphs['u_label'].value_counts(normalize=True)}\n{df_graphs['v_label'].value_counts(normalize=True)}"
    )

    # Create mapping for idxs to edges
    idxs_2_edges = {
        row["idx"]: (row["u"], row["v"], row["ts"]) for _, row in df_graphs.iterrows()
    }

    # Save outputs
    df_graphs.to_parquet(output_path / f"df_graph_{df_graph_version}.parquet")
    with h5py.File(
        features_path / f"node_feature_vectors_{df_graph_version}.h5", "w"
    ) as hf:
        for node in tqdm(feat_dict):
            for time_id in feat_dict[node]:
                hf.create_dataset(
                    f"node_{node}/ts_{time_id}", data=feat_dict[node][time_id]
                )
    # Store feature as numpy
    feat_shape = len(graph_build_config["features"])
    node_features = np.zeros((len(feat_dict), len(timestamp_2_ids), feat_shape))
    for node in feat_dict:
        for timestamp in feat_dict[node]:
            node_features[node, timestamp, :] = feat_dict[node][timestamp]
    np.save(
        features_path / f"node_feature_vectors_{df_graph_version}.npy", node_features
    )

    # Store mappings
    node_2_ids
    with open(idxs_path / "node_2_ids.json", "w") as fb:
        json.dump(node_2_ids, fb)
    with open(idxs_path / "ids_2_node.json", "w") as fb:
        json.dump(ids_2_node, fb)
    with open(idxs_path / "idxs_2_edges.json", "w") as fb:
        json.dump(idxs_2_edges, fb)
    with open(idxs_path / "timestamp_2_ids.json", "w") as fb:
        json.dump({k.strftime("%Y-%m-%d"): v for k, v in timestamp_2_ids.items()}, fb)
    with open(idxs_path / "ids_2_timestamp.json", "w") as fb:
        json.dump({k: v.strftime("%Y-%m-%d") for k, v in ids_2_timestamp.items()}, fb)

    logger.success("Data preparation completed.")


if __name__ == "__main__":
    app()
