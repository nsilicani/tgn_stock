from typing import Optional, Tuple
import h5py

from pathlib import Path

import numpy as np
import pandas as pd


class GraphData:
  def __init__(self, sources: np.ndarray,
        destinations: np.ndarray, timestamps: np.ndarray, edge_idxs: np.ndarray, source_labels: np.ndarray, dest_labels: np.ndarray, limit: Optional[int] = None):
    self.sources = sources if limit is None else sources[:limit]
    self.destinations = destinations if limit is None else destinations[:limit]
    self.timestamps = timestamps if limit is None else timestamps[:limit]
    self.edge_idxs = edge_idxs if limit is None else edge_idxs[:limit]
    self.source_labels = source_labels if limit is None else source_labels[:limit]
    self.dest_labels = dest_labels if limit is None else dest_labels[:limit]
    self.n_interactions = len(self.sources)
    self.unique_nodes = set(self.sources) | set(self.destinations)
    self.n_unique_nodes = len(self.unique_nodes)
  
  def __repr__(self):
    return f"""GraphData
    sources =  {self.sources.shape},
    destinations =  {self.destinations.shape},
    timestamps =  {self.timestamps.shape},
    edge_idxs =  {self.edge_idxs.shape},
    source_labels =  {self.source_labels.shape},
    dest_labels =  {self.dest_labels.shape},
    n_interactions =  {self.n_interactions},
    n_unique_nodes =  {self.n_unique_nodes}
    """


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times


def load_data(data_path: Path, features_path: Path, limit: Optional[int], randomize_features: bool = False, train_size: float = 0.7, test_size: float = 0.25) -> GraphData:
    df_graph = pd.read_parquet(data_path)
    edge_features = df_graph["wt"].values
    node_features = np.load(features_path)
    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])
    
    val_time, test_time = list(np.quantile(df_graph.ts, [train_size, 1-test_size]))

    sources = df_graph["u"].values
    destinations = df_graph["v"].values
    edge_idxs = df_graph["idx"].values
    source_labels = df_graph["u_label"].values
    dest_labels = df_graph["v_label"].values
    timestamps = df_graph.ts.values

    full_data = GraphData(sources=sources, destinations=destinations, timestamps=timestamps, edge_idxs=edge_idxs, source_labels=source_labels, dest_labels=dest_labels)

    train_mask = timestamps <= val_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time
    train_data = GraphData(
      sources=sources[train_mask], 
      destinations=destinations[train_mask], 
      timestamps=timestamps[train_mask], 
      edge_idxs=edge_idxs[train_mask], 
      source_labels=source_labels[train_mask], 
      dest_labels=dest_labels[train_mask],
      limit=limit,
    )
    val_data = GraphData(
      sources=sources[val_mask], 
      destinations=destinations[val_mask], 
      timestamps=timestamps[val_mask], 
      edge_idxs=edge_idxs[val_mask], 
      source_labels=source_labels[val_mask], 
      dest_labels=dest_labels[val_mask],
    )
    test_data = GraphData(
      sources=sources[test_mask], 
      destinations=destinations[test_mask], 
      timestamps=timestamps[test_mask], 
      edge_idxs=edge_idxs[test_mask], 
      source_labels=source_labels[test_mask], 
      dest_labels=dest_labels[test_mask],
    )
    return node_features, edge_features, full_data, train_data, val_data, test_data

def get_neighbor_finder(data: GraphData, uniform: bool = False, max_node_idx: Optional[int] = None) -> NeighborFinder:
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)

def compute_time_statistics(sources: np.ndarray, destinations: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float, float, float]:
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst