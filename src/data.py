import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from torch_geometric.data import Data

def build_spatial_graph(hits_path, truth_path, k=10):
    print(f"Building spatial graph from hits...")

    df_hits = pd.read_csv(hits_path)
    df_truth = pd.read_csv(truth_path)

    x = df_hits['x'].values
    y = df_hits['y'].values
    z = df_hits['z'].values

    features = np.column_stack([x / 1000.0, y / 1000.0, z / 1000.0])

    tree = KDTree(features)
    _, neighbor_idx = tree.query(features, k=k + 1)

    src = np.repeat(np.arange(len(features)), k)
    dst = neighbor_idx[:, 1:].flatten()

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    particle_ids = df_truth['particle_id'].values
    sender_particles = particle_ids[edge_index[0].numpy()]
    receiver_particles = particle_ids[edge_index[1].numpy()]
    edge_labels = ((sender_particles == receiver_particles) & (sender_particles != 0)).astype(np.float32)

    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(edge_labels, dtype=torch.float32)

    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor), df_hits, df_truth