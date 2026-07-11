import json
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def evaluate_track_reconstruction(data, model, device, df_truth, threshold):
    model.eval()
    with torch.no_grad():
        data_gpu = data.to(device)
        probs = torch.sigmoid(model(data_gpu)).cpu().numpy()

    edge_index = data.edge_index.cpu().numpy()
    valid_mask = probs > threshold
    valid_edges = edge_index[:, valid_mask]
    valid_probs = probs[valid_mask]

    num_hits = data.x.shape[0]
    distances = np.clip(1.0 - valid_probs, 1e-5, 1.0).astype(np.float32)
    
    adj_matrix = coo_matrix((distances, (valid_edges[0], valid_edges[1])), shape=(num_hits, num_hits)).tocsr()
    adj_matrix.sort_indices()
    
    dbscan = DBSCAN(eps=0.05, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(adj_matrix)

    # REMAP NOISE LABELS (-1) TO UNIQUE TRACK IDS
    max_label = labels.max()
    for i in range(len(labels)):
        if labels[i] == -1:
            max_label += 1
            labels[i] = max_label

    true_particle_ids = df_truth['particle_id'].values

    results = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_size = cluster_mask.sum()
        if cluster_size < 3:
            continue
            
        cluster_particle_ids = true_particle_ids[cluster_mask]
        cluster_particle_ids = cluster_particle_ids[cluster_particle_ids != 0]
        if len(cluster_particle_ids) == 0:
            continue
            
        values, counts = np.unique(cluster_particle_ids, return_counts=True)
        majority_particle = values[np.argmax(counts)]
        majority_count = counts.max()

        purity = majority_count / cluster_size
        total_true_hits = (true_particle_ids == majority_particle).sum()
        efficiency = majority_count / total_true_hits if total_true_hits > 0 else 0.0

        results.append({'cluster_id': cluster_id, 'size': cluster_size,
                        'matched_particle': majority_particle,
                        'purity': purity, 'efficiency': efficiency})

    results_df = pd.DataFrame(results) if results else pd.DataFrame(columns=['cluster_id', 'size', 'matched_particle', 'purity', 'efficiency'])
    
    if not results_df.empty:
        good_tracks = results_df[(results_df['purity'] > 0.75) & (results_df['efficiency'] > 0.5)]
        print(f"Reconstructed clusters (size>=3): {len(results_df)}")
        print(f"'Good' tracks: {len(good_tracks)} | Mean purity: {results_df['purity'].mean():.4f} | Mean efficiency: {results_df['efficiency'].mean():.4f}")

    return results_df, labels

def export_telemetry(data, model, device, threshold, filename="tracking_telemetry.json"):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.to(device))).cpu().numpy()
        
    valid_edges = data.edge_index.cpu().numpy()[:, probs > threshold]
    
    telemetry = {
        "metadata": {
            "total_nodes": data.x.shape[0],
            "total_predicted_edges": int(valid_edges.shape[1]),
            "confidence_threshold": threshold
        },
        "nodes": data.x.cpu().numpy().tolist(),
        "edges": valid_edges.T.tolist()
    }
    
    with open(filename, 'w') as f:
        json.dump(telemetry, f, cls=NumpyEncoder)
    print(f"[SUCCESS] JSON Telemetry exported to {filename}")