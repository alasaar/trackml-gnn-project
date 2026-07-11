import torch
import argparse
from src.data import build_spatial_graph
from src.model import TrueTrackMLGNN
from src.train import train_model
from src.evaluate import evaluate_track_reconstruction, export_telemetry

def main():
    parser = argparse.ArgumentParser(description="TrackML GNN Pipeline")
    parser.add_argument('--hits_path', type=str, required=True, help="Path to hits CSV")
    parser.add_argument('--truth_path', type=str, required=True, help="Path to truth CSV")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--weights', type=str, default='trackml_gnn_weights.pth', help="Path to saved model weights")
    parser.add_argument('--threshold', type=float, default=0.6, help="Edge probability threshold")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware locked to: {device}")

    # 1. Build Graph
    graph_data, df_hits, df_truth = build_spatial_graph(args.hits_path, args.truth_path)
    train_graphs = [graph_data]

    # 2. Initialize Model
    model = TrueTrackMLGNN().to(device)

    # 3. Train or Load
    if args.train:
        model = train_model(model, train_graphs, device, epochs=50, save_path=args.weights)
    else:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        print(f"Loaded weights from {args.weights}")

    # 4. Evaluate and Cluster
    print("\nEvaluating Track Reconstruction...")
    eval_df, labels = evaluate_track_reconstruction(graph_data, model, device, df_truth, threshold=args.threshold)

    # 5. Export JSON Telemetry
    export_telemetry(graph_data, model, device, threshold=args.threshold)

if __name__ == "__main__":
    main()