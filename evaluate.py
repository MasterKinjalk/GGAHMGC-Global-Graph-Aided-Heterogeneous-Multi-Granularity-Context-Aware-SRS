import argparse
import torch
import yaml
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data.dataset import SessionDataset, collate_fn
from models.ggahmgc import GGAHMGC
from utils.metrics import evaluate_model, calculate_coverage, calculate_diversity


def visualize_attention_weights(model, data_loader, device, num_samples=5):
    """Visualize attention weights for sample sessions"""
    model.eval()

    samples_processed = 0
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    with torch.no_grad():
        for batch in data_loader:
            if samples_processed >= num_samples:
                break

            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            _ = model.forward(batch)
            attention_weights = model.get_attention_weights()

            batch_size = batch["input_items"].shape[0]

            for i in range(min(batch_size, num_samples - samples_processed)):
                row_idx = samples_processed + i

                # Global attention (if available)
                if attention_weights["global_attention"] is not None:
                    # Average over layers and heads
                    global_attn = (
                        attention_weights["global_attention"][0][i]
                        .mean(dim=0)
                        .cpu()
                        .numpy()
                    )
                    sns.heatmap(global_attn, ax=axes[row_idx, 0], cmap="Blues")
                    axes[row_idx, 0].set_title(
                        f"Sample {row_idx + 1}: Global Attention"
                    )

                # Fusion attention
                fusion_attn = (
                    attention_weights["fusion_attention"][i]
                    .mean(dim=0)
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                sns.heatmap(
                    fusion_attn.reshape(-1, 1), ax=axes[row_idx, 1], cmap="Reds"
                )
                axes[row_idx, 1].set_title(f"Sample {row_idx + 1}: Fusion Attention")

                # Readout weights
                readout = attention_weights["readout_weights"][i].cpu().numpy()
                sns.heatmap(readout.reshape(1, -1), ax=axes[row_idx, 2], cmap="Greens")
                axes[row_idx, 2].set_title(f"Sample {row_idx + 1}: Readout Weights")

            samples_processed += batch_size

    plt.tight_layout()
    plt.savefig("attention_visualizations.png", dpi=300, bbox_inches="tight")
    plt.close()


def analyze_predictions(model, data_loader, device, item_mappings, top_k=20):
    """Analyze model predictions"""
    model.eval()

    # Track predictions
    all_predictions = []
    all_targets = []
    prediction_counts = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Analyzing predictions"):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get predictions
            scores = model.predict(batch)
            targets = batch["targets"]

            # Get top-k predictions
            _, topk_indices = torch.topk(scores, top_k, dim=1)

            # Store results
            all_predictions.extend(topk_indices.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Count predictions
            for preds in topk_indices:
                for item_id in preds:
                    item_id = item_id.item()
                    prediction_counts[item_id] = prediction_counts.get(item_id, 0) + 1

    # Analyze popularity bias
    id_to_item = item_mappings["id_to_item"]

    # Get most frequently recommended items
    sorted_items = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 Most Frequently Recommended Items:")
    print("-" * 50)
    for item_id, count in sorted_items[:10]:
        if item_id in id_to_item:
            print(f"Item {item_id}: {count} times")

    # Calculate hit rate by position
    position_hits = [0] * top_k
    total_samples = len(all_targets)

    for preds, target in zip(all_predictions, all_targets):
        target_idx = target - 1  # Adjust for 0-indexing
        if target_idx in preds:
            position = list(preds).index(target_idx)
            for i in range(position, top_k):
                position_hits[i] += 1

    # Plot hit rate by position
    plt.figure(figsize=(10, 6))
    positions = list(range(1, top_k + 1))
    hit_rates = [hits / total_samples for hits in position_hits]

    plt.plot(positions, hit_rates, marker="o")
    plt.xlabel("Top-K")
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate by Position")
    plt.grid(True, alpha=0.3)
    plt.savefig("hit_rate_by_position.png", dpi=300, bbox_inches="tight")
    plt.close()

    return prediction_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/tmall/",
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate attention visualizations"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze prediction patterns"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    if args.split == "val":
        data = pd.read_pickle(f"{args.data_dir}/validation.pkl")
    else:
        data = pd.read_pickle(f"{args.data_dir}/test.pkl")

    # Load mappings
    with open(f"{args.data_dir}/item_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    # Load global graph data
    with open(f"{args.data_dir}/global_graph_data.pkl", "rb") as f:
        global_graph_data = pickle.load(f)

    # Update config
    config["num_items"] = mappings["num_items"]

    # Create dataset and dataloader
    dataset = SessionDataset(data, mappings["num_items"])
    data_loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create and load model
    print("Loading model...")
    model = GGAHMGC(config, global_graph_data).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    metrics = evaluate_model(model, data_loader, device, [5, 10, 20])

    print("\nResults:")
    print("-" * 40)
    for k in [5, 10, 20]:
        print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
        print(f"MRR@{k}: {metrics[f'mrr@{k}']:.4f}")
        print(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        print("-" * 40)

    # Additional metrics
    print("\nAdditional Metrics:")
    coverage = calculate_coverage(model, data_loader, device, k=20)
    print(f"Item Coverage@20: {coverage:.4f}")

    diversity = calculate_diversity(model, data_loader, device, k=20)
    print(f"Recommendation Diversity@20: {diversity:.4f}")

    # Visualizations
    if args.visualize:
        print("\nGenerating attention visualizations...")
        visualize_attention_weights(model, data_loader, device)
        print("Saved to attention_visualizations.png")

    # Prediction analysis
    if args.analyze:
        print("\nAnalyzing predictions...")
        analyze_predictions(model, data_loader, device, mappings)
        print("Saved analysis plots")

    # Save results
    results = {
        "metrics": metrics,
        "coverage": coverage,
        "diversity": diversity,
        "config": config,
        "checkpoint": args.checkpoint,
    }

    with open(f"evaluation_results_{args.split}.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
