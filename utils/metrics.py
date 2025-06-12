import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def calculate_metrics(scores, targets, k_values=[10, 20]):
    """
    Calculate recommendation metrics

    Args:
        scores: [batch_size, num_items] - Predicted scores
        targets: [batch_size] - Ground truth items
        k_values: List of k values for metrics@k

    Returns:
        Dictionary of metrics
    """
    batch_size = scores.shape[0]
    metrics = {}

    # Get top-k predictions
    for k in k_values:
        _, topk_indices = torch.topk(scores, k, dim=1)

        # Calculate Recall@k
        hits = 0
        for i in range(batch_size):
            if targets[i] in topk_indices[i]:
                hits += 1
        recall_k = hits / batch_size
        metrics[f"recall@{k}"] = recall_k

        # Calculate MRR@k
        mrr_sum = 0
        for i in range(batch_size):
            target_positions = (topk_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                # Position is 1-indexed
                mrr_sum += 1.0 / (target_positions[0].item() + 1)
        mrr_k = mrr_sum / batch_size
        metrics[f"mrr@{k}"] = mrr_k

        # Calculate NDCG@k
        ndcg_sum = 0
        for i in range(batch_size):
            target_positions = (topk_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                # DCG = 1 / log2(position + 1)
                position = target_positions[0].item() + 1
                ndcg_sum += 1.0 / np.log2(position + 1)
        ndcg_k = ndcg_sum / batch_size
        metrics[f"ndcg@{k}"] = ndcg_k

    return metrics


def evaluate_model(model, data_loader, device, k_values=[10, 20]):
    """
    Evaluate model on a dataset

    Args:
        model: The model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to use
        k_values: List of k values for metrics@k

    Returns:
        Dictionary of averaged metrics
    """
    model.eval()

    all_metrics = {
        f"{metric}@{k}": [] for metric in ["recall", "mrr", "ndcg"] for k in k_values
    }

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get predictions
            scores = model.predict(batch)
            targets = batch["targets"] - 1  # Adjust for 0-indexing

            # Calculate metrics
            batch_metrics = calculate_metrics(scores, targets, k_values)

            # Accumulate metrics
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)

    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    return avg_metrics


def calculate_coverage(model, data_loader, device, k=20):
    """
    Calculate item coverage - what percentage of items are recommended

    Args:
        model: The model to evaluate
        data_loader: DataLoader
        device: Device to use
        k: Top-k items to consider

    Returns:
        coverage: Percentage of items recommended
    """
    model.eval()

    recommended_items = set()

    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get predictions
            scores = model.predict(batch)

            # Get top-k items
            _, topk_indices = torch.topk(scores, k, dim=1)

            # Add to recommended set
            for i in range(topk_indices.shape[0]):
                recommended_items.update(topk_indices[i].cpu().numpy().tolist())

    # Calculate coverage
    coverage = len(recommended_items) / model.num_items

    return coverage


def calculate_diversity(model, data_loader, device, k=20):
    """
    Calculate diversity of recommendations using intra-list diversity

    Args:
        model: The model to evaluate
        data_loader: DataLoader
        device: Device to use
        k: Top-k items to consider

    Returns:
        diversity: Average pairwise distance between recommended items
    """
    model.eval()

    # Get item embeddings
    all_items = torch.arange(1, model.num_items + 1, device=device)
    item_embeddings = model.global_encoder.item_embedding(all_items)

    # Normalize embeddings
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

    diversity_scores = []

    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get predictions
            scores = model.predict(batch)

            # Get top-k items
            _, topk_indices = torch.topk(scores, k, dim=1)

            # Calculate diversity for each session
            for i in range(topk_indices.shape[0]):
                # Get embeddings of recommended items
                rec_items = topk_indices[i]
                rec_embeddings = item_embeddings[rec_items]

                # Calculate pairwise cosine distances
                sim_matrix = torch.mm(rec_embeddings, rec_embeddings.t())

                # Get upper triangular part (excluding diagonal)
                mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
                similarities = sim_matrix[mask]

                # Diversity is 1 - average similarity
                if len(similarities) > 0:
                    diversity = 1 - similarities.mean().item()
                    diversity_scores.append(diversity)

    return np.mean(diversity_scores) if diversity_scores else 0.0
