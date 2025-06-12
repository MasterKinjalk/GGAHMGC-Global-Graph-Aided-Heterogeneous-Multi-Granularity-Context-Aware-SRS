import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from datetime import datetime

from data.dataset import SessionDataset, collate_fn
from models.ggahmgc import GGAHMGC
from utils.metrics import evaluate_model
from utils.train_utils import EarlyStopping, set_seed


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_dir):
    """Load preprocessed data"""
    # Load session data
    train_data = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
    val_data = pd.read_pickle(os.path.join(data_dir, "validation.pkl"))
    test_data = pd.read_pickle(os.path.join(data_dir, "test.pkl"))

    # Load mappings
    with open(os.path.join(data_dir, "item_mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)

    # Load global graph data
    with open(os.path.join(data_dir, "global_graph_data.pkl"), "rb") as f:
        global_graph_data = pickle.load(f)

    return train_data, val_data, test_data, mappings, global_graph_data


def train_epoch(model, train_loader, optimizer, device, epoch, writer, log_interval):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass and compute loss
        loss = model.loss(batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update parameters
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Log to tensorboard
        if batch_idx % log_interval == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar("Train/Loss", loss.item(), global_step)

            # Log learning rate
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Train/LearningRate", lr, global_step)

    return total_loss / num_batches


def validate(model, val_loader, device, epoch, writer):
    """Validate model"""
    model.eval()

    # Evaluate on validation set
    metrics = evaluate_model(model, val_loader, device, [10, 20])

    # Log metrics
    for k in [10, 20]:
        writer.add_scalar(f"Val/Recall@{k}", metrics[f"recall@{k}"], epoch)
        writer.add_scalar(f"Val/MRR@{k}", metrics[f"mrr@{k}"], epoch)
        writer.add_scalar(f"Val/NDCG@{k}", metrics[f"ndcg@{k}"], epoch)

    return metrics


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/yoochoose/",
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Device configuration
    device = torch.device(
        "cuda:1" if torch.cuda.is_available() and config["device"]["cuda"] else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_data, val_data, test_data, mappings, global_graph_data = load_data(
        args.data_dir
    )

    # Update config with data info
    config["num_items"] = mappings["num_items"]

    # Create datasets
    train_dataset = SessionDataset(train_data, mappings["num_items"])
    val_dataset = SessionDataset(val_data, mappings["num_items"])
    test_dataset = SessionDataset(test_data, mappings["num_items"])

    print(f"Train sessions: {len(train_dataset)}")
    print(f"Val sessions: {len(val_dataset)}")
    print(f"Test sessions: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["device"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["device"]["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["device"]["num_workers"],
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    model = GGAHMGC(config, global_graph_data).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_scheduler"]["step_size"],
        gamma=config["training"]["lr_scheduler"]["gamma"],
    )

    # Create tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/ggahmgc_{timestamp}")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        save_path=os.path.join(
            config["logging"]["save_dir"], f"best_model_{timestamp}.pt"
        ),
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            writer,
            config["logging"]["log_interval"],
        )

        # Validate
        val_metrics = validate(model, val_loader, device, epoch, writer)

        # Step scheduler
        scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Recall@20: {val_metrics['recall@20']:.4f}")
        print(f"  Val MRR@20: {val_metrics['mrr@20']:.4f}")
        print(f"  Val NDCG@20: {val_metrics['ndcg@20']:.4f}")

        # Early stopping check
        if early_stopping(val_metrics["recall@20"], model, optimizer, epoch):
            print("Early stopping triggered!")
            break

    # Load best model
    print("\nLoading best model...")
    best_checkpoint = torch.load(early_stopping.save_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, [10, 20])

    print("\nTest Results:")
    print(f"  Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  Recall@20: {test_metrics['recall@20']:.4f}")
    print(f"  MRR@10: {test_metrics['mrr@10']:.4f}")
    print(f"  MRR@20: {test_metrics['mrr@20']:.4f}")
    print(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"  NDCG@20: {test_metrics['ndcg@20']:.4f}")

    # Save final results
    results = {
        "config": config,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "best_epoch": best_checkpoint["epoch"],
    }

    with open(
        os.path.join(config["logging"]["save_dir"], f"results_{timestamp}.pkl"), "wb"
    ) as f:
        pickle.dump(results, f)

    # Close tensorboard writer
    writer.close()

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
