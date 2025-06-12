# GGAHMGC: Global Graph Aided Heterogeneous Multi-Granularity Context-Aware Session-based Recommendation

This repository contains a PyTorch implementation of the GGAHMGC model, which combines global graph context with multi-granularity session representations for improved session-based recommendation.

## Features

- **Global Graph Context**: Leverages item co-occurrence and transition patterns across all sessions
- **Multi-granularity Representation**: Captures user intent at different granularity levels (items, snippets, sessions)
- **Enhanced Attention Mechanism**: Custom global context attention for fusing representations
- **Heterogeneous Graph Neural Networks**: Processes multi-granularity session graphs
- **Complete Training Pipeline**: End-to-end implementation with data preprocessing, training, and evaluation

## Architecture

The GGAHMGC model consists of several key components:

1. **Global Graph Encoder**: Learns item representations using global transition patterns
2. **Hierarchical Session Graph**: Creates multi-granularity representations of sessions
3. **Global Context Attention**: Fuses global and local representations (new component)
4. **Session Readout**: Attention-based aggregation of session representations
5. **Fusion Gate**: Gated mechanism for combining different representations
6. **Target-aware Prediction**: Attention-based scoring for candidate items

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.12.0
- DGL >= 0.9.0
- NumPy, Pandas, scikit-learn
- TensorBoard for visualization

## Data Preprocessing

### Tmall Dataset

1. Download the Tmall dataset (ensure it's in CSV format with columns: user_id, item_id, category_id, behavior_type, timestamp)

2. Run preprocessing:
```bash
python preprocess_tmall.py --input path/to/tmall.csv --output datasets/tmall/
```

This will:
- Create sessions from user interactions (30-minute gap splitting)
- Filter items by frequency (min 5 occurrences)
- Create train/validation/test splits (time-based)
- Build global graph data (co-occurrence and transition patterns)
- Save processed data in pickle format

## Training

### Basic Training

```bash
python train.py --config config/config.yaml --data_dir datasets/tmall/
```

### Resume Training

```bash
python train.py --config config/config.yaml --data_dir datasets/tmall/ --resume checkpoints/best_model.pt
```

### Configuration

Key configuration options in `config/config.yaml`:

```yaml
model:
  embedding_dim: 256
  hidden_dim: 256
  num_layers: 2
  dropout: 0.2
  
  global_graph:
    num_neighbors: 12
    attention_heads: 8
    
  multi_granularity:
    max_granularity: 3
    
  attention:
    type: "multi_head"
    num_heads: 8
    temperature: 0.1

training:
  batch_size: 100
  learning_rate: 0.001
  epochs: 30
  patience: 3
```

## Model Architecture Details

### Global Graph Construction
- Builds weighted adjacency matrix from item co-occurrence and transitions
- Keeps top-k neighbors for efficiency
- Uses session-aware attention for neighbor aggregation

### Multi-Granularity Representation
- Level 1: Individual items
- Level 2: Consecutive item pairs
- Level 3: Item triplets
- Heterogeneous graph with intra and inter-granularity edges

### Enhanced Attention Mechanism
The model includes a custom Global Context Attention module that:
- Applies multi-head attention between global and local representations
- Uses temperature-scaled attention scores
- Includes residual connections and layer normalization

## Evaluation

The model is evaluated using standard metrics:
- **Recall@K**: Percentage of test cases where the ground truth item appears in top-K
- **MRR@K**: Mean Reciprocal Rank of the ground truth item
- **NDCG@K**: Normalized Discounted Cumulative Gain

Additional metrics available:
- Item coverage
- Recommendation diversity

## Results

Expected performance on Tmall dataset:
- Recall@20: ~32.5%
- MRR@20: ~15.2%

## Project Structure

```
ggahmgc/
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   ├── dataset.py          # Dataset classes
│   ├── preprocessor.py     # Data preprocessing
│   └── data_loader.py      # Data loading utilities
├── models/
│   ├── ggahmgc.py         # Main model
│   ├── global_graph.py    # Global graph components
│   ├── multi_granularity.py # Multi-granularity encoder
│   ├── attention.py       # Attention mechanisms
│   └── aggregators.py     # Aggregation functions
├── utils/
│   ├── metrics.py         # Evaluation metrics
│   ├── train_utils.py     # Training utilities
│   └── graph_utils.py     # Graph utilities
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── preprocess_tmall.py    # Preprocessing script
└── requirements.txt       # Dependencies
```

## Customization

### Adding New Datasets

1. Create a new preprocessor class inheriting from `TmallPreprocessor`
2. Implement dataset-specific session creation logic
3. Update configuration with dataset parameters

### Modifying Architecture

The modular design allows easy modifications:
- Change attention mechanism in `models/attention.py`
- Modify graph construction in `models/global_graph.py`
- Adjust multi-granularity levels in configuration

## Visualization

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir runs/
```

Visualizations include:
- Training/validation loss
- Metrics evolution
- Learning rate schedule
- Attention weight distributions

## Citation

If you use this implementation, please cite the original papers:

```bibtex
@inproceedings{ggahmgc2023,
  title={Global Graph Aided Heterogeneous Multi Granularity Context Aware Session-based Recommendation},
  author={Parth, Kinjalk and Singh, Janmajay and U, Annie},
  booktitle={SYNASC 2023},
  year={2023}
}
```

## Acknowledgments

This implementation builds upon ideas from:
- GCE-GNN: Global Context Enhanced Graph Neural Networks
- MSGIFSR: Multi-granularity Session-based Graph Neural Networks
- SR-GNN: Session-based Recommendation with Graph Neural Networks

## License

MIT License - see LICENSE file for details.