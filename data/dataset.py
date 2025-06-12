import torch
from torch.utils.data import Dataset
import numpy as np
import dgl
from collections import defaultdict


class SessionDataset(Dataset):
    """Dataset for session-based recommendation"""

    def __init__(self, sessions_df, num_items, max_length=50):
        self.sessions = sessions_df
        self.num_items = num_items
        self.max_length = max_length

        # Prepare augmented data (all subsequences)
        self.data = []
        for _, session in sessions_df.iterrows():
            items = session["items"]
            for i in range(1, len(items)):
                self.data.append(
                    {
                        "session_id": session["session_id"],
                        "input_items": items[:i],
                        "target": items[i],
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Pad sequences
        input_items = sample["input_items"][-self.max_length :]
        padded_items = [0] * (self.max_length - len(input_items)) + input_items

        # Create mask
        mask = [0] * (self.max_length - len(input_items)) + [1] * len(input_items)

        return {
            "session_id": sample["session_id"],
            "input_items": torch.LongTensor(padded_items),
            "mask": torch.BoolTensor(mask),
            "target": sample["target"],
            "session_length": len(input_items),
        }


class MultiGranularitySessionDataset(Dataset):
    """Dataset with multi-granularity session representations"""

    def __init__(self, sessions_df, num_items, max_length=50, max_granularity=3):
        self.sessions = sessions_df
        self.num_items = num_items
        self.max_length = max_length
        self.max_granularity = max_granularity

        # Prepare data
        self.data = []
        for _, session in sessions_df.iterrows():
            items = session["items"]
            for i in range(1, len(items)):
                # Create multi-granularity representations
                granularities = self._create_granularities(items[:i])

                self.data.append(
                    {
                        "session_id": session["session_id"],
                        "input_items": items[:i],
                        "granularities": granularities,
                        "target": items[i],
                    }
                )

    def _create_granularities(self, items):
        """Create multi-granularity representations of session"""
        granularities = {}

        # Level 1: Individual items
        granularities[1] = [(item,) for item in items]

        # Higher levels: Consecutive item groups
        for k in range(2, min(self.max_granularity + 1, len(items) + 1)):
            granularities[k] = []
            for i in range(len(items) - k + 1):
                granularities[k].append(tuple(items[i : i + k]))

        return granularities

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def collate_fn(batch):
    """Custom collate function for batching"""
    # Stack all tensors
    session_ids = [item["session_id"] for item in batch]
    input_items = torch.stack([item["input_items"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    targets = torch.LongTensor([item["target"] for item in batch])
    lengths = torch.LongTensor([item["session_length"] for item in batch])

    return {
        "session_ids": session_ids,
        "input_items": input_items,
        "masks": masks,
        "targets": targets,
        "lengths": lengths,
    }


def create_session_graph(items, max_granularity=3):
    """Create a heterogeneous session graph with multi-granularity nodes"""

    # Create graph
    graph_data = defaultdict(list)
    node_features = {}

    # Add nodes for each granularity level
    for k in range(1, min(max_granularity + 1, len(items) + 1)):
        node_type = f"gran_{k}"
        node_features[node_type] = []

        for i in range(len(items) - k + 1):
            if k == 1:
                # Single item
                node_features[node_type].append(items[i])
            else:
                # Multi-item snippet
                snippet = items[i : i + k]
                # Use mean of item embeddings as snippet feature
                node_features[node_type].append(snippet)

    # Add edges within same granularity (sequential)
    for k in range(1, min(max_granularity + 1, len(items) + 1)):
        node_type = f"gran_{k}"
        edge_type = (node_type, f"seq_{k}", node_type)

        for i in range(len(items) - k):
            graph_data[edge_type].append((i, i + 1))

    # Add edges between granularities (hierarchical)
    for k in range(1, min(max_granularity, len(items))):
        src_type = f"gran_{k}"
        dst_type = f"gran_{k + 1}"

        # Connect consecutive items to their snippets
        edge_type_up = (src_type, f"up_{k}", dst_type)
        edge_type_down = (dst_type, f"down_{k}", src_type)

        for i in range(len(items) - k):
            # Connect items i and i+1 to snippet starting at i
            graph_data[edge_type_up].append((i, i))
            graph_data[edge_type_up].append((i + 1, i))
            graph_data[edge_type_down].append((i, i))
            graph_data[edge_type_down].append((i, i + 1))

    # Convert to DGL format
    edge_dict = {}
    for edge_type, edges in graph_data.items():
        if edges:
            src, dst = zip(*edges)
            edge_dict[edge_type] = (torch.tensor(src), torch.tensor(dst))

    # Create heterogeneous graph
    g = dgl.heterograph(edge_dict)

    # Add node features
    for node_type, features in node_features.items():
        if node_type in g.ntypes:
            g.nodes[node_type].data["feat"] = torch.tensor(features)

    return g
