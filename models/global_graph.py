import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np
from collections import defaultdict


class GlobalGraphConstructor:
    """Constructs and maintains the global item transition graph"""

    def __init__(
        self,
        num_items,
        cooccurrence_data,
        transition_data,
        num_neighbors=12,
        weight_threshold=0.0,
    ):
        self.num_items = num_items
        self.num_neighbors = num_neighbors
        self.weight_threshold = weight_threshold

        # Build adjacency matrix
        self.adj_matrix, self.weight_matrix = self._build_adjacency(
            cooccurrence_data, transition_data
        )

        # Create DGL graph
        self.graph = self._create_dgl_graph()

    def _build_adjacency(self, cooccurrence_data, transition_data):
        """Build weighted adjacency matrix from co-occurrence and transition data"""

        # Initialize matrices
        adj_matrix = np.zeros((self.num_items + 1, self.num_items + 1))
        weight_matrix = np.zeros((self.num_items + 1, self.num_items + 1))

        # Add co-occurrence weights
        for item1, neighbors in cooccurrence_data.items():
            for item2, count in neighbors.items():
                if item1 <= self.num_items and item2 <= self.num_items:
                    weight = count / (count + 1)  # Normalize
                    adj_matrix[item1][item2] = 1
                    weight_matrix[item1][item2] += weight * 0.5

        # Add transition weights (higher weight)
        for item1, neighbors in transition_data.items():
            for item2, count in neighbors.items():
                if item1 <= self.num_items and item2 <= self.num_items:
                    weight = count / (count + 1)
                    adj_matrix[item1][item2] = 1
                    weight_matrix[item1][item2] += weight

        # Keep only top-k neighbors for each item
        for i in range(1, self.num_items + 1):
            weights = weight_matrix[i]
            if np.sum(weights) > 0:
                # Get top-k indices
                top_k_indices = np.argpartition(weights, -self.num_neighbors)[
                    -self.num_neighbors :
                ]

                # Create mask
                mask = np.zeros_like(weights)
                mask[top_k_indices] = 1

                # Apply mask
                adj_matrix[i] = adj_matrix[i] * mask
                weight_matrix[i] = weight_matrix[i] * mask

        return adj_matrix, weight_matrix

    def _create_dgl_graph(self):
        """Create DGL graph from adjacency matrix"""
        # Get edges
        src, dst = np.nonzero(self.adj_matrix)

        # Create graph
        g = dgl.graph((src, dst))

        # Add edge weights
        edge_weights = self.weight_matrix[src, dst]
        g.edata["weight"] = torch.FloatTensor(edge_weights)

        return g

    def get_neighbors(self, items):
        """Get neighbors for a batch of items"""
        # Handle batched input
        if isinstance(items, torch.Tensor):
            items = items.cpu().numpy()

        neighbors_list = []
        weights_list = []

        for item in items:
            if item > 0 and item <= self.num_items:
                # Get neighbors from adjacency matrix
                neighbors = np.nonzero(self.adj_matrix[item])[0]
                weights = self.weight_matrix[item][neighbors]

                # Pad if necessary
                if len(neighbors) < self.num_neighbors:
                    pad_size = self.num_neighbors - len(neighbors)
                    neighbors = np.pad(neighbors, (0, pad_size), constant_values=0)
                    weights = np.pad(weights, (0, pad_size), constant_values=0)

                neighbors_list.append(neighbors[: self.num_neighbors])
                weights_list.append(weights[: self.num_neighbors])
            else:
                # Padding node
                neighbors_list.append(np.zeros(self.num_neighbors, dtype=np.int64))
                weights_list.append(np.zeros(self.num_neighbors))

        return (
            torch.LongTensor(np.array(neighbors_list)),
            torch.FloatTensor(np.array(weights_list)),
        )


class GlobalGraphLayer(nn.Module):
    """Global context aggregation layer using graph attention"""

    def __init__(self, embedding_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention for global context
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Projection layers
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)

        # Session-aware attention
        self.session_attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Output projection
        self.output_proj = nn.Linear(embedding_dim * 2, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        item_embeddings,
        neighbor_embeddings,
        neighbor_weights,
        session_embedding,
        mask=None,
    ):
        """
        Args:
            item_embeddings: [batch_size, seq_len, embedding_dim]
            neighbor_embeddings: [batch_size, seq_len, num_neighbors, embedding_dim]
            neighbor_weights: [batch_size, seq_len, num_neighbors]
            session_embedding: [batch_size, embedding_dim]
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len, num_neighbors, embed_dim = neighbor_embeddings.shape

        # Reshape for attention computation
        item_emb_flat = item_embeddings.view(batch_size * seq_len, 1, embed_dim)
        neighbor_emb_flat = neighbor_embeddings.view(
            batch_size * seq_len, num_neighbors, embed_dim
        )
        neighbor_weights_flat = neighbor_weights.view(
            batch_size * seq_len, num_neighbors
        )

        # Expand session embedding
        session_emb_expanded = (
            session_embedding.unsqueeze(1)
            .expand(batch_size, seq_len, embed_dim)
            .reshape(batch_size * seq_len, embed_dim)
        )

        # Session-aware neighbor attention
        session_neighbor_concat = torch.cat(
            [
                session_emb_expanded.unsqueeze(1).expand(-1, num_neighbors, -1),
                neighbor_emb_flat,
            ],
            dim=-1,
        )

        attention_scores = self.session_attention(session_neighbor_concat).squeeze(-1)

        # Apply neighbor weights
        attention_scores = attention_scores * neighbor_weights_flat

        # Mask padding neighbors
        neighbor_mask = (neighbor_weights_flat > 0).float()
        attention_scores = attention_scores.masked_fill(neighbor_mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted aggregation
        global_context = torch.sum(
            neighbor_emb_flat * attention_weights.unsqueeze(-1), dim=1
        )

        # Reshape back
        global_context = global_context.view(batch_size, seq_len, embed_dim)

        # Combine with item embeddings
        combined = torch.cat([item_embeddings, global_context], dim=-1)
        output = self.output_proj(combined)

        # Residual connection and layer norm
        output = self.layer_norm(item_embeddings + self.dropout(output))

        return output, attention_weights.view(batch_size, seq_len, num_neighbors)


class GlobalGraphEncoder(nn.Module):
    """Multi-layer global graph encoder"""

    def __init__(
        self,
        num_items,
        embedding_dim,
        hidden_dim,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.num_layers = num_layers

        # Item embeddings
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # Global graph layers
        self.global_layers = nn.ModuleList(
            [
                GlobalGraphLayer(embedding_dim, hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Position encoding
        self.position_embedding = nn.Embedding(100, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, items, global_graph, session_embedding, mask=None):
        """
        Forward pass through global graph encoder

        Args:
            items: [batch_size, seq_len]
            global_graph: GlobalGraphConstructor instance
            session_embedding: [batch_size, embedding_dim]
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len = items.shape

        # Get item embeddings
        item_emb = self.item_embedding(items)

        # Add position embeddings
        positions = torch.arange(seq_len, device=items.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        item_emb = item_emb + pos_emb

        item_emb = self.dropout(item_emb)

        # Process through layers
        hidden = item_emb
        all_attention_weights = []

        for layer in self.global_layers:
            # Get neighbors for all items in batch
            neighbors_list = []
            weights_list = []

            for i in range(batch_size):
                for j in range(seq_len):
                    item_id = items[i, j].item()
                    if item_id > 0:
                        neighbors, weights = global_graph.get_neighbors([item_id])
                        neighbors_list.append(neighbors[0])
                        weights_list.append(weights[0])
                    else:
                        neighbors_list.append(
                            torch.zeros(global_graph.num_neighbors, dtype=torch.long)
                        )
                        weights_list.append(torch.zeros(global_graph.num_neighbors))

            # Stack neighbors
            neighbors = torch.stack(neighbors_list).view(batch_size, seq_len, -1)
            neighbor_weights = torch.stack(weights_list).view(batch_size, seq_len, -1)

            # Get neighbor embeddings
            neighbor_emb = self.item_embedding(neighbors.to(items.device))

            # Apply global graph layer
            hidden, attention_weights = layer(
                hidden, neighbor_emb, neighbor_weights, session_embedding, mask
            )

            all_attention_weights.append(attention_weights)

        return hidden, all_attention_weights
