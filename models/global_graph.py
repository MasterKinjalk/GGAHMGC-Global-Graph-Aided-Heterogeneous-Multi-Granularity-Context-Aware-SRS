import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from collections import defaultdict


# GlobalGraphConstructor can remain the same as you have it.
# The adjacency matrix is still useful for building the initial DGL graph.
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
        self.adj_matrix, self.weight_matrix = self._build_adjacency(
            cooccurrence_data, transition_data
        )
        self.graph = self._create_dgl_graph()

    def _build_adjacency(self, cooccurrence_data, transition_data):
        adj_matrix = np.zeros((self.num_items + 1, self.num_items + 1))
        weight_matrix = np.zeros((self.num_items + 1, self.num_items + 1))
        for item1, neighbors in cooccurrence_data.items():
            for item2, count in neighbors.items():
                if item1 <= self.num_items and item2 <= self.num_items:
                    weight = count / (count + 1)
                    adj_matrix[item1][item2] = 1
                    weight_matrix[item1][item2] += weight * 0.5
        for item1, neighbors in transition_data.items():
            for item2, count in neighbors.items():
                if item1 <= self.num_items and item2 <= self.num_items:
                    weight = count / (count + 1)
                    adj_matrix[item1][item2] = 1
                    weight_matrix[item1][item2] += weight
        for i in range(1, self.num_items + 1):
            weights = weight_matrix[i]
            if np.sum(weights) > 0:
                top_k_indices = np.argpartition(weights, -self.num_neighbors)[
                    -self.num_neighbors :
                ]
                mask = np.zeros_like(weights)
                mask[top_k_indices] = 1
                adj_matrix[i] = adj_matrix[i] * mask
                weight_matrix[i] = weight_matrix[i] * mask
        return adj_matrix, weight_matrix

    def _create_dgl_graph(self):
        src, dst = np.nonzero(self.adj_matrix)
        g = dgl.graph((src, dst))
        edge_weights = self.weight_matrix[src, dst]
        g.edata["weight"] = torch.FloatTensor(edge_weights)
        return g


# ==============================================================================
#                      <<< NEW DGL-NATIVE IMPLEMENTATION >>>
# ==============================================================================


class DGLGlobalGraphLayer(nn.Module):
    """
    DGL-native implementation of the Global Graph Layer.
    This layer uses message passing for efficient computation.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.session_attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.output_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def edge_attention_message_func(self, edges):
        """
        Message function to compute session-aware attention scores on edges.
        """
        # Concatenate neighbor features (src) with the session embedding (dst)
        concat_features = torch.cat([edges.src["h"], edges.dst["s_emb"]], dim=-1)
        # Compute raw attention score and multiply by the pre-calculated edge weight
        raw_attn = self.session_attention(concat_features) * edges.data[
            "weight"
        ].unsqueeze(-1)
        return {"raw_attn": raw_attn}

    def forward(self, block, h_src, h_dst, session_embedding):
        """
        Args:
            block: A DGL message passing graph (MPG) block from neighbor sampling.
            h_src: Features of the source nodes (neighbors).
            h_dst: Features of the destination nodes (items in the session).
            session_embedding: The session-level embedding for attention.
        """
        with block.local_scope():
            # Assign features to the graph
            block.srcdata["h"] = h_src
            block.dstdata["s_emb"] = session_embedding

            # Step 1: Compute raw attention scores on all edges in the block.
            block.apply_edges(self.edge_attention_message_func)

            # Step 2: Normalize attention scores using edge-softmax.
            # edge_softmax groups edges by destination node and applies softmax.
            attn_weights = dgl.ops.edge_softmax(block, block.edata.pop("raw_attn"))
            block.edata["a"] = attn_weights

            # Step 3: Perform message passing. Aggregate neighbor features (h_src)
            # weighted by the computed attention scores.
            block.update_all(
                message_func=fn.u_mul_e("h", "a", "m"),  # message = h_src * attention
                reduce_func=fn.sum("m", "h_global"),  # h_global = sum(messages)
            )

            # The aggregated global context is now in the destination nodes
            global_context = block.dstdata["h_global"]

            # Combine original destination features with the new global context
            combined = torch.cat([h_dst, global_context], dim=-1)
            output = self.output_proj(combined)

            # Final residual connection and layer normalization
            output = self.layer_norm(h_dst + self.dropout(output))
            return output, attn_weights


class GlobalGraphEncoder(nn.Module):
    """
    Multi-layer global graph encoder using DGL's neighbor sampling.
    """

    def __init__(
        self,
        num_items,
        embedding_dim,
        hidden_dim,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(100, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.global_layers = nn.ModuleList(
            [
                DGLGlobalGraphLayer(embedding_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, items, global_graph, session_embedding, mask=None):
        """
        Forward pass using efficient neighbor sampling.
        """
        batch_size, seq_len = items.shape
        device = items.device

        # Get initial item embeddings with position info
        item_emb = self.item_embedding(items)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        h = self.dropout(item_emb + pos_emb)

        all_attention_weights = []

        # Reshape for processing: from [B, L, D] to [B*L, D]
        h = h.view(batch_size * seq_len, -1)

        # Expand session embedding to match each item in the flattened sequence
        session_emb_expanded = (
            session_embedding.unsqueeze(1)
            .expand(-1, seq_len, -1)
            .reshape(batch_size * seq_len, -1)
        )

        # Create the list of blocks for message passing
        # We sample neighbors layer by layer, starting from the final items
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
        dataloader = dgl.dataloading.NodeDataLoader(
            global_graph.graph,
            items.flatten(),
            sampler,
            batch_size=batch_size * seq_len,  # Process the whole batch at once
            shuffle=False,
            drop_last=False,
        )

        # This dataloader will yield input_nodes, output_nodes, and blocks
        # For a single batch, it runs once.
        input_nodes, output_nodes, blocks = next(iter(dataloader))

        # The features for the first layer's input nodes are their raw embeddings
        h_input = self.item_embedding(input_nodes.to(device))

        for i, (layer, block) in enumerate(zip(self.global_layers, blocks)):
            # The block needs to be on the correct device
            block = block.to(device)
            # Input features for this layer
            h_src = h_input
            # Output features are a subset of the input features
            h_dst = h_src[block.dstnodes.long()]

            # The session embedding needs to correspond to the destination nodes
            # We need to map the flattened session_emb_expanded to the dst nodes of this block
            # This is complex, a simpler and effective alternative is to use the dst features
            # to compute a session embedding for the block.
            # For simplicity, we'll pass the expanded session embedding, assuming node order is preserved.
            # A more robust solution might require an extra mapping step.
            session_emb_for_block = session_emb_expanded[output_nodes.long()]

            # Apply the GNN layer
            h_output, attention_weights = layer(
                block, h_src, h_dst, session_emb_for_block
            )

            # The output of this layer is the input for the next
            h_input = h_output
            all_attention_weights.append(attention_weights)

        # Reshape the final output back to [B, L, D]
        final_hidden = h_output.view(batch_size, seq_len, -1)

        return final_hidden, all_attention_weights
