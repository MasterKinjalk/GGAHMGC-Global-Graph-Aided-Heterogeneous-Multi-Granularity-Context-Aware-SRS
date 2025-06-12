import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn import GATConv


class MultiGranularityEncoder(nn.Module):
    """Heterogeneous Graph Attention Network for multi-granularity session encoding"""

    def __init__(
        self, embedding_dim, hidden_dim, num_heads=8, max_granularity=3, dropout=0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_granularity = max_granularity

        # Heterogeneous graph convolutions for different edge types
        self.conv_layers = nn.ModuleDict()

        # Intra-granularity convolutions (sequential edges)
        for k in range(1, max_granularity + 1):
            self.conv_layers[f"seq_{k}"] = GATConv(
                embedding_dim,
                hidden_dim // num_heads,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=True,
            )

        # Inter-granularity convolutions (hierarchical edges)
        for k in range(1, max_granularity):
            self.conv_layers[f"up_{k}"] = GATConv(
                embedding_dim,
                hidden_dim // num_heads,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False,
            )
            self.conv_layers[f"down_{k}"] = GATConv(
                embedding_dim,
                hidden_dim // num_heads,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False,
            )

        # Readout functions for each granularity
        self.readout_layers = nn.ModuleDict()
        for k in range(1, max_granularity + 1):
            self.readout_layers[f"gran_{k}"] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )

        # Granularity fusion
        self.fusion_attention = nn.MultiheadAttention(
            embedding_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Intent fusion ranking weights
        self.granularity_weights = nn.Parameter(torch.ones(max_granularity))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, hetero_graph, item_embeddings):
        """
        Forward pass through multi-granularity encoder

        Args:
            hetero_graph: DGL heterogeneous graph
            item_embeddings: Initial item embeddings
        """
        # Initialize node features
        h = {}

        # Set features for each node type
        for k in range(1, self.max_granularity + 1):
            node_type = f"gran_{k}"
            if node_type in hetero_graph.ntypes:
                if k == 1:
                    # Level 1: Direct item embeddings
                    item_ids = hetero_graph.nodes[node_type].data["feat"]
                    h[node_type] = item_embeddings(item_ids)
                else:
                    # Higher levels: Average pooling of constituent items
                    # This is a simplification - in practice, you'd use learned aggregation
                    h[node_type] = torch.zeros(
                        hetero_graph.num_nodes(node_type),
                        self.embedding_dim,
                        device=item_embeddings.weight.device,
                    )

        # Message passing
        outputs = {}

        # Process each edge type
        for etype, conv in self.conv_layers.items():
            if etype.startswith("seq_"):
                # Intra-granularity edges
                k = int(etype.split("_")[1])
                node_type = f"gran_{k}"

                if (node_type, etype, node_type) in hetero_graph.canonical_etypes:
                    subgraph = hetero_graph[node_type, etype, node_type]
                    feat = conv(subgraph, h[node_type])

                    # Aggregate multi-head outputs
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    feat = feat.view(feat.shape[0], -1)

                    if node_type not in outputs:
                        outputs[node_type] = feat
                    else:
                        outputs[node_type] = outputs[node_type] + feat

            elif etype.startswith("up_") or etype.startswith("down_"):
                # Inter-granularity edges
                k = int(etype.split("_")[1])

                if etype.startswith("up_"):
                    src_type = f"gran_{k}"
                    dst_type = f"gran_{k + 1}"
                else:
                    src_type = f"gran_{k + 1}"
                    dst_type = f"gran_{k}"

                if (src_type, etype, dst_type) in hetero_graph.canonical_etypes:
                    subgraph = hetero_graph[src_type, etype, dst_type]
                    feat = conv(subgraph, (h[src_type], h[dst_type]))

                    # Aggregate multi-head outputs
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    feat = feat.view(feat.shape[0], -1)

                    if dst_type not in outputs:
                        outputs[dst_type] = feat
                    else:
                        outputs[dst_type] = outputs[dst_type] + feat

        # Apply readout and collect representations
        granularity_representations = []

        for k in range(1, self.max_granularity + 1):
            node_type = f"gran_{k}"
            if node_type in outputs:
                # Apply readout
                repr_k = self.readout_layers[node_type](outputs[node_type])

                # Pool to session level (for now, using mean pooling)
                session_repr_k = repr_k.mean(dim=0, keepdim=True)
                granularity_representations.append(session_repr_k)

        # Stack representations
        if granularity_representations:
            granularity_stack = torch.cat(granularity_representations, dim=0)

            # Apply intent fusion ranking
            weights = F.softmax(
                self.granularity_weights[: len(granularity_representations)], dim=0
            )
            fused_representation = torch.sum(
                granularity_stack * weights.unsqueeze(-1), dim=0
            )

            return fused_representation, granularity_stack, weights
        else:
            # Fallback if no valid granularities
            return (
                torch.zeros(
                    1, self.embedding_dim, device=item_embeddings.weight.device
                ),
                None,
                None,
            )


class ConsecutiveIntentUnit(nn.Module):
    """Computes intent representations for consecutive item units"""

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        # Order-invariant component (set-based)
        self.set_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Order-sensitive component (sequence-based)
        self.seq_encoder = nn.GRU(
            embedding_dim, embedding_dim, batch_first=True, bidirectional=False
        )

        # Combination layer
        self.combine = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.Tanh()
        )

    def forward(self, item_embeddings):
        """
        Args:
            item_embeddings: [batch_size, seq_len, embedding_dim]
        """
        # Set-based representation (mean pooling)
        set_repr = self.set_encoder(item_embeddings.mean(dim=1))

        # Sequence-based representation (last hidden state)
        _, seq_repr = self.seq_encoder(item_embeddings)
        seq_repr = seq_repr.squeeze(0)

        # Combine
        combined = self.combine(torch.cat([set_repr, seq_repr], dim=-1))

        return combined


class HierarchicalSessionGraph(nn.Module):
    """Constructs and processes hierarchical session graph"""

    def __init__(
        self,
        num_items,
        embedding_dim,
        hidden_dim,
        max_granularity=3,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_granularity = max_granularity

        # Item embeddings
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # Multi-granularity encoder
        self.mg_encoder = MultiGranularityEncoder(
            embedding_dim, hidden_dim, num_heads, max_granularity, dropout
        )

        # Consecutive intent units
        self.ciu_encoders = nn.ModuleDict(
            {
                f"gran_{k}": ConsecutiveIntentUnit(embedding_dim, hidden_dim)
                for k in range(2, max_granularity + 1)
            }
        )

        self.dropout = nn.Dropout(dropout)

    def create_hetero_graph(self, items, lengths):
        """Create heterogeneous graph from session items"""
        batch_size = items.shape[0]
        graphs = []

        for i in range(batch_size):
            seq_len = lengths[i].item()
            session_items = items[i, :seq_len]

            # Create graph for this session
            g = create_session_graph(session_items.cpu().numpy(), self.max_granularity)
            graphs.append(g)

        # Batch graphs
        batched_g = dgl.batch(graphs)

        return batched_g

    def forward(self, items, lengths, mask=None):
        """
        Forward pass

        Args:
            items: [batch_size, seq_len]
            lengths: [batch_size]
            mask: [batch_size, seq_len]
        """
        # Create heterogeneous graph
        hetero_graph = self.create_hetero_graph(items, lengths)

        # Get multi-granularity representations
        fused_repr, gran_reprs, weights = self.mg_encoder(
            hetero_graph, self.item_embedding
        )

        return fused_repr, gran_reprs, weights


def create_session_graph(items, max_granularity=3):
    """Create a heterogeneous session graph with multi-granularity nodes"""
    import dgl

    graph_data = {}
    node_features = {}

    # Add nodes and edges for each granularity level
    for k in range(1, min(max_granularity + 1, len(items) + 1)):
        node_type = f"gran_{k}"
        num_nodes = len(items) - k + 1

        if num_nodes > 0:
            # Sequential edges within same granularity
            if num_nodes > 1:
                edge_type = (node_type, f"seq_{k}", node_type)
                src = list(range(num_nodes - 1))
                dst = list(range(1, num_nodes))
                graph_data[edge_type] = (torch.tensor(src), torch.tensor(dst))

            # Node features
            if k == 1:
                node_features[node_type] = torch.tensor(items[:num_nodes])
            else:
                # For higher granularities, store indices
                node_features[node_type] = torch.arange(num_nodes)

    # Inter-granularity edges
    for k in range(1, min(max_granularity, len(items))):
        src_type = f"gran_{k}"
        dst_type = f"gran_{k + 1}"

        if src_type in node_features and dst_type in node_features:
            # Upward edges
            edge_type_up = (src_type, f"up_{k}", dst_type)
            src_up, dst_up = [], []

            for i in range(len(items) - k):
                src_up.extend([i, i + 1])
                dst_up.extend([i, i])

            if src_up:
                graph_data[edge_type_up] = (torch.tensor(src_up), torch.tensor(dst_up))

            # Downward edges
            edge_type_down = (dst_type, f"down_{k}", src_type)
            src_down, dst_down = [], []

            for i in range(len(items) - k):
                src_down.extend([i, i])
                dst_down.extend([i, i + 1])

            if src_down:
                graph_data[edge_type_down] = (
                    torch.tensor(src_down),
                    torch.tensor(dst_down),
                )

    # Create heterogeneous graph
    g = dgl.heterograph(graph_data)

    # Add node features
    for node_type, features in node_features.items():
        if node_type in g.ntypes:
            g.nodes[node_type].data["feat"] = features

    return g
