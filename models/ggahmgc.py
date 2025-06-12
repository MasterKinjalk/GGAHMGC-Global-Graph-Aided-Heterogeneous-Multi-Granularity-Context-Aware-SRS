import torch
import torch.nn as nn
import torch.nn.functional as F
from .global_graph import GlobalGraphConstructor, GlobalGraphEncoder
from .multi_granularity import HierarchicalSessionGraph
from .attention import (
    GlobalContextAttention,
    SessionReadout,
    FusionGate,
    TargetAwareAttention,
)


class GGAHMGC(nn.Module):
    """
    Global Graph Aided Heterogeneous Multi-Granularity Context-Aware
    Session-based Recommendation Model
    """

    def __init__(self, config, global_graph_data=None):
        super().__init__()

        # Extract config parameters
        self.num_items = config["num_items"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.hidden_dim = config["model"]["hidden_dim"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout"]

        # Global graph settings
        self.num_neighbors = config["model"]["global_graph"]["num_neighbors"]
        self.global_attention_heads = config["model"]["global_graph"]["attention_heads"]

        # Multi-granularity settings
        self.max_granularity = config["model"]["multi_granularity"]["max_granularity"]

        # Initialize global graph
        if global_graph_data:
            self.global_graph = GlobalGraphConstructor(
                self.num_items,
                global_graph_data["cooccurrence"],
                global_graph_data["transition"],
                num_neighbors=self.num_neighbors,
            )
        else:
            self.global_graph = None

        # Components
        # 1. Global Graph Encoder
        self.global_encoder = GlobalGraphEncoder(
            self.num_items,
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.global_attention_heads,
            dropout=self.dropout,
        )

        # 2. Multi-granularity Hierarchical Session Graph
        self.hierarchical_encoder = HierarchicalSessionGraph(
            self.num_items,
            self.embedding_dim,
            self.hidden_dim,
            max_granularity=self.max_granularity,
            num_heads=8,
            dropout=self.dropout,
        )

        # 3. Global Context Attention (new component as requested)
        self.global_context_attention = GlobalContextAttention(
            self.embedding_dim,
            num_heads=self.global_attention_heads,
            dropout=self.dropout,
        )

        # 4. Session Readout
        self.session_readout = SessionReadout(
            self.embedding_dim, self.hidden_dim, dropout=self.dropout
        )

        # 5. Fusion Gate
        self.fusion_gate = FusionGate(self.embedding_dim, self.hidden_dim)

        # 6. Prediction Layer
        self.prediction = TargetAwareAttention(
            self.embedding_dim, self.hidden_dim, dropout=self.dropout
        )

        # Output projection
        self.output_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, batch):
        """
        Forward pass of GGAHMGC

        Args:
            batch: Dictionary containing:
                - input_items: [batch_size, seq_len]
                - masks: [batch_size, seq_len]
                - lengths: [batch_size]
        """
        input_items = batch["input_items"]
        masks = batch["masks"]
        lengths = batch["lengths"]

        batch_size, seq_len = input_items.shape

        # 1. Get initial session embedding for global context
        item_embeddings = self.global_encoder.item_embedding(input_items)
        masked_embeddings = item_embeddings * masks.unsqueeze(-1).float()
        initial_session_emb = (
            masked_embeddings.sum(dim=1) / masks.sum(dim=1, keepdim=True).float()
        )

        # 2. Global Graph Encoding
        if self.global_graph is not None:
            global_hidden, global_attention = self.global_encoder(
                input_items, self.global_graph, initial_session_emb, masks
            )
        else:
            # Fallback if no global graph
            global_hidden = item_embeddings
            global_attention = None

        # 3. Multi-granularity Hierarchical Encoding
        mg_fused, mg_granularities, mg_weights = self.hierarchical_encoder(
            input_items, lengths, masks
        )

        # Expand mg_fused to match sequence length for attention
        mg_expanded = mg_fused.unsqueeze(1).expand(-1, seq_len, -1)

        # 4. Apply Global Context Attention (new component)
        # This fuses global graph representations with multi-granularity representations
        fused_hidden, attention_weights = self.global_context_attention(
            global_hidden, mg_expanded, masks
        )

        # 5. Session-level Readout
        session_repr, readout_weights = self.session_readout(
            fused_hidden, lengths, masks
        )

        # 6. Final Fusion with Gating
        # Get global session representation
        global_session_repr, _ = self.session_readout(global_hidden, lengths, masks)

        # Fuse global and multi-granularity session representations
        final_session_repr, gate_values = self.fusion_gate(
            global_session_repr, session_repr
        )

        # 7. Output projection
        final_session_repr = self.output_proj(final_session_repr)

        # Store intermediate results for analysis
        self.intermediate_results = {
            "global_attention": global_attention,
            "mg_weights": mg_weights,
            "fusion_attention": attention_weights,
            "readout_weights": readout_weights,
            "gate_values": gate_values,
        }

        return final_session_repr

    def predict(self, batch):
        """
        Generate predictions for all items

        Args:
            batch: Input batch

        Returns:
            scores: [batch_size, num_items] - Scores for all items
        """
        # Get session representation
        session_repr = self.forward(batch)

        # Get all item embeddings
        all_items = torch.arange(1, self.num_items + 1, device=session_repr.device)
        item_embeddings = self.global_encoder.item_embedding(all_items)

        # Compute scores using target-aware attention
        scores = self.prediction(session_repr, item_embeddings)

        return scores

    def loss(self, batch):
        """
        Compute loss for training

        Args:
            batch: Input batch with 'targets' field

        Returns:
            loss: Scalar loss value
        """
        # Get predictions
        scores = self.predict(batch)

        # Get targets
        targets = batch["targets"] - 1  # Adjust for 0-indexing

        # Compute cross-entropy loss
        loss = self.loss_fn(scores, targets)

        return loss

    def get_attention_weights(self):
        """Get attention weights for visualization"""
        return self.intermediate_results
