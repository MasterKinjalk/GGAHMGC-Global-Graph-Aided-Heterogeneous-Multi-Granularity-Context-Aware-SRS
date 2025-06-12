import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalContextAttention(nn.Module):
    """Enhanced attention mechanism for global context fusion"""

    def __init__(self, embedding_dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, (
            "Embedding dim must be divisible by num_heads"
        )

        # Multi-head attention components
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, global_repr, local_repr, mask=None):
        """
        Apply attention between global and local representations

        Args:
            global_repr: [batch_size, seq_len, embedding_dim] - Global context representations
            local_repr: [batch_size, seq_len, embedding_dim] - Local multi-granularity representations
            mask: [batch_size, seq_len] - Attention mask
        """
        batch_size, seq_len, _ = global_repr.shape

        # Project to Q, K, V
        Q = self.q_proj(local_repr).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        K = self.k_proj(global_repr).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        V = self.v_proj(global_repr).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            math.sqrt(self.head_dim) * self.temperature
        )

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Transpose and reshape
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embedding_dim)
        )

        # Output projection
        output = self.out_proj(context)

        # Residual connection and layer norm
        output = self.layer_norm(local_repr + self.dropout(output))

        return output, attn_weights


class SessionReadout(nn.Module):
    """Attention-based readout for session representation"""

    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Position-aware attention
        self.position_embedding = nn.Embedding(100, embedding_dim)

        # Last item attention
        self.last_item_weight = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, lengths, mask=None):
        """
        Compute session representation using attention readout

        Args:
            hidden_states: [batch_size, seq_len, embedding_dim]
            lengths: [batch_size] - Actual sequence lengths
            mask: [batch_size, seq_len] - Padding mask
        """
        batch_size, seq_len, embedding_dim = hidden_states.shape

        # Add position embeddings
        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        hidden_with_pos = hidden_states + pos_emb

        # Compute attention scores
        attn_scores = self.attention(hidden_with_pos).squeeze(-1)

        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Get last item representations
        last_indices = lengths - 1
        last_item_repr = hidden_states[torch.arange(batch_size), last_indices]

        # Combine attention
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention
        session_repr = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)

        # Combine with last item
        final_weight = torch.sigmoid(self.last_item_weight)
        session_repr = final_weight * last_item_repr + (1 - final_weight) * session_repr

        return session_repr, attn_weights


class FusionGate(nn.Module):
    """Gated fusion mechanism for combining global and local representations"""

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        # Gate computation
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )

        # Transform layers
        self.transform_global = nn.Linear(embedding_dim, embedding_dim)
        self.transform_local = nn.Linear(embedding_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, global_repr, local_repr):
        """
        Fuse global and local representations using gating mechanism

        Args:
            global_repr: [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
            local_repr: [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        # Compute gate values
        concat_repr = torch.cat([global_repr, local_repr], dim=-1)
        gate_values = self.gate(concat_repr)

        # Transform representations
        global_transformed = self.transform_global(global_repr)
        local_transformed = self.transform_local(local_repr)

        # Apply gating
        fused = gate_values * global_transformed + (1 - gate_values) * local_transformed

        # Layer norm
        fused = self.layer_norm(fused)

        return fused, gate_values


class TargetAwareAttention(nn.Module):
    """Target-aware attention for candidate item scoring"""

    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()

        # Attention layers
        self.W_1 = nn.Linear(embedding_dim, hidden_dim)
        self.W_2 = nn.Linear(embedding_dim, hidden_dim)
        self.W_3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, session_repr, target_embeddings):
        """
        Compute attention scores for target items

        Args:
            session_repr: [batch_size, embedding_dim]
            target_embeddings: [num_items, embedding_dim]
        """
        batch_size = session_repr.shape[0]
        num_items = target_embeddings.shape[0]

        # Expand session representation
        session_expanded = session_repr.unsqueeze(1).expand(batch_size, num_items, -1)
        target_expanded = target_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention
        session_proj = self.W_1(session_expanded)
        target_proj = self.W_2(target_expanded)

        # Combine and compute scores
        h = self.tanh(session_proj + target_proj)
        scores = self.W_3(self.dropout(h)).squeeze(-1)

        return scores
