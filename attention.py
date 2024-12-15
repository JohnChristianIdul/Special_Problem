import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _compute_feature_importance(self, Q, K):
        """
        Compute feature importance based on attention scores
        """
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Compute feature importance based on attention magnitude
        feature_importance = attn_scores.abs().mean(dim=[0, 1, 2])

        return feature_importance / feature_importance.sum()

    def _probabilistic_feature_selection(self, Q, K, sample_ratio=0.5):
        """
        Select most important features probabilistically
        """
        # Compute feature importance
        feature_importance = self._compute_feature_importance(Q, K)

        # Determine number of features to select
        num_features = max(1, int(feature_importance.size(0) * sample_ratio))

        # Select top features
        _, selected_indices = torch.topk(
            feature_importance,
            k=num_features
        )

        return selected_indices, feature_importance[selected_indices]

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose for multi-head processing
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Probabilistic feature selection
        selected_features, feature_weights = self._probabilistic_feature_selection(
            queries, keys
        )

        # Apply feature selection to queries, keys, and values
        queries_selected = queries[:, :, selected_features]
        keys_selected = keys[:, :, selected_features]
        values_selected = values[:, :, selected_features]

        # Compute attention scores
        scale = self.scale or 1. / sqrt(D)
        scores = torch.matmul(queries_selected, keys_selected.transpose(-2, -1)) * scale

        # Apply softmax
        attn = torch.softmax(scores, dim=-1)

        # Compute context
        context = torch.matmul(attn, values_selected)

        # Prepare output
        if self.output_attention:
            return (
                context.transpose(2, 1).contiguous(),
                {
                    'selected_features': selected_features,
                    'feature_weights': feature_weights,
                    'attention_matrix': attn
                }
            )
        else:
            return context.transpose(2, 1).contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project queries, keys, and values
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Compute attention
        out, additional_info = self.inner_attention(queries, keys, values)

        # Optional mixing
        if self.mix:
            out = out.transpose(2, 1).contiguous()

        # Reshape and project
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, additional_info
