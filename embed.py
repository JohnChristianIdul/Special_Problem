import torch
import torch.nn as nn
import math


import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class FeatureSelectionEmbedding(nn.Module):
    def __init__(self, c_in, d_model, selection_method='importance'):
        super(FeatureSelectionEmbedding, self).__init__()

        # Feature importance network
        self.feature_importance_network = nn.Sequential(
            nn.Linear(c_in, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, c_in),
            nn.Sigmoid()
        )

        self.selection_method = selection_method
        self.token_embedding = TokenEmbedding(c_in, d_model)

    def compute_feature_importance(self, x):
        """
        Compute feature importance scores
        """
        if self.selection_method == 'importance':
            # Learnable importance network
            importance_scores = self.feature_importance_network(
                x.mean(dim=[0, 1])
            )
        elif self.selection_method == 'variance':
            # Variance-based selection
            importance_scores = x.std(dim=[0, 1])
        else:
            # Default: uniform importance
            importance_scores = torch.ones(
                x.shape[-1],
                device=x.device
            )

        return torch.softmax(importance_scores, dim=-1)

    def select_features(self, x, threshold=0.1):
        """
        Select most important features
        """
        importance_scores = self.compute_feature_importance(x)

        top_k = max(
            1,
            int(x.shape[-1] * threshold)
        )

        _, selected_indices = torch.topk(
            importance_scores,
            k=min(top_k, x.shape[-1])
        )

        selected_features = x.index_select(
            dim=-1,
            index=selected_indices
        )

        return selected_features, selected_indices.tolist()

    def forward(self, x, threshold=0.1):
        """
        Forward pass with feature selection
        """
        # Select features
        x_selected, selected_indices = self.select_features(x, threshold)

        # Embed selected features
        embedded_features = self.token_embedding(x_selected)

        return embedded_features, selected_indices

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular'
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )

    def forward(self, x):
        # Handle different tensor dimensions
        if x.is_sparse:
            x = x.to_dense()

        # Ensure 3D tensor [batch_size, sequence_length, features]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            x = x.squeeze(-1)

        # Permute for convolution
        x = x.permute(0, 2, 1)

        # Apply convolution
        x = self.tokenConv(x).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):
    def __init__(
            self,
            c_in,
            d_model,
            embed_type='fixed',
            freq='h',
            dropout=0.1,
            selection_method='importance'
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = FeatureSelectionEmbedding(
            c_in=c_in,
            d_model=d_model,
            selection_method=selection_method
        )
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model,
            embed_type=embed_type,
            freq=freq
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, threshold=0.1):
        # Feature selection and embedding
        x_embedded, selected_indices = self.value_embedding(x, threshold)

        # Add positional and temporal embeddings
        x_embedded = x_embedded + \
                     self.position_embedding(x_embedded) + \
                     self.temporal_embedding(x_mark)

        return self.dropout(x_embedded), selected_indices