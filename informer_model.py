import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class Informer(nn.Module):

    def __init__(
            self,
            enc_in: int,  # Number of input features
            selection_method: str = 'importance',
            selection_threshold: float = 0.1,
            **kwargs
    ):
        super(Informer, self).__init__()

        # Feature selection parameters
        self.enc_in = enc_in
        self.selection_method = selection_method
        self.selection_threshold = selection_threshold

        # Feature importance network
        self.feature_importance_network = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.ReLU(),
            nn.Linear(enc_in, enc_in),
            nn.Sigmoid()
        )

        # Embedding layers
        self.enc_embedding = self._create_embedding(enc_in, kwargs.get('d_model', 512))

        # Encoder setup with feature selection focus
        self.encoder = self._create_feature_selector_encoder(
            kwargs.get('d_model', 512),
            kwargs.get('n_heads', 8)
        )

    def _create_embedding(self, input_dim: int, d_model: int):
        from embed import DataEmbedding
        return DataEmbedding(input_dim, d_model)

    def _create_feature_selector_encoder(self, d_model: int, n_heads: int):
        from encoder import Encoder, EncoderLayer
        from attention import ProbAttention, AttentionLayer

        encoder_layer = EncoderLayer(
            AttentionLayer(
                ProbAttention(),
                d_model,
                n_heads
            ),
            d_model
        )

        return Encoder([encoder_layer])

    def compute_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance scores
        """
        # Ensure x is 2D (batch, features)
        if x.dim() > 2:
            # If more than 2D, reduce to 2D by taking mean across extra dimensions
            x = x.mean(dim=tuple(range(2, x.dim())))

        # Compute feature-wise mean
        x_mean = x.mean(dim=0)

        if self.selection_method == 'importance':
            # If input is 1D, repeat to create a pseudo-batch
            if x_mean.dim() == 1:
                x_mean = x_mean.unsqueeze(0).repeat(2, 1)

            # Compute importance scores
            importance_scores = self.feature_importance_network(x_mean).mean(dim=0)
        elif self.selection_method == 'variance':
            importance_scores = x.std(dim=0)
        else:
            importance_scores = torch.ones(
                self.enc_in,
                device=x.device
            )

        return torch.softmax(importance_scores, dim=-1)

    def select_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        if x.dim() > 2:
            x = x.mean(dim=tuple(range(2, x.dim())))

        importance_scores = self.compute_feature_importance(x)

        top_k = max(
            1,
            int(self.enc_in * self.selection_threshold)
        )

        _, selected_indices = torch.topk(
            importance_scores,
            k=min(top_k, self.enc_in)
        )

        selected_features = x.index_select(
            dim=-1,
            index=selected_indices
        )

        return selected_features, selected_indices.tolist()

    def forward(
            self,
            x_enc: torch.Tensor,
            x_mark_enc: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, List[int]]:

        # Select features
        x_selected, selected_indices = self.select_features(x_enc)

        # Embed selected features
        x_embedded = self.enc_embedding(x_selected, x_mark_enc)

        # Process through encoder
        x_encoded, _ = self.encoder(x_embedded)

        return x_encoded, selected_indices