import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwisePreferenceModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        hidden_size = backbone.config.hidden_size

        self.attention_pool = nn.Linear(
            hidden_size,
            1
        )

        fusion_size = hidden_size * 4

        self.preference_head = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)
        )

    def pool(self, hidden, attention_mask):

        attention_scores = self.attention_pool(
            hidden
        ).squeeze(-1)

        # Don't attend to padding
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0,
            -1e9
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=1
        )

        pooled = torch.sum(
            hidden * attention_weights.unsqueeze(-1),
            dim=1
        )

        return pooled

    def encode(self, input_ids, attention_mask):

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = output.last_hidden_state

        pooled = self.pool(
            hidden,
            attention_mask
        )

        return pooled

    def forward(self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b
    ):
        h_a = self.encode(
            input_ids_a,
            attention_mask_a
        )

        h_b = self.encode(
            input_ids_b,
            attention_mask_b
        )

        # Pairwise interaction features
        difference = torch.abs(h_a - h_b)

        product = h_a * h_b

        combined = torch.cat(
            [
                h_a,
                h_b,
                difference,
                product
            ],
            dim=1
        )

        logits = self.preference_head(
            combined
        )

        return logits