import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        hidden_size = backbone.config.hidden_size

        self.attention_pool = nn.Linear(
            hidden_size,
            1
        )

        self.reward_head = nn.Linear(
            hidden_size,
            1
        )

        self.raw_tie_param = nn.Parameter(
            torch.tensor(0.5413)
        )

    def forward(self, input_ids, attention_mask):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = output.last_hidden_state

        attention_scores = self.attention_pool(
            hidden
        ).squeeze(-1)

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

        reward = self.reward_head(pooled)

        return reward.squeeze(-1)

    def get_tie_param(self):
        return F.softplus(self.raw_tie_param)


def preference_probabilities(reward_a, reward_b, tie_param=1.0):
    """
    Convert two reward scores into probabilities of:
    A wins, B wins, Tie
    """

    exp_a = torch.exp(reward_a)
    exp_b = torch.exp(reward_b)

    tie_term = (
        2 * tie_param *
        torch.exp((reward_a + reward_b) / 2)
    )

    denominator = exp_a + exp_b + tie_term

    prob_a = exp_a / denominator
    prob_b = exp_b / denominator
    prob_tie = tie_term / denominator

    return prob_a, prob_b, prob_tie