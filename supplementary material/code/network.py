import torch as th
from torch import nn
from torch.nn import functional as F

from minirl.common.policy import Extractor


class StateEmbeddingNet(Extractor):
    def __init__(self, input_shape) -> None:
        super().__init__(
            extractor_fn="cnn",
            extractor_kwargs={
                "input_shape": input_shape,
                "conv_kwargs": (
                    {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "out_channels": 128,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ),
                "hiddens": (),
                "activation": nn.ELU,
            },
        )


class ForwardDynamicNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.forward_dynamics = nn.Sequential(
            nn.Linear(128 + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(256, 128)

    def forward(self, state_embedding, action):
        # Embedding shape: T x B x C
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


class InverseDynamicNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(2 * 128, 256),
            nn.ReLU(),
        )
        self.id_out = nn.Linear(256, num_actions)

    def forward(self, state_embedding, next_state_embedding):
        # Embedding shape: T x B x C
        inputs = th.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits