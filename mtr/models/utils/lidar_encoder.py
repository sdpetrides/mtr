import torch
import torch.nn as nn
from ..utils import common_layers


class LidarEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, out_channels=256):
        super().__init__()
        self.dense1 = nn.Linear(
            in_features=in_channels, out_features=hidden_dim, bias=True
        )
        self.batch1 = nn.BatchNorm1d(
            hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(
            in_features=hidden_dim, out_features=out_channels, bias=True
        )
        self.batch2 = nn.BatchNorm1d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        """
        Args:
            lidar_points (batch_size, num_points, C):

        Returns:
        """
        x = self.dense1(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.batch1(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.relu1(x)
        x = self.dense2(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.batch2(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.relu2(x)

        return x
