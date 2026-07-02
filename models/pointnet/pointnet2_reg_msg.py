"""
This model is for regression on QM7, QM9 datasets using the PointNet2 architecture
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from dataclasses import dataclass

class get_model(nn.Module):
    def __init__(self, out_channels: int):

        super(get_model, self).__init__()
        self.out_channels = out_channels
        self.pointnet = PointNet2Encoder()

        self.linear_layer_1 = nn.Linear(out_channels, 64)
        self.linear_layer_cent = nn.Linear(64, 3)
        self.linear_layer_dist = nn.Linear(64, 1)


    def forward(self, pcl: torch.Tensor) -> torch.Tensor:
        n_batch, n_points, _ = pcl.shape
        out_features = self.pointnet(pcl)

        features_1 = F.relu(self.linear_layer_1(out_features))

        cent = self.linear_layer_cent(features_1)
        dist = self.linear_layer_dist(features_1)
        out = torch.cat([cent, dist],dim=-1)
        return out



@dataclass
class PointNet2EncoderConfig:
    # in_channel: int = MISSING
    in_channel: int = 6

class PointNet2Encoder(nn.Module):
    def __init__(self, cfg: PointNet2EncoderConfig = None):
        super().__init__()
        self.cfg = cfg
        if cfg is None:
            self.in_channel = 6
        else:
            self.in_channel = cfg.in_channel

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.1,
                nsample=64,
                mlp=[self.in_channel, 64, 64, 128],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=128,
                mlp=[128, 128, 128, 256],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 512]
            )
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

    def forward(self, pc):
        " pc: (B, N, 3+C) "
        xyz = pc[:, :, :3].contiguous()
        pc_extended = torch.cat([pc, pc], dim=-1)
        features = pc_extended.transpose(1, 2).contiguous() if pc_extended.size(-1) > 3 else None
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.fc_layer(features.squeeze(-1))


# class PointNet2MSGModel(nn.Module):
#     def __init__(self, 
#                     n_centroids_1: int,
#                     msg_radii_1: List[float],
#                     msg_nsample_1: List[int],
#                     n_centroids_2: int,
#                     msg_radii_2: List[float],
#                     msg_nsample_2: List[int],
#                     in_channels: int,
#                     out_channels: int):
#         super(PointNet2MSGModel, self).__init__()
#         self.normal_channel = True
#         self.sa1 = PointNetSetAbstractionMsg(n_centroids_1, 
#                                                 msg_radii_1, 
#                                                 msg_nsample_1, 
#                                                 in_channels,
#                                                 [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
#         self.sa2 = PointNetSetAbstractionMsg(n_centroids_2, 
#                                                 msg_radii_2, 
#                                                 msg_nsample_2, 
#                                                 320,
#                                                 [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
#         self.sa3 = PointNetSetAbstraction(None, 
#                                             None, 
#                                             None, 
#                                             640 + 3, 
#                                             [256, 512, 1024], 
#                                             True)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(256, out_channels)

#     def forward(self, xyz: torch.Tensor) -> torch.Tensor:
#         """Forward pass of the regression network

#         Args:
#             xyz (torch.Tensor): has shape (batch_size, max_n_atoms, 3 + feature_dim)

#         Returns:
#             torch.Tensor: size [batch_size,]
#         """
#         B, _, _ = xyz.shape
#         in_xyz = xyz[:, :, :3] # Norm are the features in the 4th and on columns
#         in_features = xyz[:, :, :3] # XYZ are the cartesian coordinates
        
#         l1_xyz, l1_features = self.sa1(in_xyz.permute(0, 2, 1), in_features.permute(0, 2, 1))
#         if torch.any(torch.isnan(l1_xyz)):
#             raise ValueError("l1_xyz contains NaNs")
        
#         if torch.any(torch.isnan(l1_features)):
#             raise ValueError("l1_features contains NaNs")
#         l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
#         l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
#         x = l3_features.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         # x = F.log_softmax(x, -1)
#         return x


class get_loss(nn.Module):
    # def __init__(self):
    #     super(get_loss, self).__init__()

    # def forward(self, pred, target):
    #     return torch.mean(torch.square(pred.flatten() - target.flatten()))
    def __init__(self):
        super(get_loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        return self.criterion(pred.flatten(), target.flatten())

