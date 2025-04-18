
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Helper Functions ---
def knn(x, k):
    # ... (keep the full knn function code here) ...
    B, F, N = x.size()
    k = min(k, N) # Ensure k does not exceed the number of points N.
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1] # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    # ... (keep the full get_graph_feature function code here) ...
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k) # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    # Ensure indices are within bounds after flattening
    idx = idx.clamp(0, x.view(batch_size * num_points, -1).shape[0] - 1)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) # Adjusted view dimensions

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # Ensure feature and x have compatible shapes for subtraction
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (B, 2*F, N, k)

    return feature


# --- Model Definition ---
class DGCNN(nn.Module):
    def __init__(self, input_channels=6, output_channels=2, k=20, emb_dims=1024):
        # ... (keep the full DGCNN class definition here) ...
        super(DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # MLP for classification
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # ... (keep the full forward method here) ...
        batch_size = x.size(0)
        num_points = x.size(2) # Get N from input [B, F, N]

        x = x.transpose(2, 1) # Convert to [B, N, F] for feature extraction helper
        x = x.transpose(1, 2) # Back to [B, F, N] for get_graph_feature

        edge_feature = get_graph_feature(x, k=self.k) # [B, 2*F, N, k]
        x1 = self.conv1(edge_feature) # [B, 64, N, k]
        x1 = x1.max(dim=-1, keepdim=False)[0] # [B, 64, N]

        edge_feature = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(edge_feature)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        edge_feature = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(edge_feature)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        edge_feature = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(edge_feature)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1) # [B, 512, N]

        x = self.conv5(x_cat) # [B, emb_dims, N]

        if num_points > 0:
            x1_pooled = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # [B, emb_dims]
            x2_pooled = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # [B, emb_dims]
            x_pooled = torch.cat((x1_pooled, x2_pooled), 1) # [B, emb_dims*2]
        else:
             return torch.zeros((batch_size, self.linear3.out_features), device=x.device)

        out = F.leaky_relu(self.bn1(self.linear1(x_pooled)), negative_slope=0.2)
        out = self.dp1(out)
        out = F.leaky_relu(self.bn2(self.linear2(out)), negative_slope=0.2)
        out = self.dp2(out)
        out = self.linear3(out)

        return out
