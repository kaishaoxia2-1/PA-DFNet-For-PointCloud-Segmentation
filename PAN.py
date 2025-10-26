import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PAN(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], last_bn=False, temp=1, num_heads=4, alpha=1.5):
        super(PAN, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.temp = temp
        self.num_heads = num_heads
        self.alpha = alpha

        # Multi-layer convolution
        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.append(out_channel)
        hidden_unit.insert(0, in_channel)

        for i in range(1, len(hidden_unit)):  # from 1st hidden to next hidden to last hidden
            # Multi-layer convolution
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            # Batch normalization layer
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))

        # Polar Transformer parameters
        head_dim = out_channel // num_heads
        self.head_dim = head_dim
        self.qg = nn.Linear(out_channel, 2 * out_channel)
        self.kv = nn.Linear(out_channel, 2 * out_channel)
        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, out_channel)))
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=5,
                             groups=head_dim, padding=5 // 2)

        # Positional encoding parameters
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, 1, out_channel)))

    def forward(self, xyz, score_norm='softmax'):

        # xyz :  (B, 3, N, K)
        B, _, N, K = xyz.size()
        scores = xyz


        for i, conv in enumerate(self.mlp_convs_hidden):
            # Not the last layer
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))
            # Special handling for the last layer
            else:
                scores = conv(scores)
                if self.last_bn:  # Optional BN
                    scores = self.mlp_bns_hidden[i](scores)

        # ========== Polar processing module ==========
        scores = scores.permute(0, 2, 3, 1).reshape(B * N, K, -1)  # Process each neighborhood
        q, g = self.qg(scores).reshape(B * N, K, 2, -1).unbind(2)
        kv = self.kv(scores).reshape(B * N, K, 2, -1).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        # Add positional encoding to key vectors
        k = k + self.positional_encoding
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * torch.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B * N, K, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B * N, K, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B * N, K, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (K ** -0.5)) @ (v1 * (K ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (K ** -0.5)) @ (v2 * (K ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B * N, K, -1)

        v = v.reshape(B * N * self.num_heads, self.head_dim, K, 1)
        v = self.dwc(v).reshape(B * N, -1, K).permute(0, 2, 1)
        x = x + v
        x = x * g

        scores = x.reshape(B, N, K, -1).permute(0, 3, 1, 2)
        # ========== End module ==========
        # Choose final normalization method
        if score_norm == 'softmax':
            scores = F.softmax(scores / self.temp, dim=1)  # + 0.5  # B*m*N*K
        elif score_norm == 'sigmoid':
            scores = torch.sigmoid(scores / self.temp)  # + 0.5  # B*m*N*K
        elif score_norm is None:
            scores = scores
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores


# Test
if __name__ == '__main__':
    import torch

    # Generate test data (B, 3, N, K)
    # B: batch size, 3: coordinate dimensions, N: number of points, K: neighborhood size
    xyz = torch.randn(2, 3, 1024, 32)

    # Initialize model
    model = PAN(in_channel=3, out_channel=16)

    # Forward pass
    output = model(xyz)

    # Print basic information
    print(f"Input shape: {xyz.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value range: [{output.min():.4f}, {output.max():.4f}]")