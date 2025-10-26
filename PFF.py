import torch
import torch.nn as nn


class PFF_PointCloud(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Keep channel dimension unchanged
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv1d(dim * 2, dim, kernel_size=1, bias=False)

        # Adjust input dimension for point attention layer
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        # Input directly uses (B, C, N) format
        # Feature concatenation and processing
        output = torch.cat([x, skip], dim=1)  # [B, 2C, N]

        # Global channel attention
        att = self.avg_pool(output)  # [B, 2C, 1]
        att = self.conv_atten(att)  # [B, 2C, 1]
        output = output * att  # [B, 2C, N]

        # Channel compression
        output = self.conv_redu(output)  # [B, C, N]

        # Spatial attention (cross-channel pooling)
        x_pool = x.mean(dim=1, keepdim=True)  # [B, 1, N]
        skip_pool = skip.mean(dim=1, keepdim=True)  # [B, 1, N]

        att = self.conv1(x_pool) + self.conv2(skip_pool)  # [B, 1, N]
        att = self.nonlin(att)

        return output * att  # Output maintains [B, C, N]

if __name__ == '__main__':
    # Generate test data in new format (B, C, N)
    input1 = torch.randn(3, 512, 1024)  # Channel dimension first
    input2 = torch.randn(3, 512, 1024)

    model = PFF_PointCloud(512)
    output = model(input1, input2)

    print("Input 1 shape:", input1.shape)  # Expected output: torch.Size([3, 512, 1024])
    print("Output shape:", output.shape)   # Expected output: torch.Size([3, 512, 1024])