import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool),
        )

    def forward(self, x):
        return self.net(x)


class ECGQualityCNN(nn.Module):
    """
    Input:  [B, 12, 5000]
    Output: [B, 3]  (Clean / Noisy / Artifact logits)

    Architecture:
        ConvBlock(12->32,  k=7, pool=2)  -> [B, 32,  2500]
        ConvBlock(32->64,  k=5, pool=2)  -> [B, 64,  1250]
        ConvBlock(64->128, k=3, pool=4)  -> [B, 128,  312]
        AdaptiveAvgPool1d(1)             -> [B, 128,    1]
        Flatten + Dropout(0.3)           -> [B, 128]
        Linear(128 -> n_classes)         -> [B, 3]

    ~38k parameters. Trains in ~15 min on CPU.
    """
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(12,  32,  7, 2),
            ConvBlock(32,  64,  5, 2),
            ConvBlock(64, 128,  3, 4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# --- Quick sanity check ---
if __name__ == '__main__':
    m = ECGQualityCNN()
    x = torch.randn(4, 12, 5000)
    out = m(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # expect [4, 3]
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")
