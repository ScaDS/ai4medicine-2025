from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, width: float = 1.0):
        super().__init__()
        f1 = int(16 * width)
        f2 = int(32 * width)
        f3 = int(64 * width)
        self.net = nn.Sequential(
            nn.Conv2d(3,  f1, 3, padding=1, bias=False), nn.BatchNorm2d(f1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 224 -> 112
            nn.Conv2d(f1, f2, 3, padding=1, bias=False), nn.BatchNorm2d(f2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 112 -> 56
            nn.Conv2d(f2, f3, 3, padding=1, bias=False), nn.BatchNorm2d(f3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                          # global average pool
            nn.Flatten(),
            nn.Linear(f3, num_classes)
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
