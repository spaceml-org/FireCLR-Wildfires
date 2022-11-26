import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    4-layer fully conv CNN
    """
    def __init__(self, n_channels, tile_size, out_dim):
        super().__init__()
        
        self.conv = nn.Sequential(
            self.double_conv(n_channels, tile_size),
            self.double_conv(tile_size, out_dim),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten()
        )

    def double_conv(self,input_size,output_size):
        
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size = 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, kernel_size = 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)