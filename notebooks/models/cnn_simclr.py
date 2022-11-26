import torch.nn as nn
from models.simple_cnn import SimpleCNN


class CNN_SimCLR(nn.Module): # we call it FireCLR

    def __init__(self, validation = False,n_channels=4, tile_size=32, out_dim=256):
        # inputting 4 channel RGB+NIR
        # the tile size (window size) is 32 by 32 pixels
        # dimension of the hidden-layer vector is 256
        super().__init__()
        self.n_channels = n_channels
        self.tile_size = tile_size
        self.out_dim = out_dim
        self.validation = validation

        # encoding part f(.)
        # a simple cnn architechture encodes the (32,32,4) to a 256-dimension hidden space
        self.encoder = self._get_cnn()
        # projection part g(.)
        # a fully connected layer transforms (256,1) hidden layer vector to (128,1)
        self.projection = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 128)
        )
        

    def _get_cnn(self):
        # import the cnn model from simple_cnn
        try:
            model = SimpleCNN(self.n_channels, self.tile_size, self.out_dim)
        except KeyError:
            raise ValueError(
                "Invalid architecture. Check the simple_cnn.py file")
        else:
            return model

    def forward(self, x):
        # return the proposed model
        # if for training, return the full model
        # if for validation, only return the encoding part f(.)
        encoding = self.encoder(x)
        projection = self.projection(encoding)

        if self.validation:
            return encoding
        else:
            return projection
