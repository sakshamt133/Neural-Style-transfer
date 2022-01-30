import torch.nn as nn
from torchvision.models import vgg19


class Model(nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.model = vgg19(pretrained=True).features
        self.layers = layers

    def forward(self, x):
        out = []

        for i, layer in enumerate(self.model):
            x = layer(x)

            if i in self.layers:
                out.append(x)

        return out
