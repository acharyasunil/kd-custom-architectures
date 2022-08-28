import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, model_layers):
        super().__init__()
        self.layers = model_layers
        #self.trace = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            #self.trace.append(x)
        return x


