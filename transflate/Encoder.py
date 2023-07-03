"""
The goal of Encoder class is to implement the stack of N layers (EncoderLayer)
test
"""
import torch.nn as nn
from copy import deepcopy
from transflate.helper import clones
from transflate.LayerNorm import LayerNorm
from transflate.EncoderLayer import EncoderLayer

class Encoder(nn.Module):

    def __init__(self, layer:EncoderLayer, N:int):
        super().__init__()
        self.layers = clones(layer, N) 
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask) # EncoderLayer
        return self.norm(x)