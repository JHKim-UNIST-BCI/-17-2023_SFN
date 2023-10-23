import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from SNN.Izhikevich import *
from SNN.receptive_field import *

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        self.sa_rf, self.ra_rf, self.sa_rf_dim, self.ra_rf_dim = generate_primary_receptive_field_weights(pixel_h=320, pixel_w=240, device='cpu')

        self.sa_layer = IzhikevichLayer(size=self.sa_rf_dim[0]*self.sa_rf_dim[1])
        self.ra_layer = IzhikevichLayer(size=self.ra_rf_dim[0]*self.ra_rf_dim[1])
        
    def forward(self, stim):
        input = stim.squeeze().reshape(-1)

        sa_input = torch.matmul(self.sa_rf, input).unsqueeze(0)
        ra_input = torch.matmul(self.ra_rf, input).unsqueeze(0)

        sa_spikes = self.sa_layer(sa_input)
        ra_spikes = self.ra_layer(ra_input)

        return sa_spikes, ra_spikes
