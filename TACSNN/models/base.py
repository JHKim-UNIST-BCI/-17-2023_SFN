import torch
from torch import nn
from TACSNN.layers import IzhikevichNeuron

class NeuralModule(nn.Module):
    def __init__(self, neuron_params, synapse_weights, buffer_size, device):
        super(NeuralModule, self).__init__()
        self.neuron = IzhikevichNeuron(**neuron_params, buffer_size=buffer_size, device=device)
        self.weights = nn.Parameter(synapse_weights, requires_grad=False)
        self.buffer_size = buffer_size

    def forward(self, x):
        post_spikes = self.neuron(x)
        post_potential = torch.matmul(self.weights, post_spikes)
        return post_potential

class SNN(nn.Module):
    def __init__(self, device='cpu'):
        super(SNN, self).__init__()
        neuron_params = {
            'a': 0.02,
            'b': 0.2,
            'c': -65,
            'd': 8,
            'v_thres': 30,
            'noise_std': 2,
            'num_neurons': 100
        }
        synapse_weights = torch.randn(100, 100)  # Random weights for demonstration
        buffer_size = 50
        self.layer1 = NeuralModule(neuron_params, synapse_weights, buffer_size, device)
        self.layer2 = NeuralModule(neuron_params, synapse_weights, buffer_size, device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x