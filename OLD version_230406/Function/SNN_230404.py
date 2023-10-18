import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SNNModel(nn.Module):
    def __init__(self, layers, synapses, device='cpu'):
        self.layers = layers
        self.device = device
        self.synapses = synapses

        self.reset_model()

        print('Initializing SNN Model with {} layers with device {}'
              .format(len(self.layers), self.device))

    def reset_model(self):
        for layer in self.layers:
            layer.reset()

    def feedforward(self, input_stim):

        stim = input_stim
        stim_len = input_stim.shape[2]
        
        self.spike_times = [torch.zeros(
            (synapse.weights.shape[0], stim_len - 1)) for synapse in self.synapses]
        self.v = torch.zeros(stim_len+1)

        for i in range(1, stim_len):
            print(stim[:, :, i].shape)
            rf_input = self.synapses[0].cal_post_input(
                stim[:, :, i].reshape(-1))
    
            self.layers[0].update(rf_input)

            self.spike_times[0][:, i-1] = self.layers[0].spikes
            self.v[i+1] = self.layers[0].v

        return self.spike_times
