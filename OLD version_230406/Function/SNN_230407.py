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
        for layer, synapse in zip(self.layers, self.synapses):
            layer.reset()
            synapse.reset()

    def feedforward(self, stim):

        input_spike = stim
        input_len = stim.shape[1]

        self.spike_times = [torch.zeros(
            (synapse.weights.shape[0], input_len)) for synapse in self.synapses]
        self.v = [torch.zeros(
            (synapse.weights.shape[0], input_len)) for synapse in self.synapses]

        for i in range(0, input_len):

            input_current = self.synapses[0].cal_post_input(stim[:, i])
            self.layers[0].update(input_current)

            self.spike_times[0][:, i] = self.layers[0].spikes
            self.v[0][:, i] = self.layers[0].v

        return self.spike_times, self.v

    def feedforward_delay(self, stim):

        input_spike = stim
        input_len = stim.shape[1]

        self.spike_times = [torch.zeros(
            (synapse.weights.shape[0], input_len)) for synapse in self.synapses]
        self.v = [torch.zeros(
            (synapse.weights.shape[0], input_len)) for synapse in self.synapses]
        
        window_size = 5
        
        for i in range(input_len - window_size + 1):
            stim_window = stim[:, i:i+window_size]

            input_current = self.synapses[0].cal_post_input_delay(stim_window, self.layers[0].buffer_size)
            self.layers[0].update(input_current)
            
            self.spike_times[0][:, i] = self.layers[0].spikes
            self.v[0][:, i] = self.layers[0].v

        return self.spike_times, self.v
