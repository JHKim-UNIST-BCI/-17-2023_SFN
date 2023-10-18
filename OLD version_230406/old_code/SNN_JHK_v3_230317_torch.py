import torch
import torch.nn as nn
import numpy as np


class SNNModel(nn.Module):
    def __init__(self, layers, device = 'cpu'):
        self.layers = layers
        self.device = device

        print('Initializing SNN Model with {} layers with device {}'
              .format(len(self.layers),self.device))
        


    def feedforward(self, input_stim, synapses, plot_spikes = False):

        stim = input_stim
        stim_len = input_stim.shape[2]

        self.spike_times = [torch.zeros((synapse.weights.shape[0], stim_len - 1)) for synapse in synapses]
        
        for i in range(1, stim_len):
            # Update receptive_input and spikes for each layer
            # print(stim[:,:,i].reshape(-1))
            rf_input = synapses[0].cal_post_input(stim[:,:,i].reshape(-1))

            self.layers[0].update(rf_input)

            rf_input_2 = synapses[1].cal_post_input(self.layers[0].spikes)

            self.layers[1].update(rf_input_2)

            self.spike_times[0][:,i-1]=self.layers[0].spikes
            self.spike_times[1][:,i-1]=self.layers[1].spikes

        return self.spike_times
        # for layer, synapse in zip(self.layers, synapses):
        #     print(layer,synapse)
        