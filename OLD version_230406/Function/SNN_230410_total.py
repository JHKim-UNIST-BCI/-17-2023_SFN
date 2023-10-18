import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SNNModel(nn.Module):
    def __init__(self, layers, synapses, device='cpu'):
        self.SA_layers = layers[0]
        self.RA_layers = layers[1]

        self.SA_synapses = synapses[0]
        self.RA_synapses = synapses[1]
        self.device = device

        self.reset_model()

        # print('Initializing SNN Model with {} layers with device {}'
        #       .format(len(self.layers), self.device))

    def reset_model(self):
        for layer_set, synapse_set in zip([self.SA_layers, self.RA_layers], [self.SA_synapses, self.RA_synapses]):
            for layer, synapse in zip(layer_set, synapse_set):
                layer.reset()
                synapse.reset()

    def feedforward(self, stim):
        print('start feedforward')

        SA_stim = stim
        RA_stim = torch.abs(torch.diff(stim, dim=2)) * 20

        self.SA_spike_times = self.process_stim(
            SA_stim, self.SA_layers, self.SA_synapses)
        self.RA_spike_times = self.process_stim(
            RA_stim, self.RA_layers, self.RA_synapses)

    def process_stim(self, stim, layers, synapses):
        stim_len = stim.shape[2]
        spike_times = [torch.zeros(
            (synapse.weights.shape[0], stim_len - 1)) for synapse in synapses]

        for i in range(1, stim_len):
            rf_input = synapses[0].cal_post_input(stim[:, :, i].reshape(-1))
            layers[0].update(rf_input)

            rf_input_in = synapses[2].cal_post_input_delay(
                layers[0].spike_buffer)
            layers[2].update(rf_input_in)

            rf_input_pn1 = rf_input_in
            rf_input_in = synapses[1].cal_post_input_delay(
                layers[2].spike_buffer)

            spike_times[0][:, i - 1] = layers[0].spikes
            spike_times[2][:, i - 1] = layers[2].spikes

        return spike_times

    def feedforward_old(self, input_stim,  plot_spikes=False):

        stim = input_stim
        stim_len = input_stim.shape[2]

        self.spike_times = [torch.zeros(
            (synapse.weights.shape[0], stim_len - 1)) for synapse in self.synapses]

        for i in range(1, stim_len):

            rf_input = self.synapses[0].cal_post_input(
                stim[:, :, i].reshape(-1))
            self.layers[0].update(rf_input)

            rf_input_2 = self.synapses[1].cal_post_input_delay(
                self.layers[0].spike_buffer)

            self.layers[1].update(rf_input_2)

            self.spike_times[0][:, i-1] = self.layers[0].spikes
            self.spike_times[1][:, i-1] = self.layers[1].spikes

            if plot_spikes and i % 200 == 0:
                plt.imshow(stim[:, :, i].cpu(), cmap='jet', vmin=0, vmax=20)
                plt.colorbar()
                plt.show()

                plt.imshow(rf_input.reshape(19, 13).cpu(),
                           cmap='jet', vmin=0, vmax=500)
                plt.yticks([0, 18])
                plt.xticks([0, 12])
                plt.colorbar()
                plt.show()

                firing_count_image = torch.sum(
                    self.spike_times[0][:, i-100:i], axis=1)
                firing_count_image = firing_count_image.reshape(19, 13)
                plt.imshow(firing_count_image, cmap='jet', vmin=0, vmax=100)
                plt.yticks([0, 18])
                plt.xticks([0, 12])
                plt.colorbar()
                plt.show()

                # plt.imshow(rf_input_2.reshape(
                #     22, 17).cpu(), cmap='jet', vmin=0,)
                # plt.yticks([0, 21])
                # plt.xticks([0, 16])
                # plt.colorbar()
                # plt.show()

                # firing_count_image = torch.sum(
                #     self.spike_times[1][:, i-100:i], axis=1)
                # firing_count_image = firing_count_image.reshape(22, 17)
                # plt.imshow(firing_count_image, cmap='jet', vmin=0)
                # plt.yticks([0, 21])
                # plt.xticks([0, 16])
                # plt.colorbar()
                # plt.show()

        return self.spike_times
