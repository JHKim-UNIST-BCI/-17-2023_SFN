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

    def feedforward(self, input_stim,  plot_spikes=False):

        stim = input_stim
        stim_len = input_stim.shape[2]

        self.spike_times = [torch.zeros(
            (synapse.weights.shape[0], stim_len - 1)) for synapse in self.synapses]

        for i in range(1, stim_len):

            rf_input = self.synapses[0].cal_post_input(
                stim[:, :, i].reshape(-1))
            print(rf_input.shape)
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