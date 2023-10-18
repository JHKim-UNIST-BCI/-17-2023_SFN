import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SNNModel(nn.Module):
    def __init__(self, layers, synapses, device='cpu'):
        self.SA_layers = layers[0]
        self.RA_layers = layers[1]
        self.CN_layers = layers[2]

        self.SA_synapses = synapses[0]
        self.RA_synapses = synapses[1]
        self.CN_synapses = synapses[2]

        self.device = device

        self.reset_model()

    def reset_model(self):
        for layer_set, synapse_set in zip([self.SA_layers, self.RA_layers, self.CN_layers], [self.SA_synapses, self.RA_synapses, self.CN_synapses]):
            for layer, synapse in zip(layer_set, synapse_set):
                layer.reset()
                synapse.reset()

    def feedforward(self, stim):
        print('start feedforward')

        SA_stim = stim
        RA_stim = torch.abs(torch.diff(stim, dim=2)) * 30

        zeros_to_insert = torch.zeros(RA_stim.size(
            0), RA_stim.size(1), 1, device=self.device)

        # Concatenate the zeros tensor and the RA_stim tensor along the second dimension
        RA_stim = torch.cat((zeros_to_insert, RA_stim), dim=2)

        [self.SA_spike_times, self.RA_spike_times, self.CN_spike_times] = self.process_stim(
            SA_stim, self.SA_layers, self.SA_synapses,
            RA_stim, self.RA_layers, self.RA_synapses,
            self.CN_layers, self.CN_synapses
        )

    def process_stim(self,
                     SA_stim, SA_layers, SA_synapses,
                     RA_stim, RA_layers, RA_synapses,
                     CN_layers, CN_synapses):

        stim_len = SA_stim.shape[2]
        SA_spike_times = [torch.zeros(
            (synapse.weights.shape[0], stim_len - 1), device=self.device) for synapse in SA_synapses]
        RA_spike_times = [torch.zeros(
            (synapse.weights.shape[0], stim_len - 1), device=self.device) for synapse in RA_synapses]
        CN_spike_times = [torch.zeros(
            (synapse.weights.shape[0], stim_len - 1), device=self.device) for synapse in CN_synapses]

        for i in range(1, stim_len):
            # SA updates
            SA_rf_input = SA_synapses[0].cal_post_input(
                SA_stim[:, :, i].reshape(-1))
            SA_layers[0].update(SA_rf_input)

            SA_rf_input_in = SA_synapses[2].cal_post_input_delay(
                SA_layers[0].spike_buffer)
            SA_layers[2].update(SA_rf_input_in * 2)

            SA_rf_input_pn1 = SA_synapses[1].cal_post_input_delay(
                SA_layers[0].spike_buffer)

            # Update the PN layer with modified input (subtracting IN layer's spikes)'
            SA_layers[1].update(SA_rf_input_pn1 * 2 - SA_layers[2].spikes * 20)
            # print(SA_rf_input_pn1 * 2 - SA_layers[2].spikes * 20)

            SA_spike_times[0][:, i - 1] = SA_layers[0].spikes
            SA_spike_times[1][:, i - 1] = SA_layers[1].spikes
            SA_spike_times[2][:, i - 1] = SA_layers[2].spikes

            # RA updates
            RA_rf_input = RA_synapses[0].cal_post_input(
                RA_stim[:, :, i].reshape(-1))
            RA_layers[0].update(RA_rf_input)

            RA_rf_input_in = RA_synapses[2].cal_post_input_delay(
                RA_layers[0].spike_buffer)
            RA_layers[2].update(RA_rf_input_in * 2)

            RA_rf_input_pn1 = RA_synapses[1].cal_post_input_delay(
                RA_layers[0].spike_buffer)

            # Update the PN layer with modified input (subtracting IN layer's spikes)
            RA_layers[1].update(RA_rf_input_pn1 * 2 -
                                torch.sum(RA_layers[2].spikes))

            RA_spike_times[0][:, i - 1] = RA_layers[0].spikes
            RA_spike_times[1][:, i - 1] = RA_layers[1].spikes
            RA_spike_times[2][:, i - 1] = RA_layers[2].spikes

            CN_rf_input_in = CN_synapses[1].cal_post_input_delay(
                torch.cat((SA_layers[1].spike_buffer, RA_layers[1].spike_buffer), dim=0))
            # print(CN_rf_input_in)
            CN_layers[1].update(CN_rf_input_in)

            CN_rf_input_pn = CN_synapses[0].cal_post_input_delay(
                torch.cat((SA_layers[1].spike_buffer, RA_layers[1].spike_buffer), dim=0))
            CN_layers[0].update(CN_rf_input_pn - CN_layers[1].spikes * 10)

            CN_spike_times[0][:, i - 1] = CN_layers[0].spikes
            CN_spike_times[1][:, i - 1] = CN_layers[1].spikes

        return SA_spike_times, RA_spike_times, CN_spike_times
