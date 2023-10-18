# Author: Jaehun Kim
# Email: rlawogns1204@unist.ac.kr
# Affiliation: UNIST BME BCILAB
# Date: 2023-05-19

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


class SNNModel(nn.Module):
    def __init__(self, layers, synapses, rf_sizes, device='cpu'):
        self.SA_layers = layers[0]
        self.RA_layers = layers[1]
        self.CN_layers = layers[2]

        self.SA_synapses = synapses[0]
        self.RA_synapses = synapses[1]
        self.CN_synapses = synapses[2]

        self.device = device
        self.rf_sizes = rf_sizes
        self.reset_model()

    def reset_model(self):
        for layer_set, synapse_set in zip([self.SA_layers, self.RA_layers, self.CN_layers], [self.SA_synapses, self.RA_synapses, self.CN_synapses]):
            for layer, synapse in zip(layer_set, synapse_set):
                layer.reset()
                synapse.reset()

    def feedforward(self, stim, stim_name = None):
        print('start feedforward with', stim_name)
        SA_stim = stim - 3
        RA_stim = torch.abs(torch.diff(stim, dim=2)) * 10
        RA_stim[:, :, :3] = 0
        zeros_to_insert = torch.zeros(RA_stim.size(0), RA_stim.size(1), 1, device=self.device)
        RA_stim = torch.cat((zeros_to_insert, RA_stim), dim=2)


        
        print(SA_stim.shape)
        plt.figure()
        plt.plot(SA_stim[40, 40, :], label='SA_stim')
        plt.plot(RA_stim[40, 40, :], label='RA_stim')
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.title('SA_stim and RA_stim')
        plt.ylim([0,10])
        plt.legend()
        plt.show()
        

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
        
        psp_sa = []
        psp_ra = []

        for i in range(1, stim_len):
            ##############################################################
            # SA updates
            SA_layers[0].update(SA_synapses[0].cal_post_input(SA_stim[:, :, i].reshape(-1)))
            SA_layers[1].update(SA_synapses[1].cal_post_input_delay(SA_layers[0].spike_buffer))

            SA_rf_input_PN1 = SA_synapses[2].cal_post_input_delay(SA_layers[0].spike_buffer)
            SA_rf_input_PN2 = SA_synapses[3].cal_post_input_delay(SA_layers[1].spike_buffer)

            SA_layers[2].update(SA_rf_input_PN1 * 2  - SA_rf_input_PN2 * 1)
            ##############################################################

            ##############################################################
            # RA updates
            RA_layers[0].update(RA_synapses[0].cal_post_input(RA_stim[:, :, i].reshape(-1)))
            RA_layers[1].update(RA_synapses[1].cal_post_input_delay(RA_layers[0].spike_buffer))

            RA_rf_input_PN1 = RA_synapses[2].cal_post_input_delay(RA_layers[0].spike_buffer)
            RA_rf_input_PN2 = RA_synapses[3].cal_post_input_delay(RA_layers[1].spike_buffer)
            RA_layers[2].update(RA_rf_input_PN1 * 2 - RA_rf_input_PN2 * 1)

            ##############################################################

            ##############################################################
            # CN updates
            CN_IN_SA_rf_input = CN_synapses[0].cal_post_input_delay(SA_layers[2].spike_buffer)
            CN_IN_RA_rf_input = CN_synapses[2].cal_post_input_delay(RA_layers[2].spike_buffer)

            CN_layers[0].update( CN_IN_SA_rf_input * 1 + CN_IN_RA_rf_input * 1)

            CN_PN_SA_rf_input = CN_synapses[1].cal_post_input_delay(SA_layers[2].spike_buffer)
            CN_PN_RA_rf_input = CN_synapses[3].cal_post_input_delay(RA_layers[2].spike_buffer)

            psp_sa.append(CN_synapses[1].psp)
            psp_ra.append(CN_synapses[3].psp)

            CN_IN_rf_input = CN_synapses[4].cal_post_input_delay(CN_layers[0].spike_buffer)

            input_value = CN_PN_SA_rf_input * 2 + CN_PN_RA_rf_input * 2 - CN_IN_rf_input * 1
            CN_layers[1].update(input_value)

            ##############################################################

            for spike_times, layers in zip([SA_spike_times, RA_spike_times, CN_spike_times], 
                                        [SA_layers, RA_layers, CN_layers]):
                for layer, spike_time in zip(layers, spike_times):
                    spike_time[:, i - 1] = layer.spikes

        return SA_spike_times, RA_spike_times, CN_spike_times
