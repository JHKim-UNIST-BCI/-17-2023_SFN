# Author: Jaehun Kim
# Email: rlawogns1204@unist.ac.kr
# Affiliation: UNIST BME BCILAB
# Date: 2023-06-12

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from Function.snn_IZHIlayers import *

class SNN(nn.Module):
    def __init__(self, R, noise_std_val = 0 ,a_decay = 1.01, device='cpu'):
        self.a_decay = a_decay

        self.sa_layers = [IzhikevichLayer(0.02, 0.2, -65, 8, len(R.sa_rf),v_thres=30, a_decay=self.a_decay , noise_std = noise_std_val, device=device),
                          IzhikevichLayer(0.1, 0.2, -65, 6, len(R.sa_cn_in_rf),v_thres=30, a_decay=1, noise_std = noise_std_val, device=device),
                          IzhikevichLayer(0.1, 0.2, -65, 6, len(R.sa_cn_pn_rf),v_thres=30, a_decay=1, noise_std = noise_std_val, device=device)]
        
        self.ra_layers = [IzhikevichLayer(0.02, 0.2, -65, 2, len(R.ra_rf),v_thres=30, a_decay=1, noise_std = noise_std_val,device=device),
                          IzhikevichLayer(0.1, 0.2, -65, 2, len(R.ra_cn_in_rf),v_thres=30, a_decay=1, noise_std=noise_std_val, device=device),
                          IzhikevichLayer(0.1, 0.2, -65, 2, len(R.ra_cn_pn_rf),v_thres=30, a_decay=1,noise_std = noise_std_val, device=device)]
        
        self.cn_layers = [IzhikevichLayer(0.02, 0.2, -65, 8, len(R.cn_in_sa_rf), v_thres=30, a_decay=1, noise_std = noise_std_val, device=device),
                          IzhikevichLayer(0.02, 0.2, -65, 8, len(R.cn_pn_sa_rf),v_thres=30, a_decay=1, noise_std = noise_std_val, device=device)]

        self.sa_synapses = [Synapse(R.sa_rf, device=device),
                            Synapse(R.sa_cn_in_rf, delays=R.sa_cn_SD, device=device),
                            Synapse(R.sa_cn_pn_rf, delays=R.sa_cn_SD, device=device),
                            Synapse(R.sa_intopn_rf, delays =R.sa_intopn_DN, tau_psp = 10, device = device)]
        self.ra_synapses = [Synapse(R.ra_rf, device=device),
                            Synapse(R.ra_cn_in_rf, delays=R.ra_cn_SD, device=device),
                            Synapse(R.ra_cn_pn_rf, delays=R.ra_cn_SD, device=device),
                            Synapse(R.ra_intopn_rf, delays = R.ra_intopn_DN, tau_psp = 10, device = device)]
        self.cn_synapses = [Synapse(R.cn_in_sa_rf, delays=R.cn_sa_SD, device=device),
                            Synapse(R.cn_pn_sa_rf, delays=R.cn_sa_SD, device=device),
                            Synapse(R.cn_in_ra_rf, delays=R.cn_ra_SD, device=device),
                            Synapse(R.cn_pn_ra_rf, delays=R.cn_ra_SD, device=device),
                            Synapse(R.cn_intopn_rf, delays = R.cn_intopn_DN, tau_psp = 10, device = device)]

        self.device = device
        self.reset_model()

    def reset_model(self):
        for layer_set, synapse_set in zip([self.sa_layers, self.ra_layers, self.cn_layers], [self.sa_synapses, self.ra_synapses, self.cn_synapses]):
            for layer, synapse in zip(layer_set, synapse_set):
                layer.reset()
                synapse.reset()

    def feedforward(self, stim, stim_name = None):
        # print('start feedforward with', stim_name)
        sa_stim = stim
        ra_stim = torch.abs(torch.diff(stim, dim=2)) * 5
        # ra_stim[:, :, :3] = 0
        zeros_to_insert = torch.zeros(ra_stim.size(0), ra_stim.size(1), 1, device=self.device)
        ra_stim = torch.cat((zeros_to_insert, ra_stim), dim=2)

        # print(sa_stim.shape)
        # plt.figure()
        # plt.plot(sa_stim[1, 1, :], label='sa_stim')
        # plt.plot(ra_stim[1, 1, :], label='ra_stim')
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Value')
        # plt.title('sa_stim and ra_stim')
        # plt.ylim([0,10])
        # plt.legend()
        # plt.show()
        
        [self.sa_spike_times, self.ra_spike_times, self.cn_spike_times] = self.process_stim(
            sa_stim, self.sa_layers, self.sa_synapses,
            ra_stim, self.ra_layers, self.ra_synapses,
            self.cn_layers, self.cn_synapses
        )

    def process_stim(self,sa_stim, sa_layers, sa_synapses,ra_stim, ra_layers, ra_synapses,cn_layers, cn_synapses):

        stim_len = sa_stim.shape[2]
        sa_spike_times = [torch.zeros(
            (layer.num_neurons, stim_len - 1), device=self.device) for layer in sa_layers]
        ra_spike_times = [torch.zeros(
            (layer.num_neurons, stim_len - 1), device=self.device) for layer in ra_layers]
        cn_spike_times = [torch.zeros(
            (layer.num_neurons, stim_len - 1), device=self.device) for layer in cn_layers]
        
        psp_sa = []
        psp_ra = []

        for i in range(1, stim_len):
            ##############################################################
            # sa updates
            sa_layers[0].update(sa_synapses[0].cal_post_input(sa_stim[:, :, i].reshape(-1)))
            sa_layers[1].update(sa_synapses[1].cal_post_input_delay(sa_layers[0].spike_buffer))

            sa_rf_input_PN1 = sa_synapses[2].cal_post_input_delay(sa_layers[0].spike_buffer)
            sa_rf_input_PN2 = sa_synapses[3].cal_post_input_delay(sa_layers[1].spike_buffer)

            sa_layers[2].update(sa_rf_input_PN1 * 1  - sa_rf_input_PN2 * 1)
            ##############################################################

            ##############################################################
            # ra updates
            ra_layers[0].update(ra_synapses[0].cal_post_input(ra_stim[:, :, i].reshape(-1)))
            ra_layers[1].update(ra_synapses[1].cal_post_input_delay(ra_layers[0].spike_buffer))

            ra_rf_input_PN1 = ra_synapses[2].cal_post_input_delay(ra_layers[0].spike_buffer)
            ra_rf_input_PN2 = ra_synapses[3].cal_post_input_delay(ra_layers[1].spike_buffer)
            ra_layers[2].update(ra_rf_input_PN1 * 1 - ra_rf_input_PN2 * 1)
            ##############################################################


            # self.cn_synapses = [Synapse(R.cn_in_sa_rf, delays=R.cn_sa_SD, device=device),
            #         Synapse(R.cn_pn_sa_rf, delays=R.cn_sa_SD, device=device),
            #         Synapse(R.cn_in_ra_rf, delays=R.cn_ra_SD, device=device),
            #         Synapse(R.cn_pn_ra_rf, delays=R.cn_ra_SD, device=device),
            #         Synapse(R.cn_intopn_rf, delays = R.cn_intopn_DN, tau_psp = 10, device = device)]

            ##############################################################
            # cn updates
            cn_IN_sa_rf_input = cn_synapses[0].cal_post_input_delay(sa_layers[2].spike_buffer)
            cn_IN_ra_rf_input = cn_synapses[2].cal_post_input_delay(ra_layers[2].spike_buffer)

            cn_layers[0].update( cn_IN_sa_rf_input * 1 + cn_IN_ra_rf_input * 1)

            cn_PN_sa_rf_input = cn_synapses[1].cal_post_input_delay(sa_layers[2].spike_buffer)
            cn_PN_ra_rf_input = cn_synapses[3].cal_post_input_delay(ra_layers[2].spike_buffer)

            psp_sa.append(cn_synapses[1].psp)
            psp_ra.append(cn_synapses[3].psp)

            cn_IN_rf_input = cn_synapses[4].cal_post_input_delay(cn_layers[0].spike_buffer)

            input_value = cn_PN_sa_rf_input * 2 + cn_PN_ra_rf_input * 2 - cn_IN_rf_input * 1
            cn_layers[1].update(input_value)
            ##############################################################

            for spike_times, layers in zip([sa_spike_times, ra_spike_times, cn_spike_times], 
                                        [sa_layers, ra_layers, cn_layers]):
                for layer, spike_time in zip(layers, spike_times):
                    spike_time[:, i - 1] = layer.spikes

        return sa_spike_times, ra_spike_times, cn_spike_times
