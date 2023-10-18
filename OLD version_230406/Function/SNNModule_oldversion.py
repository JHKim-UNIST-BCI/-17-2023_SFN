import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class IzhikevichLayer:
    def __init__(self, a, b, c, d, num_neurons, a_decay=1, device='cpu'):
        self.device = device

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.noise_std = 0
        self.buffer_size = 50
        
        self.num_neurons = num_neurons
        self.init_values()

        self.a_decay = a_decay  # decay factor for a
        print('layers initialized')

    def init_values(self):
        # initial membrane potential
        self.v = -65 * torch.ones((self.num_neurons,), device=self.device)
        self.u = self.b * self.v  # initial recovery variable
        self.spikes = torch.zeros((self.num_neurons,), device=self.device)
        # initialize decay_a to a
        self.decay_a = torch.full(
            (self.num_neurons,), self.a, device=self.device)
        self.spike_buffer = torch.zeros((self.buffer_size,self.num_neurons), device=self.device)

    def reset(self):
        self.init_values()

    def update(self, I, dt=1):
        noise = torch.randn((self.num_neurons,), device=self.device) * self.noise_std

        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I
              + noise) * dt
        du = (self.decay_a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        self.spikes = torch.where(self.v >= 30, 1, 0)
        self.decay_a = torch.where(
            self.v >= 30, self.decay_a/self.a_decay, self.decay_a)
        self.u = torch.where(self.v >= 30, self.u + self.d, self.u)
        self.v = torch.where(self.v >= 30, torch.full_like(
            self.v, self.c, device=self.device), self.v)
        
        # # Update the spike buffer
        self.spike_buffer = torch.cat(
            [self.spike_buffer[1:], self.spikes.unsqueeze(0)], dim=0)


class Synapse:
    def __init__(self, weights, delays, device='cpu'):
        self.device = device
        self.initial_weights = weights.clone().to(self.device)
        self.weights = weights.to(self.device)
        self.delays = delays
        print('Synapses initialized')
    def reset(self):
        self.weights.copy_(self.initial_weights)

    def cal_post_input(self, pre_spike_times):
        
        post_input = torch.matmul(self.weights, pre_spike_times)
        
        return post_input
    
    def cal_post_input_delay(self, pre_spike_times):
        modified_delays = 49 - self.delays.long()
        print(modified_delays.shape)
        print(pre_spike_times.shape)
        delayed_spikes = pre_spike_times.gather(0, modified_delays)
        
        delayed_post_input = torch.einsum('ij,ij->i', self.weights, delayed_spikes)
        
        return delayed_post_input

    # def cal_delayed_post_input(self, pre_spike_times):
    #     # print('cal')
