import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Enable GPU support if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class IzhikevichLayer:
    def __init__(self, a, b, c, d, num_neurons, a_decay=1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.num_neurons = num_neurons
        # initial membrane potential
        self.v = -65 * torch.ones((self.num_neurons,), device=device)
        self.u = self.b * self.v  # initial recovery variable
        self.spikes = torch.zeros((self.num_neurons,), device=device)
        # initialize decay_a to a
        self.decay_a = torch.full((self.num_neurons,), self.a, device=device)
        self.a_decay = a_decay  # decay factor for a
        print('layer initialized')

    def update(self, I, dt=1):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.decay_a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        self.spikes = torch.where(self.v >= 30, 1, 0)
        self.decay_a = torch.where(
            self.v >= 30, self.decay_a/self.a_decay, self.decay_a)
        self.v = torch.where(self.v >= 30, torch.full_like(self.v, self.c, device=device), self.v)
        self.u = torch.where(self.v >= 30, self.u + self.d, self.u)


class Synapse:
    def __init__(self, weights):
        self.weights = weights.to(device)  # Move weights to GPU if available

    def cal_post_input(self, pre_spike_times):
        pre_spike_times_float = pre_spike_times.float()
        post_input = torch.matmul(self.weights, pre_spike_times_float)
        return post_input
