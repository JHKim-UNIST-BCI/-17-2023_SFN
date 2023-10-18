import matplotlib.pyplot as plt
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class IzhikevichLayer:
    def __init__(self, a, b, c, d, num_neurons, a_decay=1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.num_neurons = num_neurons
        # initial membrane potential
        self.v = -65 * np.ones((self.num_neurons,))
        self.u = self.b * self.v  # initial recovery variable
        self.spikes = np.zeros((self.num_neurons,))
        # initialize decay_a to a
        self.decay_a = np.full((self.num_neurons,), self.a)
        self.a_decay = a_decay  # decay factor for a
        print('layer initialized')

    def update(self, I, dt=1):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.decay_a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        self.spikes = np.where(self.v >= 30, 1, 0)
        self.decay_a = np.where(
            self.v >= 30, self.decay_a/self.a_decay, self.decay_a)
        self.v = np.where(self.v >= 30, self.c, self.v)
        self.u = np.where(self.v >= 30, self.u + self.d, self.u)


class Synapse:
    def __init__(self, weights):
        self.weights = weights

    def cal_post_input(self, pre_spike_times):
        post_input = np.dot(self.weights, pre_spike_times)
        return post_input

