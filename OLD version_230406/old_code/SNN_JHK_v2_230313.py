import matplotlib.pyplot as plt
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class IzhikevichLayer:
    def __init__(self, a, b, c, d, num_neurons):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.num_neurons = num_neurons
        # initial membrane potential
        self.v = -65 * np.ones((self.num_neurons,))
        self.u = self.b * self.v  # initial recovery variable
        self.spikes = np.zeros((self.num_neurons,))
        print('layer initialized')

    def update(self, I, dt=1):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        self.spikes = np.where(self.v >= 30, 1, 0)
        self.v = np.where(self.v >= 30, self.c, self.v)
        self.u = np.where(self.v >= 30, self.u + self.d, self.u)


class Synapse:
    def __init__(self, weights):
        self.weights = weights

    def cal_post_input(self, pre_spike_times):
        post_input = np.dot(self.weights, pre_spike_times)
        return post_input


def plot_spike_times(spikes):
    neuron_spike_times = []
    for i in range(spikes.shape[0]):
        neuron_spike_times.append(list(np.where(spikes[i, :] == 1)[0]))

    plt.eventplot(neuron_spike_times, colors='k', linelengths=0.4)
    plt.xlabel('Time')
    plt.ylabel('Neuron')
    plt.title('Spike Times')
    plt.show()


def plot_spike_times_with_rate(spikes, bin_size):
    neuron_spike_times = []
    num_neurons = spikes.shape[0]
    total_time = spikes.shape[1]
    for i in range(num_neurons):
        neuron_spike_times.append(list(np.where(spikes[i, :] == 1)[0]))

    # Create a histogram of spike times with the given bin size
    bins = np.arange(0, total_time, bin_size)
    spike_counts, _ = np.histogram(neuron_spike_times, bins=bins)

    # Calculate the firing rate by dividing the spike counts by the bin size
    firing_rates = spike_counts / bin_size

    # Plot the spike times
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.eventplot(neuron_spike_times, colors='k', linelengths=0.4)
    plt.xlabel('Time')
    plt.ylabel('Neuron')
    plt.title('Spike Times')

    # Plot the firing rate
    plt.subplot(2, 1, 2)
    plt.bar(bins[:-1], firing_rates, width=bin_size)
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Firing Rate')
    plt.show()

    # Print the total number of spikes for each neuron
    print("Total number of spikes per neuron:")
    print(np.sum(neuron_spike_times, axis=1))


def pick_28_in_100():
    # create a 10x10 array filled with zeros
    arr = np.zeros((10, 10))
    # define center of the array
    center = np.array([4.5, 4.5])
    # generate indices from a Gaussian distribution centered on the center
    indices = np.random.normal(loc=center, scale=2.5, size=(28, 2)).astype(int)
    # clip indices to ensure they fall within the array boundaries
    indices = np.clip(indices, 0, 9)
    # set the values of the array to 1 for the selected indices
    values = np.random.uniform(low=0.1, high=1, size=28)
    np.put(arr, indices[:, 0] * 10 + indices[:, 1], values)
    # print the resulting array
    return arr


def generate_sa1_receptivefield(pixel_w=64, pixel_h=48, kernel_w=10, kernel_h=10, step_size=6):
    rf = []

    num_step_h = (pixel_w - kernel_w) // step_size + 1
    num_step_v = (pixel_h - kernel_h) // step_size + 1
    print(num_step_h, num_step_v)

    for step_h in range(0, num_step_h * step_size, step_size):
        for step_v in range(0, num_step_v * step_size, step_size):
            tmp = np.zeros((pixel_h, pixel_w))

            tmp_arr = pick_28_in_100()

            tmp[step_v:step_v+kernel_h, step_h:step_h+kernel_w] = tmp_arr
            tmp_weight = tmp.reshape(-1)

            rf.append(tmp_weight)
    print("Complete! Create {}x{} kernel with step size {}! Generated {} times."
          .format(kernel_h, kernel_w, step_size, len(rf)))

    rf_array = np.vstack(rf)
