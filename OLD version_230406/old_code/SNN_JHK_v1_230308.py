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

    print(np.sum(neuron_spike_times, axis=0))


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


def plot_2d_sine_wave(ncols=640, nrows=480, freq=0.1, amp=5):
    X, Y = np.meshgrid(np.arange(1, nrows+1), np.arange(1, ncols+1))

    Z = amp*np.sin(2*np.pi*freq*X + np.pi/4)*np.sin(2*np.pi*freq*Y + np.pi/4)
    plt.gray()

    # Plot the 2D sine wave
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    # Show the plot
    plt.show()
    return Z


def createReceptiveField(receptiveField, numFinger, sensor_h, sensor_w):
    rf = []
    for i in range(len(receptiveField)):
        reft_finger_rf = []
        right_finger_rf = []

        kernel = receptiveField[i]
        kernal_h, kernal_w = kernel.shape
        num_sensor_array = sensor_h * sensor_w

        num_step_h = sensor_w - kernal_w + 1
        num_step_v = sensor_h - kernal_h + 1

        for step_h in range(num_step_h):
            for step_v in range(num_step_v):
                tmp = np.zeros((sensor_h, sensor_w))

                tmp[step_v:step_v+kernal_h, step_h:step_h+kernal_w] = kernel

                tmp_weight1 = np.concatenate(
                    (tmp.reshape(-1), np.zeros(num_sensor_array)))
                tmp_weight2 = np.concatenate(
                    (np.zeros(num_sensor_array), tmp.reshape(-1)))

                reft_finger_rf.append(tmp_weight1)
                right_finger_rf.append(tmp_weight2)

        rf.append(np.concatenate((reft_finger_rf, right_finger_rf), axis=0))
        print("Complete! Create {}x{} kernel! with {} times.".format(
            kernal_h, kernal_w, len(rf[i])))

    # Stack the arrays vertically to create a single array
    rf_array = np.vstack(rf)

    return rf_array
