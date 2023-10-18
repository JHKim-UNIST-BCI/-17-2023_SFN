import matplotlib.pyplot as plt
import time
import numpy as np


def plot_spike_times(spikes, colors='k', size=(5, 3),xtick = [0, 500, 1000]):
    neuron_spike_times = []
    for i in range(spikes.shape[0]):
        neuron_spike_times.append(list(np.where(spikes[i, :] == 1)[0]))

    plt.figure(figsize=size)
    plt.eventplot(neuron_spike_times, colors=colors, linelengths=0.4)
    plt.xlabel('Time')
    plt.ylabel('Neuron index')
    plt.xticks(xtick)
    # plt.title('Spike Times')
    plt.xlim([0, spikes.shape[1]])
    plt.show()
