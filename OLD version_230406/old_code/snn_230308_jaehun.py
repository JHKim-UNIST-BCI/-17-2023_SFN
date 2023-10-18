import numpy as np
import time
import matplotlib.pyplot as plt


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
        self.spike_times = np.zeros((self.num_neurons,))
        print('layer initialized')

    def update(self, I, dt=1):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        self.spike_times = np.where(self.v >= 30, 1, 0)
        self.v = np.where(self.v >= 30, self.c, self.v)
        self.u = np.where(self.v >= 30, self.u + self.d, self.u)


class Synapse:
    def __init__(self, weights):
        self.weights = weights

    def update(self, pre_spike_times):
        post_input = np.dot(self.weights, pre_spike_times)
        return post_input


def plot_spike_times(spike_times):
    fig, ax = plt.subplots()
    ax.eventplot(spike_times, colors='k', linelengths=0)
    ax.set_xlim(0, spike_times.shape[1])
    ax.set_ylim(0, spike_times.shape[0])
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_title('Spike Times')
    plt.show()


if __name__ == '__main__':
    print('Start SNN_230307')

    # Layer Initialization
    L1_num_neurons = 1
    L1 = IzhikevichLayer(0.02, 0.2, -65, 8, L1_num_neurons)

    L2_num_neurons = 1
    L2 = IzhikevichLayer(0.02, 0.2, -65, 8, L2_num_neurons)

    weights = np.random.uniform(
        low=0.1, high=1, size=(L2_num_neurons, L1_num_neurons))
    synapse = Synapse(weights)

    # input signal
    signal_length = 10
    input_signal = np.random.uniform(low=50, high=100,
                                     size=(signal_length, L1_num_neurons))

    L1_spike_times = np.zeros((L1_num_neurons, signal_length))
    L2_spike_times = np.zeros((L2_num_neurons, signal_length))

    # simulation
    start_time = time.time()

    for i in range(len(input_signal)):
        I = input_signal[i]
        L1.update(I)
        pre_spike_times = L1.spike_times
        post_input = synapse.update(pre_spike_times)
        L2.update(post_input)
        L1_spike_times[:, i] = L1.spike_times
        L2_spike_times[:, i] = L2.spike_times

    print(L1_spike_times)
    end_time = time.time()
    vectorized_time = end_time - start_time
    # simulation end

    # Print the execution times
    print("Vectorized execution time:", vectorized_time)
    plot_spike_times(L1_spike_times)
    # plot_spike_times(L2_spike_times)
