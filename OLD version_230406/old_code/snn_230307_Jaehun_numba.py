import numpy as np
import time
from numba import njit


@njit
def izhikevich_update(v, u, I, a, b, c, d, dt):
    dv = (0.04 * v**2 + 5 * v + 140 - u + I) * dt
    du = (a * (b * v - u)) * dt
    v += dv
    u += du
    spike_times = np.where(v >= 30, 1, 0)
    v = np.where(v >= 30, c, v)
    u = np.where(v >= 30, u + d, u)
    return v, u, spike_times


class IzhikevichLayer_v3:
    def __init__(self, a, b, c, d, num_neurons, signal_lenght):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.num_neurons = num_neurons
        self.signal_lenght = signal_lenght
        # initial membrane potential
        self.v = -65 * np.ones((self.num_neurons, signal_lenght))
        self.u = self.b * self.v  # initial recovery variable
        self.spike_times = np.zeros((self.num_neurons, signal_lenght))
        print('layer initialized')

    def update(self, input, dt=1):
        for t in range(self.signal_lenght):
            I = input[:, t]
            self.v[:, t], self.u[:, t], self.spike_times[:, t] = izhikevich_update(
                self.v[:, t], self.u[:, t], I, self.a, self.b, self.c, self.d, dt)


if __name__ == '__main__':
    print('Start SNN_230307')

    L1_num_neurons = 10000
    L1_v3 = IzhikevichLayer_v3(0.02, 0.2, -65, 8, L1_num_neurons, 10000)

    input_signal = np.random.uniform(low=5, high=10,
                                     size=(10000, L1_num_neurons))

    L1_spike_times_v3 = np.zeros((L1_num_neurons, 10000))
    start_time = time.time()
    L1_v3.update(input_signal)
    L1_spike_times_v3 = L1_v3.spike_times
    end_time = time.time()
    vectorized_time_v3 = end_time - start_time

# Print the execution times
print("Vectorized execution time v3:", vectorized_time_v3)
