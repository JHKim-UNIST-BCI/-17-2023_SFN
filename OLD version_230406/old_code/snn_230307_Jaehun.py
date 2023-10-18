import numpy as np
import time
from numba import jit

class IzhikevichLayer:
    def __init__(self, a , b , c, d, num_neurons):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.num_neurons = num_neurons
        self.v = -65 * np.ones((self.num_neurons,))  # initial membrane potential
        self.u = self.b * self.v  # initial recovery variable
        self.spike_times = np.zeros((self.num_neurons,))
        print('layer initialized')

    def update(self, I, dt = 1):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        self.spike_times = np.where(self.v >= 30, 1, 0)
        self.v = np.where(self.v >= 30, self.c, self.v)
        self.u = np.where(self.v >= 30, self.u + self.d, self.u)

if __name__ == '__main__':
    print('Start SNN_230307')

    L1_num_neurons = 640*480
    L1 = IzhikevichLayer(0.02, 0.2, -65, 8, L1_num_neurons)

    signal_length = 10
    input_signal = np.random.uniform(low=5, high=10, 
                                     size=(signal_length, L1_num_neurons))

    L1_spike_times = np.zeros((L1_num_neurons, signal_length))
    print(L1_spike_times.shape)

    #######simulation
    start_time = time.time()

    for i in range(len(input_signal)):
        I = input_signal[i]
        L1.update(I)
        L1_spike_times[:,i] = L1.spike_times

    

    end_time = time.time()
    vectorized_time = end_time - start_time
    ########simulation end

    
# Print the execution times
print("Vectorized execution time:", vectorized_time)





