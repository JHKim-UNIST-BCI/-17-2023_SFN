import numpy as np
import time


class IzhikevichNeuron:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = -65
        self.u = self.b * self.v

    def update(self, I, dt):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        if self.v >= 30:
            self.v = self.c
            self.u += self.d

num_izhi = 99
# Create 10 Izhikevich neurons
neurons = [IzhikevichNeuron(0.02, 0.2, -65, 8) for i in range(num_izhi)]

# Create a random input signal with 100 time steps and 10 dimensions
input_signal = np.random.randn(100, num_izhi)
print(len(input_signal))
print(input_signal.shape)
print(input_signal[0].shape)
# Run the neurons using for loops
start_time = time.time()
for i in range(len(input_signal)):
    for j in range(len(neurons)):
        I = input_signal[i][j]
        neurons[j].update(I, 1.0)
end_time = time.time()
for_loop_time = end_time - start_time

# Run the neurons using vectorization
start_time = time.time()
v = -65 * np.ones((num_izhi,))
u = -20 * np.ones((num_izhi,))
for i in range(len(input_signal)):
    I = input_signal[i]
    dv = (0.04 * v**2 + 5 * v + 140 - u + I) * 1.0
    du = (0.02 * (0.2 * v - u)) * 1.0
    v += dv
    u += du
    v = np.where(v >= 30, -65, v)
    u = np.where(v >= 30, u + 8, u)
    print(v)
end_time = time.time()
vectorized_time = end_time - start_time

# Print the execution times
print("For loop execution time:", for_loop_time)
print("Vectorized execution time:", vectorized_time)
print("Portion: ", for_loop_time / vectorized_time)

print(np.ones((num_izhi,num_izhi)).shape)
print(np.ones(num_izhi).shape)
print(range(len(input_signal)))
print(v**2)