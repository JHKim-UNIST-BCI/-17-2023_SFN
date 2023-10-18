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

num_izhi = 2
# Create 10 Izhikevich neurons
neurons = [IzhikevichNeuron(0.02, 0.2, -65, 8) for i in range(num_izhi)]

print(neurons[0].v)
print(neurons[1].v)
print(neurons)

