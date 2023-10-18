import numpy as np

class IzhikevichNeuron:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = -65  # initial membrane potential
        self.u = self.b * self.v  # initial recovery variable

    def update(self, I, dt):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        if self.v >= 30:
            self.v = self.c
            self.u += self.d


class SpikingNeuralNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [IzhikevichNeuron(0.02, 0.2, -65, 8)
                        for i in range(num_neurons)]
        self.weights = np.random.randn(num_neurons, num_neurons)

    def simulate(self, input_signal):
        output_signal = []
        for i in range(len(input_signal)):
            # Update neuron membrane potentials
            for j in range(self.num_neurons):
                I = input_signal[i][j]
                self.neurons[j].update(I, 1.0)

            # Determine which neurons have spiked
            spikes = [int(neuron.v >= 30) for neuron in self.neurons]

            # Update synaptic weights using STDP
            for j in range(self.num_neurons):
                for k in range(self.num_neurons):
                    if j != k:
                        if spikes[j] and not spikes[k]:
                            self.weights[j][k] += 0.1
                        elif not spikes[j] and spikes[k]:
                            self.weights[j][k] -= 0.1

            # Reset neuron membrane potentials
            for j in range(self.num_neurons):
                if spikes[j]:
                    self.neurons[j].v = self.neurons[j].c
                    self.neurons[j].u += self.neurons[j].d

            output_signal.append(spikes)

        return output_signal
