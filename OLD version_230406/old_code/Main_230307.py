from snn_230307_chatgpt1 import IzhikevichNeuron

neuron = IzhikevichNeuron(0.02, 0.2, -65, 8)

print(neuron.v)

neuron.update(10,1)

print(neuron.v)