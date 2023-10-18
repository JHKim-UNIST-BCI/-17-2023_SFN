from brian2 import *

# Set up the network
start_scope()

N = 1  # number of neurons
tau = 10*ms  # membrane time constant
Vr = -70*mV  # reset potential
Vth = -40*mV  # threshold potential
El = -60*mV  # resting potential

eqs = '''
dV/dt = (El - V + I)/tau : volt (unless refractory)
I : volt
'''

G = NeuronGroup(N, eqs, threshold='V > Vth', reset='V = Vr',
                refractory=5*ms, method='euler')

G.V = Vr
G.I = 20*mV

# Define the output variables
M = StateMonitor(G, 'V', record=True)

# Run the simulation
run(50*ms)

# Plot the results
plot(M.t/ms, M.V[0])
xlabel('Time (ms)')
ylabel('Membrane potential (mV)')
show()
