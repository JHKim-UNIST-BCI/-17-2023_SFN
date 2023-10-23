import torch
from torch import Tensor
from torch.nn import Module
import math

class IzhikevichLayer(Module):
    """Izhikevich neuron model layer."""
    
    def __init__(self, a: float, b: float, c: float, d: float, num_neurons: int,
                 v_thres: float, a_decay: float = 1, buffer_size: int = 50,
                 noise_std: int = 2, device: str = 'cpu'):
        super().__init__()
        
        self.device = torch.device(device)
        self.a, self.b, self.c, self.d = a, b, c, d
        self.v_thres = v_thres
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.num_neurons = num_neurons
        self.a_decay = a_decay
        
        self.init_values()

    def init_values(self) -> None:
        """Initialize neuron values."""
        self.v = torch.full((self.num_neurons,), -65, device=self.device)
        self.u = self.b * self.v
        self.spikes = torch.zeros(self.num_neurons, device=self.device)
        self.decay_a = torch.full(self.num_neurons, self.a, device=self.device)
        self.spike_buffer = torch.zeros((self.num_neurons, self.buffer_size), device=self.device)

    def reset(self) -> None:
        """Reset neuron values."""
        self.init_values()

    def update(self, I: Tensor, dt: float = 1) -> None:
        """Update neuron states."""
        noise = torch.randn(self.num_neurons, device=self.device) * self.noise_std
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I + noise) * dt
        du = (self.decay_a * (self.b * self.v - self.u)) * dt

        self.v += dv
        self.u += du

        spike_mask = self.v >= self.v_thres
        self.spikes = spike_mask.float()
        self.decay_a = torch.where(spike_mask, self.decay_a / self.a_decay, self.decay_a)
        self.u = torch.where(spike_mask, self.u + self.d, self.u)
        self.v = torch.where(spike_mask, self.c, self.v)
        self.spike_buffer.roll(shifts=-1, dims=1)
        self.spike_buffer[:, -1] = self.spikes


class Synapse(Module):
    """Synapse module."""
    
    def __init__(self, weights: Tensor, tau_psp: float = 20, delays: Tensor = None,
                 dt: float = 1, device: str = 'cpu'):
        super().__init__()
        
        self.device = torch.device(device)
        self.tau_psp = tau_psp
        self.dt = dt
        self.delays = delays if delays is not None else torch.zeros_like(weights)
        
        self.init_values(weights)

    def init_values(self, weights: Tensor) -> None:
        """Initialize synapse values."""
        self.initial_weights = weights.clone().to(self.device)
        self.weights = weights.to(self.device)
        self.psp = torch.zeros_like(weights, device=self.device)

    def reset(self) -> None:
        """Reset synapse values."""
        self.init_values(self.initial_weights)

    def cal_post_input(self, pre_spike_times: Tensor) -> Tensor:
        """Calculate post synaptic input without delay."""
        return torch.matmul(self.weights, pre_spike_times)

    def cal_post_input_delay(self, pre_spike_times: Tensor, buffer_size: int = 50) -> Tensor:
        """Calculate post synaptic input considering delay."""
        synaptic_delay_transpose = (buffer_size - 1) - self.delays.t().long()
        spikes_with_delay = torch.gather(pre_spike_times, 1, synaptic_delay_transpose).t()
        self.psp = self.psp * math.exp(-self.dt / self.tau_psp) + spikes_with_delay
        return torch.einsum('ij,ij->i', self.weights, self.psp)
