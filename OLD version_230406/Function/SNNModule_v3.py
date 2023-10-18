import torch
from torch import Tensor
from torch.nn import Module


class IzhikevichLayer(Module):
    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        num_neurons: int,
        v_thres: float,
        a_decay: float = 1,
        buffer_size: int = 50,
        noise_std: int = 2,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = torch.device(device)

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_thres = v_thres
        self.noise_std = noise_std
        self.buffer_size = buffer_size

        self.num_neurons = num_neurons
        self.init_values()

        self.a_decay = a_decay  # decay factor for a
        print('layers initialized')

    def init_values(self) -> None:
        # initial membrane potential
        self.v = -65 * torch.ones((self.num_neurons,), device=self.device)
        self.u = self.b * self.v  # initial recovery variable
        self.spikes = torch.zeros((self.num_neurons,), device=self.device)
        # initialize decay_a to a
        self.decay_a = torch.full(
            (self.num_neurons,), self.a, device=self.device)
        self.spike_buffer = torch.zeros(
            (self.num_neurons, self.buffer_size), device=self.device)

    def reset(self) -> None:
        self.init_values()


    def update(self, I: Tensor, dt: float = 1) -> None:
        noise = torch.randn((self.num_neurons,), device=self.device) * self.noise_std
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I + noise) * dt
        du = (self.decay_a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du

        spike_mask = self.v >= self.v_thres
        self.spikes = torch.where(spike_mask, 1, 0)
        self.decay_a = torch.where(spike_mask, self.decay_a / self.a_decay, self.decay_a)
        self.u = torch.where(spike_mask, self.u + self.d, self.u)
        self.v = torch.where(spike_mask, torch.full_like(
            self.v, self.c, device=self.device), self.v)

        # Update the spike buffer
        # Update the spike buffer
        self.spike_buffer = torch.cat([self.spike_buffer[:, 1:], self.spikes.unsqueeze(-1)], dim=1)

class Synapse(Module):
    def __init__(
        self,
        weights: Tensor,
        delays: Tensor = [],
        device: str = 'cpu',
    ):
        super().__init__()
        self.device = torch.device(device)
        self.delays = delays
        self.init_values(weights)
        print('Synapses initialized')

    def init_values(self, weights: Tensor) -> None:
        self.initial_weights = weights.clone().to(self.device)
        self.weights = weights.to(self.device)

    def reset(self) -> None:
        self.init_values(self.initial_weights)

    def cal_post_input(self, pre_spike_times: Tensor) -> Tensor:
        
        post_input = torch.matmul(self.weights, pre_spike_times)
        return post_input

    def cal_post_input_delay(self, pre_spike_times: Tensor, buffer_size: int = 50) -> Tensor:
        
        synaptic_delay_transpose = (buffer_size-1) - self.delays.t().long()
        spikes_with_delay = torch.gather(
            pre_spike_times, 1, synaptic_delay_transpose).t()
        
        delayed_post_input = torch.einsum('ij,ij->i', self.weights, spikes_with_delay)

        return delayed_post_input
