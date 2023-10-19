import torch
from torch import Tensor
from torch.nn import Module
import math

class IzhikevichLayer(Module):
    def __init__(
        self, a: float, b: float, c: float, d: float,
        num_neurons: int,
        v_thres: float,
        a_decay: float = 1,
        buffer_size: int = 50,
        noise_std: int = 2,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = torch.device(device)

        self.a, self.b, self.c, self.d = a, b, c, d
        self.v_thres = v_thres
        self.noise_std = noise_std
        self.buffer_size = buffer_size

        self.num_neurons = num_neurons
        self.init_values()

        self.a_decay = a_decay
        
        # print('layers initialized')

    def init_values(self) -> None:
        self.v = -65 * torch.ones((self.num_neurons,), device=self.device)
        self.u = self.b * self.v
        self.spikes = torch.zeros((self.num_neurons,), device=self.device)
        self.decay_a = torch.full(
            (self.num_neurons,), self.a, device=self.device)
        self.spike_buffer = torch.zeros(
            (self.num_neurons, self.buffer_size), device=self.device)

    def reset(self) -> None:
        self.init_values()

    def update(self, I: Tensor, dt: float = 1) -> None:
        noise = torch.randn((self.num_neurons,),
                            device=self.device) * self.noise_std
        # noise = 0
        dv = (0.04 * torch.square(self.v) + 5 * self.v + 140 - self.u + I + noise) * dt
        du = (self.decay_a * (self.b * self.v - self.u)) * dt
        self.v.add_(dv)
        self.u.add_(du)

        spike_mask = self.v >= self.v_thres
        self.spikes = spike_mask.to(torch.float32)
        self.decay_a = torch.where(spike_mask, self.decay_a / self.a_decay, self.decay_a)
        self.u = torch.where(spike_mask, self.u + self.d, self.u)
        self.v = torch.where(spike_mask, torch.full_like(self.v, self.c, device=self.device), self.v)
        self.spike_buffer = torch.roll(self.spike_buffer, shifts=-1, dims=1)
        self.spike_buffer[:, -1] = self.spikes

class Synapse(Module):
    def __init__(
        self,
        weights: Tensor,
        tau_psp: float = 20,
        delays: Tensor = [],
        dt: float = 1,
        device: str = 'cpu',
    ):
        super().__init__()
        self.device = torch.device(device)
        self.delays = delays
        self.tau_psp = tau_psp
        self.dt = dt
        self.init_values(weights)
        # print('Synapses initialized')

    def init_values(self, weights: Tensor) -> None:
        self.initial_weights = weights.clone().to(self.device)
        self.weights = weights.to(self.device)
        self.psp = torch.zeros(
            (weights.shape[0], weights.shape[1]), device=self.device)

    def reset(self) -> None:
        self.init_values(self.initial_weights)

    def cal_post_input(self, pre_spike_times: Tensor) -> Tensor:
        post_input = torch.matmul(self.weights, pre_spike_times)
        return post_input
    
    def cal_post_input_delay(self, pre_spike_times: Tensor, buffer_size: int = 50) -> torch.Tensor:
        synaptic_delay_transpose = (buffer_size-1) - self.delays.t().long()
        spikes_with_delay = torch.gather(pre_spike_times, 1, synaptic_delay_transpose).t()
        self.psp.mul_(math.exp(-self.dt / self.tau_psp)).add_(spikes_with_delay)
        delayed_post_input = torch.einsum('ij,ij->i', self.weights, self.psp)
        return delayed_post_input
