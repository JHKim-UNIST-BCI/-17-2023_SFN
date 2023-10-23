import torch
import torch.nn as nn

class IzhikevichLayer(nn.Module):
    def __init__(self, size, a=0.02, b=0.2, c=-65, d=8, dt=1.0):
        super(IzhikevichLayer, self).__init__()
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.size = size
        
        self.v = torch.full((self.size,), self.c, dtype=torch.float32)
        self.u = torch.full((self.size,), self.b * self.c, dtype=torch.float32)
        
    def forward(self, I):
        I = I.squeeze()
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I) * self.dt
        du = (self.a * (self.b * self.v - self.u)) * self.dt
        
        self.v += dv
        self.u += du
        spikes = (self.v >= 30).float()
        
        self.v[self.v >= 30] = self.c
        self.u[self.v >= 30] += self.d
        
        return spikes
