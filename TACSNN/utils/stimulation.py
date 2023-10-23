import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import io
from PIL import Image
import os

def generate_stimuli(angle, num_stim, pixel_h, pixel_w, F=5, plot_stimuli=False, device = 'cpu'):
    stimulation = torch.zeros((pixel_h, pixel_w, num_stim), device=device)
    theta = angle * np.pi / 180.0

    for i in range(num_stim):
        x0 = 500 * 0.3 / 17  # Move the stimulus by 0.3mm for each frame
        y0 = 500 * 0.3 / 15
        stim_dot = edge_stim(x0, y0, F=F, pixel_h=pixel_h, pixel_w=pixel_w, theta=theta, w=1)
        stimulation[:, :, i] = stim_dot

        if plot_stimuli and i in [500]:
            plt.imshow(stimulation[:, :, i].cpu(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.show()

    return stimulation