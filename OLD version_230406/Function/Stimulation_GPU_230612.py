# Stimulation_GPU function 23-03-17
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import io
from PIL import Image
import os


def dot_stim(x0, y0, sensor_h=19, sensor_w=16, pixel_h=64, pixel_w=48, F=10, r=1):

    sigma_x = r
    sigma_y = r*16/19

    y, x = torch.meshgrid(torch.linspace(0, sensor_h, pixel_h),
                          torch.linspace(0, sensor_w, pixel_w))

    return F * torch.exp(-(torch.square(x - x0) / (2 * np.square(sigma_x))) -
                          (torch.square(y - y0) / (2 * np.square(sigma_y))))

def generate_stimuli_dot(num_stim, x0=8, y0=8 , pixel_h=64, pixel_w=48, F=10, r=1, plot_stimuli=False, device = 'cpu'):
    stimulation = torch.zeros((pixel_h, pixel_w, num_stim), device=device)

    for i in range(num_stim):
  
        stim_dot = dot_stim(x0, y0, pixel_h=pixel_h, pixel_w=pixel_w, F=F, r=r)
        stimulation[:, :, i] = stim_dot

        if plot_stimuli and i in [500]:
            plt.imshow(stimulation[:, :, i].cpu(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.show()

    return stimulation

def edge_stim(x0, y0, sensor_h=19, sensor_w=16, pixel_h=640, pixel_w=480, w=1, F=10, theta=0):
    y, x = torch.meshgrid(torch.linspace(0, sensor_h, pixel_h),
                          torch.linspace(0, sensor_w, pixel_w))
    x = x - x0  # shift origin to (x0, y0)
    y = y - y0

    return F * torch.exp(-torch.square(x*np.sin(theta) + y*np.cos(theta))/(2*np.square(w)))

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


def full_stim(sensor_h=19, sensor_w=16, pixel_h=640, pixel_w=480, F=10):

    return F * torch.ones((pixel_h, pixel_w))


def plot_2d_sine_wave(ncols=640, nrows=480, freq=0.1, amp=5):
    X, Y = np.meshgrid(np.arange(1, nrows+1), np.arange(1, ncols+1))
    Z = amp*np.sin(2*np.pi*freq*X + np.pi/4)*np.sin(2*np.pi*freq*Y + np.pi/4)

    # Plot the 2D sine wave
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    fig2 = plt.figure()
    plt.imshow(Z, cmap='gray')
    # Show the plot
    plt.show()
    return Z


def create_stimulus(ncols=640, nrows=480, theta=0, width=20):
    # Convert angle to radians
    theta_rad = np.deg2rad(theta)

    # Calculate the number of pixels needed to represent the line
    max_width = int(np.ceil(np.abs(width * np.sin(theta_rad))))
    max_height = int(np.ceil(np.abs(width * np.cos(theta_rad))))

    print(max_width, max_height)

    # Create an empty matrix for the stimulus
    stimulus = np.zeros((ncols, nrows))

    # Calculate the coordinates for the line
    x0 = int((640 - max_width) / 2)
    y0 = int((480 - max_height) / 2)
    x1 = x0 + max_width
    y1 = y0 + max_height

    # Set the values in the stimulus matrix to 1 along the line
    for i in range(x0, x1+1):
        j = int(np.round((i-x0) * np.tan(theta_rad))) + y0
        if j >= 0 and j < 480:
            stimulus[j, i] = 1

    # Show the stimulus plot
    plt.imshow(stimulus, cmap='gray')
    plt.show()

    return stimulus


def elongated_stim(F, pixel_h, pixel_w, orientation='horizontal'):
    stim = torch.zeros((pixel_h, pixel_w))
    h_center = pixel_h // 2
    w_center = pixel_w // 2

    if orientation == 'horizontal':
        h_length = pixel_h // 5
        w_length = pixel_w
    elif orientation == 'vertical':
        h_length = pixel_h
        w_length = pixel_w // 5
    else:
        raise ValueError(
            "Invalid orientation. Choose 'horizontal' or 'vertical'.")

    h_start = h_center - h_length // 2
    h_end = h_center + h_length // 2
    w_start = w_center - w_length // 2
    w_end = w_center + w_length // 2

    stim[h_start:h_end, w_start:w_end] = F
    return stim


def elongated_stim_v2(F, pixel_h, pixel_w, position='middle', orientation='horizontal'):
    stim = torch.zeros((pixel_h, pixel_w))
    h_center = pixel_h // 2
    w_center = pixel_w // 2

    if orientation == 'horizontal':
        h_length = pixel_h // 5
        w_length = pixel_w
    elif orientation == 'vertical':
        h_length = pixel_h
        w_length = pixel_w // 5
    else:
        raise ValueError(
            "Invalid orientation. Choose 'horizontal' or 'vertical'.")

    if position == 'middle':
        h_start = h_center - h_length // 2
    elif position == 'above':
        h_start = h_center - (4 * h_length // 2)
    elif position == 'below':
        h_start = h_center + h_length // 2 - 3
    else:
        raise ValueError(
            "Invalid position. Choose 'above', 'middle', or 'below'.")

    h_end = h_start + h_length*2
    w_start = w_center - w_length // 2
    w_end = w_center + w_length // 2

    percentage = 10
    noise = torch.randn(h_end - h_start, w_end - w_start) * (F * 10 / 100)
    stim[h_start:h_end, w_start:w_end] = F 
    return stim

def elongated_stim_v2(F, pixel_h, pixel_w, position='middle', orientation='horizontal'):
    stim = torch.zeros((pixel_h, pixel_w))
    h_center = pixel_h // 2
    w_center = pixel_w // 2

    if orientation == 'horizontal':
        h_length = pixel_h // 5
        w_length = pixel_w
    elif orientation == 'vertical':
        h_length = pixel_h
        w_length = pixel_w // 5
    else:
        raise ValueError(
            "Invalid orientation. Choose 'horizontal' or 'vertical'.")

    if position == 'middle':
        h_start = h_center - h_length // 2
    elif position == 'above':
        h_start = h_center - (4 * h_length // 2)
    elif position == 'below':
        h_start = h_center + h_length // 2 - 3
    else:
        raise ValueError(
            "Invalid position. Choose 'above', 'middle', or 'below'.")

    h_end = h_start + h_length*2
    w_start = w_center - w_length // 2
    w_end = w_center + w_length // 2

    percentage = 10
    noise = torch.randn(h_end - h_start, w_end - w_start) * (F * 10 / 100)
    stim[h_start:h_end, w_start:w_end] = F 
    return stim

def full_stim(F, pixel_h, pixel_w):
    stim = torch.zeros((pixel_h, pixel_w))

    # Fill the entire area with the stimulus intensity
    stim[:, :] = F 

    return stim

def generate_slip_dot(num_stim=1000, F=10, pixel_h=64, pixel_w=48, device='cpu'):
    stimulation_slip_dot = torch.zeros(
        (pixel_h, pixel_w, num_stim), device=device)

    for i in range(num_stim):
        if i < num_stim/2:
            x0 = 8  # Move the stimulus by 0.3mm for each frame
            y0 = 9
        else:
            x0 = 8
            y0 = i*0.3/15
        stim_dot = dot_stim(x0, y0, F=F, pixel_h=pixel_h, pixel_w=pixel_w, r=2)
        stimulation_slip_dot[:, :, i] = stim_dot
        if i in [100, 500, 900]:  # 100, 500, 900에서만 플롯을 그립니다.
            plt.imshow(stimulation_slip_dot[:, :, i].cpu(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.show()
    
    return stimulation_slip_dot


def recording_stimulation(stim , filename='recording_stimulation.gif'):
    frames = []

    for i in range(stim.shape[2]):
        fig, ax = plt.subplots()
        ax.imshow(stim[:, :, i].cpu(), cmap='gray')
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(f'{i} ms', xy=(0.5, 1.05),
                    xycoords='axes fraction', fontsize=12, ha='center')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        im = Image.open(buf)
        if i % 10 == 0:
            frames.append(im)

        plt.close(fig)

    frames[0].save(os.path.join('plot', 'GIF', filename), format='GIF',
                   append_images=frames[1:], save_all=True, duration=10, loop=0)


def add_gaussian_noise(tensor, std_dev=0.1):
    noise = torch.randn_like(tensor) * std_dev
    return torch.clamp(tensor + noise, 0, 1)
