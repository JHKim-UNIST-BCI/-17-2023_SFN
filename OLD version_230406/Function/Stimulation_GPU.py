# Stimulation_GPU function 23-03-17
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def dot_stim(x0, y0, sensor_h=19, sensor_w=16, pixel_h=640, pixel_w=480, F=10, r=1):

    sigma_x = r
    sigma_y = r*16/19

    y, x = torch.meshgrid(torch.linspace(0, sensor_h, pixel_h),
                          torch.linspace(0, sensor_w, pixel_w))

    return F * torch.exp(-(torch.square(x - x0) / (2 * np.square(sigma_x))) -
                          (torch.square(y - y0) / (2 * np.square(sigma_y))))


def edge_stim(x0, y0, sensor_h=19, sensor_w=16, pixel_h=640, pixel_w=480, w=1, F=10, theta=0):
    y, x = torch.meshgrid(torch.linspace(0, sensor_h, pixel_h),
                          torch.linspace(0, sensor_w, pixel_w))
    x = x - x0  # shift origin to (x0, y0)
    y = y - y0

    return F * torch.exp(-torch.square(x*np.sin(theta) + y*np.cos(theta))/(2*np.square(w)))


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
    h_center=pixel_h // 2
    w_center=pixel_w // 2

    if orientation == 'horizontal':
        h_length=pixel_h // 5
        w_length=pixel_w
    elif orientation == 'vertical':
        h_length=pixel_h
        w_length=pixel_w // 5
    else:
        raise ValueError(
            "Invalid orientation. Choose 'horizontal' or 'vertical'.")

    h_start=h_center - h_length // 2
    h_end=h_center + h_length // 2
    w_start=w_center - w_length // 2
    w_end=w_center + w_length // 2

    stim[h_start:h_end, w_start:w_end]=F
    return stim
