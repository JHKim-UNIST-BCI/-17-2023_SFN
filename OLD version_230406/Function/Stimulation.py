# Stimulation function 23-03-13

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def dot_stim(x0, y0, sensor_h=19, sensor_w=16, pixel_h=640, pixel_w=480, F=10, r=1):

    sigma_x = r
    sigma_y = r*16/19

    x, y = np.meshgrid(np.linspace(0, sensor_w, pixel_w),
                       np.linspace(0, sensor_h, pixel_h))

    return F * np.exp(-(np.square(x - x0)/(2*np.square(sigma_x))) -
                      (np.square(y - y0)/(2*np.square(sigma_y))))


def edge_stim(x0, y0, sensor_h=19, sensor_w=16, pixel_h=640, pixel_w=480, w=1, F=10, theta=0):
    x, y = np.meshgrid(np.linspace(0, sensor_w, pixel_w),
                       np.linspace(0, sensor_h, pixel_h))
    x = x - x0  # shift origin to (x0, y0)
    y = y - y0

    return F * np.exp(-np.square(x*np.sin(theta) + y*np.cos(theta))/(2*np.square(w)))


def full_stim(sensor_h=19, sensor_w=16, pixel_h=640, pixel_w=480, F=10):

    return F * np.ones((pixel_h, pixel_w))


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


if __name__ == '__main__':
    F = 1.0
    x0 = 8
    y0 = 10
    sigma_x = 0.5
    sigma_y = 0.5

    t = np.linspace(0, 1, 1000)
    w = 0.01
    theta = np.linspace(0, 2*np.pi, 1000)
    x = x0 + sigma_x*np.cos(theta)
    y = y0 + sigma_y*np.sin(theta)

    Z = edge_stim(x, y, w, theta, t)

    plt.imshow(Z, cmap='jet')
    plt.colorbar()
    plt.show()

    X, Y = np.meshgrid(np.linspace(0, 16, 480), np.linspace(0, 19, 640))
    Z = dot_stim(X, Y, F=1.0, x0=8, y0=10)

    plt.imshow(Z, cmap='jet')
    plt.colorbar()
    plt.show()
