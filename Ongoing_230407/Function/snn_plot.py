from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import io
from PIL import Image
import os

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import io
from PIL import Image
import os

from scipy.ndimage.filters import gaussian_filter1d
from scipy.fft import fft
from scipy.signal import stft
import scipy.ndimage



def plot_spike_times(
        spikes: torch.Tensor,
        colors: str = 'k',
        size: Tuple[float, float] = (20, 3),
        xtick: List[int] = [0, 500, 1000],
        line_lengths: float = 0.4,
        save_fig: int = 0
) -> None:

    # Convert spikes to CPU if on a CUDA device
    spikes = spikes.cpu() if spikes.is_cuda else spikes

    neuron_spike_times = [list(np.where(spikes[i, :] == 1)[0])
                          for i in range(spikes.shape[0])]
    # print(neuron_spike_times)
    # Reverse the order of neuron_spike_times
    neuron_spike_times.reverse()
    plt.figure(figsize=size)
    plt.eventplot(neuron_spike_times, colors=colors, linelengths=line_lengths, linewidths=2)
    # plt.xlabel('Time')
    # plt.ylabel('Neuron')
    plt.xticks(xtick)
    plt.xlim([0, spikes.shape[1]+1])

    if save_fig == 1:
        plt.savefig("spike_times_plot.svg", format="svg")

    plt.show()


def plot_spike_times_gif(
        spikes: torch.Tensor,
        colors: str = 'k',
        size: Tuple[float, float] = (8, 3),
        line_lengths: float = 0.4,
        xtick: List[int] = [0, 500, 1000],
        save_gif: bool = False,
        file_name: str = 'spike_times_animation.gif'
) -> None:

    # Convert spikes to CPU if on a CUDA device
    spikes = spikes.cpu() if spikes.is_cuda else spikes

    frames = []

    for i in range(spikes.shape[1]):

        neuron_spike_times = [list(np.where(spikes[ii, :i] == 1)[0])
                              for ii in range(spikes.shape[0])]

        fig, ax = plt.subplots(figsize=size)
        plt.eventplot(neuron_spike_times, colors=colors, linelengths=0.4)
        plt.xlim([0, spikes.shape[1]+1])
        # Add a vertical red line at the position of i
        ax.axvline(i, color='red', linestyle='--')

        ax.axis('off')

        if save_gif:
            # Save the current frame as an in-memory image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            im = Image.open(buf)
            if i % 10 == 0:
                frames.append(im)

        # Clear the current plot
        plt.close(fig)

    if save_gif:
        # Save frames as a GIF
        frames[0].save(os.path.join('plot', 'GIF', file_name), format='GIF',
                       append_images=frames[1:], save_all=True, duration=0.5, loop=0)
    else:
        plt.show()


def plot_total_spike_2D(spikes, size):

    spikes = torch.sum(spikes, axis=1)
    spikes = torch.reshape(spikes, size)
    # Convert the spike count tensor to a numpy array
    spikes_np = spikes.cpu().numpy()
    size = (3, 5)
    # Plot the heatmap using matplotlib
    plt.figure(figsize=size)
    plt.imshow(spikes_np, cmap='hot', aspect='auto',vmax = 100)
    plt.axis('off')
    # plt.colorbar(label='Spike Count')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('2D Spike Count Heatmap')
    # plt.savefig("colorbar.svg", format="svg")
    plt.show()
    

def plot_total_spike_2D_gif(spikes, size, save_gif=False, file_name='heatmap_animation.gif'):

    frames = []

    for i in range(spikes.shape[1]):
        spikes_slice = torch.sum(spikes[:, max(0, i-100):i], axis=1) * 10
        spikes_slice = torch.reshape(spikes_slice, size)
        # Convert the spike count tensor to a numpy array
        spikes_np = spikes_slice.cpu().numpy()

        # Plot the heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(3, 5))
        plt.imshow(spikes_np, cmap='hot', aspect='auto', vmin=0, vmax=100)
        ax.axis('off')

        if save_gif:
            # Save the current frame as an in-memory image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            im = Image.open(buf)
            if i % 10 == 0:
                frames.append(im)

        # Clear the current plot
        plt.close(fig)

    if save_gif:
        # Save frames as a GIF
        frames[0].save(os.path.join('plot', 'GIF', file_name), format='GIF',
                       append_images=frames[1:], save_all=True, duration=100, loop=0)
    else:
        plt.show()

def plot_mean_firing_across_channels(spike_times, bin_size=5, size=(10, 10), plot_figure = True):
    neuron_count, time_length = spike_times.shape
    firing_rates = spike_times.unfold(1, bin_size, 1).sum(2) / (bin_size / 1000)

    mean_firing_rates = torch.mean(firing_rates, dim=1)

    if plot_figure:
        plt.figure(figsize=size)
        plt.plot(range(neuron_count), mean_firing_rates.numpy(), linestyle='-', marker='', linewidth = 2)
        plt.show()

    return mean_firing_rates

def plot_mean_firing_across_channels(spike_times, bin_size=5, size=(10, 10), plot_figure = True):
    neuron_count, time_length = spike_times.shape
    firing_rates = spike_times.unfold(1, bin_size, 1).sum(2) / (bin_size / 1000)

    mean_firing_rates = torch.mean(firing_rates, dim=1)

    if plot_figure:
        plt.figure(figsize=size)
        plt.plot(range(neuron_count), mean_firing_rates.numpy(), linestyle='-', marker='', linewidth = 2)
        plt.show()

    return mean_firing_rates

def plot_SNN(S, fig_size = (3,1), layers = [], plot_figure = True):
    plt.rcParams['font.size'] = 10
    fig_size = fig_size
    line_lengths = 0.4

    layers_mapping = {'sa': S.sa_spike_times, 
                      'ra': S.ra_spike_times, 
                      'cn': S.cn_spike_times}

    colors_mapping = {'sa': 'tab:blue',
                      'ra': 'tab:red',
                      'cn': 'k'}
    if plot_figure:  
        for layer in layers:
            for i in range(len(layers_mapping[layer])):
                plot_spike_times(layers_mapping[layer][i], size=fig_size, colors=colors_mapping[layer], line_lengths=line_lengths,xtick = [0,10000,20000])
            
    mean_firing_rates = plot_mean_firing_across_channels(S.cn_spike_times[1],size = fig_size, plot_figure = plot_figure)
    
    return mean_firing_rates

def compute_individual_firing_rates(spike_times, window_size):

    assert window_size <= spike_times.shape[1], "Window size must be less than or equal to time length."
    total_spikes_per_neuron = torch.sum(torch.from_numpy(spike_times[:, :window_size]), axis=1)
    window_size_s = window_size / 1000
    individual_firing_rates = total_spikes_per_neuron / window_size_s

    return individual_firing_rates

def plot_membrane_potential(v, size=(5, 3), xtick=[0, 500, 1000]):
    # Convert v to CPU if on a CUDA device
    if v.is_cuda:
        v = v.cpu()

    num_neurons = v.shape[0]
    time_length = v.shape[1]
    # Adjust the value 100 to control the vertical spacing
    offset = np.arange(num_neurons) * 50

    plt.figure(figsize=size)

    for i in range(num_neurons):
        plt.plot(v[i, :] + offset[i])

    plt.xlabel('Time')
    plt.ylabel('Membrane potential')
    plt.xticks(xtick)
    plt.xlim([0, time_length])
    plt.show()


def plot_firing_rates(spike_times, bin_size=50, fig_size = (10,2)):
    # Get neuron count and time length from the shape of spike_times
    neuron_count, time_length = spike_times.shape

    # Calculate firing rates using a sliding window
    firing_rates = torch.zeros((neuron_count, time_length - bin_size + 1))
    for neuron in range(neuron_count):
        for t in range(time_length - bin_size + 1):
            firing_rates[neuron, t] = torch.sum(
                spike_times[neuron, t:t+bin_size]) / (bin_size/1000)

    # Smooth firing rates using a rolling average
    smoothed_firing_rates = torch.zeros(
        (neuron_count, time_length - bin_size + 1))
    window_size = 1
    for neuron in range(neuron_count):
        for t in range(time_length - bin_size - window_size + 2):
            smoothed_firing_rates[neuron, t] = torch.mean(
                firing_rates[neuron, t:t+window_size])

    # Plot firing rates
    plt.figure(figsize=fig_size)
    for neuron in range(neuron_count):
        plt.plot(smoothed_firing_rates[neuron], label=f'Neuron {neuron + 1}')
    plt.ylim([0, 120])
    plt.xlim([0, time_length])
    plt.xlabel('Time', fontsize=10)  # Adjust fontsize here
    plt.ylabel('Firing rate (spikes/bin)', fontsize=10)  # Adjust fontsize here
    plt.title('Firing rates of neurons', fontsize=10)  # Adjust fontsize here
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.show()

def plot_mean_firing_rates(spike_times, bin_size=50, fig_size=(10, 2), plot_figure = True):
    neuron_count, time_length = spike_times.shape

    firing_rates = torch.zeros((neuron_count, time_length - bin_size + 1))
    for neuron in range(neuron_count):
        for t in range(time_length - bin_size + 1):
            firing_rates[neuron, t] = torch.sum(
                spike_times[neuron, t:t + bin_size]) / (bin_size / 1000)

    smoothed_firing_rates = torch.zeros(
        (neuron_count, time_length - bin_size + 1))
    window_size = 5
    for neuron in range(neuron_count):
        for t in range(time_length - bin_size - window_size + 2):
            smoothed_firing_rates[neuron, t] = torch.mean(
                firing_rates[neuron, t:t + window_size])

    mean_firing_rates = torch.mean(smoothed_firing_rates, dim=0)
    variance_firing_rates = torch.var(smoothed_firing_rates, dim=0, unbiased=False)

    if plot_figure:
        plt.figure(figsize=fig_size)
        plt.plot(mean_firing_rates, label='Mean firing rate')
        # plt.plot(variance_firing_rates, label='Variance of firing rate')
        plt.ylim([0, 200])
        plt.xlim([0, time_length])
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Firing rate (Hz)', fontsize=10)
        plt.title('Mean firing rates across channels', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.show()


    return mean_firing_rates

def plot_mean_firing_rates_with_gaussian(spike_times, bin_size=50, kernel_width=50, fig_size=(10, 2)):
    # Convert torch tensor to numpy array if it's not
    if torch.is_tensor(spike_times):
        spike_times = spike_times.numpy()

    # Get neuron count and time length from the shape of spike_times
    neuron_count, time_length = spike_times.shape

    # Calculate firing rates using a bin
    firing_rates = np.zeros((neuron_count, time_length - bin_size + 1))
    for neuron in range(neuron_count):
        for t in range(time_length - bin_size + 1):
            firing_rates[neuron, t] = np.sum(
                spike_times[neuron, t:t+bin_size]) / (bin_size/1000)

    # Define your gaussian kernel
    gaussian_kernel = np.zeros(kernel_width)
    gaussian_kernel[kernel_width // 2] = 1  # Impulse in the middle
    gaussian_kernel = gaussian_filter1d(gaussian_kernel, kernel_width / 7)  # Gaussian filter with SD = size/7

    # Smooth firing rates using Gaussian kernel
    smoothed_firing_rates = np.zeros_like(firing_rates)
    for neuron in range(neuron_count):
        smoothed_firing_rates[neuron] = np.convolve(firing_rates[neuron], gaussian_kernel, mode='same')

    mean_firing_rates = np.mean(smoothed_firing_rates, axis=0)
    variance_firing_rates = np.var(smoothed_firing_rates, axis=0, ddof=0)

    # Plot firing rates
    plt.figure(figsize=fig_size)
    plt.plot(mean_firing_rates, label='Mean firing rate')
    plt.ylim([0, 200])
    plt.xlim([0, time_length])
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Firing rate (Hz)', fontsize=10)
    plt.title('Mean firing rates across channels', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()

    return mean_firing_rates


def plot_isi_firing_rates(spike_times, fig_size=(10, 2), plot_figure=True):
    neuron_count, time_length = spike_times.shape
    firing_rates = torch.zeros((neuron_count, time_length))

    # calculate ISI for each neuron
    for neuron in range(neuron_count):
        spike_indices = torch.nonzero(spike_times[neuron, :]).squeeze()
        if spike_indices.numel() > 1:
            isi = spike_indices[1:] - spike_indices[:-1]
            firing_rates[neuron, spike_indices[1:]] = 1000 / isi  # convert to Hz

    mean_firing_rates = torch.mean(firing_rates, dim=0)
    variance_firing_rates = torch.var(firing_rates, dim=0, unbiased=False)

    if plot_figure:
        plt.figure(figsize=fig_size)
        plt.plot(mean_firing_rates, label='Mean firing rate')
        # plt.plot(variance_firing_rates, label='Variance of firing rate')
        plt.ylim([0, 200])
        plt.xlim([0, time_length])
        plt.xlabel('Time (ms)', fontsize=10)
        plt.ylabel('Firing rate (Hz)', fontsize=10)
        plt.title('Mean firing rates across channels', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.show()

    return mean_firing_rates