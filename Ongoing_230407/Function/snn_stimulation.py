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

def generate_speed_bar_stimuli(num_stim=1000, speed=1, direction='vert', angle=0, pixel_h=64, pixel_w=48, F=10, show_frames=[0, 500, 999], device='cpu'):
    # Convert speed from mm/sec to pixel/frame
    if direction == 'vert':
        total_distance_in_mm = 19  # The total distance in the y direction is 19 mm
    else:  # hori
        total_distance_in_mm = 16  # The total distance in the x direction is 16 mm

    speed_pixel_per_frame = speed/num_stim
    
    stimulations = torch.zeros((pixel_h, pixel_w, num_stim), device=device)

    for i in range(num_stim):
        if direction == 'vert':
            x0 = 8  # Initial position for x
            y0 = i * speed_pixel_per_frame  # Move the stimulus by speed pixel/frame in y direction
        else:  # hori
            x0 = i * speed_pixel_per_frame  # Move the stimulus by speed pixel/frame in x direction
            y0 = 8  # Initial position for y
            
        stim_dot = edge_stim(x0, y0, F=F, pixel_h=pixel_h,
                             pixel_w=pixel_w, theta=angle, w=1)
        stimulations[:, :, i] = stim_dot

        if i in show_frames:
            plt.imshow(stimulations[:, :, i].cpu(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            print("Current y position in mm:", y0)  # Print the current y position in mm
            plt.show()

    return stimulations

def generate_single_frequency_stimulation(frequency, num_stim=1000, pixel_h=64, pixel_w=48, F=0.5,plot_stimuli = False, device='cpu'):
    # Initialize the stimulus for this frequency
    stimulus = torch.zeros((pixel_h, pixel_w, num_stim), device=device)

    # Convert the frequency to radians
    frequency_radians = 2 * np.pi * frequency / 1000

    for i in range(num_stim):
        stim_freq = full_stim(F=F, pixel_h=pixel_h, pixel_w=pixel_w)
        
        # Apply sine wave form with the given frequency
        intensity = np.sin(frequency_radians * i)
        stimulus[:, :, i] = stim_freq * intensity  # Apply frequency code to stimulus

    if plot_stimuli == True:
        plt.figure(figsize= (20,1))
        plt.plot(stimulus[0, 0, :], linewidth=0.5)
        plt.plot(torch.abs(torch.diff(stimulus[40, 40, :])) * 5, linewidth=0.5)
        plt.title("Frequency: " + str(frequency) + " Hz",fontsize = 12)
        plt.tick_params(axis='both', which='major', labelsize=8)  # adjust the size here
        plt.xlim([0, num_stim])
        plt.show()

    return stimulus

def generate_2d_sine_wave_stimulation(speed_mm_s =19, frequency= 1, amplitude=1, num_stim=1000, pixel_h=64, pixel_w=48, plot_stimuli = True, show_frames=[0, 500, 999], device='cpu'):
    # Convert speed from mm/s to pixel/s
    speed_pixel_s = speed_mm_s * (64 / 19.0)

    # Initialize the stimulus for this frequency
    stimulation = torch.zeros((pixel_h, pixel_w, num_stim), device=device)
    for i in range(num_stim):
        # Convert current time step to seconds
        time_s = i / 1000.0
        # Calculate phase shift based on speed
        phase_shift = speed_pixel_s * time_s/2

        # Generate 2D sine wave grid
        # Apply frequency code to stimulus
        stim_tensor = generate_2d_sine_wave(phase_shift, frequency, amplitude, pixel_h, pixel_w)
        stimulation[:, :, i] = stim_tensor

        if plot_stimuli and i in show_frames:
            plt.imshow(stimulation[:, :, i].cpu(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.show()

    # Create SA and RA stimuli
    sa_stim = stimulation
    ra_stim = torch.abs(torch.diff(stimulation, dim=2)) * 5
    zeros_to_insert = torch.zeros(ra_stim.size(0), ra_stim.size(1), 1, device=device)
    ra_stim = torch.cat((zeros_to_insert, ra_stim), dim=2)
    
    # Plot SA and RA stimuli at the center of the sensor array
    plt.figure(figsize=(20,1))
    plt.plot(sa_stim[32, 24, :], label='SA_stim')
    plt.plot(ra_stim[32, 24, :], label='RA_stim')
    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    plt.title('SA_stim and RA_stim at center')
    # plt.ylim([0,10])
    plt.legend()
    plt.show()
    return stimulation

def generate_2d_sine_wave(phase_shift,frequency=1 , amp=1,  pixel_h = 64, pixel_w = 48, device='cpu'):
    # Create grid
    y = torch.linspace(0, 1, pixel_h, device=device)
    x = torch.linspace(0, 1, pixel_w, device=device)
    Y, X = torch.meshgrid(y, x)

    # Create a 2D sine wave grid stimulation with the pattern repeated across columns
    stimulation = amp * torch.sin(2 * np.pi * frequency * Y - phase_shift)

    return stimulation


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

def create_alphabet_stimuli(height=48, width=64, bg_size=300, amplitude=10):
    # 알파벳 리스트 생성
    alphabet_list = [chr(i) for i in range(65, 91)]

    # 각 알파벳에 대한 촉각 이미지 자극물 저장
    stimuli = {}

    for i, alpha in enumerate(alphabet_list):
        # 배경 이미지 생성
        background = np.zeros((height, bg_size))

        # 알파벳 이미지 생성
        alphabet_img = np.zeros((height, width))
        cv2.putText(alphabet_img, alpha, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, amplitude, 2, cv2.LINE_AA)

        # 알파벳 이미지를 배경에 추가
        background[:, :width] = alphabet_img

        # dictionary에 저장
        stimuli[i] = background

    return stimuli



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
