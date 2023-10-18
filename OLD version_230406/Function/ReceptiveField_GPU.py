import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def pick_28_in_100(length=10):
    arr = torch.zeros((length, length), device=device)
    center = np.array([length / 2, length / 2])
    indices = np.random.normal(
        loc=center, scale=length / 6, size=(28, 2)).astype(int)
    indices = np.clip(indices, 0, length - 1)
    values = torch.rand(28).uniform_(0.1, 1).to(device)
    arr[indices[:, 0], indices[:, 1]] = values
    return arr


def generate_mechanoreceptor_to_afferent_rf(pixel_h=64, pixel_w=48, kernel_w=10, kernel_h=10, step_size=6):
    num_step_h = (pixel_w - kernel_w) // step_size + 1
    num_step_v = (pixel_h - kernel_h) // step_size + 1

    receptive_fields = []
    for step_v in range(0, num_step_v * step_size, step_size):
        for step_h in range(0, num_step_h * step_size, step_size):
            temp_rf = torch.zeros((pixel_h, pixel_w), device=device)
            temp_arr = pick_28_in_100(length=kernel_h)
            temp_rf[step_v:step_v + kernel_h,
                    step_h:step_h + kernel_w] = temp_arr
            receptive_fields.append(temp_rf)

    stacked_rf = torch.stack(receptive_fields)
    reshaped_rf = stacked_rf.reshape(stacked_rf.shape[0], -1)

    print(
        f"Complete! Generated {len(receptive_fields)} receptive fields from mechanoreceptor to afferents with kernel size {kernel_h}x{kernel_w}.")
    return reshaped_rf, [num_step_h, num_step_v]


def generate_afferent_to_top_rf(rf, pixel_h=64, pixel_w=48, step_size=2, device='cpu'):

    rf_array = []
    rf_length = []

    for r in rf:
        kernel_h, kernel_w = r.shape
        num_step_h = (pixel_w - kernel_w) // step_size + 1
        num_step_v = (pixel_h - kernel_h) // step_size + 1
        rf_length.append(num_step_h * num_step_v)

        for step_v in range(0, num_step_v * step_size, step_size):
            for step_h in range(0, num_step_h * step_size, step_size):
                temp_rf = torch.zeros((pixel_h, pixel_w), device=device)
                temp_rf[step_v:step_v + kernel_h, step_h:step_h + kernel_w] = r
                rf_array.append(temp_rf)

    stacked_rf = torch.stack(rf_array).to(device)
    reshaped_rf = stacked_rf.reshape(stacked_rf.shape[0], -1)
    print(
        f"Complete! Generated {len(rf_array)} receptive fields from afferents to the top with kernel size {kernel_h}x{kernel_w}. with step size {step_v}x{step_h}")
    return reshaped_rf, rf_length
