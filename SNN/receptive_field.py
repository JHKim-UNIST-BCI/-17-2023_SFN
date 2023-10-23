import torch
import numpy as np

def pick_points_in_rf(num_points=28, kernel_h=10, kernel_w=10, device='cpu'):
    arr = torch.zeros((kernel_h, kernel_w), device=device)
    center = np.array([kernel_h / 2, kernel_w / 2])
    scale = min(kernel_h, kernel_w) / 8
    
    indices = np.random.normal(loc=center, scale=scale, size=(num_points, 2)).astype(int)
    indices = np.clip(indices, 0, np.array([kernel_h, kernel_w]) - 1)
    outer_indices = np.random.normal(loc=center, scale=scale*2, size=(num_points, 2)).astype(int)
    outer_indices = np.clip(outer_indices, 0, np.array([kernel_h, kernel_w]) - 1)
    
    values = torch.rand(num_points).uniform_(0.1, 1).to(device)
    
    arr[indices[:, 0], indices[:, 1]] = values
    arr[outer_indices[:, 0], outer_indices[:, 1]] = values
    
    return arr

def generate_mechanoreceptor_to_afferent_rf(pixel_h=320, pixel_w=240, kernel_w=9, kernel_h=11, step_size=6, device = 'cpu'):
    num_step_h = (pixel_w - kernel_w) // step_size + 1
    num_step_v = (pixel_h - kernel_h) // step_size + 1

    receptive_fields = []
    for step_v in range(0, num_step_v * step_size, step_size):
        for step_h in range(0, num_step_h * step_size, step_size):
            temp_rf = torch.zeros((pixel_h, pixel_w), device=device)
            temp_arr = pick_points_in_rf(num_points=28, kernel_h=kernel_h, kernel_w=kernel_w, device=device)
            temp_rf[step_v:step_v + kernel_h, step_h:step_h + kernel_w] = temp_arr
            receptive_fields.append(temp_rf)

    stacked_rf = torch.stack(receptive_fields)
    reshaped_rf = stacked_rf.reshape(stacked_rf.shape[0], -1)

    return reshaped_rf, [num_step_v, num_step_h]

def generate_primary_receptive_field_weights(pixel_h=320, pixel_w=240, device='cpu'):
    scale_factor = 5 if pixel_h == 320 else 1

    sa_kernel_h, sa_kernel_w = 11 * scale_factor, 9 * scale_factor
    ra_kernel_h, ra_kernel_w = 14 * scale_factor, 11 * scale_factor

    sa_rf, sa_rf_dim = generate_mechanoreceptor_to_afferent_rf(
        pixel_h=pixel_h, pixel_w=pixel_w, kernel_w=sa_kernel_w, kernel_h=sa_kernel_h,
        step_size=5 * scale_factor, device=device
    )
    ra_rf, ra_rf_dim = generate_mechanoreceptor_to_afferent_rf(
        pixel_h=pixel_h, pixel_w=pixel_w, kernel_w=ra_kernel_w, kernel_h=ra_kernel_h,
        step_size=4 * scale_factor, device=device
    )
    
    return sa_rf, ra_rf, sa_rf_dim, ra_rf_dim