import torch
import numpy as np

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def pick_points_in_rf(num_points=28, kernel_h=10, kernel_w=10, device='cpu'):
    arr = torch.zeros((kernel_h, kernel_w), device=device)
    center = np.array([kernel_h / 2, kernel_w / 2])
    indices = np.random.normal(
        loc=center, scale=min(kernel_h, kernel_w) / 6, size=(num_points, 2)).astype(int)
    indices = np.clip(indices, 0, np.array([kernel_h, kernel_w]) - 1)
    values = torch.rand(num_points).uniform_(0.1, 1).to(device)
    arr[indices[:, 0], indices[:, 1]] = values
    return arr


def generate_mechanoreceptor_to_afferent_rf(pixel_h=64, pixel_w=48, kernel_w=10, kernel_h=10, step_size=6, device = 'cpu'):
    num_step_h = (pixel_w - kernel_w) // step_size + 1
    num_step_v = (pixel_h - kernel_h) // step_size + 1

    receptive_fields = []
    for step_v in range(0, num_step_v * step_size, step_size):
        for step_h in range(0, num_step_h * step_size, step_size):
            temp_rf = torch.zeros((pixel_h, pixel_w), device=device)
            temp_arr = pick_points_in_rf(
                num_points=28, kernel_h=kernel_h, kernel_w = kernel_w, device=device)
            temp_rf[step_v:step_v + kernel_h,
                    step_h:step_h + kernel_w] = temp_arr
            receptive_fields.append(temp_rf)

    stacked_rf = torch.stack(receptive_fields)
    reshaped_rf = stacked_rf.reshape(stacked_rf.shape[0], -1)

    # print(
    #     f"Complete! Generated {len(receptive_fields)} receptive fields from mechanoreceptor to afferents with kernel size {kernel_h}x{kernel_w}.")
    return reshaped_rf, [num_step_v,num_step_h]


def generate_weight(rf, pixel_h=64, pixel_w=48, step_size=2, device='cpu'):

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
    # print(
    #     f"Complete! Generated {len(rf_array)} receptive fields from afferents to the top with kernel size {kernel_h}x{kernel_w}. with step size {num_step_v}x{num_step_h}")
    return reshaped_rf, [num_step_v,num_step_h] 


def create_weight_matrix(input_neurons, output_neurons, connection_probability=0.2, device = 'cpu'):
    """
    Create a weight matrix with the specified connection probability and synaptic weights.

    Args:
        input_neurons (int): Number of input neurons.
        output_neurons (int): Number of output neurons.
        connection_probability (float): Probability of connection between input and output neurons.
        device (str): Device to create tensors on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The final weight matrix with the desired connectivity and synaptic weights.
    """

    # Create binary connectivity matrix
    connectivity = torch.rand(output_neurons, input_neurons, device=device)
    connectivity = (connectivity < connection_probability).float()

    # Create weight matrix
    weights = torch.rand(output_neurons, input_neurons, device=device)

    # Multiply connectivity with weights
    final_weights = connectivity * weights

    # Create delay matrix with the given delay value
    delays = torch.full((output_neurons, input_neurons), 0, device=device)

    return final_weights, delays
