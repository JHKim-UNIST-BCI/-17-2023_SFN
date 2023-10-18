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


def generate_receptive_field(rf, pixel_h=64, pixel_w=48, step_size=2, device='cpu'):

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


def generate_neuron_connection_weight(input_neurons, output_neurons, connection_probability=0.2, device = 'cpu'):
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

def generate_primary_receptive_field_weights(pixel_h = 64, pixel_w = 48, device = 'cpu'):
    scale_factor = 5 if pixel_h == 320 else 1

    sa_rf, [sa_rf_height, sa_rf_width] = generate_mechanoreceptor_to_afferent_rf(pixel_h = pixel_h, pixel_w = pixel_w, kernel_w=9*scale_factor, kernel_h=11*scale_factor, step_size=5*scale_factor, device=device)
    ra_rf, [ra_rf_height, ra_rf_width] = generate_mechanoreceptor_to_afferent_rf(pixel_h = pixel_h, pixel_w = pixel_w, kernel_w=11*scale_factor, kernel_h=14*scale_factor, step_size=4*scale_factor, device=device)

    print(sa_rf.shape)
    # Print the shape of the sa_rf variable
    print("sa_rf shape:", sa_rf.shape, 'with height =',sa_rf_height, 'with width =', sa_rf_width)
    print("ra_rf shape:", ra_rf.shape, 'with height =',ra_rf_height, 'with width =', ra_rf_width)

    cn_pn_rf = [torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]],device=device) * 4]
    cn_in_RF = [torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]],device=device)]
    cn_SD = [torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=device)]

    cn_intopn_RF = []

    # Check if the sizes of the inner tensors are different and print the index
    for i, (pn,IN) in enumerate(zip(cn_pn_rf, cn_in_RF)):
        if pn.size() != IN.size():
            raise ValueError(
                f"The inner tensors at index {i} have different sizes: {pn.size()} != {IN.size()}")
        
    sa_cn_pn_rf, [sa_cn_pn_step_height, sa_cn_pn_step_width] = generate_receptive_field(cn_pn_rf, pixel_h=sa_rf_height,pixel_w=sa_rf_width, step_size=1, device=device)
    sa_cn_in_RF, [sa_cn_in_step_height, sa_cn_in_step_width] = generate_receptive_field(cn_in_RF, pixel_h=sa_rf_height,pixel_w=sa_rf_width, step_size=1, device=device)
    sa_cn_SD, [sa_cn_SD_step_height, sa_cn_SD_step_width]  = generate_receptive_field(cn_SD, pixel_h=sa_rf_height,pixel_w=sa_rf_width, step_size=1, device=device)
    ra_cn_pn_rf, [ra_cn_pn_step_height, ra_cn_pn_step_width] = generate_receptive_field(cn_pn_rf, pixel_h=ra_rf_height,pixel_w=ra_rf_width, step_size=1, device=device)
    ra_cn_in_RF, [ra_cn_in_step_height, ra_cn_in_step_width] = generate_receptive_field(cn_in_RF, pixel_h=ra_rf_height,pixel_w=ra_rf_width, step_size=1, device=device)
    ra_cn_SD, [ra_cn_SD_step_height, ra_cn_SD_step_width] = generate_receptive_field(cn_SD, pixel_h=ra_rf_height, pixel_w=ra_rf_width, step_size=1, device=device)

    sa_intopn_RF, sa_intopn_DN = generate_neuron_connection_weight(len(sa_cn_in_RF), len(sa_cn_pn_rf), connection_probability=0.2, device=device)
    ra_intopn_RF, ra_intopn_DN = generate_neuron_connection_weight(len(ra_cn_in_RF), len(ra_cn_pn_rf), connection_probability=0.2, device=device)

    print("sa_cn_pn_rf shape: ", sa_cn_pn_rf.shape,"sa_cn_pn_step_height:", sa_cn_pn_step_height,"sa_cn_pn_step_width:", sa_cn_pn_step_width)
    print("sa_cn_in_RF shape: ", sa_cn_in_RF.shape,"sa_cn_in_step_height:", sa_cn_in_step_height,"sa_cn_in_step_width:", sa_cn_in_step_width)
    print("ra_cn_pn_rf shape: ", ra_cn_pn_rf.shape,"ra_cn_pn_step_height:", ra_cn_pn_step_height,"ra_cn_pn_step_width:", ra_cn_pn_step_width)
    print("ra_cn_in_RF shape: ", ra_cn_in_RF.shape,"ra_cn_in_step_height:", ra_cn_in_step_height,"ra_cn_in_step_width:", ra_cn_in_step_width)
    print("sa_intopn_RF shape: ", sa_intopn_RF.shape)
    print("ra_intopn_RF shape: ", ra_intopn_RF.shape)

    return (sa_rf, ra_rf, sa_rf_height, sa_rf_width, ra_rf_height, ra_rf_width, sa_cn_pn_rf, sa_cn_in_RF, 
sa_cn_SD, ra_cn_pn_rf, ra_cn_in_RF, ra_cn_SD, sa_cn_pn_step_height, sa_cn_pn_step_width, 
sa_cn_in_step_height, sa_cn_in_step_width, sa_cn_SD_step_height, sa_cn_SD_step_width,
ra_cn_pn_step_height, ra_cn_pn_step_width, ra_cn_in_step_height, ra_cn_in_step_width,
ra_cn_SD_step_height, ra_cn_SD_step_width, sa_intopn_RF, sa_intopn_DN, ra_intopn_RF, ra_intopn_DN)
