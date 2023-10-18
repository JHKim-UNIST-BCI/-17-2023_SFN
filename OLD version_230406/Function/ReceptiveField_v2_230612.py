import torch
import numpy as np

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class ReceptiveField_v2:

    def __init__(self, device = 'cpu'):
        self.SA_RF, [self.SA_rf_height, self.SA_rf_width] = self.generate_mechanoreceptor_to_afferent_rf(kernel_w=9, kernel_h=11, step_size=5, device=device)
        self.RA_RF, [self.RA_rf_height, self.RA_rf_width] = self.generate_mechanoreceptor_to_afferent_rf(kernel_w=11, kernel_h=14, step_size=4, device=device)

        print("SA_rf shape:", self.SA_RF.shape, 'with height =', self.SA_rf_height, 'with width =', self.SA_rf_width)
        print("RA_rf shape:", self.RA_RF.shape, 'with height =',self.RA_rf_height, 'with width =', self.RA_rf_width)

        #2nd layer
        ############################################################################################################################################################
        # Define optimized receptive fields and synaptic delays
        self.CN_PN_RF = [torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]],device=device) * 4]
        self.CN_IN_RF = [torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]],device=device)]
        self.CN_SD = [torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=device)]
        self.CN_INtoPN_RF = []

        # Check if the sizes of the inner tensors are different and print the index
        for i, (PN, IN) in enumerate(zip(self.CN_PN_RF, self.CN_IN_RF)):
            if PN.size() != IN.size():
                raise ValueError(
                    f"The inner tensors at index {i} have different sizes: {PN.size()} != {IN.size()}")

        SA_CN_PN_RF, [SA_CN_PN_step_height, SA_CN_PN_step_width] = self.generate_weight(CN_PN_RF, pixel_h=SA_rf_height,pixel_w=SA_rf_width, step_size=1, device=device)
        SA_CN_IN_RF, [SA_CN_IN_step_height, SA_CN_IN_step_width] = self.generate_weight(CN_IN_RF, pixel_h=SA_rf_height,pixel_w=SA_rf_width, step_size=1, device=device)
        SA_CN_SD, [SA_CN_SD_step_height, SA_CN_SD_step_width]  = self.generate_weight(CN_SD, pixel_h=SA_rf_height,pixel_w=SA_rf_width, step_size=1, device=device)
        RA_CN_PN_RF, [RA_CN_PN_step_height, RA_CN_PN_step_width] = self.generate_weight(CN_PN_RF, pixel_h=RA_rf_height,pixel_w=RA_rf_width, step_size=1, device=device)
        RA_CN_IN_RF, [RA_CN_IN_step_height, RA_CN_IN_step_width] = self.generate_weight(CN_IN_RF, pixel_h=RA_rf_height,pixel_w=RA_rf_width, step_size=1, device=device)
        RA_CN_SD, [RA_CN_SD_step_height, RA_CN_SD_step_width] = self.generate_weight(CN_SD, pixel_h=RA_rf_height, pixel_w=RA_rf_width, step_size=1, device=device)

        SA_INtoPN_RF, SA_INtoPN_DN = create_weight_matrix(len(SA_CN_IN_RF), len(SA_CN_PN_RF), connection_probability=0.2, device=device)
        RA_INtoPN_RF, RA_INtoPN_DN = create_weight_matrix(len(RA_CN_IN_RF), len(RA_CN_PN_RF), connection_probability=0.2, device=device)

        print("SA_CN_PN_RF shape: ", SA_CN_PN_RF.shape,"SA_CN_PN_step_height:", SA_CN_PN_step_height,"SA_CN_PN_step_width:", SA_CN_PN_step_width)
        print("SA_CN_IN_RF shape: ", SA_CN_IN_RF.shape,"SA_CN_IN_step_height:", SA_CN_IN_step_height,"SA_CN_IN_step_width:", SA_CN_IN_step_width)
        print("RA_CN_PN_RF shape: ", RA_CN_PN_RF.shape,"RA_CN_PN_step_height:", RA_CN_PN_step_height,"RA_CN_PN_step_width:", RA_CN_PN_step_width)
        print("RA_CN_IN_RF shape: ", RA_CN_IN_RF.shape,"RA_CN_IN_step_height:", RA_CN_IN_step_height,"RA_CN_IN_step_width:", RA_CN_IN_step_width)
        print("SA_INtoPN_RF shape: ", SA_INtoPN_RF.shape)
        print("RA_INtoPN_RF shape: ", RA_INtoPN_RF.shape)
        ############################################################################################################################################################
    
    
    def pick_points_in_rf(self, num_points=28, kernel_h=10, kernel_w=10, device='cpu'):
        arr = torch.zeros((kernel_h, kernel_w), device=device)
        center = np.array([kernel_h / 2, kernel_w / 2])
        indices = np.random.normal(
            loc=center, scale=min(kernel_h, kernel_w) / 6, size=(num_points, 2)).astype(int)
        indices = np.clip(indices, 0, np.array([kernel_h, kernel_w]) - 1)
        values = torch.rand(num_points).uniform_(0.1, 1).to(device)
        arr[indices[:, 0], indices[:, 1]] = values
        return arr


    def generate_mechanoreceptor_to_afferent_rf(self, pixel_h=64, pixel_w=48, kernel_w=10, kernel_h=10, step_size=6, device = 'cpu'):
        num_step_h = (pixel_w - kernel_w) // step_size + 1
        num_step_v = (pixel_h - kernel_h) // step_size + 1

        receptive_fields = []
        for step_v in range(0, num_step_v * step_size, step_size):
            for step_h in range(0, num_step_h * step_size, step_size):
                temp_rf = torch.zeros((pixel_h, pixel_w), device=device)
                temp_arr = self.pick_points_in_rf(num_points=28, kernel_h=kernel_h, kernel_w = kernel_w, device=device)
                temp_rf[step_v:step_v + kernel_h,
                        step_h:step_h + kernel_w] = temp_arr
                receptive_fields.append(temp_rf)

        stacked_rf = torch.stack(receptive_fields)
        reshaped_rf = stacked_rf.reshape(stacked_rf.shape[0], -1)

        # print(
        #     f"Complete! Generated {len(receptive_fields)} receptive fields from mechanoreceptor to afferents with kernel size {kernel_h}x{kernel_w}.")
        return reshaped_rf, [num_step_v,num_step_h]


    def generate_weight(self, rf, pixel_h=64, pixel_w=48, step_size=2, device='cpu'):

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


    def create_weight_matrix(self, input_neurons, output_neurons, connection_probability=0.2, device = 'cpu'):
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
