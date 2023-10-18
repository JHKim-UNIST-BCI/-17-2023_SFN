import matplotlib.pyplot as plt
import time
import numpy as np


def pick_28_in_100(len=10):
    # create a 10x10 array filled with zeros
    arr = np.zeros((len, len))
    # define the center of the array
    center = np.array([len/2, len/2])
    # generate indices from a Gaussian distribution centered on the center
    indices = np.random.normal(
        loc=center, scale=len/6, size=(28, 2)).astype(int)
    # clip indices to ensure they fall within the array boundaries
    indices = np.clip(indices, 0, len-1)
    # set the values of the array to random values between 0.1 and 1 for the selected indices
    values = np.random.uniform(low=0.1, high=1, size=28)
    arr[indices[:, 0], indices[:, 1]] = values
    # return the resulting array
    return arr


def generate_receptivefield(pixel_h=64, pixel_w=48, kernel_w=10, kernel_h=10, step_size=6):
    rf = []

    num_step_h = (pixel_w - kernel_w) // step_size + 1
    num_step_v = (pixel_h - kernel_h) // step_size + 1
    print(num_step_v, num_step_h)
    for step_h in range(0, num_step_h * step_size, step_size):
        for step_v in range(0, num_step_v * step_size, step_size):
            tmp = np.zeros((pixel_h, pixel_w))

            tmp_arr = pick_28_in_100(len=kernel_h)

            tmp[step_v:step_v+kernel_h, step_h:step_h+kernel_w] = tmp_arr
            tmp_weight = tmp.reshape(-1)

            rf.append(tmp_weight)

    print("Complete! Created {}x{} kernel with step size {}! Generated {} times."
          .format(kernel_h, kernel_w, step_size, len(rf)))

    rf_array = np.vstack(rf)
    return rf_array


def generate_receptivefield_2(rf, pixel_h=64, pixel_w=48, step_size=2):
    rf_array = []
    rf_length = []
    for r in rf:
        kernel_h, kernel_w = r.shape
        num_step_h = (pixel_w - kernel_w) // step_size + 1
        num_step_v = (pixel_h - kernel_h) // step_size + 1
        print(num_step_v, num_step_h)
        rf_length.append((num_step_h)*(num_step_v))
        for step_h in range(0, num_step_h * step_size, step_size):
            for step_v in range(0, num_step_v * step_size, step_size):

                tmp = np.zeros((pixel_h, pixel_w))
                tmp[step_v:step_v+kernel_h, step_h:step_h+kernel_w] = r
                tmp_weight = tmp.reshape(-1)
                rf_array.append(tmp_weight)

    print("Complete! Generated {} receptive fields.".format(len(rf_array)))
    return np.vstack(rf_array), rf_length
