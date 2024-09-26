#==============================================================================
# ==============================================================================
import numpy as np
import os
import random
import h5py

from tqdm import tqdm
from src.utils import get_system_and_backend, paths
from src.data.EnsembleNV_MWbroadband_addressing_time_domain import awgn
from datetime import datetime
get_system_and_backend()
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ndata = 20000  # number of weight-vector pairs to generate
    save = True
    plot = False
    B_low = 0
    B_high = 10
    snr = 100 # db
    downsampling = 2

    # h5_filename = f"n{ndata}_{B_low}_to_{B_high}_snr{snr}_long.h5"
    h5_filename = f"n{ndata}_{B_low}_to_{B_high}_snr{snr}_long.h5"

    data_dir = paths.get("raw").joinpath("clean_full")
    # data_dir = paths.get("raw").joinpath("clean")
    filenames = os.listdir(data_dir)

    # Get today's data
    date = datetime.now().strftime("%Y%m%d")

    header = {
        "date": date,
        "ndata": ndata,
        "B_low": B_low,
        "B_high": B_high,
        "snr": snr,
    }


    inputs = []
    labels = []

    for i in tqdm(range(0, ndata)):
        idx = random.randint(0, len(filenames) - 1)
        filename = filenames[idx]
        B_value = float(filename.split("G.dat")[0])
        data = np.loadtxt(data_dir.joinpath(filename))

        # if len(data) != 201:
        #     print(f"Data length is {len(data)}")
        #     continue

        # Downsample data
        data = data[::downsampling]

        # # Remove last entry if odd
        # if len(data) % 2 != 0:
        #     data = data[:-1]

        # Add noise
        data_noisy = awgn(data[:, 1], snr)

        inputs.append(data_noisy)
        labels.append(B_value)

    time = data[:, 0]


    if save:
        with h5py.File(paths["datasets"].joinpath(h5_filename), "w") as f:
            f.create_dataset("labels", data=labels)
            f.create_dataset("inputs", data=inputs)
            f.create_dataset("Time", data=time)
            for key, value in header.items():
                f.attrs[key] = value

        print(f"Data saved to {h5_filename}")

    # A function to inspect a .h5 file that prints the header attributes
    def inspect_h5_file(h5_filename):
        with h5py.File(paths["datasets"].joinpath(h5_filename), "r") as f:
            print(f"Header attributes of {h5_filename}:")
            for key, value in f.attrs.items():
                print(f"{key}: {value}")

    # inspect_h5_file(h5_filename)