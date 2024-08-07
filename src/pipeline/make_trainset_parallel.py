import h5py
import numpy as np
from src.data.EnsembleNV_MWbroadband_addressing_time_domain_parallel import intensity_time_noisy
from src.utils import paths
from src.visualization.visualize import plot_adj_matrix
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
import time

from src.utils import get_system_and_backend
get_system_and_backend()

ndata = 256 * 100  # number of weight-vector pairs to generate
save = True
plot = False
B_low = 0
B_high = 10

filename = f"trainset_20240728_n{ndata}_{B_low}_to_{B_high}.h5"

def intensity_time_noisy_wrapper(B):
    try:
        return intensity_time_noisy(B)
    except Exception as e:
        print(f"Error processing B={B}: {e}")
        return None


if __name__ == "__main__":
    # Generate B_values randomly between 0 and 2

    B_values = np.random.uniform(B_low, B_high, ndata).astype(np.float32)

    # Generate the data
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for processing...")

    with Pool(num_cores) as pool:
        start_time = time.time()
        data_arrays = list(tqdm(pool.imap(intensity_time_noisy_wrapper, B_values), total=ndata))
        end_time = time.time()
        print(f"Data generation completed in {end_time - start_time} seconds")

    # Filter out None results
    data_arrays = [d for d in data_arrays if d is not None]

    if len(data_arrays) < ndata:
        print(f"Warning: Only {len(data_arrays)} out of {ndata} were processed successfully.")

    time_array = data_arrays[0][:, 0]
    intensity_array = np.zeros((len(data_arrays), len(time_array)))
    for i, data in enumerate(data_arrays):
        intensity_array[i] = data[:, 1]

    # Some plotting to verify
    if plot:
        plt.plot(time_array, intensity_array[0])
        plt.show()

    if save:
        with h5py.File(paths["datasets"].joinpath(filename), "w") as f:
            f.create_dataset("labels", data=B_values)
            f.create_dataset("inputs", data=intensity_array)
            f.create_dataset("Time", data=time_array)

        print(f"Data saved to {filename}")
