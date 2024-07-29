import h5py
import numpy as np

from src.data.EnsembleNV_MWbroadband_addressing_time_domain_parallel import intensity_time_noisy
from src.utils import paths
from src.visualization.visualize import plot_adj_matrix
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count


ndata = 20  # number of weight-vector pairs to generate
save  = False
plot  = False

filename = f"trainset_20240728_n{ndata}.h5"

if __name__ == "__main__":
    # Generate B_values randomly between 0 and 2
    B_values = np.random.uniform(0, 2, ndata).astype(np.float32)

    # Generate the data
    num_cores = cpu_count()
    with Pool(num_cores) as pool:
        data_arrays = pool.map(intensity_time_noisy, B_values)


    time_array = data_arrays[0][:, 0]
    intensity_array = np.zeros((ndata, len(time_array)))
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