import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time 

start_time = time.time()
# Read input and output file addresses
if len(sys.argv) >= 3:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
else:
    input_file = input("Enter the path to the input HDF5 file: ")
    output_file = input("Enter the path to the output HDF5 file: ")

def get_sphere(mask, center, radius, z_min, z_max):
    """Efficiently create a spherical mask within a bounding box."""
    x, y, z = np.indices(mask.shape)
    distance = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    sphere = (distance <= radius ** 2)
    sphere[:, :, :z_min] = 0
    sphere[:, :, z_max:] = 0
    return sphere

def process_frame(h5, NumberOfNeurons, radius, T, C):
    ActivityMax = np.zeros((2, NumberOfNeurons + 2, T))
    ActivityMean = np.zeros((2, NumberOfNeurons + 2, T))
    ActivityTop30 = np.zeros((2, NumberOfNeurons + 2, T))
    ActivityGr = np.zeros((2, NumberOfNeurons + 2, T))
    VolumeTrack = np.zeros((NumberOfNeurons + 2, T))

    for i in range(T):
        frame_data = np.array(h5[str(i)+"/frame"]).astype(np.int16)  # Preload frame data for efficiency
        mask_data = np.array(h5[str(i)+"/mask"]) if str(i) + "/mask" in h5.keys() else np.zeros_like(frame_data)

        for cell in np.unique(mask_data):
            if cell == 0:
                continue
            
            submask = (mask_data == cell)
            c1, c2, c3 = np.array(np.mean(np.nonzero(submask), axis=1), dtype=int)
            sphere_mask = get_sphere(submask, [c1, c2, c3], radius, max(0, c3 - 1), min(mask_data.shape[2], c3 + 2))

            VolumeTrack[cell, i] = sphere_mask.sum()

            # Extract pixel intensities within the spherical mask
            cellInt = frame_data[:, sphere_mask]

            # Compute maximum intensity
            ActivityMax[:, cell, i] = np.max(cellInt, axis=1)

            # Compute mean activity
            ActivityMean[:, cell, i] = np.mean(cellInt, axis=1)

            # Compute mean of the top 30% intensity values
            top_pix = max(1, int(VolumeTrack[cell, i] * 0.333))  # At least 1 pixel
            top_values = np.partition(cellInt, -top_pix, axis=1)[:, -top_pix:]
            ActivityTop30[:, cell, i] = np.mean(top_values, axis=1)
            ActivityTop30[1, cell, i] = ActivityTop30[1, cell, i]/ActivityTop30[0, cell, i]
            ActivityGr[0, cell, i] = ActivityTop30[0, cell, i]
            ActivityGr[1, cell, i] = ActivityTop30[1, cell, i]

    return ActivityMax, ActivityMean, ActivityTop30, VolumeTrack, ActivityGr
# Main workflow
with h5py.File(input_file, "r") as h5:
    T, C = h5.attrs["T"], h5.attrs["C"]
    print(C)
    T= 10
    radius = 10
    NumberOfNeurons = 21
    ActivityMax, ActivityMean, ActivityTop30, VolumeTrack, ActivityGr = process_frame(
        h5, NumberOfNeurons, radius, T, C
    )
end_time = time.time()
total_time = end_time - start_time
print(f"Script execution time: {total_time:.2f} seconds")
# Save results
with h5py.File(output_file, "w") as hf:
    hf.create_dataset("ActivityMax", data=ActivityMax)
    hf.create_dataset("ActivityMean", data=ActivityMean)
    hf.create_dataset("ActivityTop30", data=ActivityTop30)
    hf.create_dataset("ActivityGr", data=ActivityGr)
    hf.create_dataset("VolumeTrack", data=VolumeTrack)

# Plotting
time = np.linspace(0, T - 1, T)  # Time array for plotting
NeuronsToPlot = range(1, NumberOfNeurons + 1)
print(NeuronsToPlot)
print(ActivityMean[0, 2, :])
# Plot Mean Activity
fig1, axs1 = plt.subplots(len(NeuronsToPlot), 1)#, figsize=(10, len(NeuronsToPlot) * 2))
for idx, neuron in enumerate(NeuronsToPlot):
    #axs1[idx].plot(time, ActivityMean[0, neuron, :], label="Red Mean", color="red")
    if C > 1:
        axs1[idx].plot(time, ActivityMean[1, neuron, :], label="Green Mean", color="green")
    #axs1[idx].set_title(f"Neuron {neuron} - Mean Activity")
    #axs1[idx].legend()
#plt.tight_layout()
plt.show()

# Plot Top 30% Activity
fig2, axs2 = plt.subplots(len(NeuronsToPlot), 1)#, figsize=(10, len(NeuronsToPlot) * 2))
for idx, neuron in enumerate(NeuronsToPlot):
    #axs2[idx].plot(time, ActivityTop30[0, neuron, :], label="Red Top 30%", color="red")
    if C > 1:
        axs2[idx].plot(time, ActivityTop30[1, neuron, :], label="Green Top 30%", color="green")
    #axs2[idx].set_title(f"Neuron {neuron} - Top 30% Activity")
    #axs2[idx].legend()
#plt.tight_layout()
plt.show()

# Plot Volume Track
fig3, axs3 = plt.subplots(len(NeuronsToPlot), 1)#, figsize=(10, len(NeuronsToPlot) * 2))
for idx, neuron in enumerate(NeuronsToPlot):
    axs3[idx].plot(time, VolumeTrack[neuron, :], label="Volume", color="blue")
    #axs3[idx].set_title(f"Neuron {neuron} - Volume Track")
    #axs3[idx].legend()
#plt.tight_layout()
plt.show()
