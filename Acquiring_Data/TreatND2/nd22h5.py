import sys
import os
import numpy as np
import h5py
from nd2reader import ND2Reader


nd2filename=sys.argv[1]
hdfout=sys.argv[2]
X_pix_cut=int(input("How many pixels to cut in x direction?(default=2)"))
with ND2Reader(nd2filename) as images:
    c = images.sizes['c']
    x = images.sizes['x']
    y = images.sizes['y']
    z = images.sizes['z']
    time_series = images.get_timesteps()
    print(time_series[0])
    print(time_series[1])
    print(time_series[2])
    time_series = (time_series - time_series[0])/1000#start time at zero and convert ms to s
    time_origin = int(z/2)#slice whose time we are reporting
    images.iter_axes = 't'
    images.bundle_axes = 'xyz'

    # compute number of frames, which is wrong in the metadata   (by dichotomy)
    max_n_frames = len(images.metadata["frames"])
    mi = 0
    ma = max_n_frames
    prev_t = -1  # anything that is not the starting point
    while True:
        t = mi + (ma - mi) // 2
        try:
            images[t]
            mi = t
        except:
            ma = t
        if prev_t == t:
            break
        prev_t = t
    t += 1

    with h5py.File(hdfout, 'w') as hf:
        for i1 in range(t):
            im3d = np.zeros((c, x, y, z))
            dset = hf.create_dataset(str(i1) + "/frame", (c, x-X_pix_cut, y, z), dtype="i2", compression="gzip")
            hf.create_dataset(str(i1) + "/time", data = time_series[time_origin+i1*z])
            if c>1:
                for c1 in range(c):
                    images.default_coords['c'] = c1
                    im3d[1-c1] = np.array(images[i1]).astype(np.int32)
            else:
                #images.default_coords['c'] = 1
                im3d[0] = np.array(images[i1]).astype(np.int32)
            dset[...] = im3d[:,:-X_pix_cut,:,:]
        name1 = os.path.basename(hdfout)
        name1 = name1.split(".")
        name = name1[0]
        print(name)
        hf.attrs["name"]=name#os.path.basename(hdfout)
        hf.attrs["C"]=c
        hf.attrs["W"]=x-X_pix_cut
        hf.attrs["H"]=y
        hf.attrs["D"]=z
        hf.attrs["T"]=t
        #hf.attrs['Time_series'] = time_series
        hf.attrs["N_neurons"]=0
