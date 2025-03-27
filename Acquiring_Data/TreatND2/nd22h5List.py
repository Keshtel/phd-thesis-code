import sys
import os
import numpy as np
import h5py
from nd2reader import ND2Reader
from os import listdir
from os.path import isfile, join
mypath = sys.argv[1]
X_pix_cut=int(input("How many pixels to cut in x direction?(default=2)"))
onlyfiles = [f for f in listdir(mypath)  if ".nd2" in f]

for f in onlyfiles:
    nd2filename=f
    out = f.split("_")
    output_name = out[1]+".h5"
    hdfout =  os.path.join(mypath,output_name)
    nd2filename = os.path.join(mypath,f)
    print(hdfout)
    with ND2Reader(nd2filename) as images:
        c = images.sizes['c']
        x = images.sizes['x']
        y = images.sizes['y']
        z = images.sizes['z']
        time_series = images.get_timesteps()
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
                #print(hf[str(i1)+ "/time"])
                for c1 in range(c):
                    images.default_coords['c'] = c1
                    im3d[1-c1] = np.array(images[i1]).astype(np.int16)
                dset[...] = im3d[:,:-X_pix_cut,:,:]
            name1 = os.path.basename(hdfout)
            name1 = name1.split(".")
            name = name1[0]
            print(name)
            hf.attrs["name"]=name#os.path.basename(hdfout)
            hf.attrs["C"]=c
            hf.attrs["W"]=int(x-X_pix_cut)
            hf.attrs["H"]=y
            hf.attrs["D"]=z
            hf.attrs["T"]=t
            #hf.attrs['Time_series'] = time_series
            hf.attrs["N_neurons"]=20
    hf.close()
