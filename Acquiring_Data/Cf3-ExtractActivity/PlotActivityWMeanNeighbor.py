'''
This script computes the activity in a circle or cylinder volume around each neurons center.
It computes the activity only in one z-stack where the neuron or neurite is most present.
It should be applied to an H5 file (Infilename) that contains both green and red channels.
The output is saved to an H5 file defined by hdfout.
It processes all the frames and uses the available ground truth masks,
saved as h5[str(i) + '/mask'], where i is the frame number.
'''


import sys
import os
import h5py
import importlib
import numpy as np
import scipy.ndimage as sim
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import time

start_time = time.time()
Infilename = sys.argv[1]
hdfout = sys.argv[2]

NumberOfNeurons = 18#number of neurons+1
NeuronsToPlot = range(1,NumberOfNeurons+1)
NeuronList = dict.fromkeys(range(1,NumberOfNeurons+1))
for d in range(1,NumberOfNeurons+1):
	NeuronList[d] = str(d)

radius = 30
Zmi,Zma = 3,3
NeuronList[1] = 'RIA'
NeuronList[2] = 'RIA'
NeuronList[3] = 'RIB'
NeuronList[4] = 'RIM'
NeuronList[6] = 'RIM'
NeuronList[7] = 'RIB'
NeuronList[10] = 'senso'
NeuronList[NumberOfNeurons] = 'mean'
print(NeuronList)
############functions####################
def get_mode_Z(mask,neurite):
    submask = (mask == neurite)
    x,y,z = np.nonzero(submask)
    stack = stats.mode(z)[0]
    return stack

def get_center(mask,neurite):
    submask = (mask == neurite)
    x,y,z = np.nonzero(submask)
    x_m,y_m,z_m = np.mean(x),np.mean(y),np.mean(z)
    return int(x_m),int(y_m),int(z_m)

def get_sphere(mask,c,radius,z0=1,z1=2):
    'c: center of the sphere'
    # Create a grid of indices
    x, y, z = np.indices(mask.shape)
    # Calculate the distance from the center
    distance = np.sqrt((x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2)
    # Set the points within the sphere to 1
    mask[distance <= radius] = 1
    mask[distance > radius] = 0
    zmin=np.max([c[2]-z0,0])
    zmax=np.min([c[2]+z1,mask.shape[2]-1])
    mask[:,:,:zmin]=0
    mask[:,:,zmax:]=0
    return mask

def save_in_h5(h5File,dsetname,data):
    '''
    saves the dataset data under the name dsetname in the h5 file h5File
    '''
    if not dsetname in h5File.keys():
        dset = h5File.create_dataset(dsetname, (np.shape(data)), dtype="f4", compression="gzip")
    dset[...] = data

Mode = 2  # o0:max Mode, 1: mean mode,2:Mean+G/R ratio
dependencies = ["W", "H", "D", "C", "T", "N_neurons"]
h5 = h5py.File(Infilename, "r")
if 'net' in h5.keys():
	NNname = list(h5['net'].keys())
for dep in dependencies:
    if dep not in h5.attrs.keys():
        h5.close()
        assert False, "Dependency "+dep+" not in  attributes"

T = h5.attrs["T"]
# x,y,z ordering
C, W, H, D = h5.attrs["C"], h5.attrs["W"], h5.attrs["H"], h5.attrs["D"]

Activity = np.zeros([2, NumberOfNeurons+2, T])#activity of green channel devided by red channel
ActivityMax = np.zeros([2, NumberOfNeurons+2, T])
ActivityGr = np.zeros([2, NumberOfNeurons+2, T])#activity of green channel
ActivityMean = np.zeros([2, NumberOfNeurons+2, T]) #mean activity of each mask region

time_scale = 600/1715# 600/1715#600/1715#convert to second unit
timet = time_scale*np.array(range(T))
real_time = np.zeros((1,T))
VolumeTrack = np.zeros((NumberOfNeurons+2,T))
save_realtime = False

if "0/time" in h5.keys():
	print("saving real time")
	save_realtime=True

for i in range(T):
    print(i)
    if save_realtime:
        real_time[0,i] = np.array(h5[str(i)+"/time"])
        #print(real_time[0,i])
    frame = np.array(h5[str(i)+"/frame"]).astype(np.int16)
    MaxFrR = np.max(frame[0])
    '''First, save the maximum and mean pixel values of the frame'''
    if C > 1:
        MaxFrGr = np.max(frame[1])
    mkey = str(i) + "/mask"  # (i in traininds) or (i>1714):
    if mkey in h5.keys():
        mask = np.array(h5[str(i)+"/mask"])#.astype(np.int16)
    else:
        mask = np.zeros(np.shape(frame))
    ActivityMax[0, NumberOfNeurons, i] = MaxFrR
    if C > 1:
        ActivityMax[1, NumberOfNeurons, i] = MaxFrGr
    ActivityGr[0, NumberOfNeurons, i] = np.mean(frame[0])
    if C > 1:
        ActivityGr[1, NumberOfNeurons, i] = np.mean(frame[1])
    else:
        ActivityGr[1, NumberOfNeurons, i] = np.mean(frame[0])
    Activity[0, NumberOfNeurons, i] = np.mean(frame[0])
    if C > 1:
        Activity[1, NumberOfNeurons, i] = np.mean(frame[1])/np.mean(frame[0])
    else:
        Activity[1, NumberOfNeurons, i] = np.mean(frame[0])
    for cell in np.unique(mask):
        if not cell == 0:
            c1,c2,c3 = get_center(mask,cell)
            submask0 = (mask == cell)
            submask = get_sphere(submask0,[c1,c2,c3],radius,Zmi,Zma)
            cellCoorX, cellCoorY, cellCoorZ = np.nonzero(submask)

            cellInt = np.zeros([C, len(cellCoorX)])#intensity of each pixel in the mask of cell
            Volume = len(cellCoorX)
            VolumeTrack[cell,i] = len(cellCoorX)
            for j in range(len(cellCoorX)):
                cellInt[0, j] = frame[0, cellCoorX[j], cellCoorY[j],cellCoorZ[j]]  # red channel intensity
                if C > 1:
                    # green channel intensity
                    cellInt[1, j] = frame[1, cellCoorX[j],cellCoorY[j], cellCoorZ[j]]

            ActivityMax[0, cell, i] = np.max(cellInt[0])  # /MaxFrR
            if C > 1:
                ActivityMax[1, cell, i] = np.max(cellInt[1])  # /MaxFrGr
            ActivityMean[1,cell,i] = np.mean(cellInt[1,:])
            ActivityMean[0,cell,i] = np.mean(cellInt[0,:])
			# MeanMode
            if True:
                TopPix = int(Volume/3)+3#top 30 percent
                Sort = np.argsort(cellInt[0, :])#sort the red channel activity
                if C>1:
                    cellIntSorted = cellInt[1, Sort]#top in green
                cellIntSorted_R = cellInt[0,Sort]#top in red
                Activity[0, cell, i] = np.mean(cellIntSorted_R[-TopPix:])#red channel intensity
                if C>1:
                    ActivityGr[1,cell,i] = np.mean(cellIntSorted[-TopPix:])
                    Activity[1,cell,i] = np.mean(cellIntSorted[-TopPix:])/Activity[0,cell,i]
                else:
                    ActivityGr[1,cell,i] = np.mean(cellIntSorted_R[-TopPix:])
                    Activity[1,cell,i] = np.mean(cellIntSorted_R[-TopPix:])
            else:
                Activity[0,cell,i] = np.mean(cellInt[0,:])
                if C>1:
                    ActivityGr[1,cell,i] = np.mean(cellInt[1,:])
                    Activity[1,cell,i] = np.mean(cellInt[1,:]) / Activity[0,cell,i]
ActivityP = Activity
Activity_d = Activity
timeP = timet
VolumeTrackP = VolumeTrack
end_time = time.time()
total_time = end_time - start_time
print(f"Script execution time: {total_time:.2f} seconds")
###########plottings###################3
fig, axs = plt.subplots(len(NeuronsToPlot), 1)
p=0
for l in NeuronsToPlot:
    axs[p].set_ylabel(NeuronList[l], fontsize=5, labelpad=26,rotation=0)
    axs[p].yaxis.set_tick_params(labelsize=6)
    #axs[p].yaxis.labelpad = 26
    seqR = np.nonzero(ActivityP[0,l,:])#time points where the neuron l is in the mask
    seqG = np.nonzero(ActivityP[1,l,:])
    if not Mode == 2:#only plot red channel when you are not in the ratio mode
    	axs[p].plot(timeP[seqR],ActivityP[0,l,seqR],color = 'r')
    if C>1:
        axs[p].plot(timeP[seqG[0]],ActivityP[1,l,seqG[0]],color = 'g')
    p = p+1
plt.show()

#plotting points plots
p=0
fig2, axs2 = plt.subplots(len(NeuronsToPlot), 1)
for l in NeuronsToPlot:
    #axs2[p].set(ylabel=NeuronList[l])
    axs2[p].set_ylabel(NeuronList[l], fontsize=6, labelpad=2,rotation=0)
    axs2[p].yaxis.set_tick_params(labelsize=4)
    axs2[p].yaxis.labelpad = 26
    seqR = np.nonzero(ActivityP[0,l,:])#time points where the neuron l is in the mask
    seqG = np.nonzero(ActivityP[1,l,:])
    if not Mode == 2:
    	axs2[p].plot(timeP[seqR],ActivityP[0,l,seqR],color = 'r')
    if C>1:
        axs2[p].plot(timeP[seqG],ActivityP[1,l,seqG].transpose(),color = 'g')
        te=40
    if len(timeP[seqG])>1:
        while te<np.max(timeP[seqG]):
            axs2[p].axvline(x=te)
            te= te+60
    p = p+1
plt.show()
#plotting mean activity
p=0
fig2, axs2 = plt.subplots(len(NeuronsToPlot), 1)
for l in NeuronsToPlot:
    #axs2[p].set(ylabel=NeuronList[l])
    axs2[p].set_ylabel(NeuronList[l], fontsize=6, labelpad=2,rotation=0)
    axs2[p].yaxis.set_tick_params(labelsize=4)
    axs2[p].yaxis.labelpad = 26
    seqR = np.nonzero(ActivityMean[0,l,:])#time points where the neuron l is in the mask
    seqG = np.nonzero(ActivityMean[1,l,:])
    if not Mode == 2:
    	axs2[p].plot(timeP[seqR],ActivityMean[0,l,seqR],color = 'r')
    if C>1:
        axs2[p].plot(timeP[seqG],ActivityMean[1,l,seqG].transpose(),color = 'g')
        te=40
    if len(timeP[seqG])>1:
        while te<np.max(timeP[seqG]):
            axs2[p].axvline(x=te)
            te= te+60
    p = p+1
plt.show()

p=0
fig3, axs3 = plt.subplots(len(NeuronsToPlot), 1)
#fig3.suptitle("Volume tracks")
for l in NeuronsToPlot:
    #axs2[p].set(ylabel=NeuronList[l])
    axs3[p].set_ylabel(NeuronList[l], fontsize=6, labelpad=2,rotation=0)
    axs3[p].yaxis.set_tick_params(labelsize=4)
    axs3[p].yaxis.labelpad = 26
    seqR = np.nonzero(ActivityP[0,l,:])#time points where the neuron l is in the mask
    seqG = np.nonzero(ActivityP[1,l,:])
    VolSeq = VolumeTrackP[l,seqR]
    VolSeq =VolSeq.T
    axs3[p].plot(timeP[seqR],VolSeq,color = 'r')
    p = p+1
plt.show()
print(np.mean(VolumeTrack,1))


###############save as a new h5 file######################3
hf = h5py.File(hdfout, 'w')#make the h5 file where you want to save the results
name1 = Infilename.split(".")
name2 = name1[0]
name3 = name2.split("/")
name = name3[-1]
print(name)
hf.attrs["name"]=name
hf.attrs["neuron_Number"]=int(NumberOfNeurons+1)
for k in NeuronList.keys():
    hf.attrs["Neuron_id/"+str(k)] = NeuronList[k]

if "Merge_Order" in h5.attrs.keys():
	hf.attrs["Merge_Order"] = h5.attrs["Merge_Order"]
	print(h5.attrs["Merge_Order"])
	hf.attrs["Merged_Movies_Size"] = h5.attrs["Merged_Movies_Size"]

save_in_h5(hf,"Time_series-s",timet)
save_in_h5(hf,"Real_time_series",real_time)
save_in_h5(hf,"Mean_30perc/Red",Activity[0,:,:])
save_in_h5(hf,"Mean_30perc/Green",Activity[1,:,:])
save_in_h5(hf,"Max/Red",ActivityMax[0,:,:])
save_in_h5(hf,"Max/Green",ActivityMax[1,:,:])
save_in_h5(hf,"Mean/Red",ActivityGr[0,:,:])
save_in_h5(hf,"Mean/Green",ActivityGr[1,:,:])
save_in_h5(hf,"RealMean/Red",ActivityMean[0,:,:])
save_in_h5(hf,"RealMean/Green",ActivityMean[1,:,:])

save_in_h5(hf,"Volume tracks",VolumeTrack)

hf.close()
