'''
This script computes the activity of neurons in both channels.
It computes the activity only in one z-stack where the neuron or neurite is most present.
It should be applied to an H5 file (Infilename) that contains both green and red channels.
The output is saved to an H5 file defined by hdfout.
It processes all the frames and uses the available ground truth masks,
saved as h5[str(i) + '/mask'], where i is the frame number.

The output is an h5 file with the following datasets:
"Real_time_series":time stamp of each frame. a vector of length # T, T is the
total number of frames in each movie
"Mean_30perc/Red" : top 30% pixels in red channel for each neuron mask (NxT)
"Mean_30perc/Green": (green channel intensity of the same top 30% pixels)/Mean_30perc_Red (NxT)
"Mean/Green": green channel intensity of the same top 30% pixels (NxT)
"Max/Red": maximum intensity of red pixels (channel 0) in each neurons mask (NxT)
"Max/Green": maximum intensity of pixels (channel 1) in each neurons mask (NxT)
"Mean/Red": average intensity of red pixels (channel 0) in each neurons mask (NxT)
"Volume tracks": number of pixels in each neurons mask (NxT)
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

Infilename = sys.argv[1]
hdfout = sys.argv[2]


NumberOfNeurons = 21#number of neurons+1
OnlyEvenFrames=False


NeuronsToPlot = range(1,NumberOfNeurons+1)
#insert the name of the neurons you had identified:
NeuronList = dict.fromkeys(range(1,NumberOfNeurons+1))
for d in range(1,NumberOfNeurons+1):
	NeuronList[d] = str(d)
NeuronList[NumberOfNeurons] = 'mean'

def get_mode_Z(mask,neurite):
    submask = (mask == neurite)
    x,y,z = np.nonzero(submask)
    stack = stats.mode(z)[0]
    return stack

def save_in_h5(h5File,dsetname,data):
    '''
    saves the dataset data under the name dsetname in the h5 file h5File
    '''
    if not dsetname in h5File.keys():
        dset = h5File.create_dataset(dsetname, (np.shape(data)), dtype="f4", compression="gzip")
    dset[...] = data



Mode = 2  # o0:max Mode, 1: mean mode,2:Mean+G/R ratio
dependencies = ["W", "H", "D", "C", "T", "N_neurons"]
h5 = h5py.File(Infilename, "r+")
if 'net' in h5.keys():
	NNname = list(h5['net'].keys())
for dep in dependencies:
    if dep not in h5.attrs.keys():
        h5.close()
        assert False, "Dependency "+dep+" not in  attributes"
T = h5.attrs["T"]
# x,y,z ordering
C, W, H, D = h5.attrs["C"], h5.attrs["W"], h5.attrs["H"], h5.attrs["D"]
k = 0
#T= 200#Test
Activity = np.zeros([2, NumberOfNeurons+2, T])#activity of green channel devided by red channel
ActivityMax = np.zeros([2, NumberOfNeurons+2, T])
ActivityMean = np.zeros([2, NumberOfNeurons+2, T])#activity of green channel

time_scale = 600/1715# 600/1715#600/1715#convert to second unit
time = time_scale*np.array(range(T))
real_time = np.zeros((1,T))
VolumeTrack = np.zeros((NumberOfNeurons+2,T))
save_realtime=False
if "0/time" in h5.keys():
	print("saving real time")
	save_realtime=True
if True:
    for i in range(T):
        print(i)
        if save_realtime:
            real_time[0,i] = np.array(h5[str(i)+"/time"])
            print(real_time[0,i])
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
        ActivityMean[0, NumberOfNeurons, i] = np.mean(frame[0])
        if C > 1:
            ActivityMean[1, NumberOfNeurons, i] = np.mean(frame[1])
        else:
            ActivityMean[1, NumberOfNeurons, i] = np.mean(frame[0])
        Activity[0, NumberOfNeurons, i] = np.mean(frame[0])
        if C > 1:
             Activity[1, NumberOfNeurons, i] = np.mean(frame[1])
        else:
             Activity[1, NumberOfNeurons, i] = np.mean(frame[0])
        for cell in np.unique(mask):
            if not cell == 0:
                stack = get_mode_Z(mask,cell)
                submask = (mask == cell)
                for z in range(np.shape(submask)[2]):
                    if not z==stack:
                       submask[:,:,z] =0
                cellCoorX, cellCoorY, cellCoorZ = np.nonzero(submask)
                #print((cellCoorZ))
                cellInt = np.zeros([C, len(cellCoorX)])#intensity of each pixel in the mask of cell
                Volume = len(cellCoorX)
                VolumeTrack[cell,i] = len(cellCoorX)
                for j in range(len(cellCoorX)):
                    cellInt[0, j] = frame[0, cellCoorX[j], cellCoorY[j],cellCoorZ[j]]  # red channel intensity
                    if C > 1:
                        # green channel intensity
                        cellInt[1, j] = frame[1, cellCoorX[j],cellCoorY[j], cellCoorZ[j]]
                    #else:
                    #    cellInt[1, j] = cellInt[0, j]


				# maxMode
                if True:#Mode==0:
                   ActivityMax[0, cell, i] = np.max(cellInt[0])  # /MaxFrR
                   if C > 1:
                      ActivityMax[1, cell, i] = np.max(cellInt[1])  # /MaxFrGr
				# MeanMode
                if Volume > 20:
                    TopPix = int(Volume/3) + 3#top 30 percent
                    Sort = np.argsort(cellInt[0, :])#sort the red channel activity
                    if C>1:
                          cellIntSorted = cellInt[1, Sort]#top in green
                    cellIntSorted_R = cellInt[0,Sort]#top in red
                    Activity[0, cell, i] = np.mean(cellIntSorted_R[-TopPix:])#red channel intensity
                    if C>1:
                        ActivityMean[1,cell,i] = np.mean(cellIntSorted[-TopPix:])
                        Activity[1,cell,i] = np.mean(cellIntSorted[-TopPix:])/Activity[0,cell,i]
                    else:
                        ActivityMean[1,cell,i] = np.mean(cellIntSorted_R[-TopPix:])
                        Activity[1,cell,i] = np.mean(cellIntSorted_R[-TopPix:])
                else:
                    Activity[0,cell,i] = np.mean(cellInt[0,:])
                    if C>1:
                        ActivityMean[1,cell,i] = np.mean(cellInt[1,:])
                        Activity[1,cell,i] = np.mean(cellInt[1,:]) / Activity[0,cell,i]
ActivityP = Activity
Activity_d = Activity


timeP = time
if OnlyEvenFrames:
	Evenindices = 2*np.array(range(int(len(time)/2)))

	timeP = time[Evenindices]
	ActivityP = ActivityP[:,:,Evenindices]
	VolumeTrackP = VolumeTrack[:,Evenindices]
else:
	VolumeTrackP = VolumeTrack


fig, axs = plt.subplots(len(NeuronsToPlot), 1)
p=0
for l in NeuronsToPlot:
    axs[p].set_ylabel(NeuronList[l], fontsize=5, labelpad=26,rotation=0)
    axs[p].yaxis.set_tick_params(labelsize=6)
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

    axs2[p].set_ylabel(NeuronList[l], fontsize=6, labelpad=2,rotation=0)
    axs2[p].yaxis.set_tick_params(labelsize=4)
    axs2[p].yaxis.labelpad = 26
    seqR = np.nonzero(ActivityP[0,l,:])#time points where the neuron l is in the mask
    seqG = np.nonzero(ActivityP[1,l,:])
    if not Mode == 2:
    	axs2[p].scatter(timeP[seqR],ActivityP[0,l,seqR],s=1,color = 'r')
    if C>1:
        axs2[p].scatter(timeP[seqG],ActivityP[1,l,seqG],s=1,color = 'g')
    axs2[p].axvline(x=40)
    axs2[p].axvline(x=100)
    axs2[p].axvline(x=160)
    axs2[p].axvline(x=220)
    axs2[p].axvline(x=280)
    axs2[p].axvline(x=340)
    axs2[p].axvline(x=400)
    axs2[p].axvline(x=460)
    axs2[p].axvline(x=520)
    axs2[p].axvline(x=580)
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

save_in_h5(hf,"Time_series-s",time)
save_in_h5(hf,"Real_time_series",real_time)
save_in_h5(hf,"Mean_30perc/Red",Activity[0,:,:])
save_in_h5(hf,"Mean_30perc/Green",Activity[1,:,:])
save_in_h5(hf,"Max/Red",ActivityMax[0,:,:])
save_in_h5(hf,"Max/Green",ActivityMax[1,:,:])
save_in_h5(hf,"Mean/Red",ActivityMean[0,:,:])
save_in_h5(hf,"Mean/Green",ActivityMean[1,:,:])
save_in_h5(hf,"Volume tracks",VolumeTrack)

hf.close()
