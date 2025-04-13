import sys
import os
import h5py
import importlib
import numpy as np
import scipy.ndimage as sim
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import linalg as LA
import math
from scipy.ndimage import affine_transform
import imutils
from scipy import ndimage
save_annotations =False

Infilename = sys.argv[1]
ref_fr =1200# 0
cell1 =1#8
cell2 =2#7
cell3  =7#9
cell3_0  =cell3
cell4 = 7#12
cell4_0 = cell4

#python3 rotateh5_0401.py name of the file.h5
#2105,2108,2190,2181,2170,2162,2126,2125,2123,2122,2120,2119,2099,2098,2086,2083,2034,2033,2032,2019,2018,2011,2007,1972,1767,1758,1757,1754,1750,1749,1748

h5 = h5py.File(Infilename,"r")
#hf = h5py.File((Infilename.split('.'))[0]+'_rotated_moreSeg.h5',"w")
if save_annotations:
    hf = h5py.File((Infilename.split('.'))[0]+'_rotated_mask.h5',"w")
else:
    hf = h5py.File((Infilename.split('.'))[0]+'_rotated.h5',"w")

start_pix_x = 344
end_pix_x = 600
start_pix_y= 282
end_pix_y= 538

if save_annotations:
    hf.create_group('original_match')

fr_i=0
fr_f= int(h5.attrs["T"])

maskref = np.array(h5[str(ref_fr)+"/mask"])
neuron1ref = (maskref == cell1)
neuron2ref = (maskref == cell2)
neuron3ref = (maskref == cell3)
neuron4ref = (maskref == cell4)

neuron1ref_coor = np.nonzero(neuron1ref)
neuron2ref_coor = np.nonzero(neuron2ref)
neuron3ref_coor = np.nonzero(neuron3ref)
neuron4ref_coor = np.nonzero(neuron4ref)
# each neuron's center of mass:
mean1ref = np.array([np.mean(neuron1ref_coor[0]), np.mean(neuron1ref_coor[1]), np.mean(neuron1ref_coor[2])])
mean2ref = np.array([np.mean(neuron2ref_coor[0]), np.mean(neuron2ref_coor[1]), np.mean(neuron2ref_coor[2])])
mean3ref = np.array([np.mean(neuron3ref_coor[0]),np.mean(neuron3ref_coor[1]),np.mean(neuron3ref_coor[2])])
mean4ref = np.array([np.mean(neuron4ref_coor[0]),np.mean(neuron4ref_coor[1]),np.mean(neuron4ref_coor[2])])

line13ref0 = (mean3ref - mean1ref)[:2]
line14ref0 = (mean4ref - mean1ref)[:2]
line13ref = line13ref0.copy()
W = h5.attrs["W"]
H = h5.attrs["H"]

def FindCenters(mask,cell1,cell2,cell3):
    neuron1 = (mask == cell1)
    neuron2 = (mask == cell2)
    neuron3 = (mask == cell3)
    neuron1_coor = np.nonzero(neuron1)
    neuron2_coor = np.nonzero(neuron2)
    neuron3_coor = np.nonzero(neuron3)
    # each neuron's center of mass:
    mean1 = np.array([np.mean(neuron1_coor[0]), np.mean(neuron1_coor[1]), np.mean(neuron1_coor[2])])
    mean2 = np.array([np.mean(neuron2_coor[0]), np.mean(neuron2_coor[1]), np.mean(neuron2_coor[2])])
    mean3 = np.array([np.mean(neuron3_coor[0]),np.mean(neuron3_coor[1]),np.mean(neuron3_coor[2])])
    return mean1, mean2, mean3

def calculate_sine(arrow10, arrow20):
    # Compute the cross product
    arrow1 = [arrow10[0],arrow10[1],0]
    arrow2 = [arrow20[0],arrow20[1],0]
    cross_product = np.cross(arrow1, arrow2)

    # Calculate the magnitude of the cross product
    cross_product_magnitude = np.linalg.norm(cross_product)

    # Calculate the magnitudes of the arrows
    arrow1_magnitude = np.linalg.norm(arrow1)
    arrow2_magnitude = np.linalg.norm(arrow2)
    print(cross_product)
    # Calculate the sine
    sine = cross_product[2] / (arrow1_magnitude * arrow2_magnitude)

    return sine

count = 0
changeAng=0
fr_range = range(fr_i,fr_f)
if save_annotations:
    fr_range = [100,2000,1200,1300]

for t in fr_range:
    print('t: '+str(t))
    frame0 = np.array(h5[str(t)+"/frame"])
    frame = frame0[0]
    frameG = frame0[1]
    d_3_transform = np.zeros((3, 4))
    if str(t)+"/mask" in h5.keys():
        mask = np.array(h5[str(t)+"/mask"])
        if cell1 in np.unique(mask) and cell2 in np.unique(mask):
            if not cell3 in np.unique(mask):
                cell3 = cell4_0
                line13ref = line14ref0.copy()
                changeAng=1
                print("neuron 8 not found")
            if cell3 in np.unique(mask):
                mean1, mean2, mean3 = FindCenters(mask,cell1,cell2,cell3)

                #line connecting neurons
                line12 = (mean2 - mean1)[:2]
                line13 = (mean3 - mean1)[:2]
                print(line13)
                print(line13ref)
                cosine = np.inner(line13,line13ref)/(LA.norm(line13)*LA.norm(line13ref))
                sine = calculate_sine(line13,line13ref)
                print(sine)
                print(cosine)
                if cosine>1:
                    cosine=1
                if sine>0:
                    angle0=math.acos(cosine)
                elif sine<0:
                    angle0=-math.acos(cosine)
                else:
                    angle0=0

                #cosine13 = np.inner(line13,[1,0])/(LA.norm(line13))#the angle of neuron 13 line from horizontal axis
                #cosine13ref = np.inner(line13ref,[1,0])/(LA.norm(line13ref))
                #if cosine13>1:
                #    cosine13=1
                #if line13[1]*line13ref[1]>0: # if the line 13 in the ref frame is in the same direction(+ or -) as the current frame
                #    angle0 = math.acos(cosine13ref)-math.acos(cosine13)#math.acos(cosine)
                #else:
                #    angle0 = (math.acos(cosine13ref)+math.acos(cosine13))#math.acos(cosine)

                d_3_transform[:2, :2] = [[math.cos(angle0),-math.sin(angle0)],[math.sin(angle0),math.cos(angle0)]]
                d_3_transform[2, 2] = 1.0

                frame_rot= np.zeros((2,np.shape(frame0)[1],np.shape(frame0)[2],np.shape(frame0)[3]))


                offset = -d_3_transform[:,3]
                cval = np.median(frame)
                rot1 =  [[1,0,0],[0,1,0],[0,0,1]]
                rot = d_3_transform[:,:3]

                mode0 = 'constant'
                P = 100
                frame_temp = np.zeros(np.shape(frame))
                frame_temp_G = np.zeros(np.shape(frameG))
                mask_temp = np.zeros(np.shape(mask))
                frame_temp = np.pad(frame,((P,P),(P,P),(0,0)), 'median')#affine_transform(frame,rot1,offset,mode=mode0,cval = cval, order=3)
                frame_temp_G = np.pad(frameG,((P,P),(P,P),(0,0)), 'median')
                mask_temp= np.pad(mask,((P,P),(P,P),(0,0)), 'median')#affine_transform(mask,rot1,offset,mode=mode0,cval = 0, order=0)
                if changeAng == 0:
                    angleDeg = angle0*180/math.pi
                else:
                    angleDeg = angle0*180/math.pi
                    changeAng = 0
                print("angle")
                print(angleDeg)
                frame_temp[:,:,:] = ndimage.rotate(frame_temp[:,:,:], angle=angleDeg, reshape=False,mode=mode0,cval = cval, order=3)
                frame_temp_G[:,:,:] = ndimage.rotate(frame_temp_G[:,:,:], angle=angleDeg, reshape=False,mode=mode0,cval = cval, order=3)
                mask_temp[:,:,:] = ndimage.rotate(mask_temp[:,:,:], angle=angleDeg, reshape=False,mode=mode0,cval = 0, order=0)


                mean1t, mean2t, mean3t = FindCenters(mask_temp,cell1,cell2,cell3)
                mean1tref, mean2tref, mean3tref = FindCenters(np.pad(maskref,((P,P),(P,P),(0,0)), 'median'),cell1,cell2,cell3)
                d_3_transform[:2, 3] = (mean1tref - mean1t)[:2]
                offset = -d_3_transform[:,3]

                frame_temp = affine_transform(frame_temp,rot1,offset,mode=mode0,cval = cval, order=3)
                frame_temp_G = affine_transform(frame_temp_G,rot1,offset,mode=mode0,cval = cval, order=3)
                frame_rot[0,:,:,:] =frame_temp[P:-P,P:-P,:]
                frame_rot[1,:,:,:] =frame_temp_G[P:-P,P:-P,:]
                mask_temp = affine_transform(mask_temp,rot1,offset,mode=mode0,cval = 0, order=0)
                mask_rot = mask_temp[P:-P,P:-P,:]
                hf.create_dataset(str(count) + "/frame",data=frame_rot[:,start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], compression="gzip")
                if not save_annotations:
                    hf.create_dataset("net/CZANet_Final/"+str(count)+"/predmask",data=mask_rot[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
                else:
                    hf.create_dataset(str(count) + "/mask",data=mask_rot[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
                hf.create_dataset(str(count) + "/time", data = h5[str(t) + "/time"])
                hf.create_dataset(str(count) + "/transform1/angle", data = angleDeg)
                hf.create_dataset(str(count) + "/transform1/offset", data = offset)
            else:
                hf.create_dataset(str(count) + "/frame",data=frame0[:,start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], compression="gzip")
                if not save_annotations:
                    hf.create_dataset("net/CZANet_Final/"+str(count)+"/predmask",data=mask[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
                else:
                    hf.create_dataset(str(count) + "/mask",data=mask_rot[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
                hf.create_dataset(str(count) + "/time", data = h5[str(t) + "/time"])
                hf.create_dataset(str(count) + "/transform1/angle", data =  0)#new
                hf.create_dataset(str(count) + "/transform1/offset", data = [0,0,0])#new

        else:
            hf.create_dataset(str(count) + "/frame",data=frame0[:,start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], compression="gzip")
            if not save_annotations:
                hf.create_dataset(str(count) + "net/CZANet_Final/"+str(count)+"/predmask",data=mask[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
            else:
                hf.create_dataset(str(count) + "/mask",data=mask_rot[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
            hf.create_dataset(str(count) + "/time", data = h5[str(t) + "/time"])
            hf.create_dataset(str(count) + "/transform1/angle", data =  0)#new
            hf.create_dataset(str(count) + "/transform1/offset", data = [0,0,0])#new
    else:
        mask= np.zeros(np.shape(frame))
        mask_rot= np.zeros(np.shape(frame))
        print("mask doesnt exist for t= "+str(t))
        hf.create_dataset(str(count) + "/frame",data=frame0[:,start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], compression="gzip")
        if not save_annotations:
            hf.create_dataset(str(count) + "/net/CZANet_Final/"+str(count)+"/predmask",data=mask[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
        else:
            hf.create_dataset(str(count) + "/mask",data=mask_rot[start_pix_x:end_pix_x,start_pix_y:end_pix_y,:], dtype="i2", compression="gzip")
        hf.create_dataset(str(count) + "/time", data = h5[str(t) + "/time"])
        hf.create_dataset(str(count) + "/transform1/angle", data =  0)#new
        hf.create_dataset(str(count) + "/transform1/offset", data = [0,0,0])#new

    if save_annotations:
        hf['original_match'].attrs[str(count)]=int(t)
    print(cell3)
    cell3 = cell3_0
    line13ref = line13ref0.copy()
    count=count+1
hf.attrs["name"] = h5.attrs["name"]
hf.attrs["C"] = 2#h5.attrs["C"]
hf.attrs["W"] = -int(start_pix_x-end_pix_x)#h5.attrs["W"]
hf.attrs["H"] = -int(start_pix_y-end_pix_y)#h5.attrs["H"]
hf.attrs["D"] = h5.attrs["D"]
hf.attrs["T"] = len(fr_range)#h5.attrs["T"]
#hf.attrs['Time_series'] = time_series
hf.attrs["N_neurons"]= h5.attrs["N_neurons"]
hf.attrs["crop_coord"]= [[start_pix_x,end_pix_x],[start_pix_y,end_pix_y]]
hf.attrs['ROI']=np.array([start_pix_x,end_pix_x,start_pix_y,end_pix_y])
