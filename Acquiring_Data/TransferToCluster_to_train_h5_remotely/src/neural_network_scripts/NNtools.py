import glob
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
import torch
import sklearn.metrics as skmet
import skimage.feature as skfeat
import cc3d
import scipy.spatial as spat
import os
import h5py
import scipy.spatial as spat
import matplotlib.pyplot as plt
import FourierAugment
import scipy.ndimage as sim
import sys
from scipy.optimize import fmin_l_bfgs_b
from skimage.measure import find_contours
import conv_autoenc1
import torch.nn as nn
import umap.umap_ as umap

class TrainDataset(Dataset):
    def __init__(self,dirname,shape,meansub=False,high=False,maskdirname="masks"):
        super().__init__()
        self.high=high
        self.dirname=dirname
        self.maskdirname=maskdirname
        self.shape=shape
        self.filelist=glob.glob(os.path.join(self.dirname,self.maskdirname,"*"))
        self.indlist=[int(name.strip().split("/")[-1].split("_")[-1].split(".")[0]) for name in self.filelist]   # Todo cleaner with path.??
        self.indlist=np.array(self.indlist)
        self.num_frames_tot=len(self.indlist)
        self.meansub=meansub
        inf={}
        inf["mask"]="mask"

        self.trffull=A.Compose([
            A.ShiftScaleRotate(shift_limit=0.12,scale_limit=0.12,rotate_limit=30,interpolation=1,border_mode=0,value=0,mask_value=0,p=1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
            A.OpticalDistortion(border_mode=0,p=0.7),
            A.MotionBlur(blur_limit=8,p=0.7),
            ],
            additional_targets=inf
            )
        self.trfaff=A.Compose([
            A.ShiftScaleRotate(shift_limit=0.12,scale_limit=0.12,rotate_limit=30,interpolation=1,border_mode=0,value=0,mask_value=0,p=1),
            ],
            additional_targets=inf
            )
        self.fourier_augment_strong=FourierAugment.FourierDeformationPk(self.shape[1:],k_cut_dimless=2.5,Pk=lambda k: 2/(k**2+0.1),dimscale=(1,1,5.),defscale=(0.5,0.5,0.25))
        self.fourier_augment_weak=FourierAugment.FourierDeformationPk(self.shape[1:],k_cut_dimless=2.5,Pk=lambda k: 1/(k**3+0.1),dimscale=(1,1,5.),defscale=(0.25,0.25,0.125))
        self.p_four=0.5
        self.augment="full"#"full","aff","none","aff_cut","aff_cut_four"
        self.grid2d=None

    def cut(self,feed,sh2d,p=1):
        if torch.rand(1).item()<p:
            point=torch.rand(2).numpy()*np.array(sh2d)
            m=np.tan(np.pi*(torch.rand(1).item()-0.5))
            if np.isnan(m):
                m=1
            top_or_bottom=np.random.random()>0.5
            if self.grid2d is None:
                self.grid2d=np.array(np.meshgrid(np.arange(sh2d[0]),np.arange(sh2d[1]),indexing="ij"))
            if top_or_bottom:
                valid=((self.grid2d[1]-point[1])>m*(self.grid2d[0]-point[0]))
            else:
                valid=((self.grid2d[1]-point[1])<m*(self.grid2d[0]-point[0]))
            feed["image"]*=valid[:,:,None]
            feed["mask"]*=valid[:,:,None]
        return feed

    def get_trf(self,fr,mask,mode):
        if mode=="none":
            return torch.tensor(fr),torch.tensor(mask)
        feed={}
        feed["mask"]=mask#z is automatically channel

        if mode=="full":
            res=self.trffull(image=fr.transpose(1,2,0,3).reshape(fr.shape[1],fr.shape[2],-1),**feed)
            res=self.cut(res,fr.shape[1:3],p=1)
            fr_aug=res["image"].reshape(fr.shape[1],fr.shape[2],fr.shape[0],-1).transpose(2,0,1,3)
            mask_aug=res["mask"]
            fr_aug,mask_aug=torch.tensor(fr_aug),torch.tensor(mask_aug)
            if torch.rand(1).item()<self.p_four:
                fr_aug,mask_aug=self.fourier_augment_strong(fr_aug,mask=mask_aug,single_batch=True)
        elif mode=="aff_cut":
            res=self.trfaff(image=fr.transpose(1,2,0,3).reshape(fr.shape[1],fr.shape[2],-1),**feed)
            res=self.cut(res,fr.shape[1:3],p=1)
            fr_aug=res["image"].reshape(fr.shape[1],fr.shape[2],fr.shape[0],-1).transpose(2,0,1,3)
            mask_aug=res["mask"]
            fr_aug,mask_aug=torch.tensor(fr_aug),torch.tensor(mask_aug)
        elif mode=="aff_cut_four":
            res=self.trfaff(image=fr.transpose(1,2,0,3).reshape(fr.shape[1],fr.shape[2],-1),**feed)
            res=self.cut(res,fr.shape[1:3],p=1)
            fr_aug=res["image"].reshape(fr.shape[1],fr.shape[2],fr.shape[0],-1).transpose(2,0,1,3)
            mask_aug=res["mask"]
            fr_aug,mask_aug=torch.tensor(fr_aug),torch.tensor(mask_aug)
            if torch.rand(1).item()<self.p_four:
                fr_aug,mask_aug=self.fourier_augment_weak(fr_aug,mask=mask_aug,single_batch=True)
        elif mode=="aff":
            res=self.trfaff(image=fr.transpose(1,2,0,3).reshape(fr.shape[1],fr.shape[2],-1),**feed)
            fr_aug=res["image"].reshape(fr.shape[1],fr.shape[2],fr.shape[0],-1).transpose(2,0,1,3)
            mask_aug=res["mask"]
            fr_aug,mask_aug=torch.tensor(fr_aug),torch.tensor(mask_aug)
        else:
            assert False,"wrong trf(augmentation) mode"

        return fr_aug,mask_aug

    def __getitem__(self,i):
        assert 0<=i<self.num_frames_tot
        ii=self.indlist[i]
        fr=np.load(self.dirname+"/frames/frame_"+str(ii)+".npy")[:self.shape[0]]/255
        mask=np.load(os.path.join(self.dirname,self.maskdirname,"mask_"+str(ii)+".npy"))
        if self.high:
            high=np.load(self.dirname+"/highs/high_"+str(ii)+".npy")/255
            fr=fr*(np.sum(high,axis=0)>0.5)[None,:,:,None]
        fr=fr.astype(np.float32)
        if self.meansub:
            fr=fr-np.mean(fr,axis=(1,2,3))[:,None,None,None]
        fr,mask=self.get_trf(fr,mask,self.augment)
        return [fr,mask]

    def __len__(self):
        return self.num_frames_tot

    def change_augment(self,augment):
        self.augment=augment
        return self.augment

    def real_ind_to_dset_ind(self,real_inds):
        query={}
        for i,ind in enumerate(self.indlist):
            query[ind]=i
        dset_inds=[]
        for ind in real_inds:
            dset_inds.append(query[ind])
        return dset_inds

class EvalDataset(Dataset):
    def __init__(self,dirname,shape,high=True,meansub=False):
        super().__init__()
        self.shape=shape
        self.dirname=dirname
        self.meansub=meansub
        self.high=high

    def __getitem__(self,ii):
        fr=np.load(self.dirname+"/frames/frame_"+str(ii)+".npy")[:self.shape[0]]/255
        try:
            mask=np.load(self.dirname+"/masks/mask_"+str(ii)+".npy")
            mask=torch.Tensor(mask)
        except FileNotFoundError:
            mask=None
        if self.high:
            high=np.load(self.dirname+"/highs/high_"+str(ii)+".npy")/255
            fr=fr*(np.sum(high,axis=0)>0.5)[None,:,:,None]
        fr=fr.astype(np.float32)
        if self.meansub:
            fr=fr-np.mean(fr,axis=(1,2,3))[:,None,None,None]
        return [torch.Tensor(fr),mask]

def repack(h5fn):
    h5=h5py.File(h5fn,"r")
    h5new=h5py.File(h5fn+"_temp","w")
    for key,val in h5.items():
        h5.copy(key,h5new)
    for key,val in h5.attrs.items():
        h5new.attrs[key]=val
    h5.close()
    h5new.close()
    os.remove(h5fn)
    os.rename(h5fn+"_temp",h5fn)

def save_into_h5(h5,state_dict):
    for key,val in state_dict.items():
        dset=h5.create_dataset(key,tuple(val.size()),dtype="f4")
        dset[...]=val.cpu().detach().numpy()

def load_from_h5(h5):
    res={}
    for key,val in h5.items():
        res[key]=torch.tensor(np.array(val))
    return res

def get_ious(preds,mask,skip=False):
    num_classes=preds.size(1)
    if skip:
        return np.full(num_classes,np.nan)
    maskgot=torch.argmax(preds,dim=1)
    ioubins=np.zeros(num_classes)
    for i in range(num_classes):
        thismask=(mask==i)
        if torch.sum(thismask).item()==0:
            ioubins[i]=np.nan
            continue
        thismaskgot=(maskgot==i)
        intersection=torch.sum(thismask&thismaskgot).item()
        union=torch.sum(thismask|thismaskgot).item()
        ioubins[i]=intersection/union
    return ioubins

def selective_ce(pred_raw,target_mask):
    existing=torch.unique(target_mask)
    with torch.no_grad():
        trf=torch.zeros(torch.max(existing)+1).to(device=pred_raw.device,dtype=torch.long)
        trf[existing]=torch.arange(0,len(existing)).to(device=pred_raw.device,dtype=torch.long)
        mask=trf[target_mask]
        mask.requires_grad=False
    return torch.nn.functional.cross_entropy(pred_raw[:,existing],mask)

def reconstruction_loss(pred_raw,fr,reconstruction_channel=0):
    rec=torch.sum(torch.sigmoid(pred_raw),dim=1)
    return torch.nn.functional.mse_loss(rec,fr[:,reconstruction_channel])

def gen_tracked_points(frame,pred,thres,grid,weight=False):
    cutpred=(frame>thres)*pred
    res=cc3d.connected_components(cutpred, connectivity=6)
    labels=[]
    sizes=[]
    cache_dict={}
    for i in range(np.max(res)+1):
        one=(res==i)
        cache_dict[i]=one
        labels.append(cutpred[one][0])
        sizes.append(np.sum(one))
    args=np.argsort(sizes)[::-1]
    pts_dict={0:None}
    if weight:
        weight_dict={}

    wgrid=grid*frame[...,None]
    for i in args:#decending
        l=labels[i]
        if l in pts_dict.keys():
            continue
        wvec=np.sum(wgrid[cache_dict[i]],axis=0)
        if np.sum(wvec)==0:
            continue
        norm=np.sum(frame[cache_dict[i]])
        pts_dict[l]=wvec/norm
        if weight:
            weight_dict[l]=norm
    if weight:
        return pts_dict,weight_dict
    return pts_dict

def get_pts_iou(package):
    fr,predmask,mask,grid,num_classes,thres,weight=package
    labels=np.arange(num_classes)
    if mask is None:
        iou=np.full((num_classes,num_classes),np.nan)
    else:
        iou=skmet.confusion_matrix(mask.flatten(),predmask.flatten(),labels=labels)
    pts=np.full((num_classes,3),np.nan)
    if weight:
        weights=np.full(num_classes,np.nan)
        pts_dict,weight_dict=gen_tracked_points(fr,predmask,thres=thres,grid=grid,weight=weight)
    else:
        pts_dict=gen_tracked_points(fr,predmask,thres=thres,grid=grid,weight=weight)
    for key,val in pts_dict.items():
        if key!=0:
            pts[key]=val
            if weight:
                weights[key]=weight_dict[key]
    if weight:
        return [pts,weights,iou]
    return [pts,iou]

def get_pts_from_masks(mask,num_class):
    '''
    MB added: to extract points assigne to the center of each mask object, and point assigned to center of mass of each two object.
    '''

    Cells = np.unique(mask)
    Cells = np.sort(Cells)
    Vol = np.zeros(len(Cells))
    CoM = np.zeros([len(Cells),3])
    for i in range(len(Cells)):
        Vol[i] = np.sum(mask==Cells[i])
        Coor = np.nonzero(mask==Cells[i])
        CoM[i] = [np.mean(Coor[0]), np.mean(Coor[1]), np.mean(Coor[2])]
    num_pts = num_class+(num_class*num_class//2)
    ptsCoor=np.full((num_pts,3),np.nan)
    for i in range(1,len(Cells)):#exclude zero
        C = Cells[i]
        Pos = num_class*(C-1)-(C*(C-1)//2)
        ptsCoor[Pos+i] = CoM[i]
        for j in range(i, len(Cells)):
            ptsCoor[Pos+j] = (CoM[i]*Vol[i]+CoM[j]*Vol[j])//(Vol[i]+Vol[j])
    return ptsCoor


def get_pts_from_masks0(mask,num_class):
    '''
    MB added: to extract points assigne to the center of each mask object, and point assigned to center of mass of each two object.
    '''

    Cells = np.unique(mask)
    Cells = np.sort(Cells)
    Vol = np.zeros(len(Cells))
    CoM = np.zeros([len(Cells),3])
    for i in range(len(Cells)):
        Vol[i] = np.sum(mask==Cells[i])
        Coor = np.nonzero(mask==Cells[i])
        CoM[i] = [np.mean(Coor[0]), np.mean(Coor[1]), np.mean(Coor[2])]
    num_pts = num_class-1
    ptsCoor=np.full((num_pts,3),np.nan)
    for i in range(1,len(Cells)):#exclude zero
        C = Cells[i]
        ptsCoor[i-1] = CoM[i]
    return ptsCoor


def get_pts_from_masksJV(maskFrom,maskTo):
    '''
    MB added: applies Jian-Vermuri registration between points on the border of
    maskFrom and points on the border of maskTo
    maskFrom: mask asarray
    maskTo: maskArray
    returns:
    ptto: list of points on the border of mask objects of maskTo
    after_tps: transformed result of the set "points"
    '''

    ptfrom = load_single_contour_Mask(maskFrom)# the frame in the training set
    ptto = load_single_contour_Mask(maskTo)# the new augmented frame
    ctrl_pts = ptto
    points, ref_seg, after_tps, func_min = registration_JV(ptfrom, ptto, ctrl_pts)

    return points, after_tps, ptto

def get_pts_from_masks2(mask,num_class):
    '''
    MB added: to extract points assigne to the center of each mask object, and point assigned to center of mass of each two object.
    '''
    import math
    Cells = np.unique(mask)
    Cells = np.sort(Cells)
    Vol = np.zeros(len(Cells))
    CoM = np.zeros([len(Cells),3])
    num_pts = (num_class-1)*(num_class-1)
    ptsCoor=np.full((num_pts,3),np.nan)
    for i in range(1,len(Cells)): #exclude zero
        Vol[i] = np.sum(mask==Cells[i])
        Coor = np.nonzero(mask==Cells[i])
        CoM[i] = [np.mean(Coor[0]), np.mean(Coor[1]), np.mean(Coor[2])]
        #distFromCenter = np.zeros(int(Vol[i]))# to get the point with highest distance from center
        #for j in range(int(Vol[i])):
        #    distFromCenter[j] = math.sqrt((Coor[0][j]-CoM[i][0])**2+(Coor[1][j]-CoM[i][0])**2+(Coor[2][j]-CoM[i][0])**2)
        C = Cells[i]
        #Pos = num_class*(C-1)-(C*(C-1)//2)
        ptsCoor[(num_class-1)*(C-1)+C-1] = CoM[i]
        for j in range(1, len(Cells)):
            if j != i :
                distvec = CoM[j]-CoM[i]
                directionVec = 5*(distvec)/np.max(abs(distvec))
                directionVec[0] = int(directionVec[0])
                directionVec[1] = int(directionVec[1])
                directionVec[2] = int(directionVec[2])
                border = CoM[i]
                border[0] = int(border[0])
                border[1] = int(border[1])
                border[2] = int(border[2])
                check = 0
                while ((border[0] in Coor[0]) and (border[1] in Coor[1]) and (border[2] in Coor[2])):
                    border[0] = border[0]+directionVec[0]
                    border[1] = border[1]+directionVec[1]
                    border[2] = border[2]+directionVec[2]
                    check = 1
                if check ==1 :
                    border[0] = border[0]-directionVec[0]
                    border[1] = border[1]-directionVec[1]
                    border[2] = border[2]-directionVec[2]
                    ptsCoor[(num_class-1)*(C-1)+Cells[j]-1] = border
    return ptsCoor

def registration_JV(points, reference, ctrl_pts):
    """
    :param points: (n,3) numpy array, the model to register
    :param reference: (n,3) numpy array, the reference model
    :param ctrl_pts: (n,3) numpy array, sample points of the model used for registration.
    :return: points, reference and the transformed points and the registration loss.
    after_tps: are the points after transformation to fit the ref
    """
    import gmmreg._core as core
    level = 4
    scales = [ .6, .3, .2, .1]
    lambdas = [ 0.1, .01, .001, .001]
    iters = [ 100, 100, 500, 300]
    [points, c_m, s_m] = core.normalize(points)
    [reference, c_s, s_s] = core.normalize(reference)
    [ctrl_pts, c_c, s_c] = core.normalize(ctrl_pts)
    after_tps, x0, loss = run_multi_level(points, reference, ctrl_pts, level, scales, lambdas, iters)
    points = core.denormalize(points, c_m, s_m)
    reference = core.denormalize(reference, c_s, s_s)
    after_tps = core.denormalize(after_tps, c_s, s_s)
    return points, reference, after_tps, loss

def run_multi_level(model, scene, ctrl_pts, level, scales, lambdas, iters):
    """
    The point set registration by Jian Vemuri, check https://pubmed.ncbi.nlm.nih.gov/21173443/
    :param model:(n,3) numpy array, the reference model
    :param scene: (n,3) numpy array, the scene
    :param ctrl_pts: (n,3) control points to register
    :param level: Integer,
    :param scales: list of scales of length level, Gaussian variance at each level,
    :param lambdas: list of double of length level, Bending regularizer at each level.related to the energy of the nonlinear transformation
    :param iters: list of Integers of length level,Number of iterations to run for each level
    :return: the transformed points, also the registration loss
    """
    import gmmreg._core as core
    [n, d] = ctrl_pts.shape
    x0 = core.init_param(n, d)
    [basis, kernel] = core.prepare_TPS_basis(model, ctrl_pts)
    loss = 1
    for i in range(level):
        x = fmin_l_bfgs_b(core.obj_L2_TPS, x0, None, args=(basis, kernel, scene, scales[i], lambdas[i]),
                          maxfun=iters[i])
        x0 = x[0]
        loss = x[1]
    after_tps = core.transform_points(x0, basis)
    return after_tps, x0, loss

def rotation_translation(Initial, final):
    """
    compute the max x,y rotation R and translation T, such that final - R@Initial + offset
    :param Initial: (n,3) numpy array, where n is number of 3D points
    :param final: (n,3) numpy array, where n is number of 3D points.
    :return: the concatenated numpy array
    """

    Initial = Initial[:, :2]
    final = final[:, :2]
    c_i = np.mean(Initial, axis=0)
    Initial = Initial - c_i
    c_f = np.mean(final, axis=0)
    final = final - c_f
    H = Initial.T @ final
    U, D, V = np.linalg.svd(H)
    det_sign = np.prod(D) > 0
    D = np.diag([1, det_sign])
    R = V.T @ D @ (U.T)
    offset = c_f - np.dot(R, c_i)
    transform_points = np.matmul(Initial, R.T)
    loss = np.linalg.norm(final - transform_points) / np.linalg.norm(final)
    # offset = offset.reshape((offset.shape[0],1))
    d_3_transform = np.zeros((3, 4))
    d_3_transform[:2, :2] = R
    d_3_transform[2, 2] = 1.0
    d_3_transform[:2, 3] = offset
    return d_3_transform, loss

def find_3D_contour(segment_binary):
    """
    Finds the 3d contour of a neuron.
    :param segment_binary: binary mask of one neuron
    :return: np array [[x1, y1, z1],...], a list of coordinates of points of the contour
    """
    contour_points = []
    for z in range(segment_binary.shape[2]):
        z_conts = find_contours(segment_binary[:, :, z], 0.5)
        for cont in z_conts:  # several contours
            for pt in cont:  # all points in contour
                contour_points.append([*pt, z])
    return contour_points


def sample_points_from_contour(contour):
    """
    Returns coordinates of points from the segment to be used for point set registration.
    :param contour: array [[x1, y1, z1],...], a list of the coordinates of all points of the contour.
    :return: list [[x1, y1, z1],...], a list of coordinates of approx n_points points from the contour
    MB: returns evenly distributed points on the xy projection of the contour
    """
    num_samples_per_neuron = 10
    n_points = num_samples_per_neuron
    sampled_points = []
    contour = np.asarray(contour)
    all_xs = np.unique(contour[:, 0])  # is sorted
    x_step = int(np.ceil(len(all_xs) / np.sqrt(n_points / 2)))
    sample_xs = all_xs[(x_step - 1) // 2::x_step]
    for x in sample_xs:
        sample_x = contour[contour[:, 0] == x]
        all_ys = np.unique(sample_x[:, 1])
        y_step = int(np.ceil(len(all_ys) / np.sqrt(n_points / 2)))
        sample_ys = all_ys[(y_step - 1) // 2::y_step]
        for y in sample_ys:
            sample_y = sample_x[sample_x[:, 1] == y]
            mi = min(sample_y[:, 2])
            sampled_points.append([x, y, mi])
            ma = max(sample_y[:, 2])
            if ma != mi:
                sampled_points.append([x, y, ma])
    return sampled_points
def contour_of_segment(segdf):
    """
    :param segdf: numpy array, the mask of the segment
    :return:
    """
    countor = find_3D_contour(segdf)
    segdf = sample_points_from_contour(countor)
    return segdf

def load_single_contour_Mask(mask):
    """
    Returns a sample of points from the contour of the segments,
    loads the entire segment file and filters the segment, to load for a set of frames use the load batch instead.
    :param frame: Integer, the time
    """
    frame_segment = mask
    segments_in_frame = np.unique(frame_segment)
    points = []
    for seg in segments_in_frame:
        if True:
            segdf = (np.array(frame_segment) == seg)#MB added np.array
            segdf = contour_of_segment(segdf.astype(int))
            if len(points) == 0:
                points = segdf
            else:
                points = np.append(points, segdf, axis=0)
    return points

def pack(h5,identifier,i,grid,num_classes,thres=4,weight=False):
    fr=np.array(h5[str(i)+"/frame"])[0]
    predmask=np.array(h5[identifier][str(i)+"/predmask"])
    if str(i)+"/mask" in h5.keys():
        mask=np.array(h5[str(i)+"/mask"])
    else:
        mask=None
    return [fr,predmask,mask,grid,num_classes,thres,weight]

def get_mask(im,pts,num_classes,grid,thres=4,distthres=4):
    valid=(np.isnan(pts[:,0])!=1)
    pts=pts[valid]
    if len(pts)==0:
        return np.zeros(im.shape).astype(np.int16)
    labs=np.arange(num_classes)[valid]
    tree=spat.cKDTree(pts)
    sh=im.shape
    ds,iis=tree.query(grid,k=1)
    mask=labs[iis].reshape(sh)
    return mask*(im>thres)*(ds.reshape(sh)<distthres)

def get_distmat(mode="multipoles_peak",**kwargs):
    assert False,"not implemented"
    if mode=="multipoles_peak":
        assert all([el in kwargs for el in ["thres,min_distance"]])
        skfeat.peak_local_max()

class RGNTrainDataset(Dataset):
    def __init__(self,dirname,shape,n,select_channel=0,eps=0.00001,augvec=np.array([1,1,10]),augmag=np.pi/2):
        super().__init__()
        self.dirname=dirname
        self.shape=shape
        self.n=n
        self.select_channel=select_channel
        self.eps=eps
        self.indlist=[int(name.strip().split("/")[-1].split("_")[-1].split(".")[0]) for name in glob.glob(self.dirname+"/masks/*")]
        self.indlist=np.array(self.indlist)
        self.grid=grid=np.array(np.meshgrid(*[np.linspace(-1,1,s) for s in self.shape[1:]],indexing="ij")).reshape(3,-1)
        self.num_frames_tot=len(self.indlist)

        self.augment="full"#"none"
        self.augvec=augvec
        self.augmag=augmag

    def get_trf(self,pts):
        if self.augment=="none":
            return pts
        elif self.augment=="full":
            trans=0.2*(2*np.random.random(3)-1)
            rotvec=self.augvec*(2*np.random.random(3)-1)
            norm=np.sqrt(np.sum(np.square(rotvec)))
            if norm==0:
                rotvec=(np.random.random()>0.5)*np.array([0,0,1])
            else:
                rotvec/=norm
            mag=self.augmag*np.random.random()
            rot=spat.transform.Rotation.from_rotvec(mag*rotvec)
            return rot.apply(pts.T).T+trans[:,None]
        else:
            assert False,"wrong trf(augmentation) mode"

    def __getitem__(self,i):
        assert 0<=i<self.num_frames_tot
        ii=self.indlist[i]
        fr_flat=np.load(self.dirname+"/frames/frame_"+str(ii)+".npy")[:self.shape[0]].reshape(self.shape[0],-1)/255
        mask_flat=np.load(self.dirname+"/masks/mask_"+str(ii)+".npy").flatten()
        pts_ind=self.point_ind_sample_single(fr_flat[self.select_channel])
        pts=self.get_trf(self.grid[:,pts_ind])
        return torch.tensor(np.concatenate([pts,fr_flat[:,pts_ind]],axis=0)),torch.tensor(mask_flat[pts_ind])

    def point_ind_sample_single(self,fr_flat):
        fr_flat+=self.eps
        posterior=(fr_flat/np.sum(fr_flat))
        return np.random.choice(self.grid.shape[1],self.n,replace=False,p=posterior)

    def __len__(self):
        return self.num_frames_tot

    def change_augment(self,augment):
        if augment in ["full","none"]:
            self.augment=augment
            return self.augment
        return "Fail to change still: "+self.augment

    def real_ind_to_dset_ind(self,real_inds):
        query={}
        for i,ind in enumerate(self.indlist):
            query[ind]=i
        dset_inds=[]
        for ind in real_inds:
            dset_inds.append(query[ind])
        return dset_inds

class RGNEvalDataset(Dataset):
    def __init__(self,dirname,shape,n,select_channel=0,eps=0.00001,eval_rep=4):
        super().__init__()
        self.dirname=dirname
        self.shape=shape
        self.n=n
        self.select_channel=select_channel
        self.eps=eps
        self.eval_rep=eval_rep
        self.grid=grid=np.array(np.meshgrid(*[np.linspace(-1,1,s) for s in self.shape[1:]],indexing="ij")).reshape(3,-1)

    def __getitem__(self,ii):
        fr_flat=np.load(self.dirname+"/frames/frame_"+str(ii)+".npy")[:self.shape[0]].reshape(self.shape[0],-1)/255
        pts_inds=[self.point_ind_sample_single(fr_flat[self.select_channel]) for _ in range(self.eval_rep)]
        pts=[self.grid[:,pts_inds[i]] for i in range(self.eval_rep)]
        try:
            mask_flat=np.load(self.dirname+"/masks/mask_"+str(ii)+".npy").flatten()
            labs=torch.tensor(np.array([mask_flat[pts_inds[i]] for i in range(self.eval_rep)]))
        except FileNotFoundError:
            labs=None
        return torch.tensor(np.concatenate([np.array(pts),np.array([fr_flat[:,pts_inds[i]] for i in range(self.eval_rep)])],axis=1)),labs


    def point_ind_sample_single(self,fr_flat):
        fr_flat+=self.eps
        posterior=(fr_flat/np.sum(fr_flat))
        return np.random.choice(self.grid.shape[1],self.n,replace=False,p=posterior)

def select_additional(T,traininds,distmat,num_additional):
    traininds_aug=np.array(traininds).copy()
    for i in range(num_additional):
        not_tr=np.ones(T,dtype=bool)
        not_tr[traininds_aug]=0
        not_traininds=np.nonzero(not_tr)[0]
        subdist=distmat[traininds_aug]
        subdist=subdist[:,not_tr]
        dists=np.min(subdist,axis=0)
        new=not_traininds[np.argmax(dists)]
        traininds_aug=np.append(traininds_aug,new)
    return traininds_aug

def get_deformation(ptfrom,ptto,sh,k_cut_dimless=2.5,lr=0.1,print_plot=False,iterations=200,lambda_div=1,scale=(1,1,1),at_least=4,device="cpu"):
    vecs=(ptto-ptfrom)
    valids=np.nonzero(np.all(np.isnan(vecs)==0,axis=1))[0]
    if len(valids)<at_least:
        return None,None,None,"Not enough points"
    vecs=vecs[valids][:,:]
    locs=ptfrom[valids][:,:]
    W,H,D=sh


    f=FourierAugment.Fourier((W,H,D),k_cut_dimless=k_cut_dimless)

    locs_gridded=2*(torch.tensor(locs)[:,None,None,None,:]/(np.array([W,H,D])[None,None,None,None,:]-1))-1
    locs_gridded=locs_gridded[...,[2,1,0]].to(device=device,dtype=torch.float32)
    vecs_target=torch.tensor(vecs).to(device=device,dtype=torch.float32)

    x=torch.zeros(1,3,f.dim)
    x[0,:,0]=torch.tensor(np.mean(vecs,axis=0))*np.prod([W,H,D])
    x=x.to(device=device)
    x=torch.nn.Parameter(x)
    opt=torch.optim.Adam([x],lr=W*H*D*lr)
    losses=[]
    for iters in range(iterations):
        deformation=f(x)
        deformation=torch.mean(deformation,dim=4,keepdim=True).repeat(1,1,1,1,deformation.size(4))
        vecs_sampled=torch.nn.functional.grid_sample(deformation.repeat(locs_gridded.size(0),1,1,1,1),locs_gridded, mode='bilinear', padding_mode='border',align_corners=True)[:,:,0,0,0]
        loss=torch.nn.functional.l1_loss(vecs_sampled,vecs_target)
        gx=deformation[:,0,2:,1:-1,1:-1]-deformation[:,0,:-2,1:-1,1:-1]
        gy=deformation[:,1,1:-1,2:,1:-1]-deformation[:,1,1:-1,:-2,1:-1]
        gz=deformation[:,2,1:-1,1:-1,2:]-deformation[:,2,1:-1,1:-1,:-2]
        divergence=scale[0]*gx+scale[1]*gy+scale[2]*gz
        loss+=lambda_div*torch.mean(torch.abs(divergence))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if print_plot:
            print(loss.item())
    if print_plot:
        plt.plot(losses)
        plt.yscale("log")
    return deformation,f,x,0

def deform(sh,deformation,fr,mask=None):
    W,H,D=sh
    grid=torch.stack(torch.meshgrid(*[torch.arange(s) for s in (W,H,D)])).to(dtype=torch.float32,device=deformation.device)
    normten=(torch.tensor([W,H,D])[None,:,None,None,None]-1).to(dtype=torch.float32,device=deformation.device)
    moved=(2*((grid[None]-deformation)/normten)-1).to(dtype=torch.float32,device=deformation.device)
    moved=moved.permute(0,2,3,4,1)[...,[2,1,0]]
    fr_aug=torch.nn.functional.grid_sample(fr.to(torch.float32),moved.repeat(fr.size(0),1,1,1,1), mode='bilinear', padding_mode="border",align_corners=True)
    if mask is not None:
        mask_aug=torch.nn.functional.grid_sample(mask.unsqueeze(1).to(torch.float32),moved.repeat(mask.size(0),1,1,1,1), mode='nearest', padding_mode='zeros',align_corners=True)[:,0].to(torch.long)
        return fr_aug,mask_aug
    return fr_aug

def train(h5,identifier,device,net,optimizer,criterion,get_ious,scheduler,allset,traindataloader,num_epochs,\
aug_dict,log,verbose,losses,iouss,inds,print_iou_every,digits,\
num_trains,vnum=0,valdataloader=None,digits_v=None,num_vals=None,write_log=False,logfn=None):
    gc=0
    for epoch in range(num_epochs):
        if epoch in aug_dict.keys():
            text="augment is now:"+allset.change_augment(aug_dict[epoch])
            log+=text
            if verbose:
                print(text)
        log+="Epoch: "+str(epoch)+" lr: "+str(optimizer.param_groups[0]['lr'])+"\n"
        if verbose:
            print("Epoch: "+str(epoch)+" lr: "+str(optimizer.param_groups[0]['lr']))
        net.train()
        eploss=0
        count=0
        for i,(fr,mask) in enumerate(traindataloader):
            fr = fr.to(device=device, dtype=torch.float32)
            mask= mask.to(device=device, dtype=torch.long)
            preds=net(fr)
            loss=criterion(preds,mask,fr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """
            plt.subplot(131)
            plt.imshow(np.max(fr[0,0].cpu().detach().numpy(),axis=2).T)
            plt.colorbar()
            plt.subplot(132)
            plt.imshow(np.max(fr[0,1].cpu().detach().numpy(),axis=2).T)
            plt.subplot(133)
            plt.imshow(np.max(mask[0].cpu().detach().numpy(),axis=2).T,cmap="nipy_spectral",interpolation="none")
            plt.show()
            """

            eploss+=loss.item()
            count+=1
            losses.append(loss.item())
            ious=get_ious(preds,mask,((gc%print_iou_every)!=0))
            iouss.append(ious)
            inds.append(0)

            txt="    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item())+" nanmeaniou: "+str(np.nanmean(ious) if np.sum(np.isnan(ious)!=1)>0 else np.nan )
            if verbose:
                print(txt)
            log+=(txt+"\n")
            gc+=1
        eploss=eploss/count
        if allset.augment!="full":
            if scheduler is not None:
                scheduler.step(eploss)#step scheduler by epoch loss
        log+="Epoch Loss: "+str(eploss)+"\n"+"\n"

        if vnum>0:
            net.eval()
            log+="Validation:"+"\n"
            eploss=0
            count=0
            for i,(fr,mask) in enumerate(valdataloader):
                fr = fr.to(device=device, dtype=torch.float32)
                mask= mask.to(device=device, dtype=torch.long)
                with torch.no_grad():
                    preds=net(fr)
                    loss=criterion(preds,mask,fr)
                losses.append(loss.item())
                eploss+=loss.item()
                count+=1

                ious=get_ious(preds,mask,False)
                iouss.append(ious)

                inds.append(1)

                txt="    val"+str(i+1).zfill(digits_v)+"/"+str(num_vals)+" loss: "+str(loss.item())+" nanmeaniou: "+str(np.nanmean(ious) if np.sum(np.isnan(ious)!=1)>0 else np.nan )
                if verbose:
                    print(txt)
                log+=(txt+"\n")
            eploss=eploss/count
            log+="Mean Validation Loss: "+str(eploss)+"\n"+"\n"

        #save net in h5
        if "net" in h5[identifier].keys():
            del h5[identifier]["net"]
        h5.create_group(identifier+"/net")
        save_into_h5(h5[identifier+"/net"],net.state_dict())

        #save loss and iou
        if "loss_iou" in h5[identifier].keys():
            del h5[identifier]["loss_iou"]
        dset=h5[identifier].create_dataset("loss_iou",(len(losses),2+len(ious)),dtype="f4",compression="gzip")
        dset[...]=np.concatenate((np.array(inds)[:,None],np.array(losses)[:,None],np.array(iouss)),axis=1).astype(np.float32)

        log+="Results saved."+"\n"+"\n"
        h5[identifier].attrs["log"]=log
        if write_log:
            logform="Prepare={:04.4f} Train={:04.4f} Predict={:04.4f} GetPoints={:04.4f}"
            with open(logfn,"w") as f:
                f.write(logform.format(1.,epoch/num_epochs,0.,0.)+"\n")

        #CRITICAL, emergency break
        if os.path.exists("STOP"):
            break


def to_np(ten):
    return ten.cpu().detach().numpy()
def standardize(vecs):
    m=np.mean(vecs,axis=0)
    s=np.std(vecs,axis=0)
    return (vecs-m)/(s+1e-8)

def Compute_distmat(h5,T,W,H,batch_size=20,n_z=31,n_channels=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class DS(torch.utils.data.Dataset):
        def __init__(self):
            super(DS,self).__init__()
        def __getitem__(self,i):
            return torch.tensor(np.max(np.array(h5[str(i)+"/frame"]),axis=3)/255)
        def __len__(self):
            return T
    ds=DS()
    dl=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True)
    net=conv_autoenc1.Net(n_channels=n_channels,n_z=n_z,sh_2d=(W,H))
    net.to(device=device)
    None
    num_epochs=40
    opt=torch.optim.Adam(net.parameters())
    losses=[]
    for epoch in range(num_epochs):
        print("\r Epoch "+str(epoch+1)+"/"+str(num_epochs),end="")
        for i,ims in enumerate(dl):
            ims=ims.to(device=device,dtype=torch.float32)
            res,latent=net(ims)
            loss=nn.functional.mse_loss(res,ims)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    net.eval()
    vecs=[]
    with torch.no_grad():
        for i in range(T):
            if (i+1)%100==0:
                print("\r"+str(i)+"/"+str(T),end="")
            _,latent=net(ds[i].unsqueeze(0).to(device=device,dtype=torch.float32))
            vecs.append(to_np(latent[0]))
    vecs=np.array(vecs)
    key="vecs"
    if key in h5.keys():
        del h5[key]
    ds=h5.create_dataset(key,shape=(T,n_z),dtype="f4")
    ds[...]=vecs.astype(np.float32)
    vecs=standardize(vecs)
    u_map=umap.UMAP(n_components=2)
    res=u_map.fit_transform(vecs)
    distmat=spat.distance_matrix(res,res)
    key="distmat"
    if key in h5.keys():
        del h5[key]
    ds=h5.create_dataset(key,shape=(T,T),dtype="f4")
    ds[...]=distmat.astype(np.float32)
    print("distmat created")
    return distmat.astype(np.float32)
