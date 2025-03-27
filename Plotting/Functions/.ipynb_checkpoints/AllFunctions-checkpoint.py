from sklearn.cluster import KMeans
import numpy as np
import h5py
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib import style
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
#from dtaidistance import dtw
#from dtaidistance.dtw import constrained
from pytwed import twed

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import umap
#import umap.umap_ as umap
import pandas as pd
import seaborn as sns
import copy
from scipy.stats import ranksums

from scipy.stats import ks_2samp
import warnings
from scipy.signal import find_peaks
import scipy
# Suppress all warnings
#warnings.filterwarnings("ignore")



def bin_and_plot(X,Y,bins=10,label='',plot=False):
    #bins X and plots average of Y in each bin
    # Creating a DataFrame
    data = {
    'time': X,
    'activity': Y,
        }

    df = pd.DataFrame(data)
    df['vel_Bin'] = pd.cut(df['time'], bins=bins)

    # Calculate the mean of 'activity' in each bin
    mean_activity_by_bin = df.groupby('vel_Bin')['activity'].mean()
    mean_time_by_bin = df.groupby('vel_Bin')['time'].mean()

    # Extract bin edges and calculate bin width
    bin_edges = pd.cut(df['time'], bins=bins).unique()
    bin_width = bin_edges[0].right - bin_edges[0].left
    if plot:
        # Make the plot larger
        plt.figure(figsize=(12, 8))
        # Plotting the mean activity in each bin
        plt.bar(mean_time_by_bin, mean_activity_by_bin, width = bin_width, align='center')
        plt.title(label)
        plt.xlabel('time Bins')
        plt.ylabel('Mean Activity')
        plt.xticks(rotation=45, ha='right')
        plt.show()
    return df,mean_time_by_bin, mean_activity_by_bin,bin_width

def estimate_y(X,theta,y):
    #estimates y from the linear regressions coefficients
    yhat = X @ theta
    error = (np.linalg.norm(y-yhat))**2
    return yhat,error


def ordinary_least_squares(X, y):
    #y=X*\theta+\epsilon
    #performs linear regression and returns the coefficients
    temp1 = (X.T) @ y
    temps2 = (X.T) @ X
    temp3 = np.linalg.inv(temps2)
    theta_hat =  temp3 @ temp1
    return theta_hat

def find_stretches(valid_indices):
    stretches = []
    start = None
    for i, valid in enumerate(valid_indices):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            stretches.append((start, i))
            start = None
    if start is not None:
        stretches.append((start, len(valid_indices)))
    return stretches


def plot_crosscorr_for_diff_stretches(x,y,stretches,threshold=20,odor_bins=0):
    print(len(stretches))
    for (start, end) in stretches:
        print("stretches length:"+str(end-start))
        if (end-start)>threshold:
            print(start)
            x_valid = x[start:end]
            y_valid = y[start:end]
            cross_correlation = norm_cross_corr(x_valid, y_valid, mode='full')
            lags = np.arange(-len(x_valid) + 1, len(x_valid))
            # Plotting the cross-correlation
            plt.figure(figsize=(10, 6))
            plt.plot(lags, cross_correlation)
            plt.xlabel('Lag')
            if np.sum(odor_bins):
                plt.ylabel(np.average(odor_bins[start:end]))
            else:
                plt.ylabel("cross correlation")
            plt.title(f'Cross-corr,segment(Stretch {start} to {end-1})')
            plt.grid()
            plt.show()

def averaging_crosscorr_diff_stretches(x,y,stretches,threshold=20):
    count = 0
    collected_matrices = []
    collected_weights = []
    for (start, end) in stretches:
        leng = end-start
        delta = leng-threshold
        print("stretches length:" +str(leng))
        if leng > threshold:
            print('accepted')
            count=count+1
            x_valid = x[start:end]
            y_valid = y[start:end]
            cross_correlation = norm_cross_corr(x_valid, y_valid, mode='full')
            st_ind = delta
            end_ind = delta+2*threshold
            collected_matrices.append(cross_correlation[st_ind:end_ind])
            collected_weights.append(leng)
    # Convert the list of matrices to a numpy array
    total_matrix = np.vstack(collected_matrices)
    total_weight = np.vstack(collected_weights)
    if True:#total_matrix.shape[0] == count*threshold*2:
        final_matrix = total_matrix.reshape(count, threshold*2)
        print("Final matrix shape:", final_matrix.shape)
    else:
        print("Error: The total number of rows is not correct")
    return final_matrix,total_weight

def closest_index(A, b):
    #returns the index of vector A that has the closest value to b
    # Calculate the absolute differences between b and each element in A
    diff = np.abs(A - b)

    # Find the index of the element in A with the smallest difference
    idx = np.argmin(diff)

    # Return the index
    return idx

def compute_vector_z_score(vector,exclude_zero=True):

    if not exclude_zero:
        mean = np.mean(vector)
        std=np.std(vector)
        vector_zscore= (vector-mean)/std
    else:
        vector_zscore = np.zeros(np.shape(vector))
        mean = np.mean(vector[np.nonzero(vector)[0]])
        std=np.std(vector[np.nonzero(vector)[0]])
        vector_zscore[np.nonzero(vector)[0]] = (vector[np.nonzero(vector)[0]]-mean)/std
    return vector_zscore


def compute_r_squared(y, yhat):
    # Mean of actual values
    y_mean = np.mean(y)
    # Total sum of squares (proportional to the variance of the data)
    ss_total = np.sum((y - y_mean) ** 2)
    # Residual sum of squares
    ss_residual = np.sum((y - yhat) ** 2)
    # R-squared
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def get_nonzero_start_end(turn_timepoints):
    start_indices = []
    end_indices = []
    if not turn_timepoints[0]==0:
        start_indices.append(0)
    for i in range(len(turn_timepoints) - 1):
        if turn_timepoints[i] == 0 and not(turn_timepoints[i+1] == 0):
            start_indices.append(i+1)
        if not(turn_timepoints[i] == 0) and (turn_timepoints[i+1] == 0):
            end_indices.append(i)
    if not turn_timepoints[-1]==0:
        end_indices.append(len(turn_timepoints)-1)
    assert len(end_indices)==len(start_indices)
    return start_indices,end_indices

def compute_2rows(row1, row2):
    row1[np.isnan(row1)] = 0
    row2[np.isnan(row2)] = 0
    time_intersection = (np.abs(row1)>0)&(np.abs(row2)>0)
    if (np.sum(time_intersection))>0:
        correlation = np.corrcoef(row1[time_intersection], (row2[time_intersection]))[0, 1]
        correlation2 = scipy.stats.pearsonr(compute_vector_z_score(row1[time_intersection]), (row2[time_intersection]))[0]
    else:
        correlation = 0
        correlation2 = 0
        print("no simultaneous data")
    return correlation,correlation2

def get_nonzero_start_end_long(turn_timepoints,thresh=5):
    start_indices = []
    end_indices = []
    if not turn_timepoints[0]==0:
        start_indices.append(0)
    for i in range(len(turn_timepoints) - 1):
        if turn_timepoints[i] == 0 and not(turn_timepoints[i+1] == 0):
            start_indices.append(i+1)
        if not(turn_timepoints[i] == 0) and (turn_timepoints[i+1] == 0):
            end_indices.append(i)
    if not turn_timepoints[-1]==0:
        end_indices.append(len(turn_timepoints)-1)
    assert len(end_indices)==len(start_indices)
    delInde = []
    delInds = []
    if len(start_indices)>0:
        for i in range(len(start_indices)-1):
            if (start_indices[i+1]-end_indices[i]) < thresh:
                delInde.append(i)
                delInds.append(i+1)
        start_indices = np.delete(start_indices, delInds)
        end_indices = np.delete(end_indices, delInde)
    return start_indices,end_indices

def get_omega_start_end(turn_timepoints):
    start_indices = []
    end_indices = []
    for i in range(len(turn_timepoints) - 1):
        if turn_timepoints[i][1] == 'Omegi' and turn_timepoints[i+1][1] == 'Omegf':
            start_indices.append(turn_timepoints[i][0])
            end_indices.append(turn_timepoints[i+1][0])
        if turn_timepoints[i][1] == 'SOmegi' and turn_timepoints[i+1][1] == 'SOmegf':
            start_indices.append(turn_timepoints[i][0])
            end_indices.append(turn_timepoints[i+1][0])
    assert len(end_indices)==len(start_indices)
    return start_indices,end_indices


def interpolate_nanValues(x,y):
    x_interp = np.interp(np.arange(len(x)), np.arange(len(x))[~np.isnan(x)], x[~np.isnan(x)])
    y_interp = np.interp(np.arange(len(y)), np.arange(len(y))[~np.isnan(y)], y[~np.isnan(y)])
    return x_interp, y_interp

def plot1d_withAnnotations_scatter(timevec,Valvec,All_ind,h=200,plotFB=False,yloc=0,draw_lines=False,
                                   multi=False,multivec=[0],multivec_pos=[1,1,1,1,1,1,1],timevec_orig=[0]):
    #All_ind:list of labels
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.scatter(timevec,Valvec)
    color=['r','g','cyan','brown']
    if multi:
        multishape = np.shape(multivec)
        if len(multishape)>1:
            multilen = multishape[0]
            for m in range(multilen):
                if multishape[1]==len(timevec):
                    ax.scatter(timevec,multivec[m,:]+multivec_pos[m],color=color[m])
                else:
                    ax.scatter(np.arange(multishape[1]),multivec[m,:]+multivec_pos[m],color=color[m])
        else:
            if multishape[0]==len(timevec):
                ax.scatter(timevec,multivec+multivec_pos[0],color=color[0])
            else:
                ax.scatter(timevec_orig,multivec+multivec_pos[0],color=color[0])
    velo = Valvec
    for i in range(np.shape(All_ind)[0]):
        if All_ind[i][1]=='B':
            ax.text(All_ind[i][0],yloc+0.5,All_ind[i][1],rotation='vertical',fontsize=10,color='red')#'xx-small')
            if draw_lines:
                ax.axvline(x=All_ind[i][0],c ='red',alpha=0.4,lw=1)#,yloc,All_ind[i][1],rotation='vertical',fontsize=10)#'xx-small')
        elif All_ind[i][1]=='F':
            ax.text(All_ind[i][0],yloc-0.5,All_ind[i][1],rotation='vertical',fontsize=10,color='green')#'xx-small')
            if draw_lines:
                ax.axvline(x=All_ind[i][0],c ='g',alpha=0.4,lw=1)
        else:
            ax.text(All_ind[i][0],yloc,All_ind[i][1],rotation='vertical',fontsize=10)#'xx-small')
    plt.show()

def plotBar_withAnnotations(timevec,Valvec,widthVec,All_ind,h=0.5,plotFB=False,yloc=0,draw_lines=False,
                                   timevec_orig=[0],allindex=0):
    #All_ind:list of labels
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.bar(timevec,Valvec, width=widthVec, align='center')
    color=['r','g','cyan','brown']
    velo = Valvec
    for i in range(np.shape(All_ind)[0]):
        if All_ind[i][1]=='B':
            ax.text(All_ind[i][0],yloc-h,All_ind[i][1],rotation='horizontal',fontsize=10,color='red')#'xx-small')
            if draw_lines and i<np.shape(All_ind)[0]-1:
                if All_ind[i+1][1]=='B':
                    ax.axvspan(All_ind[i][0],All_ind[i+1][0],color ='red',alpha=0.4,lw=1)#,yloc,All_ind[i][1],rotation='vertical',fontsize=10)#'xx-small')
        elif All_ind[i][1]=='F':
            ax.text(All_ind[i][0],yloc+h,All_ind[i][1],rotation='horizontal',fontsize=10,color='green')#'xx-small')
            if draw_lines and i<np.shape(All_ind)[0]-1:
                if All_ind[i+1][1]=='F':
                    ax.axvspan(All_ind[i][0],All_ind[i+1][0],color ='g',alpha=0.4,lw=1)#
        elif allindex>0:
            ax.text(All_ind[i][0],yloc,All_ind[i][1],rotation='vertical',fontsize=10)#'xx-small')
    plt.show()

def plot2BarsOverlap(timevec,Valvec1,Valvec2,widthVec,ylab='bars',xlab='time',title = 'bars',leg = False):
    #All_ind:list of labels
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.bar(timevec,Valvec1, width=widthVec, align='center')
    ax.bar(timevec,Valvec2,alpha=0.6, width=widthVec, align='center')
    if leg:
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        ax.set_title(titre)
    plt.show()

def plot2BarsSingle(timevec,Valvec1,widthVec,ylab='bars',xlab='time',title = 'bars',leg = False):
    #All_ind:list of labels
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.bar(timevec,Valvec1, width=widthVec, align='center')
    if leg:
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        ax.set_title(titre)
    plt.show()

def plot1d_withAnnotations(timevec,Valvec,All_ind,h=0.5,plotFB=False,yloc=0,draw_lines=False):
    #draw lines: whether to draw red or green lines on forward or backward states
    #yloc:location of B and F labels
    #plotFB: whether to   plot forward and backward labels
    #All_ind:list of labels
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.plot(timevec,Valvec)
    velo = Valvec
    if plotFB:
        forward_times = np.nonzero(velo>1)
        forward = np.ones(len(forward_times[0]))*h
        backward_times = np.nonzero(velo<-1)
        backward = np.ones(len(backward_times[0]))*(-h)
        length = len(velo)
        ax.set_xlabel('t')
        ax.set_ylabel('v')
        # Create a list of colors for each segment of the line
        colors = ['black' if velo[i] < 0 else 'red' for i in range(length)]
        #ax.hlines(20, 0, length, colors=colors, linewidth=5)
        ax.scatter(forward_times[0],forward,c='red',s=2)
        ax.scatter(backward_times[0],backward,c= 'green',s=2)

    for i in range(np.shape(All_ind)[0]):
        if All_ind[i][1]=='B':
            ax.text(All_ind[i][0],yloc+h,All_ind[i][1],rotation='vertical',fontsize=10,color='red')#'xx-small')
            if draw_lines and i<np.shape(All_ind)[0]-1:
                ax.axvline(x=All_ind[i][0],c ='r',alpha=0.4,lw=1)
        elif All_ind[i][1]=='F':
            ax.text(All_ind[i][0],yloc+h,All_ind[i][1],rotation='vertical',fontsize=10,color='green')#'xx-small')
            if draw_lines and i<np.shape(All_ind)[0]-1:
                ax.axvline(x=All_ind[i][0],c ='g',alpha=0.4,lw=1)
        else:
            ax.text(All_ind[i][0],yloc,All_ind[i][1],rotation='vertical',fontsize=10)#'xx-small')

    plt.show()

def plot_2Bars(y_train2,y_pred,time,Rsq,ylab,msqE='msqE'):
    plt.figure(figsize=(10, 3))
    plt.scatter(time,y_train2, label='Actual', marker='o')
    plt.scatter(time,y_pred, label='Predicted', marker='x')
    print(np.mean(y_train2))
    #plt.axhline(y_train2.mean(), color='orange', linestyle='dashed', linewidth=2)
    plt.xlabel(msqE)
    plt.ylabel(ylab)
    plt.title('Rsq: '+str(Rsq))
    plt.legend()
    plt.show()

def plot_2BarsBars(y_train20,y_pred0,time,Rsq,ylab,msqE='msqE',w=5):
    plt.figure(figsize=(10, 3))
    y_pred = [float(y) for y in y_pred0]
    y_train2 = [float(y) for y in y_train20]
    plt.bar(np.array(time),np.array(y_train2),width=w)#, label='Actual')
    plt.bar(np.array(time),np.array(y_pred), alpha=0.6,width=w)
    print(np.mean(y_train2))

    plt.xlabel(msqE)
    plt.ylabel(ylab)
    plt.title('Rsq: '+str(Rsq))
    plt.legend()
    plt.show()

def plot1d_withOdor(timevec,Valvec,Odorvec,title):
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.scatter(timevec,Valvec)
    velo = Valvec
    for t in np.nonzero(Odorvec==1)[0]:
        ax.axvline(x=timevec[t],c ='g',alpha=0.05,lw=1)#axvspan
    for t in np.nonzero(Odorvec==2)[0]:
        ax.axvline(x=timevec[t],c ='r',alpha=0.05,lw=1)#axvspan
    ax.set_title(title)
    plt.show()
    return fig

def plot_corr_bin_heat(corr_neuronsRI_ind_Vel1,titre='Vel',leg=False,save=False):
    Wormnum = corr_neuronsRI_ind_Vel1.shape[1]
    Binnum = corr_neuronsRI_ind_Vel1.shape[0]
    fig, axs = plt.subplots(Wormnum,1, figsize=(12, 10))
    for d in range(Wormnum):
        matrix = corr_neuronsRI_ind_Vel1[:,d,:]
        cax1 = axs[d].imshow(matrix, cmap='viridis', aspect='auto')
        fig.colorbar(cax1, ax=axs[d])
        title0 = 'worm: '+ str(d)
        title = titre + title0
        if leg:
            axs[d].set_title(title)
            axs[d].set_ylabel('bin number')
            axs[d].set_xlabel('neuron')

    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(title +'.png')
    return fig

ID0 = ['RIML','RIMR','RIBL','RIBR','interc','R^2']

def plot_fewBars(Rsqrt0,ID=ID0,save=False,saveName = 'fewBars',titre='bars'):

    if len(Rsqrt0.shape)>1:
        Rsqrt = Rsqrt0[0,:]
    else:
        Rsqrt = Rsqrt0
    Num = len(Rsqrt)
    fig = plt.figure(figsize=(Num+3 ,5))
    ax = fig.add_subplot(111)


    ax.set_title(titre)
    if len(Rsqrt0.shape)>1:
        ax.bar(ID, Rsqrt, yerr=Rsqrt0[1,:], capsize=5, color='skyblue')
    else:
        ax.bar(ID, Rsqrt, color='skyblue')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
    ax.set_xticklabels(ID, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    saveName0 = 'plots/'+saveName
    if save:
        fig.savefig(saveName0 + '.png')
    plt.tight_layout()
    plt.show()
    return fig

def plot_Rsqrd_alltrials(Rsqrt,ytick=False,ytick_loc=[0,1,2,3,4],ytickLab = [1,2,3,4,5]):
    WrmNum = Rsqrt.shape[0]
    TrialNum = Rsqrt.shape[2]
    fig, axs = plt.subplots(WrmNum,1, figsize=(int(TrialNum/2), WrmNum*2+2))
    for d in range(WrmNum):
        matrix = Rsqrt[d,:,:]
        cax1 = axs[d].imshow(matrix, cmap='viridis', aspect='auto')
        divider = make_axes_locatable(axs[d])
        # Add a new axis for the colorbar to the right with specified width
        cax = divider.append_axes("right", size="5%",pad=0.1)
        fig.colorbar(cax1, cax=cax)# ax=axs[d])
        title = 'R^2-W:  '+ str(d)
        axs[d].set_title(title)
        if ytick:
            axs[d].set_yticks(ytick_loc)
            axs[d].set_yticklabels(ytickLab)
        else:
            axs[d].yaxis.set_visible(False)

        axs[d].set_xlabel('trials')
        axs[d].set_ylabel('worms')

    plt.tight_layout()
    plt.show()
    return fig

ID0 = ['RIML','RIMR','RIBL','RIBR','interc','R^2']

def plot_Rsqrd_Coeff_distr(Rsqrt,coeffRIBRIM,ID=ID0,save=False,saveName = 'RsqrdAndCoef'):
    #distribution of coeff. in the linear regression

    WrmNum = Rsqrt.shape[0]
    TrialNum = Rsqrt.shape[2]
    fig, axes = plt.subplots(WrmNum, 1, figsize=(10, WrmNum*4))
    # Iterate over the 5x5 subplots
    for d in range(WrmNum):
        # Extract the 10 data points for the current subplot
        rsq = np.mean(np.delete(Rsqrt[d,:,:],d,axis=0),axis=0)
        rsq = rsq.reshape(TrialNum,1)
        data_points = np.concatenate((coeffRIBRIM[d, :, :], rsq), axis=1)
        means = np.mean(data_points, axis=0)
        std_devs = np.std(data_points, axis=0)
        axes[d].set_title('Worm '+ str(d))
        axes[d].bar(ID, means, yerr=std_devs, capsize=5, color='skyblue')
        axes[d].axhline(0, color='black', linewidth=1)
        #axes[d-1].set_ylabel('mean coefficient')
        axes[d].grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
        axes[d].set_xticklabels(ID, fontsize=20)
        axes[d].tick_params(axis='y', labelsize=20)


    if save:
        fig.savefig(saveName + '.png')
    plt.tight_layout()
    plt.show()
    return fig

def plot_Rsqrd_Coeff_Max(Rsqrt,coeffRIBRIM,ID=ID0,save=False,saveName = 'RsqrdAndCoef'):
    #distribution of coeff. in the linear regression

    WrmNum = Rsqrt.shape[0]
    TrialNum = Rsqrt.shape[2]
    fig, axes = plt.subplots(WrmNum, 1, figsize=(10, WrmNum*4))
    # Iterate over the 5x5 subplots
    for d in range(WrmNum):
        # Extract the 10 data points for the current subplot
        rsq = np.mean(np.delete(Rsqrt[d,:,:],d,axis=0),axis=0)
        rsq = rsq.reshape(TrialNum,1)
        indmax = np.argmax(rsq)
        data_points = np.concatenate((coeffRIBRIM[d, indmax, :], rsq[indmax]))

        axes[d].set_title('Worm '+ str(d))
        axes[d].bar(ID, data_points, color='skyblue')
        axes[d].axhline(0, color='black', linewidth=1)
        #axes[d-1].set_ylabel('mean coefficient')
        axes[d].grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
        axes[d].set_xticklabels(ID, fontsize=20)
        axes[d].tick_params(axis='y', labelsize=20)


    if save:
        fig.savefig(saveName + '.png')
    plt.tight_layout()
    plt.show()
    return fig

def norm_cross_corr(signal1,signal2,mode='full'):
    signal1n = (signal1-np.mean(signal1))/np.std(signal1)
    signal2n = (signal2-np.mean(signal2))/np.std(signal2)
    norm_cross_corr = np.correlate(signal1n, signal2n, mode='full')/len(signal1n)
    return norm_cross_corr

def compute_2rows(row1, row2):
    row1[np.isnan(row1)] = 0
    row2[np.isnan(row2)] = 0
    time_intersection = (np.abs(row1)>0)&(np.abs(row2)>0)
    correlation = np.corrcoef(row1[time_intersection], (row2[time_intersection]))[0, 1]
    correlation2 = scipy.stats.pearsonr(compute_vector_z_score(row1[time_intersection]), (row2[time_intersection]))[0]
    return correlation,correlation2

def Regress_with_coeff(X_line,all_means):
    coefficients = all_means[:4]
    intercept = all_means[4]
    y_line = np.zeros(np.shape(X_line)[0])
    if True:#for t in range(len(y_line)):
        y_line = np.dot(X_line,coefficients) + intercept
    return y_line

def smooth_vec(rowsArr_floatloc0,sig=1,inter=0):
    rowsArr_floatloc1 = np.copy(rowsArr_floatloc0)
    smoothed_arr=np.copy(rowsArr_floatloc0)
    starts,end = get_nonzero_start_end(rowsArr_floatloc1)
    for t in range(len(starts)):
        if end[t]-starts[t]>3:
            smoothed_arr[starts[t]:end[t]] = gaussian_filter1d((rowsArr_floatloc1[starts[t]:end[t]]), sigma=sig)
        smoothed_arr_f = [np.float(x) for x in smoothed_arr]
    return smoothed_arr_f


def save_csv(activity_bins,Address_savebins,m,varname=''):
    df = pd.DataFrame(activity_bins)
    Address_savebins2 = Address_savebins + varname
    df.to_csv(Address_savebins2+str(m)+'.csv', index=False)


def smooth_vec(rowsArr_floatloc0,sig=2):
    rowsArr_floatloc1 = copy.deepcopy(rowsArr_floatloc0)
    smoothed_arr=copy.deepcopy(rowsArr_floatloc0)
    starts,end = get_nonzero_start_end(rowsArr_floatloc1)
    if len(starts)>0:
        for t in range(len(starts)):
            if end[t]-starts[t]>3:
                smoothed_arr[starts[t]:end[t]] = gaussian_filter1d(rowsArr_floatloc1[starts[t]:end[t]], sigma=sig)
        smoothed_arr_f = [np.float(x) for x in smoothed_arr]
        return smoothed_arr_f
    else:
        return smoothed_arr

def smooth_vec_long(rowsArr_floatloc0,sig=2,thresh=5):
    rowsArr_floatloc1 = copy.deepcopy(rowsArr_floatloc0)
    smoothed_arr=copy.deepcopy(rowsArr_floatloc0)
    starts,end = get_nonzero_start_end_long(rowsArr_floatloc1,thresh)
    if len(starts)>0:
        for t in range(len(starts)):
            if end[t]-starts[t]>3:
                if 0 in rowsArr_floatloc1[starts[t]:end[t]]:
                    Vec = rowsArr_floatloc1[starts[t]:end[t]]
                    nonz = np.nonzero(Vec)
                    Vec_sm = np.array(smooth_vec(Vec[nonz[0]],sig))
                    nonz_shifted = [int(starts[t]+k) for k in nonz[0]]
                    smoothed_arr[nonz_shifted] = Vec_sm
                else:
                    smoothed_arr[starts[t]:end[t]] = gaussian_filter1d(rowsArr_floatloc1[starts[t]:end[t]], sigma=sig)
        smoothed_arr_f = np.array([float(x) for x in smoothed_arr])
        return smoothed_arr_f
    else:
        return smoothed_arr
