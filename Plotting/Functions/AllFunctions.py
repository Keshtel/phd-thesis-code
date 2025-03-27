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

def angle_between_twovec(v1,v2):
    cosine = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if np.abs(cosine)>1:
        print(cosine)
        cosine=1
    theta = math.acos(cosine)*180/math.pi
    return theta

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
            cross_correlation = norm_cross_corr(x_valid, y_valid)
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

def binarize_with_threshold(Vector,threshPos=0,threshNeg=0):
    VectorOut = np.zeros(len(Vector))
    VectorOut[Vector<threshNeg] = -1
    VectorOut[Vector>threshPos] = 1
    return VectorOut


def closest_index(A, b):
    #returns the index of vector A that has the closest value to b
    # Calculate the absolute differences between b and each element in A
    diff = np.abs(A - b)

    # Find the index of the element in A with the smallest difference
    idx = np.argmin(diff)

    # Return the index
    return idx


def compute_2rows(row1, row2):
    #compute correlation of 2 signals at the intersection time
    row1[np.isnan(row1)] = 0
    row2[np.isnan(row2)] = 0
    time_intersection = (np.abs(row1)>0)&(np.abs(row2)>0)
    if (np.sum(time_intersection))>0:
        correlation = np.corrcoef(row1[time_intersection], (row2[time_intersection]))[0, 1]
        correlation2 = scipy.stats.pearsonr((row1[time_intersection]), compute_vector_z_score(row2[time_intersection]))[0]
    else:
        correlation = 0
        correlation2 = 0
        print("no simultaneous data")
    return correlation,correlation2


def compute_everymin_z_score(vector0,exclude_zero=True,minlength=60):
    sets = int(len(vector0)/minlength)
    vector_zscore = copy.deepcopy(vector0)
    for n in range(sets):
        n0 = n*minlength
        if n<sets-1:
            n1= (n+1)*minlength
        else:
            n1=-1
        vector = vector0[n0:n1]
        if not exclude_zero:
            mean = np.mean(vector)
            std=np.std(vector)
            vector_zscore= (vector-mean)/std
        else:
            #vector_zscore = np.zeros(np.shape(vector))
            mean = np.mean(vector[np.nonzero(vector)[0]])
            std=np.std(vector[np.nonzero(vector)[0]])
            vector_zscore[n0+(np.nonzero(vector)[0])] = (vector[np.nonzero(vector)[0]]-mean)/std
    return vector_zscore

def compute_everymin_maxmin(vector0,exclude_zero=True,minlength=60):
    #takes a sliding window and computes max and min in each window
    sets = int(len(vector0)/minlength)
    vector_zscore = copy.deepcopy(vector0)
    for n in range(sets):
        n0 = n*minlength
        if n<sets-1:
            n1= (n+1)*minlength
        else:
            n1=-1
        vector = vector0[n0:n1]
        if not exclude_zero:
            maxi = np.max(vector)
            mini = np.min(vector)
            vector_zscore= (vector-mini)/maxi
        else:
            mini = np.min(vector[np.nonzero(vector)[0]])
            maxi = np.max(vector[np.nonzero(vector)[0]])
            vector_zscore[n0+(np.nonzero(vector)[0])] = (vector[np.nonzero(vector)[0]]-mini)/(maxi-mini)
    return vector_zscore



def compute_row_correlation(matrix0, row_x, rows_y,smoothZscore=False,scaleZscore=False):
    matrix = copy.deepcopy(matrix0)
    matrix[np.isnan(matrix)] = 0
    row1 = copy.deepcopy(matrix[:,row_x])
    correlation = np.zeros(len(rows_y))
    correlation2 = np.zeros(len(rows_y))
    count=0
    scale = np.percentile(np.abs(row1), 95)
    for n in rows_y:
        row20 = copy.deepcopy(matrix[:,n])#
        time_intersection = (np.abs(row1)>0)&(np.abs(row20)>0)
        row2 = row20[time_intersection]
        if smoothZscore:
            print("smoothZscore")
            row2 = smooth_vec(compute_vector_z_score(row20[time_intersection]),2)
        elif scaleZscore:
            print("scaleZscore")
            row2 = scale*(compute_vector_z_score(row20[time_intersection]))
        correlation[count] = np.corrcoef(row1[time_intersection], row2)[0, 1]
        correlation2[count] = scipy.stats.pearsonr(row1[time_intersection], compute_vector_z_score(row2))[0]
        count=count+1
    return correlation,correlation2

def compute_row_correlation_sliding(matrix,matrix200, behavior, neurons,w=100,
            smoothZscore=False,scaleZscore=False):
    row10 = copy.deepcopy(matrix[:,behavior])
    matrix20=copy.deepcopy(matrix200)
    correlation = np.zeros((len(neurons),len(row10)))
    correlation2 = np.zeros((len(neurons),len(row10)))
    count=0
    row10[np.isnan(row10)] = 0
    scale = np.percentile(np.abs(row10), 95)
    matrix20[np.isnan(matrix20)] = 0
    matrix2 = copy.deepcopy(matrix20)
    for n in neurons:
        if smoothZscore:
            matrix2[:,n] = smooth_vec(compute_vector_z_score(matrix20[:,n]),2)
        elif scaleZscore:
            matrix2[:,n] = scale*(compute_vector_z_score(matrix20[:,n]))
        for t in range(w,len(row10)-w):
            row1 =row10[t-w:t+w]
            row2 = matrix2[t-w:t+w,n]
            time_intersection = (np.abs(row1)>0)&(np.abs(row2)>0)
            if np.sum(time_intersection)>2:
                correlation[count,t] = np.corrcoef(row1[time_intersection], row2[time_intersection])[0, 1]
                correlation2[count,t] = scipy.stats.pearsonr(row1[time_intersection], compute_vector_z_score(row2[time_intersection]))[0]
            else:
                print("no time intersection")
        count=count+1
    return correlation,correlation2


def compute_derivatives(MegaData,sm,L,ZscoreList=[0,1],exclude_worm=[100],
                            exclude_columns=[10,11,12,55,56,57,58,59]):
    '''
    Computes derivative of each column (3rd dimension) of MegaData
    MegaData has the shape number_Worms x time_Poins X num_columns
    sm:vector containing the delta_t used for derivative calculation for each
    worm
    L: vector of number of lower camera frames for each worm
    ZscoreList: list of columns (usually neuron numbers) that are Zscored before
    derivative computation
    '''
    number_Worms = MegaData.shape[0]
    number_columns = MegaData.shape[2]
    filtered_range = [i for i in range(number_Worms) if i not in exclude_worm]
    filtered_columns = [i for i in range(number_columns) if i not in exclude_columns]
    MegaData_derivative0 = copy.deepcopy(MegaData)
    MegaData_derivative0[MegaData_derivative0==0] = np.nan
    MegaData_derivative = np.zeros(np.shape(MegaData_derivative0))
    print('filtered_columns:')
    print(filtered_columns)
    for d in filtered_range:
        start = sm[d]
        end = L[d]-sm[d]
        for k in filtered_columns:
            if k in [0,1,2,3,4,5,6,7,8,9,27,28,39,40]:
                Zscored = compute_vector_z_score(MegaData[d,:L[d],k])*20
                Zscored[Zscored==0]=np.nan
                MegaData_derivative[d,start:L[d],k] = Zscored[start:L[d]]-Zscored[:end]
                temp = MegaData_derivative[d,start:L[d],k]
                maxpos = np.nanmax(temp)
                maxneg = -np.nanmin(temp)
                ratio = maxpos/maxneg
                temp[temp<0] = temp[temp<0]*ratio
                MegaData_derivative[d,start:L[d],k] = temp
            else:
                MegaData_derivative[d,start:L[d],k] = MegaData_derivative0[d,start:L[d],k]-MegaData_derivative0[d,:end,k]
    MegaData_derivative[np.isnan(MegaData_derivative)] = 0
    return MegaData_derivative

def compute_derivatives_future(MegaData,sm,L,ZscoreList=[0,1],exclude_worm=[100],
                            exclude_columns=[10,11,12,55,56,57,58,59]):
    '''
    Computes derivative of each column (3rd dimension) of MegaData
    MegaData has the shape number_Worms x time_Poins X num_columns
    sm:vector containing the delta_t used for derivative calculation for each
    worm
    L: vector of number of lower camera frames for each worm
    ZscoreList: list of columns (usually neuron numbers) that are Zscored before
    derivative computation
    '''
    number_Worms = MegaData.shape[0]
    number_columns = MegaData.shape[2]
    filtered_range = [i for i in range(number_Worms) if i not in exclude_worm]
    filtered_columns = [i for i in range(number_columns) if i not in exclude_columns]
    MegaData_derivative0 = copy.deepcopy(MegaData)
    MegaData_derivative0[MegaData_derivative0==0] = np.nan
    MegaData_derivative = np.zeros(np.shape(MegaData_derivative0))
    for d in filtered_range:
        start = 0
        end = L[d]-sm[d]
        for k in filtered_columns:
            if k in [0,1,2,3,4,5,6,7,8,9,27,28,39,40]:
                Zscored = compute_vector_z_score(MegaData[d,:L[d],k])*20
                Zscored[Zscored==0]=np.nan
                MegaData_derivative[d,start:end,k] = Zscored[sm[d]:L[d]] - Zscored[start:end]
                temp = MegaData_derivative[d,start:end,k]
                maxpos = np.nanmax(temp)
                maxneg = -np.nanmin(temp)
                ratio = maxpos/maxneg
                temp[temp<0] = temp[temp<0]*ratio
                MegaData_derivative[d,start:end,k] = temp
            else:
                MegaData_derivative[d,start:end,k] = MegaData_derivative0[d,sm[d]:L[d],k]-MegaData_derivative0[d,start:end,k]
    MegaData_derivative[np.isnan(MegaData_derivative)] = 0
    return MegaData_derivative


def compute_vector_z_score(vector0,exclude_zero=True):
    vector = copy.deepcopy(vector0)
    nanLoc = np.isnan(vector)
    vector[nanLoc] = 0
    if not exclude_zero:
        mean = np.mean(vector)
        std=np.std(vector)
        vector_zscore= (vector-mean)/std
    else:
        vector_zscore = np.zeros(np.shape(vector))
        vector_zscore[nanLoc] = np.nan
        mean = np.nanmean(vector[np.nonzero(vector)[0]])
        std=np.nanstd(vector[np.nonzero(vector)[0]])
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

def convert_lower_to_upper(lowertime,MegaData):
    ind = closest_index(MegaData[:,12], lowertime)
    upper_ind = MegaData[ind,11]
    return upper_ind

def convert_lower_to_upper_series(lowertimes,MegaData):
    upper_series = [convert_lower_to_upper(t,MegaData) for t in lowertimes]
    return upper_series


def convert_states2Freq(vector,window_size):
    output_vector = np.zeros_like(vector)

    # Compute the sum of 1's in the window for each position
    for i in range(len(vector)):
        start = max(0, i - window_size // 2)
        end = min(len(vector), i + window_size // 2 + 1)
        output_vector[i] = np.sum(vector[start:end])/(end-start)
    return output_vector

def convertToFformat(ActVector,thresh=0.01):
    '''
    convert the intensity vector F to the format (F-F0)/F0
    '''
    if len(np.nonzero(ActVector)[0])>100:
        Sort = np.argsort(ActVector)
        print(np.shape(ActVector))
        ActVector_s = ActVector[Sort]
        ActVector_s = ActVector_s[(ActVector_s>0)]
        lowerBound = np.max(ActVector_s[:int(len(ActVector_s)*thresh)])#mean of the lowwest dropTresh percent of the activity
        seq = (np.nonzero(ActVector))[0]
        print(np.shape(seq))
        print(np.shape(ActVector))
        ActVector[seq]=(ActVector[seq]-lowerBound*np.ones(len(seq)))/lowerBound
        print('negative vals')
        print(len(ActVector[ActVector<0]))
        print(np.shape(ActVector))
    return ActVector

def convert_properly(ActVector_withZeros, thresh=0.01):
    seqG = np.nonzero(ActVector_withZeros)[0]
    ActVector_withZeros[seqG]=convertToFformat(ActVector_withZeros[seqG],thresh)
    return ActVector_withZeros


def cross_product_2d(u, v):
    return u[0] * v[1] - u[1] * v[0]

def cross_product_vector(Vec, v):
    #Vec is a vector of shape Tx2
    out_v = np.zeros(Vec.shape[0])
    for t in range(len(out_v)):
        out_v[t] = (Vec[t,0] * v[1] - Vec[t,1] * v[0])/np.linalg.norm(Vec[t,:])
    return out_v

def cross_product_vector_angle(Vec, v):
    #Vec is a vector of shape Tx2
    out_v = np.zeros(Vec.shape[0])
    for t in range(len(out_v)):
        out_v[t] = (Vec[t,0] * v[1] - Vec[t,1] * v[0])/np.linalg.norm(Vec[t,:])
    return np.degrees(np.arcsin(out_v))




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

def droptop5perc(velocityVec0,thresh=0.05):
    #drops the highest 5 percent
    velocityVec = copy.deepcopy(velocityVec0)
    Sort = np.argsort(velocityVec)
    velocityVec_s = copy.deepcopy(velocityVec[Sort])
    velocityVec_s = velocityVec_s[velocityVec_s>0]
    upperBound = np.min(velocityVec_s[-int(len(velocityVec_s)*thresh):])#mean of the lowwest dropTresh percent of the activity
    above_thresh = (velocityVec > upperBound)
    velocityVec[above_thresh]=0
    velocityVec[above_thresh]=0
    return velocityVec

def dropbottom5perc(velocityVec0,thresh=0.05):
    velocityVec = copy.deepcopy(velocityVec0)
    Sort = np.argsort(velocityVec)
    velocityVec_s = np.copy(velocityVec[Sort])
    velocityVec_s = velocityVec_s[(velocityVec_s>0) + (velocityVec_s<0)]
    lowerBound = np.max(velocityVec_s[:int(len(velocityVec_s)*thresh)])#mean of the lowwest dropTresh percent of the activity
    below_thresh = (velocityVec < lowerBound)
    #print("lowerBound"+str(lowerBound))
    velocityVec[below_thresh]=0
    velocityVec[below_thresh]=0
    return velocityVec




def estimate_y(X,theta,y):
    #estimates y from the linear regressions coefficients
    yhat = X @ theta
    error = (np.linalg.norm(y-yhat))**2
    return yhat,error




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

def fill_in_gaps(x0, thresh=3):
    'x: vector where invalid components are set to zero'
    x= copy.deepcopy(x0)
    start_indices,end_indices = get_nonzero_start_end(x==0)#,thresh=3)
    length = np.array(end_indices) - np.array(start_indices)
    if thresh>10:
        print("only thresholds smaller than 10 are supported")
        print("setting thresh=10")
        thresh=10
    for v in range(len(length)-1):
        if length[v]==0:
            x[start_indices[v]] = x[start_indices[v]-1]
        if length[v]==1 and thresh>1:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st+2]
        if length[v]==2 and thresh>2:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = (x[st-1]+x[st+3])/2
            x[st+2] = x[st+3]
        if length[v]==3 and thresh>3:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st-1]
            x[st+2] = x[st+4]
            x[st+3] = x[st+4]
        if length[v]==4 and thresh>4:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st-1]
            x[st+2] = (x[st-1]+x[st+5])/2
            x[st+3] = x[st+5]
            x[st+4] = x[st+5]
        if length[v]==5 and thresh>5:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st-1]
            x[st+2] = (x[st-1]+x[st+6])/2
            x[st+3] = (x[st-1]+x[st+6])/2
            x[st+4] = x[st+6]
            x[st+5] = x[st+6]
        if length[v]==6 and thresh>6:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st-1]
            x[st+2] = (x[st-1]+x[st+7])/2
            x[st+3] = (x[st-1]+x[st+7])/2
            x[st+4] = (x[st-1]+x[st+7])/2
            x[st+5] = x[st+7]
            x[st+6] = x[st+7]
        if length[v]==7 and thresh>7:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st-1]
            x[st+2] = (x[st-1]+x[st+8])/2
            x[st+3] = (x[st-1]+x[st+8])/2
            x[st+4] = (x[st-1]+x[st+8])/2
            x[st+5] = x[st+8]
            x[st+6] = x[st+8]
            x[st+7] = x[st+8]
        if length[v]==8 and thresh>8:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st-1]
            x[st+2] = x[st-1]
            x[st+3] = (x[st-1]+x[st+9])/2
            x[st+4] = (x[st-1]+x[st+9])/2
            x[st+5] = (x[st-1]+x[st+9])/2
            x[st+6] = x[st+9]
            x[st+7] = x[st+9]
            x[st+8] = x[st+9]
        if length[v]==9 and thresh>9:
            st = start_indices[v]
            x[st] = x[st-1]
            x[st+1] = x[st-1]
            x[st+2] = x[st-1]
            x[st+3] = (x[st-1]+x[st+10])/2
            x[st+4] = (x[st-1]+x[st+10])/2
            x[st+5] = (x[st-1]+x[st+10])/2
            x[st+6] = (x[st-1]+x[st+10])/2
            x[st+7] = x[st+10]
            x[st+8] = x[st+10]
            x[st+9] = x[st+10]
    return x
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

def ordinary_least_squares(X, y):
    #y=X*\theta+\epsilon
    #performs linear regression and returns the coefficients
    temp1 = (X.T) @ y
    temps2 = (X.T) @ X
    temp3 = np.linalg.inv(temps2)
    theta_hat =  temp3 @ temp1
    return theta_hat

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
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    valid1 = ~np.isnan(Valvec1)
    valid2 = ~np.isnan(Valvec2)
    ax.bar(timevec[valid1],Valvec1[valid1], width=widthVec[valid1], align='center')
    ax.bar(timevec[valid2],Valvec2[valid2],alpha=0.3, width=widthVec[valid2], align='center')
    if leg:
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        ax.set_title(titre)
    plt.grid()
    plt.show()

def plot2BarsSingle(timevec,Valvec1,widthVec,ylab='bars',xlab='time',title = 'bars',leg = False):
    #All_ind:list of labels
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
    plt.xlabel(msqE)
    plt.ylabel(ylab)
    plt.title('Rsq: '+str(Rsq))
    plt.legend()
    plt.show()

def plot_autocorr(lags, autocorr,hl=1/np.e,titre="Autocorrelation of the Signal",ylab="Autocorrelation",xlab="Lag"):
    plt.figure(figsize=(10,4))
    plt.stem(lags, autocorr, use_line_collection=True)
    plt.title(titre)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.axhline(y= hl, color='green', linestyle='--')#, label='y = 1.96')
    plt.grid()
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
    fig, axs = plt.subplots(Wormnum,1, figsize=(14, 15))
    for d in range(Wormnum):
        matrix = corr_neuronsRI_ind_Vel1[:,d,:]
        cax1 = axs[d].imshow(matrix, cmap='viridis', aspect='auto')
        divider = make_axes_locatable(axs[d])
        # Add a new axis for the colorbar to the right with specified width
        cax = divider.append_axes("right", size="5%",pad=0.1)
        fig.colorbar(cax1, cax=cax)
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

def plot_fewBars_wtrials(Rsqrt0,ID=ID0,save=False,saveName = 'fewBars',titre='bars'):

    means = np.mean(Rsqrt0, axis=1)
    std_devs = np.std(Rsqrt0, axis=1)
    Num = Rsqrt0.shape[0]
    fig = plt.figure(figsize=(Num+3 ,5))
    ax = fig.add_subplot(111)


    ax.set_title(titre)
    ax.bar(ID, means, yerr=std_devs, capsize=5, color='skyblue')

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

def plot_fewBars_wpoints(Rsqrt0,save=False,saveName = 'fewBars',
                    titre='bars',xtick = ['neu+v', 'neu', 'v']):
    numWorms = Rsqrt0.shape[0]
    columns = Rsqrt0.shape[1]

    # Create a figure with 5 subplots (1 column, 5 rows)
    fig, axes = plt.subplots(numWorms, 1, figsize=(8, numWorms*4), sharex=True)

    for i in range(numWorms):
        ax = axes[i]
        submatrix = Rsqrt0[i,:,:]  # Extract the 3x10 submatrix
        means = np.mean(submatrix, axis=1)
        stds = np.std(submatrix, axis=1)

        # Create the bar plot for the submatrix
        x = np.arange(submatrix.shape[0])  # Indices for the rows of submatrix
        ax.bar(xtick, means, yerr=stds, capsize=5, alpha=0.7, label='Mean ± Std')
        # Add data points for the submatrix
        for j, row in enumerate(submatrix):
            x_jitter = np.random.normal(loc=x[j], scale=0.05, size=row.size)
            ax.scatter(x_jitter, row, color='red', s=20, alpha=0.7, label='Data Points' if j == 0 else "")


        ax.set_title(f'Worm {i+1}')
        ax.set_ylabel(titre)
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    return fig

def plot_histogram_subplots(MegaData0,L,columnToPlot,perc=10,start=[0,0,0,0,0,0,0,0],
                ylab='Frequency',b=[20,20,20,20,20,20,20,20],
                removeTopBottom=np.zeros((9,2))):
    numWrm = MegaData0.shape[0]
    MegaData = copy.deepcopy(MegaData0)
    MegaData[np.isnan(MegaData)]=0
    fig, axes = plt.subplots(numWrm, 1, figsize=(8, 2*numWrm), sharex=True)

    for d in range(numWrm):
        row = MegaData[d, start[d]:L[d],columnToPlot]
        nonzero_values = row[row != 0]  # Filter out zero values

        if removeTopBottom[d,0]:
            nonzero_values = dropbottom5perc(nonzero_values,removeTopBottom[d,0])
        if removeTopBottom[d,1]:
            nonzero_values = droptop5perc(nonzero_values,removeTopBottom[d,1])
        # Calculate mean and percentiles
        mean = np.mean(nonzero_values)
        p10 = np.percentile(nonzero_values, perc)
        p90 = np.percentile(nonzero_values, 100-perc)
        upperc = 100-perc
        # Plot histogram
        ax = axes[d]
        ax.hist(nonzero_values, bins=b[d], color='blue', alpha=0.7, edgecolor='black')

        # Mark mean and percentiles
        ax.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean:.2f}')
        ax.axvline(p10, color='green', linestyle='--', linewidth=1.5, label=f'{perc}th: {p10:.2f}')
        ax.axvline(p90, color='orange', linestyle='--', linewidth=1.5, label=f'{upperc}th: {p90:.2f}')
        buffer = (nonzero_values.max() - nonzero_values.min()) * 0.1  # Add 10% buffer
        ax.set_xlim(nonzero_values.min() - buffer, nonzero_values.max() + buffer)
        # Customize subplot
        ax.set_title(f'Worm {d}')
        ax.set_ylabel(ylab)
        ax.legend()

    # Add a common x-axis label
    #plt.xlabel('Value')
    #plt.tight_layout()
    plt.show()
    return fig,axes

def plot_points_timecolored(X,Y,time,lines=False):
    # Create a scatter plot with color coding based on time
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X,Y, c=time, cmap='viridis', s=50)  # s adjusts point size

    # Add a colorbar to represent the time scale
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Time')
    if lines:
        plt.plot(X,Y,lw=1)
    # Label the axes
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Color-Coded Position by Time')

    # Show the plot
    plt.show()



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

def plot_Rsqrd_Coeff_distr_self(Rsqrt,coeffRIBRIM,ID=ID0,save=False,saveName = 'RsqrdAndCoef'):
    #distribution of coeff. in the linear regression

    WrmNum = Rsqrt.shape[0]
    TrialNum = Rsqrt.shape[2]
    fig, axes = plt.subplots(WrmNum, 1, figsize=(10, WrmNum*4))
    # Iterate over the 5x5 subplots
    for d in range(WrmNum):
        # Extract the 10 data points for the current subplot
        rsq = ((Rsqrt[d,d,:]))
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

def plot_Rsqrd_Coeff_Max_self(Rsqrt,coeffRIBRIM,ID=ID0,save=False,saveName = 'RsqrdAndCoef'):
    #distribution of coeff. in the linear regression

    WrmNum = Rsqrt.shape[0]
    TrialNum = Rsqrt.shape[2]
    fig, axes = plt.subplots(WrmNum, 1, figsize=(10, WrmNum*4))
    # Iterate over the 5x5 subplots
    for d in range(WrmNum):
        # Extract the 10 data points for the current subplot
        rsq = Rsqrt[d,d,:]
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

def plot_scatter_overlap(vec1,vec2,time,xlab='Vector1',ylab='Vector2',titre='Scatter Plot Colored by Time'):
    plt.figure(figsize=(8, 8))
    nonzero_mask = (vec1 != 0) & (vec2 != 0)
    vec1_nonzero = vec1[nonzero_mask]
    vec2_nonzero = vec2[nonzero_mask]
    time_nonzero = time[nonzero_mask]

    # Create the scatter plot
    plt.scatter(vec1_nonzero, vec2_nonzero, c=time_nonzero, cmap='viridis', s=50, alpha=0.7)

    # Add a colorbar to show the time mapping
    plt.colorbar(label='Time')

    # Customize the plot
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titre)
    plt.grid(True)
    plt.show()


def plot_scatter_with_fit(vec1, vec2, time,xlab='Vector1',ylab='Vector2',
    titre='Scatter Plot Colored by Time',quad=False,fonts=20,r=0):
    fig = plt.figure(figsize=(10, 8))
    # Filter nonzero elements
    nonzero_mask = (vec1 != 0) & (vec2 != 0)
    vec1_nonzero = vec1[nonzero_mask]
    if not hasattr(vec1_nonzero, 'reshape'):
        vec1_nonzero = vec1_nonzero.to_numpy().reshape(-1, 1)
    else:
        vec1_nonzero = vec1_nonzero.reshape(-1, 1)
    vec2_nonzero = vec2[nonzero_mask]
    time_nonzero = time[nonzero_mask]

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(vec1_nonzero, vec2_nonzero)
    y_pred = model.predict(vec1_nonzero)
    r2 = r2_score(vec2_nonzero, y_pred)  # Calculate R^2

    # Scatter plot
    plt.scatter(vec1_nonzero, vec2_nonzero, c=time_nonzero, cmap='viridis', s=50, alpha=0.7, label='Data points')

    # Plot fitted line
    x_line = np.linspace(vec1_nonzero.min(), vec1_nonzero.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='red', label=f'Fit: $R^2 = {r2:.2f}$')

    # Add colorbar
    plt.colorbar(label='Time')
     # Calculate counts in each quadrant
    vec1_nonzero = vec1[nonzero_mask]
    quad1 = ((vec1_nonzero > 0) & (vec2_nonzero > 0)).sum()
    quad2 = ((vec1_nonzero < 0) & (vec2_nonzero > 0)).sum()
    quad3 = ((vec1_nonzero < 0) & (vec2_nonzero < 0)).sum()
    quad4 = ((vec1_nonzero > 0) & (vec2_nonzero < 0)).sum()
    # Annotate each quadrant
    x_range = vec1_nonzero.min(), vec1_nonzero.max()
    y_range = vec2_nonzero.min(), vec2_nonzero.max()

    x_mid = (x_range[0] + x_range[1]) / 2 +1
    y_mid = (y_range[0] + y_range[1]) / 2 +1
    if quad:
        plt.text(x_mid , y_mid , f'Q1: {quad1}', color='black', fontsize=10, ha='center')
        plt.text(-x_mid , y_mid , f'Q2: {quad2}', color='black', fontsize=10, ha='center')
        plt.text(-x_mid , -y_mid , f'Q3: {quad3}', color='black', fontsize=10, ha='center')
        plt.text(x_mid , -y_mid , f'Q4: {quad4}', color='black', fontsize=10, ha='center')


    # Customize the plot
    plt.xlabel(xlab,fontsize=fonts)
    plt.xticks(fontsize=fonts,rotation=r)
    plt.yticks(fontsize=fonts)
    plt.ylabel(ylab,fontsize=fonts)
    plt.title(titre,fontsize=fonts+5)
    plt.legend()
    plt.grid(True)
    plt.show()
    return fig

def plot_scatter_overlap_jitter(vec1,vec2,time,xlab='Vector1',ylab='Vector2',
            titre='Scatter Plot Colored by Time',jitter=0.1):
    plt.figure(figsize=(6, 6))
    nonzero_mask = (vec1 != 0) & (vec2 != 0)
    vec1_nonzero = vec1[nonzero_mask]
    vec2_nonzero = vec2[nonzero_mask]
    time_nonzero = time[nonzero_mask]

    vec1_jittered = vec1_nonzero + np.random.uniform(-jitter, jitter, size=vec1_nonzero.size)
    vec2_jittered = vec2_nonzero + np.random.uniform(-jitter, jitter, size=vec2_nonzero.size)

    # Create the scatter plot
    plt.scatter(vec1_jittered, vec2_jittered, c=time_nonzero, cmap='viridis', s=50, alpha=0.4)

    # Add a colorbar to show the time mapping
    plt.colorbar(label='Time')
     # Calculate counts in each quadrant
    quad1 = ((vec1_nonzero > 0) & (vec2_nonzero > 0)).sum()
    quad2 = ((vec1_nonzero < 0) & (vec2_nonzero > 0)).sum()
    quad3 = ((vec1_nonzero < 0) & (vec2_nonzero < 0)).sum()
    quad4 = ((vec1_nonzero > 0) & (vec2_nonzero < 0)).sum()

    # Annotate each quadrant
    x_range = vec1_nonzero.min(), vec1_nonzero.max()
    y_range = vec2_nonzero.min(), vec2_nonzero.max()

    x_mid = (x_range[0] + x_range[1]) / 2 +1
    y_mid = (y_range[0] + y_range[1]) / 2 +1

    plt.text(x_mid , y_mid , f'Q1: {quad1}', color='black', fontsize=10, ha='center')
    plt.text(-x_mid , y_mid , f'Q2: {quad2}', color='black', fontsize=10, ha='center')
    plt.text(-x_mid , -y_mid , f'Q3: {quad3}', color='black', fontsize=10, ha='center')
    plt.text(x_mid , -y_mid , f'Q4: {quad4}', color='black', fontsize=10, ha='center')

    # Customize the plot
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titre)
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_scatter_3d(x,y,z,time,xlab='X',ylab='Y',zlab='Z',titre='3d'):
    common_nonzero = (x != 0) & (y != 0) & (z != 0)

    # Extract common nonzero values
    x_common = x[common_nonzero]
    y_common = y[common_nonzero]
    z_common = z[common_nonzero]

    # Plot 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(x_common, y_common, z_common, c=time[common_nonzero], cmap='viridis', s=50)

    # Add color bar
    cb = plt.colorbar(sc, ax=ax, shrink=0.6, label='Z Value')

    # Customize the plot
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    ax.set_title(titre)

    plt.tight_layout()
    plt.show()

def plot_scatter_3dpercentile(x,y,z,time,perc1=90,perc2=100,xlab='X',ylab='Y',titre='3d'):

    common_nonzero = (x != 0) & (y != 0) & (z != 0)

    # Extract common nonzero values
    x_common = x[common_nonzero]
    y_common = y[common_nonzero]
    z_common = z[common_nonzero]
    t_common = time[common_nonzero]
    z_50 = np.percentile(z_common, perc1)
    z_75 = np.percentile(z_common, perc2)
    z_filter = (z_common >= z_50) & (z_common <= z_75)
    print(z_50)
    print(z_75)
    # Extract corresponding x and y values
    x_filtered = x_common[z_filter]
    y_filtered = y_common[z_filter]
    t_filtered = t_common[z_filter]
    # Plot 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    sc = ax.scatter(x_filtered, y_filtered, c=t_filtered, cmap='viridis', s=50)
    cb = plt.colorbar(sc, ax=ax, shrink=0.6, label='Z Value')

    # Customize the plot
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(titre)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def nan_correlate(x, y, mode='full'):
    """
    Compute correlation of two 1D arrays while ignoring NaN values.

    Parameters:
        x (array-like): First input array.
        y (array-like): Second input array.
        mode (str): 'full', 'valid', or 'same' (default is 'full').

    Returns:
        np.ndarray: Correlation result, ignoring NaN values.
    """
    signal1 = np.asarray(x)
    signal2 = np.asarray(y)
    x = (signal1-np.nanmean(signal1))/np.nanstd(signal1)
    y = (signal2-np.nanmean(signal2))/np.nanstd(signal2)
    # Ensure x and y are 1D
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be 1D arrays.")

    # Initialize output
    n = len(x)
    m = len(y)
    if mode == 'full':
        out_length = n + m - 1
    elif mode == 'valid':
        out_length = max(n, m) - min(n, m) + 1
    elif mode == 'same':
        out_length = n
    else:
        raise ValueError("mode must be 'full', 'valid', or 'same'.")

    # Pad x and y for 'full' mode
    if mode == 'full':
        x_pad = np.pad(x, (m - 1, m - 1), constant_values=np.nan)
    elif mode == 'valid' or mode == 'same':
        x_pad = x

    # Compute correlation
    result = []
    for lag in range(out_length):
        # Shift y with respect to x_pad
        y_shifted = np.roll(np.pad(y, (lag, out_length - lag - 1), constant_values=np.nan), shift=lag)

        # Identify non-NaN overlaps
        valid_mask = ~np.isnan(x_pad) & ~np.isnan(y_shifted)
        x_valid = x_pad[valid_mask]
        y_valid = y_shifted[valid_mask]

        # Compute dot product if valid elements exist
        if len(x_valid) > 0 and len(y_valid) > 0:
            result.append(np.dot(x_valid, y_valid)/len(y_valid))
        else:
            result.append(np.nan)

    return np.array(result)



def nan_correlate2(x, y, mode='full'):
    """
    Compute correlation of two 1D arrays while ignoring NaN values.

    Parameters:
        x (array-like): First input array.
        y (array-like): Second input array.
        mode (str): 'full', 'valid', or 'same' (default is 'full').

    Returns:
        tuple: (correlation_values, lags)
    """
    signal1 = np.asarray(x)
    signal2 = np.asarray(y)
    x = (signal1-np.nanmean(signal1))/np.nanstd(signal1)
    y = (signal2-np.nanmean(signal2))/np.nanstd(signal2)

    # Ensure x and y are 1D
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be 1D arrays.")

    n = len(x)
    m = len(y)

    # Determine the range of lags based on the mode
    if mode == 'full':
        lags = np.arange(-(m - 1), n)
    elif mode == 'valid':
        lags = np.arange(0, n - m + 1)
    elif mode == 'same':
        lags = np.arange(-(m // 2), n - m // 2)
    else:
        raise ValueError("mode must be 'full', 'valid', or 'same'.")

    # Initialize the result array
    result = []

    # Compute correlation for each lag
    for lag in lags:
        if lag < 0:
            x_shifted = x[-lag:]  # Align x for negative lag
            y_shifted = y[:len(x_shifted)]
        elif lag > 0:
            x_shifted = x[:n - lag]  # Align x for positive lag
            y_shifted = y[lag:]
        else:
            x_shifted = x
            y_shifted = y

        # Mask out NaN values
        valid_mask = ~np.isnan(x_shifted) & ~np.isnan(y_shifted)
        x_valid = x_shifted[valid_mask]
        y_valid = y_shifted[valid_mask]

        # Compute the dot product if there are valid elements
        if len(x_valid) > 0 and len(y_valid) > 0:
            result.append(np.dot(x_valid, y_valid)/len(y_valid))
        else:
            result.append(np.nan)

    return np.array(result), lags

def norm_cross_corr(signal1,signal2,mod='full'):
    signal1n = (signal1-np.mean(signal1))/np.std(signal1)
    signal2n = (signal2-np.mean(signal2))/np.std(signal2)
    norm_cross_corr = np.correlate(signal1n, signal2n, mode=mod)/len(signal1n)
    return norm_cross_corr

def norm_cross_corr_nan(signal1,signal2,mod='full'):
    signal1n = (signal1-np.nanmean(signal1))/np.nanstd(signal1)
    signal2n = (signal2-np.nanmean(signal2))/np.nanstd(signal2)
    valid_mask = ~np.isnan(signal1n) & ~np.isnan(signal2n)
    norm_cross_corr = np.correlate(signal1n[valid_mask], signal2n[valid_mask], mode=mod)/len(signal1n[valid_mask])
    return norm_cross_corr

def Regress_with_coeff(X_line,all_means):
    coefficients = all_means[:4]
    intercept = all_means[4]
    y_line = np.zeros(np.shape(X_line)[0])
    if True:#for t in range(len(y_line)):
        y_line = np.dot(X_line,coefficients) + intercept
    return y_line

def remove_nearZero(Vector,threshPos=0.01,threshNeg=-0.01):
    VectorOut = copy.deepcopy(Vector)
    VectorOut[(Vector>threshNeg) & (Vector<threshPos)] = 0
    return VectorOut

def smooth_vec(rowsArr_floatloc0,sig=1,inter=0):
    rowsArr_floatloc0[np.isnan(rowsArr_floatloc0)] = 0
    rowsArr_floatloc1 = copy.deepcopy(rowsArr_floatloc0)
    smoothed_arr=copy.deepcopy(rowsArr_floatloc0)
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

def self_R2(matrix):
    numW = matrix.shape[1]
    outMat = np.zeros((matrix.shape[0],numW,matrix.shape[3]))
    for d in range(numW):
        outMat[:,d,:] = matrix[:,d,d,:]
    return outMat

def smooth_vec(rowsArr_floatloc0,sig=2):
    rowsArr_floatloc0[np.isnan(rowsArr_floatloc0)] = 0
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
    rowsArr_floatloc0[np.isnan(rowsArr_floatloc0)] = 0
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
