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
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, BSpline
import matplotlib.cm as cm

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
from matplotlib import style
import matplotlib.pyplot as plt


def angle_between_twovec(v1,v2):
    cosine = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if np.abs(cosine)>1:
        print(cosine)
        cosine=1
    theta = math.acos(cosine)*180/math.pi
    return theta

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
        plt.bar(mean_time_by_bin, mean_activity_by_bin, width=bin_width, align='center')
        plt.title(label)
        plt.xlabel('time Bins')
        plt.ylabel('Mean Activity')
        plt.xticks(rotation=45, ha='right')
        plt.show()
    return df,mean_time_by_bin, mean_activity_by_bin,bin_width


def closest_index(A, b):
    #returns the index of vector A that has the closest value to b
    # Calculate the absolute differences between b and each element in A
    diff = np.abs(A - b)

    # Find the index of the element in A with the smallest difference
    idx = np.argmin(diff)

    # Return the index
    return idx

def compute_length(points):
    dx = np.diff(every_second_component(points,1))
    dy = np.diff(every_second_component(points,0))
    distances = np.sqrt(dx**2 + dy**2)

    # Compute total length of the curve
    total_length = np.sum(distances)

    print("Total length of the curve:", total_length)
    return total_length

def compute_mean_everySec(matrix):
    #computes the com of the worm
    even_columns = matrix[:, ::2]  # All rows, even columns
    odd_columns = matrix[:, 1::2]  # All rows, odd columns
    # Compute mean of even columns
    mean_even_columns = np.mean(even_columns, axis=1)
    mean_odd_columns = np.mean(odd_columns, axis=1)
    return mean_even_columns, mean_odd_columns

def compute_perpendicular_vector(v):
    perpendicular_vector = np.array([-v[1], v[0]])

    return perpendicular_vector


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


def convert20to80(Xcoord,Ycoord):
    t = np.linspace(0, 19, len(Xcoord))

    # Create cubic spline interpolators for x and y as functions of t
    spline_x = interp1d(t, Xcoord, kind='cubic')
    spline_y = interp1d(t, Ycoord, kind='cubic')

    # Create new parameterization for the 20 equally spaced points
    new_t = np.linspace(0, 19, 80)

    # Calculate the 20 points' x and y coordinates
    new_x = spline_x(new_t)
    new_y = spline_y(new_t)
    return new_x, new_y

def convert20to80_equidistant(Xcoord,Ycoord):

    dx = np.diff(Xcoord)
    dy = np.diff(Ycoord)
    ds = np.sqrt(dx**2 + dy**2)

    # Compute cumulative arc length
    s = np.concatenate([[0], np.cumsum(ds)])

    # Interpolate the parameter to get equal distance segments
    num_points = 80  # Number of points with equal distance segments
    s_equal = np.linspace(0, s[-1], num_points)
    x_interp = interp1d(s, Xcoord, kind='linear')(s_equal)
    y_interp = interp1d(s, Ycoord, kind='linear')(s_equal)
    return x_interp,y_interp

def convert20toN_equidistant(Xcoord,Ycoord,N=80):

    dx = np.diff(Xcoord)
    dy = np.diff(Ycoord)
    ds = np.sqrt(dx**2 + dy**2)

    # Compute cumulative arc length
    s = np.concatenate([[0], np.cumsum(ds)])

    # Interpolate the parameter to get equal distance segments
    num_points = N  # Number of points with equal distance segments
    s_equal = np.linspace(0, s[-1], num_points)
    x_interp = interp1d(s, Xcoord, kind='linear')(s_equal)
    y_interp = interp1d(s, Ycoord, kind='linear')(s_equal)
    return x_interp,y_interp

def convert20toN_equidistant_smooth(Xcoord,Ycoord,N=80,sm=0,deg=2):

    dx = np.diff(Xcoord)
    dy = np.diff(Ycoord)
    ds = np.sqrt(dx**2 + dy**2)

    # Compute cumulative arc length
    s = np.concatenate([[0], np.cumsum(ds)])

    # Interpolate the parameter to get equal distance segments
    num_points = N  # Number of points with equal distance segments
    s_equal = np.linspace(0, s[-1], num_points)
    x_interp = interp1d(s, Xcoord, kind='linear')(s_equal)
    y_interp = interp1d(s, Ycoord, kind='linear')(s_equal)
    return x_interp,y_interp

def convert_properly(ActVector_withZeros, thresh=0.1):
    seqG = np.nonzero(ActVector_withZeros)[0]
    print(seqG)
    ActVector_withZeros[seqG]=convertToFformat(ActVector_withZeros[seqG],thresh)
    return ActVector_withZeros

def cross_product_2d(u, v):
    return u[0] * v[1] - u[1] * v[0]

def convertToFformat(ActVector,thresh=0.1):
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
        print(lowerBound)
        ActVector[seq] = (ActVector[seq]-lowerBound*np.ones(len(seq)))/lowerBound

    return ActVector

def curvature(x, y):
    #x and y are vectors of x coord. and y coord. respectively
    dx_ds = np.gradient(x)
    dy_ds = np.gradient(y)
    d2x_ds2 = np.gradient(dx_ds)
    d2y_ds2 = np.gradient(dy_ds)

    numerator = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
    denominator = (dx_ds**2 + dy_ds**2)**1.5

    curvature = numerator / denominator
    R = 1/curvature
    return curvature, R

def droptop5perc(velocityVec,thresh=0.05):
    #drops the highest 5 percent
    Sort = np.argsort(velocityVec)
    velocityVec_s = np.copy(velocityVec[Sort])
    velocityVec_s = velocityVec_s[velocityVec_s>0]
    upperBound = np.min(velocityVec_s[-int(len(velocityVec_s)*thresh):])#mean of the lowwest dropTresh percent of the activity
    above_thresh = (velocityVec > upperBound)
    velocityVec[above_thresh]=0
    velocityVec[above_thresh]=0
    return velocityVec
def dropbottom5perc(velocityVec,thresh=0.05):
    Sort = np.argsort(velocityVec)
    velocityVec_s = np.copy(velocityVec[Sort])
    velocityVec_s = velocityVec_s[(velocityVec_s>0) + (velocityVec_s<0)]
    lowerBound = np.max(velocityVec_s[:int(len(velocityVec_s)*thresh)])#mean of the lowwest dropTresh percent of the activity
    below_thresh = (velocityVec < lowerBound)
    #print("lowerBound"+str(lowerBound))
    velocityVec[below_thresh]=0
    velocityVec[below_thresh]=0
    return velocityVec

def every_second_component(vector,odd=0):
    new_vector = []
    if odd==0:
        for i in range(1, len(vector), 2):
            new_vector.append(vector[i])
    else:
        for i in range(0, len(vector), 2):
            new_vector.append(vector[i])
    return new_vector

def extract_coordinates(Address,num=1):
    #returns the data in csv file
    file = open(Address[0])
    type(file)

    csvreader = csv.reader(file)
    np.shape(csvreader)
    rows = []

    for row in csvreader:
        rows.append(row)

    print(np.shape(rows))

    if num==3:
        rowsArr0=np.array(rows[1:])
        rowsArr=np.array(rowsArr0[:,:-2])
        #print(rowsArr)
    else:
        rowsArr=np.array(rows[1:])


    shapeMat = np.shape(rowsArr)

    rowsArr_float = np.zeros(shapeMat)
    for i in range(shapeMat[0]):
        for j in range(shapeMat[1]):
            #print(rowsArr[i,j])
            if rowsArr[i,j]=='NA':
                rowsArr_float[i,j]= 0
            elif rowsArr[i,j]=='NaN':
                rowsArr_float[i,j]= 0
            else:
                rowsArr_float[i,j]=float(rowsArr[i,j])

    return rows,rowsArr_float



def make_panda_bin(Xvec,Yvec,b=20):
    data = {
        'velocity': Xvec,
        'activity': Yvec,
            }
    df = pd.DataFrame(data)
    # Bin 'Dimension2'
    df['vel_Bin'] = pd.cut(df['velocity'], bins=b)

    # Calculate the mean of 'activity' in each bin
    mean_activity_by_bin = df.groupby('vel_Bin')['activity'].mean()
    # Extract bin edges and calculate bin width
    #bin_edges = pd.cut(df['velocity'], bins=b).unique().mid
    #bin_width = bin_edges[1] - bin_edges[0]

    # Calculate x-tick positions at the center of each bin
    #xtick_positions = bin_edges + bin_width / 2
    return df,mean_activity_by_bin



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

def interpolate_nanValues(x,y):
    x_interp = np.interp(np.arange(len(x)), np.arange(len(x))[~np.isnan(x)], x[~np.isnan(x)])
    y_interp = np.interp(np.arange(len(y)), np.arange(len(y))[~np.isnan(y)], y[~np.isnan(y)])
    return x_interp, y_interp


def plot1d_withOdor(timevec,Valvec,Odorvec,title):

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot()
    ax.scatter(timevec,Valvec)
    starts,end = get_nonzero_start_end(Odorvec)
    for t in range(len(start)):
        if Odorvec[starts[t]]==1:
            ax.axvspan(timevec[starts[t]],timevec[end[t]],c ='g',alpha=0.2)#axvspan
        if Odorvec[starts[t]]==2:
            ax.axvspan(timevec[starts[t]],timevec[end[t]],c ='r',alpha=0.2)#axvspan
    ax.set_title(title)
    plt.grid()
    plt.show()

def plot1d(timevec,Valvec,title='plot'):

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot()
    ax.scatter(timevec,Valvec)
    ax.set_title(title)
    plt.grid()
    plt.show()


def plot_allworms_box(neurons,MegaData, L,vel3first=13,Zscore= True):
    fig,ax = plt.subplots(len(neurons),5,figsize=(10,len(neurons)*3))
    label= ['RIA','RIA','RIM','RIM','RIB','RIB','Sensory','term1','term2']

    for n in neurons:
        for d in range(5):
            rowsArr_float = MegaData[d,:L[d],:]
            nonz1  = np.nonzero(rowsArr_float1[:,n])

            Vel5 = dropbottom5perc(droptop5perc(rowsArr_float[:,vel3first],thresh=0.001),thresh=0.001)

            if Zscore:
                df,mean_time, mean_activity,width = bin_and_plot(compute_vector_z_score(Vel5[nonz1[0]]),compute_vector_z_score(rowsArr_float[nonz1[0],n]))
                #df = make_panda_bin(compute_vector_z_score(Vel5[nonz1[0]]),compute_vector_z_score(rowsArr_float[nonz1[0],n]))
            else:
                df,mean_time, mean_activity,width = bin_and_plot((Vel5[nonz1[0]]),(rowsArr_float[nonz1[0],n]))

            ax[n,d].bar(mean_time, mean_activity, width=width, align='center',edgecolor='white')

        ax[n,0].set_title(label[n])
    plt.show()

def smooth_vec(rowsArr_floatloc0,sig=1,inter=0):
    rowsArr_floatloc0[np.isnan(rowsArr_floatloc0)] = 0
    rowsArr_floatloc1 = copy.deepcopy(rowsArr_floatloc0)
    smoothed_arr=np.copy(rowsArr_floatloc0)
    starts,end = get_nonzero_start_end(rowsArr_floatloc1)
    for t in range(len(starts)):
        if end[t]-starts[t]>3:
            smoothed_arr[starts[t]:end[t]] = gaussian_filter1d((rowsArr_floatloc1[starts[t]:end[t]]), sigma=sig)
        smoothed_arr_f = [np.float(x) for x in smoothed_arr]
    return smoothed_arr_f
