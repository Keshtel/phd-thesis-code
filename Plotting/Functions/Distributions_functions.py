from sklearn.cluster import KMeans
import numpy as np
import h5py
import csv
import math
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats
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
warnings.filterwarnings("ignore")
# Suppress specific warning (e.g., UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.mixture import GaussianMixture
import Functions.AllFunctions as AllF


def plot_crosscorrelation(Cross_corr_neurons,b,limits=10,n_title=['']):
    #Cross_corr_neurons HAS DIMENSION OF DxNXlagged bins
    #N is number of neurons
    D,N = np.shape(Cross_corr_neurons)[0],np.shape(Cross_corr_neurons)[1]
    for d in range(D):
        fig,ax = plt.subplots(1,N,figsize=(4*N,4))
        for n in range(N):
            if len(np.nonzero(Cross_corr_neurons[d,n,:])[0])>1:
                time_lags = np.arange(-(b[d]) + 1, (b[d]))
                Leng = int(2*b[d]-1)
                ax[n].bar(time_lags, Cross_corr_neurons[d,n,:Leng])
                ax[n].set_xlim(left=-limits)
                ax[n].set_xlim(right=limits)
                ax[n].set_xlim(left=-limits)
                ax[n].set_xlim(right=limits)
                if n<len(n_title):
                    ax[n].set_title(n_title[n]+'-worm:'+str(d))
        plt.show()


def plot_allworms_box(neurons,MegaData, L,vel3first=13,Zscore= True):
    fig,ax = plt.subplots(len(neurons),5,figsize=(10,len(neurons)*3))
    label= ['RIA','RIA','RIM','RIM','RIB','RIB','Sensory','term1','term2']

    for n in neurons:
        for d in range(5):
            rowsArr_float = MegaData[d,:L[d],:]
            nonz1  = np.nonzero(rowsArr_float1[:,n])
            #ax.scatter(rowsArr_float1[nonz1[0],neuron],rowsArr_float1[nonz1[0],vel3first])
            Vel5 = dropbottom5perc(droptop5perc(rowsArr_float[:,vel3first],thresh=0.001),thresh=0.001)

            if Zscore:
                df,mean_time, mean_activity,width = bin_and_plot(compute_vector_z_score(Vel5[nonz1[0]]),compute_vector_z_score(rowsArr_float[nonz1[0],n]))
                #df = make_panda_bin(compute_vector_z_score(Vel5[nonz1[0]]),compute_vector_z_score(rowsArr_float[nonz1[0],n]))
            else:
                df,mean_time, mean_activity,width = bin_and_plot((Vel5[nonz1[0]]),(rowsArr_float[nonz1[0],n]))
                #df = make_panda_bin(Vel5[nonz1[0]],rowsArr_float[nonz1[0],n])
            #sns.barplot(x = 'vel_Bin', y = 'activity', data=df,ax=ax[n,d])
            ax[n,d].bar(mean_time, mean_activity, width=width, align='center',edgecolor='white')

            #ax[n,d].set_xticks([])
        ax[n,0].set_title(label[n])
    plt.show()

def plot_hist_distr1(subset_data,b=200):
    plt.figure(figsize=(10, 6))
    plt.hist(subset_data, bins=b, density=True, alpha=0.6, color='purple', label='Combined Data')
    gmm_single = GaussianMixture(n_components=1, covariance_type='full')
    gmm_single.fit(subset_data.reshape(-1, 1))
    # Fit a distribution to the combined data
    mu, std = stats.norm.fit(subset_data)
    # Plot the PDF of the fitted distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Fit: mu = {mu:.2f}, std = {std:.2f}')

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram with Fitted Distribution')
    plt.legend()

    # Show the plot
    #plt.show()

def plot_histogram_distr(data,b=200,s=True,d=True,tr=True,tits='fits',
            xlab='x',ylab='y',titre='distribution'):
    data_reshaped = data.reshape(-1, 1)

    # Fit Gaussian Mixture Models
    gmm_single = GaussianMixture(n_components=1, covariance_type='full').fit(data_reshaped)
    gmm_double = GaussianMixture(n_components=2, covariance_type='full').fit(data_reshaped)
    gmm_triple = GaussianMixture(n_components=3, covariance_type='full').fit(data_reshaped)

    # Calculate BIC and AIC for each model
    bic_single, aic_single = gmm_single.bic(data_reshaped), gmm_single.aic(data_reshaped)
    bic_double, aic_double = gmm_double.bic(data_reshaped), gmm_double.aic(data_reshaped)
    bic_triple, aic_triple = gmm_triple.bic(data_reshaped), gmm_triple.aic(data_reshaped)

    # Print the BIC and AIC values
    results = pd.DataFrame({
    'Model': ['Single Gaussian', 'Double Gaussian', 'Triple Gaussian'],
    'BIC': [bic_single, bic_double, bic_triple],
    'AIC': [aic_single, aic_double, aic_triple]
    })
    #print(results)

    # Plot the histogram of the data
    #plt.figure(figsize=(10, 6))
    plt.hist(data, bins=b, density=True, alpha=0.6, color='gray', label='Data')

    # Generate points for plotting PDFs
    x = np.linspace(min(data), max(data), 1000)

    if s:
        # Plot single Gaussian fit
        pdf_single = np.exp(gmm_single.score_samples(x.reshape(-1, 1)))
        plt.plot(x, pdf_single, label='Single Gaussian', color='blue')
    if d:
        # Plot double Gaussian fit
        pdf_double = np.exp(gmm_double.score_samples(x.reshape(-1, 1)))
        plt.plot(x, pdf_double, label='Double Gaussian', color='red')
    if tr:
        # Plot triple Gaussian fit
        pdf_triple = np.exp(gmm_triple.score_samples(x.reshape(-1, 1)))
        plt.plot(x, pdf_triple, label='Triple Gaussian', color='green')

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()

def plot_2mixedGaussian(data,b=200,label=True,xlab='x',ylab='y',titre='distribution'):

    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(data.reshape(-1, 1))

    # Extract the means, covariances, and weights of the fitted Gaussians
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # Generate a range of values for plotting the PDF
    x = np.linspace(min(data), max(data), 1000)

    # Calculate the PDF for each Gaussian component
    pdf_1 = weights[0] * stats.norm.pdf(x, means[0], np.sqrt(covariances[0]))
    pdf_2 = weights[1] * stats.norm.pdf(x, means[1], np.sqrt(covariances[1]))
    pdf_bimodal = pdf_1 + pdf_2

    # Plot the histogram of the data

    plt.hist(data, bins=b, density=True, alpha=0.6, color='gray', label='Data')

    # Plot the individual Gaussian components
    plt.plot(x, pdf_1, 'r--', label=f'Component 1: mean={means[0]:.2f}, w={weights[0]:.2f}')
    plt.plot(x, pdf_2, 'b--', label=f'Component 2: mean={means[1]:.2f}, w={weights[1]:.2f}')
    # Plot the combined bimodal distribution
    plt.plot(x, pdf_bimodal, 'k-', label='2-component fit')
    if label:
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(titre)
        plt.legend()

def plot_4mixedGaussian(data,b=200,label=True,xlab='x',ylab='y',titre='distribution'):

    gmm = GaussianMixture(n_components=4, covariance_type='full')
    gmm.fit(data.reshape(-1, 1))

    # Extract the means, covariances, and weights of the fitted Gaussians
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # Generate a range of values for plotting the PDF
    x = np.linspace(min(data), max(data), 1000)

    # Calculate the PDF for each Gaussian component
    pdf_1 = weights[0] * stats.norm.pdf(x, means[0], np.sqrt(covariances[0]))
    pdf_2 = weights[1] * stats.norm.pdf(x, means[1], np.sqrt(covariances[1]))
    pdf_3 = weights[2] * stats.norm.pdf(x, means[2], np.sqrt(covariances[2]))
    pdf_4 = weights[3] * stats.norm.pdf(x, means[3], np.sqrt(covariances[3]))
    pdf_bimodal = pdf_1 + pdf_2 + pdf_3 + pdf_4

    # Plot the histogram of the data

    plt.hist(data, bins=b, density=True, alpha=0.6, color='gray', label='Data')

    # Plot the individual Gaussian components
    plt.plot(x, pdf_1, 'r--', label=f'Component 1: mean={means[0]:.2f}, w={weights[0]:.2f}')
    plt.plot(x, pdf_2, 'b--', label=f'Component 2: mean={means[1]:.2f}, w={weights[1]:.2f}')
    plt.plot(x, pdf_3, 'g--', label=f'Component 3: mean={means[2]:.2f}, w={weights[2]:.2f}')
    plt.plot(x, pdf_4, 'y--', label=f'Component 4: mean={means[3]:.2f}, w={weights[3]:.2f}')
    # Plot the combined bimodal distribution
    plt.plot(x, pdf_bimodal, 'k-', label='4-component fit')
    if label:
        # Add labels and legend
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(titre)
        plt.legend()


def plot_5mixedGaussian(data,b=200,xlab='x',ylab='y',titre='distribution'):

    gmm = GaussianMixture(n_components=5, covariance_type='full')
    gmm.fit(data.reshape(-1, 1))
    # Extract the means, covariances, and weights of the fitted Gaussians
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # Generate a range of values for plotting the PDF
    x = np.linspace(min(data), max(data), 1000)
    # Calculate the PDF for each Gaussian component
    pdf_1 = weights[0] * stats.norm.pdf(x, means[0], np.sqrt(covariances[0]))
    pdf_2 = weights[1] * stats.norm.pdf(x, means[1], np.sqrt(covariances[1]))
    pdf_3 = weights[2] * stats.norm.pdf(x, means[2], np.sqrt(covariances[2]))
    pdf_4 = weights[3] * stats.norm.pdf(x, means[3], np.sqrt(covariances[3]))
    pdf_5 = weights[4] * stats.norm.pdf(x, means[4], np.sqrt(covariances[4]))
    pdf_bimodal = pdf_1 + pdf_2 + pdf_3 + pdf_4 +pdf_5

    # Plot the histogram of the data
    plt.hist(data, bins=b, density=True, alpha=0.6, color='gray', label='Data')

    # Plot the individual Gaussian components
    plt.plot(x, pdf_1, 'r--', label=f'Component 1: mean={means[0]:.2f}, w={weights[0]:.2f}')
    plt.plot(x, pdf_2, 'b--', label=f'Component 2: mean={means[1]:.2f}, w={weights[1]:.2f}')
    plt.plot(x, pdf_3, 'g--', label=f'Component 3: mean={means[2]:.2f}, w={weights[2]:.2f}')
    plt.plot(x, pdf_4, 'y--', label=f'Component 4: mean={means[3]:.2f}, w={weights[3]:.2f}')
    plt.plot(x, pdf_5, 'c--', label=f'Component 5: mean={means[4]:.2f}, w={weights[4]:.2f}')
    # Plot the combined bimodal distribution
    plt.plot(x, pdf_bimodal, 'k-', label='5-component fit')

    # Add labels and legend
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titre)
    plt.legend()


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

def compute_vector_maxmin(vector,exclude_zero=True):
    if not exclude_zero:
        mini = np.min(vector)
        maxi = np.max(vector)
        vector_zscore= (vector-mini)/(maxi-mini)
    else:
        vector_zscore = np.zeros(np.shape(vector))
        mini = np.min(vector[np.nonzero(vector)[0]])
        maxi = np.max(vector[np.nonzero(vector)[0]])
        vector_zscore[(np.nonzero(vector)[0])] = (vector[np.nonzero(vector)[0]]-mini)/(maxi-mini)
    return vector_zscore


def compute_vector_maxmin_neg(vector,exclude_zero=True):
    if not exclude_zero:
        mini = np.min(vector)
        maxi = np.max(vector)
        vector_zscore= 2*(vector-mini)/(maxi-mini) - 1
    else:
        vector_zscore = np.zeros(np.shape(vector))
        mini = np.min(vector[np.nonzero(vector)[0]])
        maxi = np.max(vector[np.nonzero(vector)[0]])
        vector_zscore[(np.nonzero(vector)[0])] = 2*(vector[np.nonzero(vector)[0]]-mini)/(maxi-mini) -1
    return vector_zscore

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


def interpolate_nanValues(x,y):
    x_interp = np.interp(np.arange(len(x)), np.arange(len(x))[~np.isnan(x)], x[~np.isnan(x)])
    y_interp = np.interp(np.arange(len(y)), np.arange(len(y))[~np.isnan(y)], y[~np.isnan(y)])
    return x_interp, y_interp

def plot_3D_with_color(pc1, pc2, pc3,backward_bins,lines=False):


    # Step 3: Plot the data in 3D PCA space
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pc1, pc2, pc3, c=backward_bins, cmap='viridis', marker='o', label='Data Points')
    if lines:
        ax.plot(pc1, pc2, pc3,lw=1)

    cbar = plt.colorbar(scatter)#, cax=cax1)
    cbar.set_label('Color Vector Values')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_title('3D PCA Space')

    plt.legend()
    plt.show()

def plot_2D_with_color(pc1, pc2, pc3,backward_bins,lines=False):

    # Step 3: Plot the data in 3D PCA space
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(131)
    scatter = ax.scatter(pc1, pc2, c=backward_bins, cmap='viridis', marker='o', label='Data Points')
    cbar = plt.colorbar(scatter)
    #cbar.set_label('Color Vector Values')

    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(pc1, pc3, c=backward_bins, cmap='viridis', marker='o', label='Data Points')
    #cbar2 = plt.colorbar(scatter2)

    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(pc2, pc3, c=backward_bins, cmap='viridis', marker='o', label='Data Points')
    #cbar3 = plt.colorbar(scatter3)
    if lines:
        ax.plot(pc1, pc2,lw=1)
        ax2.plot(pc1, pc3,lw=1)
        ax3.plot(pc2, pc3,lw=1)



    #ax.set_xlabel('PC 1')
    #ax.set_ylabel('PC 2')
    #ax.set_title('')

    #plt.legend()
    plt.show()


def norm_cross_corr(signal1,signal2,mode='full'):
    signal1n = (signal1-np.mean(signal1))/np.std(signal1)
    signal2n = (signal2-np.mean(signal2))/np.std(signal2)
    norm_cross_corr = np.correlate(signal1n, signal2n, mode='full')/len(signal1n)
    return norm_cross_corr

def plot_basic_heatmap(activity,curve):
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(311)
    nonz7 = np.nonzero(activity)[0]
    nonz8 = np.nonzero(curve)[0]
    indices = [int(x) for x in set(nonz7).intersection(set(nonz8))]#when there is odor and rest is tracked
    indices = np.sort(indices)
    image = np.zeros((2,len(indices)))
    image[0,:] = activity[indices]
    image[1,:] = curve[indices]
    heatmap3 = ax.imshow(image, aspect='auto',interpolation='none', cmap='inferno')#)
    cbar3 = fig.colorbar(heatmap3, ax=ax)
    plt.tight_layout()
    plt.show()

def fit_line(x,y):
    coefficients = np.polyfit(x, y, 1)
    fitting_line_x = np.linspace(min(x), max(x), 100)
    fitting_line_y = np.polyval(coefficients, fitting_line_x)
    return fitting_line_x,fitting_line_y

def set_negatives_tominus1(vector):
    vector[vector<0]=-1
    vector[vector>0]= 1
    return vector

def fit_line(x,y):
    coefficients = np.polyfit(x, y, 1)
    fitting_line_x = np.linspace(min(x), max(x), 100)
    fitting_line_y = np.polyval(coefficients, fitting_line_x)
    return coefficients, fitting_line_x,fitting_line_y
