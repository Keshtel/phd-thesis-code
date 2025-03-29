import h5py
import numpy as np
import copy
import importlib
import matplotlib.pyplot as plt
import Functions.AllFunctions as AllF
import pandas as pd
from scipy.signal import correlate
from scipy.optimize import curve_fit
import math
from scipy.signal import find_peaks
import seaborn as sns

def autocorrelation_s(x0):
    x = copy.deepcopy(x0)
    nonz = np.nonzero(x0)
    x[nonz[0]] = (x0[nonz[0]] - np.mean(x0[nonz[0]]))/np.std(x0[nonz[0]])
    raw_autocorr = np.correlate(x, x, mode='full')  # Standard autocorrelation

    valid_mask = x0 != 0  # Mask where data is valid (nonzero)
    norm_factor = np.correlate(valid_mask.astype(int), valid_mask.astype(int), mode='full')  # Count of nonzero contributions per lag
    print(len(raw_autocorr))
    print(len(norm_factor))
    print(np.unique(valid_mask.astype(int)))
    norm_autocorr = np.divide(raw_autocorr, norm_factor, where=norm_factor != 0)  # Normalize where valid
    # norm_autocorr = raw_autocorr # Left to check whether this correction makes any difference

    #norm_autocorr = norm_autocorr[norm_autocorr.size // 2:]  # Take only non-negative lags
    #norm_autocorr /= norm_autocorr[norm_autocorr.size // 2]  # Normalize by lag-0 autocorrelation

    return norm_autocorr,norm_factor


def averaging_crosscorr_diff_stretches(x,y,stretches,threshold=20):
    count = 0
    collected_matrices = []
    collected_weights = []
    for (start, end) in stretches:
        leng = end-start
        delta = leng-threshold
        if leng > threshold:
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

def exponential_decay(tau, A, tau_0):
    return A * np.exp(-tau / tau_0)

def exponential1_decay(tau, tau_0):
    return np.exp(-tau / tau_0)

def fall_below_thresh(time,acf_centered,threshold = 1 / np.e):


    # Find the first index where acf falls below 1/e
    below = np.where(acf_centered < threshold)[0]

    if len(below) > 0:
        lag_index = time[below[0]]
    else:
        lag_index = np.nan
    return lag_index

def double_exponential_decay(tau, A1, tau1, A2, tau2):
    return A1 * np.exp(-tau / tau1) + A2 * np.exp(-tau / tau2)

def stretched_exponential_decay(tau, A, tau0, beta):
    return A * np.exp(-((tau / tau0) ** beta))

def power_law_decay(tau, A,tau0, gamma):
    return A/ ((1 + np.abs(tau / tau0)) ** gamma)

def gaussian_decay(tau, A, tau0):
    return A * np.exp(-((tau / tau0) ** 2))


def get_time_scale(MegaData,L,TimeMat, allfit=False,thresh=1500,neurite=False):
    ML = int(np.max(L))
    numW = MegaData.shape[0]
    num_neu = MegaData.shape[2]
    FullAutocorr = np.zeros((numW,2*ML-1,num_neu))
    FullLags = np.zeros((numW,2*ML-1,num_neu))
    Time_constant = np.zeros((numW,num_neu,2))#0time,1scale
    exp1_decay = np.zeros((numW,num_neu))#0time,1scale
    double_exponential = np.zeros((numW,num_neu,4))#0time,1scale
    stretched_exponential = np.zeros((numW,num_neu,3))#
    power_law = np.zeros((numW,num_neu,3))#
    gaussian = np.zeros((numW,num_neu,2))#
    threshold_t = np.zeros((numW,num_neu))

    for d in range(numW):
        for n in range(num_neu):
            if d>0 or not (n in [3,5]) or neurite:
                x= copy.deepcopy(MegaData[d,:L[d],n])
                x[x==0]= np.nan
                autocorrnan2,lagsnan2 = nan_correlate2(x,x)#correlate(x, x, mode='full', method='auto')
                # Center the result
                lags = np.zeros(2*len(x)-1)
                FullAutocorr[d,:len(lags),n] = autocorrnan2
                lags[:len(x)] = -TimeMat[d,:L[d]][::-1]#-MegaData[d,:L[d],12][::-1]
                lags[len(x):] = TimeMat[d,1:L[d]]#MegaData[d,1:L[d],12]
                FullLags[d,:len(lags),n] = lags
                time = lags[len(x)-1:]
                valid= np.nonzero(~np.isnan(autocorrnan2[len(x)-1:]))[0]
                if allfit:
                    valid0= np.nonzero(~np.isnan(autocorrnan2[len(x)-1:]))[0]
                    valid = [i for i in valid0 if TimeMat[d,i] < thresh]
                    valid = np.array(valid)

                # Fit the exponential model to the autocorrelation

                exp1_decay[d,n], pcov = curve_fit(exponential1_decay, time[valid], autocorrnan2[len(x)-1+valid], p0=20)  # Initial guess for A and tau_0


                popt2, pcov = curve_fit(exponential_decay, time[valid], autocorrnan2[len(x)-1+valid], p0=[1, 20])  # Initial guess for A and tau_0
                Time_constant[d,n,1], Time_constant[d,n,0] = popt2

                threshold_t[d,n] = fall_below_thresh(time[valid], autocorrnan2[len(x)-1+valid],threshold = 1 / np.e)

                popt_gauss, pcov5 = curve_fit(gaussian_decay, time[valid], autocorrnan2[len(x)-1+valid], p0=[1, 1])
                gaussian[d,n,1], gaussian[d,n,0] = popt_gauss

                #popt_double, pcov2 = curve_fit(double_exponential_decay, time[valid], autocorrnan2[len(x)-1+valid], p0=[1, 1, 1, 1])
                #double_exponential[d,n,2],double_exponential[d,n,0],double_exponential[d,n,3],double_exponential[d,n,1]= popt_double

                if allfit:
                    popt_stretch, pcov3 = curve_fit(stretched_exponential_decay, time[valid],
                                    autocorrnan2[len(x)-1+valid], p0=[1, 20, 1],maxfev=5000)
                    stretched_exponential[d,n,2],stretched_exponential[d,n,0],stretched_exponential[d,n,1]= popt_stretch

                    popt_power, pcov4 = curve_fit(power_law_decay, time[valid], autocorrnan2[len(x)-1+valid], p0=[1,1, 1],
                                              bounds=([0, 0,0], [np.inf, np.inf, np.inf]),  # Non-negative constraints
                                                maxfev=5000)
                    power_law[d,n,2],power_law[d,n,0],power_law[d,n,1]= popt_power


    if allfit:
        return FullLags,FullAutocorr,Time_constant,exp1_decay,gaussian,stretched_exponential,power_law,threshold_t
    else:
        return FullLags,FullAutocorr,Time_constant,exp1_decay,gaussian,threshold_t

def get_all_cross(MegaData,m,L,TimeMat,thresh=1500,neurite=False):
    'm: the neuron whose cross correlation you want to compute with others'

    ML = int(np.max(L))
    numW = MegaData.shape[0]
    num_neu = MegaData.shape[2]
    FullAutocorr = np.zeros((numW,2*ML-1,num_neu))
    FullLags = np.zeros((numW,2*ML-1,num_neu))
    Time_constant = np.zeros((numW,num_neu,2))#0time,1scale
    exp1_decay = np.zeros((numW,num_neu))#0time,1scale
    double_exponential = np.zeros((numW,num_neu,4))#0time,1scale
    stretched_exponential = np.zeros((numW,num_neu,3))#
    power_law = np.zeros((numW,num_neu,3))#
    gaussian = np.zeros((numW,num_neu,2))#
    top_peak3 = np.zeros((numW,num_neu,8))
    top_peak3[top_peak3==0] = np.nan
    for d in range(numW):
        y= copy.deepcopy(MegaData[d,:L[d],m])
        y[y==0]= np.nan
        for n in range(num_neu):
            if d>0 or not (n in [3,5]) or neurite:
                x= copy.deepcopy(MegaData[d,:L[d],n])
                x[x==0]= np.nan
                autocorrnan2,lagsnan2 = nan_correlate2(x,y)#correlate(x, x, mode='full', method='auto')

                # Center the result
                lags = np.zeros(2*len(x)-1)
                FullAutocorr[d,:len(lags),n] = autocorrnan2
                lags[:len(x)] = -TimeMat[d,:L[d]][::-1]#-MegaData[d,:L[d],12][::-1]
                lags[len(x):] = TimeMat[d,1:L[d]]#MegaData[d,1:L[d],12]
                FullLags[d,:len(lags),n] = lags
                time = lags[len(x)-1:]
                valid= np.nonzero(~np.isnan(autocorrnan2[len(x)-1:]))[0]
                peaks0, _ = find_peaks(autocorrnan2, distance=5)#, prominence=0.3, distance=10)  # `distance` avoids closely spaced peaks
                peaks1 = peaks0[peaks0>(L[d]-25)]
                peaks = peaks1[peaks1<(L[d]+25)]
                if len(peaks)>0:
                    # Sort peaks by power and get the top 3
                    lim=np.min([len(peaks),8])
                    top_peak3[d,n,:lim] = peaks[np.argsort(autocorrnan2[peaks])[-8:]]

    return FullLags,FullAutocorr,top_peak3



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
        lags = np.arange(-(m - 1), n, dtype=float)
    elif mode == 'valid':
        lags = np.arange(0, n - m + 1, dtype=float)
    elif mode == 'same':
        lags = np.arange(-(m // 2), n - m // 2, dtype=float)
    else:
        raise ValueError("mode must be 'full', 'valid', or 'same'.")

    # Initialize the result array
    result = []

    # Compute correlation for each lag
    for lag0 in lags:
        lag = int(lag0)
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
    # this definition is good if you dont have any  nan values
    signal1n = (signal1-np.mean(signal1))/np.std(signal1)
    signal2n = (signal2-np.mean(signal2))/np.std(signal2)
    norm_cross_corr = np.correlate(signal1n, signal2n, mode=mod)/len(signal1n)
    return norm_cross_corr

def norm_cross_corr_nan(signal1,signal2,mod='full'):
    # this is better for signals with nan values but has an intrinsic error
    # corresponding unequal time intervals
    signal1n = (signal1-np.nanmean(signal1))/np.nanstd(signal1)
    signal2n = (signal2-np.nanmean(signal2))/np.nanstd(signal2)
    valid_mask = ~np.isnan(signal1n) & ~np.isnan(signal2n)
    norm_cross_corr = np.correlate(signal1n[valid_mask], signal2n[valid_mask], mode=mod)/len(signal1n[valid_mask])
    return norm_cross_corr,valid_mask


def plot_autocorr(lags, autocorr,hl=1/np.e,titre="Autocorrelation of the Signal",ylab="Autocorrelation",xlab="Lag"):
    plt.figure(figsize=(10,4))
    plt.stem(lags, autocorr, use_line_collection=True)
    plt.title(titre)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.axhline(y= hl, color='green', linestyle='--')#, label='y = 1.96')
    plt.grid()
    plt.show()

def plot_parameters_heat(data,Neur_lab, WormLabel, paramName='τ Value',fs = 20,titre = 'τ - exp'):

    fig = plt.figure(figsize=(8, 8))
    heatmap = plt.imshow(data, cmap='viridis', aspect='auto', origin='lower')
    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label(paramName, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    plt.xticks(ticks=np.arange(len(Neur_lab)), labels = Neur_lab, rotation=45, fontsize=fs)
    plt.yticks(ticks=np.arange(len(WormLabel)), labels = WormLabel, fontsize=fs)
    plt.title(titre, fontsize=fs)
    plt.grid(visible=False)
    plt.show()
    return fig

def plot_parameters_errbar(data,Neur_lab, paramName='τ Value',fs = 16):
    row_means = np.nanmean(data, axis=0)  # Mean across columns, ignoring NaNs
    row_sems = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
    rows = np.arange(data.shape[1])  # Row indices for the x-axis
    plt.figure(figsize=(8, 4))
    plt.errorbar(rows, row_means, yerr=row_sems, fmt='o-', capsize=5, label='Mean ± SEM',lw=4)
    plt.xticks(rows,labels=Neur_lab,fontsize=fs)  # Set custom x labels with font size
    plt.yticks(fontsize=fs)
    plt.title(paramName, fontsize=fs)
    plt.grid(True)

def plot_parameters_errbar2(data1,data2,Neur_lab, paramName='τ Value',fs = 16):
    row_means1 = np.nanmean(data1, axis=0)  # Mean across rows, ignoring NaNs
    row_sems1 = np.nanstd(data1, axis=0) / np.sqrt(np.sum(~np.isnan(data1), axis=0))  # SEM calculation
    rows1 = np.arange(data1.shape[1])

    row_means2 = np.nanmean(data2, axis=0)  # Mean across rows, ignoring NaNs
    row_sems2 = np.nanstd(data2, axis=0) / np.sqrt(np.sum(~np.isnan(data2), axis=0))  # SEM calculation
    rows2 = np.arange(data2.shape[1])
    fig = plt.figure(figsize=(6, 4))
    plt.errorbar(rows1, row_means1, yerr=row_sems1, fmt='o', capsize=10, label='n',lw=4)
    plt.errorbar(rows2, row_means2, yerr=row_sems2, fmt='o', capsize=10, label='dn',lw=4)
    plt.xticks(rows1,labels=Neur_lab,fontsize=fs)  # Set custom x labels with font size
    plt.yticks(fontsize=fs)
    plt.title(paramName, fontsize=fs)
    plt.grid(True)
    return fig

def plot_parameters_bar(data, Neur_lab, paramName='τ Value', fs=16,r=0,median=0):
    """
    Plot bar plots with error bars for the given data.

    Args:
        data: 2D array-like, rows represent samples, columns represent categories.
        Neur_lab: Labels for categories (x-axis).
        paramName: Label for the y-axis.
        fs: Font size for labels and ticks.
    """
    if median:
        row_means = np.nanmedian(data, axis=0)  # Median across rows, ignoring NaNs
    else:
        row_means = np.nanmean(data, axis=0)  # Mean across rows, ignoring NaNs
    row_sems = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))  # SEM calculation
    rows = np.arange(data.shape[1])  # Row indices for the x-axis

    fig = plt.figure(figsize=(8, 6))
    plt.bar(rows, row_means, yerr=row_sems, capsize=5, alpha=0.8, color='skyblue', edgecolor='black')
    plt.xticks(rows, labels=Neur_lab, fontsize=fs,rotation=r)
    plt.yticks(fontsize=fs)
    plt.ylabel(paramName, fontsize=fs)
    plt.title(paramName, fontsize=fs)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    return fig

def plot_parameters_bar2(data1,data2, Neur_lab, paramName='τ Value', fs=16,r=0):
    """
    Plot bar plots with error bars for the given data.

    Args:
        data: 2D array-like, rows represent samples, columns represent categories.
        Neur_lab: Labels for categories (x-axis).
        paramName: Label for the y-axis.
        fs: Font size for labels and ticks.
    """
    row_means1 = np.nanmean(data1, axis=0)  # Mean across rows, ignoring NaNs
    row_sems1 = np.nanstd(data1, axis=0) / np.sqrt(np.sum(~np.isnan(data1), axis=0))  # SEM calculation
    rows1 = np.arange(data1.shape[1])

    row_means2 = np.nanmean(data2, axis=0)  # Mean across rows, ignoring NaNs
    row_sems2 = np.nanstd(data2, axis=0) / np.sqrt(np.sum(~np.isnan(data2), axis=0))  # SEM calculation
    rows2 = np.arange(data2.shape[1])
    fig = plt.figure(figsize=(8, 4))
    plt.bar(rows1, row_means1, yerr=row_sems1, capsize=5, alpha=0.8, color='skyblue', edgecolor='black')
    plt.bar(rows2, row_means2, yerr=row_sems2, capsize=5, alpha=0.8, color='green', edgecolor='black')
    plt.xticks(rows, labels=Neur_lab, fontsize=fs,rotation=r)
    plt.yticks(fontsize=fs)
    plt.ylabel(paramName, fontsize=fs)
    plt.title(paramName, fontsize=fs)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    return fig


def plot_parameters_violin(data, Neur_lab, paramName='τ Value', fs=16,r=0):
    """
    Plot violin plots for the given data.

    Args:
        data: 2D array-like, rows represent samples, columns represent categories.
        Neur_lab: Labels for categories (x-axis).
        paramName: Label for the y-axis.
        fs: Font size for labels and ticks.
    """
    fig = plt.figure(figsize=(data.shape[1], 4))
    sns.violinplot(data=data, inner='point')  # Add points to show mean/median within the distribution
    plt.xticks(np.arange(data.shape[1]), labels=Neur_lab, fontsize=fs,rotation=r)
    plt.yticks(fontsize=fs)
    plt.ylabel(paramName, fontsize=fs)
    plt.title(paramName, fontsize=fs)
    plt.grid(True)
    return fig


def plot_AllneuronsFit(FullLags,FullAutocorr,Neur_lab,Time_constant,L,exp=0):
    #exp: 0 if  A*exp(t/t0) is fitted and 1 if exp(t/t0) is fitted
    numW = FullLags.shape[0]
    num_n = FullLags.shape[2]
    fig, axes = plt.subplots(numW, num_n, figsize=(20, 10), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Loop over all combinations of n and d
    for n in range(num_n):
        for d in range(numW):
            ax = axes[d, n]  # Access the subplot

            # Extract data for current n and d
            temp = FullLags[d, L[d]:2*L[d]-1, n]
            autocorr = FullAutocorr[d, L[d]:2*L[d]-1, n]
            ax.scatter(temp, autocorr, label="Data", color="blue", alpha=0.6)
            if exp==0:
                popt2 = Time_constant[d, n, 1], Time_constant[d, n, 0]
                ax.plot(temp, exponential_decay(temp, *popt2), label="Fit", color="red", alpha=0.8,lw=4)
                tau_value = Time_constant[d, n, 0]
            if exp==1:
                popt2 = Time_constant[d, n],
                ax.plot(temp, exponential1_decay(temp, *popt2), label="Fit", color="red", alpha=0.8,lw=4)
                tau_value = Time_constant[d, n]

            # Add titles and grid
            ax.set_title(Neur_lab[n]+f", W={d}", fontsize=15)
            ax.grid(alpha=0.5)
            ax.set_ylim(-1, 1)
            ax.set_xlim(right= 1000)

            ax.text(0.3, 0.7, f"τ={tau_value:.2f}", transform=ax.transAxes, fontsize=10, color="black")
            # Remove ticks for better readability
            ax.tick_params(axis='both', which='major', labelsize=15)

    # Set shared labels
    fig.text(0.5, 0.04, 'Lag', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Autocorrelation', va='center', rotation='vertical', fontsize=20)

    return fig
