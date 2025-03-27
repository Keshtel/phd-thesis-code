import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import Functions.AllFunctions as AllF

def bin_and_plot(X,Y,bins=10,label='',plot=False):
    #bins X and plots average of Y in each bin
    # Creating a DataFrame
    data = {
    't': X,
    'a': Y,
        }

    df = pd.DataFrame(data)
    df['v'] = pd.cut(df['t'], bins=bins)

    # Calculate the mean of 'activity' in each bin
    mean_activity_by_bin = df.groupby('v')['a'].mean()
    mean_time_by_bin = df.groupby('v')['t'].mean()

    # Extract bin edges and calculate bin width
    bin_edges = pd.cut(df['t'], bins=bins).unique()
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

def bin_and_plot_majority(X, Y, bins=10, label='', plot=False):
    # Bins X and calculates majority of Y in each bin
    # Creating a DataFrame
    data = {
        't': X,
        'a': Y,
    }
    df = pd.DataFrame(data)
    df['v'] = pd.cut(df['t'], bins=bins)

    # Calculate the majority value (1 or 0) of 'activity' in each bin
    majority_activity_by_bin = (
        df.groupby('v')['a']
        .apply(lambda x: 1 if ((x[~np.isnan(x)]).sum() > ((~np.isnan(x)).sum()/ 2)) else 0)
    )
    mean_time_by_bin = df.groupby('v')['t'].mean()

    # Extract bin edges and calculate bin width
    bin_edges = pd.cut(df['t'], bins=bins).unique()
    bin_width = bin_edges[0].right - bin_edges[0].left

    if plot:
        # Make the plot larger
        plt.figure(figsize=(12, 8))
        # Plotting the majority activity in each bin
        plt.bar(mean_time_by_bin, majority_activity_by_bin, width=bin_width, align='center')
        plt.title(label)
        plt.xlabel('time Bins')
        plt.ylabel('Majority Activity (1 or 0)')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    return df, mean_time_by_bin, majority_activity_by_bin, bin_width

def bin_and_plot_checkNan(X,Y,bins=10,label='',plot=False):
    #bins X and plots average of Y in each bin
    # Creating a DataFrame
    data = {
    't': X,
    'a': Y,
        }

    df = pd.DataFrame(data)
    df['v'] = pd.cut(df['t'], bins=bins)

    # Calculate the mean of 'activity' in each bin
    mean_activity_by_bin = df.groupby('v')['a'].mean()
    mean_time_by_bin = df.groupby('v')['t'].mean()

    #Identify bins where all values are NaN
    bins_with_all_nan = df.groupby('v').apply(lambda group: group['a'].isna().all())

    mean_activity_by_bin[bins_with_all_nan] = np.nan

    bin_edges = pd.cut(df['t'], bins=bins).unique()
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

def compute_2rows(row10, row20):
    row1 = copy.deepcopy(row10)
    row2 = copy.deepcopy(row20)
    row1[np.isnan(row1)] = 0
    row2[np.isnan(row2)] = 0
    time_intersection = (np.abs(row1)>0)&(np.abs(row2)>0)
    if (np.sum(time_intersection)) > 2:
        correlation = np.corrcoef(row1[time_intersection], (row2[time_intersection]))[0, 1]
        correlation2 = scipy.stats.pearsonr(((row1[time_intersection])), row2[time_intersection])[0]
    else:
        correlation = 0
        correlation2 = 0
        print("no simultaneous data")
    return correlation,correlation2


def compute_vector_z_score(vector,exclude_zero=True):

    if not exclude_zero:
        mean = np.mean(vector)
        std=np.std(vector)
        vector_zscore= (vector-mean)/std
    else:
        vector2 = copy.deepcopy(vector)
        vector2[np.isnan(vector2)]=0
        vector_zscore = np.zeros(np.shape(vector))
        mean = np.nanmean(vector[np.nonzero(vector)[0]])
        std=np.nanstd(vector[np.nonzero(vector)[0]])
        vector_zscore[np.nonzero(vector2)[0]] = (vector[np.nonzero(vector2)[0]]-mean)/std
    return vector_zscore

def compute_vector_z_scoreNan(vector,exclude_zero=True):

    if not exclude_zero:
        mean = np.mean(vector)
        std=np.std(vector)
        vector_zscore= (vector-mean)/std
    else:
        vector2 = copy.deepcopy(vector)
        vector2[np.isnan(vector2)]=0
        vector_zscore = np.zeros(np.shape(vector))
        vector_zscore[vector_zscore==0]=np.nan
        mean = np.nanmean(vector[np.nonzero(vector)[0]])
        std=np.nanstd(vector[np.nonzero(vector)[0]])
        vector_zscore[np.nonzero(vector2)[0]] = (vector[np.nonzero(vector2)[0]]-mean)/std
    return vector_zscore

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


def Compute_bins(MegaData,b,L,neurons,forward_vec,backward_vec,speed_vec,turn_vec,exceptt=0,except_data=[0]):
    max_neurons = len(neurons)
    dataset_num = len(b)
    if not exceptt:
        filtered_range = range(dataset_num)
    else:
        filtered_range = [i for i in range(dataset_num) if i not in except_data]
    corr_neuronsRI_ind_Vel1 = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_Vel2 = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_curv = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_acc = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_head = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_theta = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_turn = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_speed = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_Back = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_Forward = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_BackFreq = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_ForwardFreq = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_head = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_headAngle = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_headAngle2 = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind= np.zeros((dataset_num,max_neurons,max_neurons))


    activity_bins = np.zeros((dataset_num,max_neurons,np.max(b)))
    forward_bins = np.zeros((dataset_num,np.max(b)))
    Velocity_bin = np.zeros((dataset_num,np.max(b)))
    backward_bins = np.zeros((dataset_num,np.max(b)))
    backwardFreq_bins = np.zeros((dataset_num,np.max(b)))
    forwardFreq_bins = np.zeros((dataset_num,np.max(b)))

    odor_bins = np.zeros((dataset_num,np.max(b)))
    speed_bins = np.zeros((dataset_num,np.max(b)))
    acceleration_bins = np.zeros((dataset_num,np.max(b)))
    head_bins = np.zeros((dataset_num,np.max(b)))
    headAngle_bins = np.zeros((dataset_num,np.max(b)))
    headAngle2_bins = np.zeros((dataset_num,np.max(b)))
    headVector_bins = np.zeros((dataset_num,np.max(b)))
    head_bins_signed= np.zeros((dataset_num,np.max(b)))
    highspeed_bins = np.zeros((dataset_num,np.max(b)))
    speed_bins_im = np.zeros((dataset_num,np.max(b)))
    theta_bins = np.zeros((dataset_num,np.max(b)))
    turn_bins = np.zeros((dataset_num,np.max(b)))
    time_bins = np.zeros((dataset_num,np.max(b)))
    width_bins = np.zeros((dataset_num,np.max(b)))
    time_binsA = np.zeros((dataset_num,np.max(b)))
    curve_bins = np.zeros((dataset_num,np.max(b)))

    ForwVelocity_bin = np.zeros((dataset_num,np.max(b)))
    BackwVelocity_bin = np.zeros((dataset_num,np.max(b)))

    time2bin = np.zeros((dataset_num,np.max(b)))
    width2bin = np.zeros((dataset_num,np.max(b)))

    for d in filtered_range:
        print("****d*****"+str(d))
        VelF = forward_vec[d,:L[d]]
        VelB = backward_vec[d,:L[d]]
        speed = speed_vec[d,:L[d]]
        Velthet = turn_vec[d,:L[d]]
        x0 = [0,1]
        VecHead = AllF.cross_product_vector_angle(MegaData[d,:L[d],49:51], x0)
        dfF,time_bins[d,:b[d]], mean_forward_by_bin,width_bins[d,:b[d]] = bin_and_plot(MegaData[d,:L[d],12],VelF,bins=b[d])
        dfB,mean_time_by_bin, mean_backward_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],VelB,bins=b[d])
        dfsi,mean_time_by_bin, highspeed_bins[d,:b[d]],width = bin_and_plot_majority(MegaData[d,:L[d],12],speed,bins=b[d])#ratio of high speed movement
        nonz_theta = np.nonzero(~np.isnan(MegaData[d,:L[d],46]))
        #dfTh,mean_time_by_bin, mean_theta_by_bin,width = bin_and_plot(MegaData[d,nonz_theta[0],12],MegaData[d,nonz_theta[0],46],bins=b[d])#average theta
        dfTh,mean_time_by_bin, mean_theta_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],46],bins=b[d])#average theta
        dfC,mean_time_by_bin, mean_curve_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],26],bins=b[d])#average head curvature
        dfA,mean_time_by_bin, mean_acc_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],48],bins=b[d])#average acceleration
        dfVcen,mean_time_by_bin, mean_VcenSigned_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],42],bins=b[d])#mean velocity
        nonz_head = np.nonzero(~np.isnan(MegaData[d,:L[d],25]))
        dfTh,mean_time_by_bin, mean_headV_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],25],bins=b[d])#head velocity
        #dfTh,mean_time_by_bin, mean_headV_by_bin,width = bin_and_plot(MegaData[d,nonz_head[0],12],MegaData[d,nonz_head[0],25],bins=b[d])#head velocity
        dfTh,mean_time_by_bin, mean_headVsigned_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],22],bins=b[d])

        dfTh,mean_time_by_bin, headAngle_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],38],bins=b[d])
        dfTh,mean_time_by_bin, headAngle2_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],37],bins=b[d])
        dfTh,mean_time_by_bin, headVector_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],VecHead,bins=b[d])

        dfTh,mean_time_by_bin, mean_turn_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],Velthet,bins=b[d])
        dfod,time_bins[d,:b[d]], mean_odor_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],10],bins=b[d])
        nonz_speed = np.nonzero(~np.isnan(MegaData[d,:L[d],15]))
        #dfsp,mean_time_by_bin, speed_bins[d,:b[d]],width = bin_and_plot(MegaData[d,nonz_speed[0],12],MegaData[d,nonz_speed[0],15],bins=b[d])#speed value
        dfsp,mean_time_by_bin, speed_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],15],bins=b[d])#speed value
        odor_bins[d,:b[d]] = mean_odor_by_bin

        theta_bins[d,:b[d]] = mean_theta_by_bin
        turn_bins[d,:b[d]] = mean_turn_by_bin
        curve_bins[d,:b[d]] = mean_curve_by_bin
        acceleration_bins[d,:b[d]] = mean_acc_by_bin
        Velocity_bin[d,:b[d]] = mean_VcenSigned_by_bin

        backwardFreq_bins[d,:b[d]] = mean_backward_by_bin
        forwardFreq_bins[d,:b[d]] = mean_forward_by_bin

        head_bins[d,:b[d]] = mean_headV_by_bin#interpolate_nanValues(mean_headV_by_bin,mean_headV_by_bin)
        head_bins_signed[d,:b[d]] = mean_headVsigned_by_bin#interpolate_nanValues(mean_headVsigned_by_bin,mean_headVsigned_by_bin)
        MegaData_2 = copy.deepcopy(MegaData)
        MegaData_2[np.isnan(MegaData_2)]=0
        rowsArr_float = copy.deepcopy(MegaData_2[d,:L[d],:])
        for j in range(len(neurons)):
            n = neurons[j]
            activity = rowsArr_float[:,n]
            if len(np.nonzero(rowsArr_float[:,n])[0])>1:
                activity[np.isinf(activity)] = 0
                nonz = np.nonzero(activity)
                df2,TimeAct,activity_bins[d,j,:b[d]],width = bin_and_plot(MegaData[d,nonz[0],12],compute_vector_z_score(activity[nonz[0]]),bins=b[d])
                if n==1:
                    time_binsA[d,:b[d]] = TimeAct
        rowsArr_float = copy.deepcopy(MegaData_2[d,:L[d],:])
        dfVcen,time2bin[d,:b[d]], mean_VcenSigned_by_bin,width2bin[d,:b[d]] = bin_and_plot(MegaData[d,:L[d],12],MegaData_2[d,:L[d],42],bins=b[d])#mean velocity
        for i in range(len(mean_VcenSigned_by_bin)):
            if (mean_VcenSigned_by_bin.iloc[i]) < 0:
                mean_VcenSigned_by_bin.iloc[i] = 0
        ForwVelocity_bin[d,:b[d]] = mean_VcenSigned_by_bin
        dfVcen,mean_time_by_bin, mean_VcenSigned_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],42],bins=b[d])#mean velocity
        for i in range(len(mean_VcenSigned_by_bin)):
            if (mean_VcenSigned_by_bin.iloc[i]) > 0:
                mean_VcenSigned_by_bin.iloc[i] = 0
        BackwVelocity_bin[d,:b[d]] = mean_VcenSigned_by_bin
        for n0 in range(len(neurons)):
            corr_neuronsRI_ind_Vel1[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], Velocity_bin[d,:b[d]])
            corr_neuronsRI_ind_speed[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], speed_bins[d,:b[d]])
            corr_neuronsRI_ind_theta[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], theta_bins[d,:b[d]])
            corr_neuronsRI_ind_turn[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], turn_bins[d,:b[d]])
            corr_neuronsRI_ind_acc[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], acceleration_bins[d,:b[d]])
            corr_neuronsRI_ind_curv[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], curve_bins[d,:b[d]])
            corr_neuronsRI_ind_Back[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], BackwVelocity_bin[d,:b[d]])
            corr_neuronsRI_ind_Forward[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], ForwVelocity_bin[d,:b[d]])
            corr_neuronsRI_ind_head[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], head_bins_signed[d,:b[d]])
            corr_neuronsRI_ind_ForwardFreq[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], forwardFreq_bins[d,:b[d]])
            corr_neuronsRI_ind_BackFreq[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], backwardFreq_bins[d,:b[d]])
            for n1 in range(len(neurons)):
                corr_neuronsRI_ind[d,n0,n1],x = compute_2rows(activity_bins[d,n0,:b[d]], activity_bins[d,n1,:b[d]])

    results= {
                'ForwardVel': ForwVelocity_bin,
                'BackwardVel': BackwVelocity_bin,
                'ForwardFreq':forwardFreq_bins,
                'BackwardFreq':backwardFreq_bins,
                'Speed': speed_bins,
                'Theta': theta_bins,
                'Turn': turn_bins,
                'Curve': curve_bins,
                'Velocity': Velocity_bin,
                'Odor':odor_bins,
                'Acceleration': acceleration_bins,
                'time': time_bins,
                'timeA': time_binsA,
                'width': width_bins,
                'head': head_bins,
                'head_signed': head_bins_signed,
                'head_angle':headAngle_bins,
                'head_angle2':headAngle2_bins,
                'head_vector':headVector_bins,
                'activity': activity_bins,
                'highspeed':highspeed_bins
            }
    results_correlation = {
                            'Vel1':corr_neuronsRI_ind_Vel1,
                            'speed':corr_neuronsRI_ind_speed,
                            'theta':corr_neuronsRI_ind_theta,
                            'turn':corr_neuronsRI_ind_turn,
                            'acc':corr_neuronsRI_ind_acc,
                            'curv':corr_neuronsRI_ind_curv,
                            'Back':corr_neuronsRI_ind_Back,
                            'Forward':corr_neuronsRI_ind_Forward,
                            'head':corr_neuronsRI_ind_head,
                            'ForwardFreq':corr_neuronsRI_ind_ForwardFreq,
                            'BackFreq':corr_neuronsRI_ind_BackFreq,
                            'neurons':corr_neuronsRI_ind
                        }

    return results_correlation, results

def Compute_binsWnans(MegaData,b,L,neurons,forward_vec,backward_vec,speed_vec,turn_vec,exceptt=0,except_data=[0]):
    max_neurons = len(neurons)
    dataset_num = len(b)
    if not exceptt:
        filtered_range = range(dataset_num)
    else:
        filtered_range = [i for i in range(dataset_num) if i not in except_data]
    corr_neuronsRI_ind_Vel1 = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_Vel2 = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_curv = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_acc = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_head = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_theta = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_turn = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_speed = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_Back = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_Forward = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_BackFreq = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_ForwardFreq = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_head = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_headAngle = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind_headAngle2 = np.zeros((dataset_num,max_neurons))
    corr_neuronsRI_ind= np.zeros((dataset_num,max_neurons,max_neurons))


    activity_bins = np.zeros((dataset_num,max_neurons,np.max(b)))
    forward_bins = np.zeros((dataset_num,np.max(b)))
    Velocity_bin = np.zeros((dataset_num,np.max(b)))
    backward_bins = np.zeros((dataset_num,np.max(b)))
    backwardFreq_bins = np.zeros((dataset_num,np.max(b)))
    forwardFreq_bins = np.zeros((dataset_num,np.max(b)))

    odor_bins = np.zeros((dataset_num,np.max(b)))
    speed_bins = np.zeros((dataset_num,np.max(b)))
    acceleration_bins = np.zeros((dataset_num,np.max(b)))
    head_bins = np.zeros((dataset_num,np.max(b)))
    headAngle_bins = np.zeros((dataset_num,np.max(b)))
    headAngle2_bins = np.zeros((dataset_num,np.max(b)))
    headVector_bins = np.zeros((dataset_num,np.max(b)))
    head_bins_signed= np.zeros((dataset_num,np.max(b)))
    highspeed_bins = np.zeros((dataset_num,np.max(b)))
    speed_bins_im = np.zeros((dataset_num,np.max(b)))
    theta_bins = np.zeros((dataset_num,np.max(b)))
    turn_bins = np.zeros((dataset_num,np.max(b)))
    time_bins = np.zeros((dataset_num,np.max(b)))
    width_bins = np.zeros((dataset_num,np.max(b)))
    time_binsA = np.zeros((dataset_num,np.max(b)))
    curve_bins = np.zeros((dataset_num,np.max(b)))

    ForwVelocity_bin = np.zeros((dataset_num,np.max(b)))
    BackwVelocity_bin = np.zeros((dataset_num,np.max(b)))

    time2bin = np.zeros((dataset_num,np.max(b)))
    width2bin = np.zeros((dataset_num,np.max(b)))

    for d in filtered_range:
        print("****d*****"+str(d))
        VelF = forward_vec[d,:L[d]]
        VelB = backward_vec[d,:L[d]]
        speed = speed_vec[d,:L[d]]
        Velthet = turn_vec[d,:L[d]]
        x0 = [0,1]
        VecHead = AllF.cross_product_vector_angle(MegaData[d,:L[d],49:51], x0)
        dfF,time_bins[d,:b[d]], mean_forward_by_bin,width_bins[d,:b[d]] = bin_and_plot(MegaData[d,:L[d],12],VelF,bins=b[d])
        dfB,mean_time_by_bin, mean_backward_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],VelB,bins=b[d])
        dfsi,mean_time_by_bin, highspeed_bins[d,:b[d]],width = bin_and_plot_majority(MegaData[d,:L[d],12],speed,bins=b[d])#ratio of high speed movement
        nonz_theta = np.nonzero(~np.isnan(MegaData[d,:L[d],46]))
        #dfTh,mean_time_by_bin, mean_theta_by_bin,width = bin_and_plot(MegaData[d,nonz_theta[0],12],MegaData[d,nonz_theta[0],46],bins=b[d])#average theta
        dfTh,mean_time_by_bin, mean_theta_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],46],bins=b[d])#average theta
        dfC,mean_time_by_bin, mean_curve_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],26],bins=b[d])#average head curvature
        dfA,mean_time_by_bin, mean_acc_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],48],bins=b[d])#average acceleration
        dfVcen,mean_time_by_bin, mean_VcenSigned_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],42],bins=b[d])#mean velocity
        nonz_head = np.nonzero(~np.isnan(MegaData[d,:L[d],25]))
        #dfTh,mean_time_by_bin, mean_headV_by_bin,width = bin_and_plot(MegaData[d,nonz_head[0],12],MegaData[d,nonz_head[0],25],bins=b[d])#head velocity
        dfTh,mean_time_by_bin, mean_headV_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],25],bins=b[d])#head velocity
        dfTh,mean_time_by_bin, mean_headVsigned_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],22],bins=b[d])

        dfTh,mean_time_by_bin, headAngle_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],38],bins=b[d])
        dfTh,mean_time_by_bin, headAngle2_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],37],bins=b[d])
        dfTh,mean_time_by_bin, headVector_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],VecHead,bins=b[d])

        dfTh,mean_time_by_bin, mean_turn_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],Velthet,bins=b[d])
        dfod,time_bins[d,:b[d]], mean_odor_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],10],bins=b[d])
        nonz_speed = np.nonzero(~np.isnan(MegaData[d,:L[d],15]))
        #dfsp,mean_time_by_bin, speed_bins[d,:b[d]],width = bin_and_plot(MegaData[d,nonz_speed[0],12],MegaData[d,nonz_speed[0],15],bins=b[d])#speed value
        dfsp,mean_time_by_bin, speed_bins[d,:b[d]],width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],15],bins=b[d])#speed value
        odor_bins[d,:b[d]] = mean_odor_by_bin

        theta_bins[d,:b[d]] = mean_theta_by_bin
        turn_bins[d,:b[d]] = mean_turn_by_bin
        curve_bins[d,:b[d]] = mean_curve_by_bin
        acceleration_bins[d,:b[d]] = mean_acc_by_bin
        Velocity_bin[d,:b[d]] = mean_VcenSigned_by_bin

        backwardFreq_bins[d,:b[d]] = mean_backward_by_bin
        forwardFreq_bins[d,:b[d]] = mean_forward_by_bin

        head_bins[d,:b[d]] = mean_headV_by_bin#interpolate_nanValues(mean_headV_by_bin,mean_headV_by_bin)
        head_bins_signed[d,:b[d]] = mean_headVsigned_by_bin#interpolate_nanValues(mean_headVsigned_by_bin,mean_headVsigned_by_bin)
        MegaData_2 = copy.deepcopy(MegaData)
        MegaData_2[np.isnan(MegaData_2)]=0
        rowsArr_float = copy.deepcopy(MegaData[d,:L[d],:])
        for j in range(len(neurons)):
            n = neurons[j]
            activity = rowsArr_float[:,n]
            if len(np.nonzero(rowsArr_float[:,n])[0])>1:
                #activity[np.isinf(activity)] = 0
                nonz = np.nonzero(activity)
                df2,TimeAct,activity_bins[d,j,:b[d]],width = bin_and_plot_checkNan(MegaData[d,:L[d],12],compute_vector_z_scoreNan(activity[:L[d]]),bins=b[d])
                #df2,TimeAct,activity_bins[d,j,:b[d]],width = bin_and_plot(MegaData[d,nonz[0],12],compute_vector_z_score(activity[nonz[0]]),bins=b[d])
                if n==1:
                    time_binsA[d,:b[d]] = TimeAct

        rowsArr_float = copy.deepcopy(MegaData_2[d,:L[d],:])
        dfVcen,time2bin[d,:b[d]], mean_VcenSigned_by_bin,width2bin[d,:b[d]] = bin_and_plot(MegaData[d,:L[d],12],MegaData_2[d,:L[d],42],bins=b[d])#mean velocity

        for i in range(len(mean_VcenSigned_by_bin)):
            if (mean_VcenSigned_by_bin.iloc[i]) < 0:
                mean_VcenSigned_by_bin.iloc[i] = 0
        ForwVelocity_bin[d,:b[d]] = mean_VcenSigned_by_bin
        dfVcen,mean_time_by_bin, mean_VcenSigned_by_bin,width = bin_and_plot(MegaData[d,:L[d],12],MegaData[d,:L[d],42],bins=b[d])#mean velocity
        for i in range(len(mean_VcenSigned_by_bin)):
            if (mean_VcenSigned_by_bin.iloc[i]) > 0:
                mean_VcenSigned_by_bin.iloc[i] = 0
        BackwVelocity_bin[d,:b[d]] = mean_VcenSigned_by_bin
        for n0 in range(len(neurons)):
            corr_neuronsRI_ind_Vel1[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], Velocity_bin[d,:b[d]])
            corr_neuronsRI_ind_speed[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], speed_bins[d,:b[d]])
            corr_neuronsRI_ind_theta[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], theta_bins[d,:b[d]])
            corr_neuronsRI_ind_turn[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], turn_bins[d,:b[d]])
            corr_neuronsRI_ind_acc[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], acceleration_bins[d,:b[d]])
            corr_neuronsRI_ind_curv[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], curve_bins[d,:b[d]])
            corr_neuronsRI_ind_Back[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], BackwVelocity_bin[d,:b[d]])
            corr_neuronsRI_ind_Forward[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], ForwVelocity_bin[d,:b[d]])
            corr_neuronsRI_ind_head[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], head_bins_signed[d,:b[d]])
            corr_neuronsRI_ind_ForwardFreq[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], forwardFreq_bins[d,:b[d]])
            corr_neuronsRI_ind_BackFreq[d,n0],x = compute_2rows(activity_bins[d,n0,:b[d]], backwardFreq_bins[d,:b[d]])
            for n1 in range(len(neurons)):
                corr_neuronsRI_ind[d,n0,n1],x = compute_2rows(activity_bins[d,n0,:b[d]], activity_bins[d,n1,:b[d]])

    results= {
                'ForwardVel': ForwVelocity_bin,
                'BackwardVel': BackwVelocity_bin,
                'ForwardFreq':forwardFreq_bins,
                'BackwardFreq':backwardFreq_bins,
                'Speed': speed_bins,
                'Theta': theta_bins,
                'Turn': turn_bins,
                'Curve': curve_bins,
                'Velocity': Velocity_bin,
                'Odor':odor_bins,
                'Acceleration': acceleration_bins,
                'time': time_bins,
                'timeA': time_binsA,
                'width': width_bins,
                'head': head_bins,
                'head_signed': head_bins_signed,
                'head_angle':headAngle_bins,
                'head_angle2':headAngle2_bins,
                'head_vector':headVector_bins,
                'activity': activity_bins,
                'highspeed':highspeed_bins
            }
    results_correlation = {
                            'Vel1':corr_neuronsRI_ind_Vel1,
                            'speed':corr_neuronsRI_ind_speed,
                            'theta':corr_neuronsRI_ind_theta,
                            'turn':corr_neuronsRI_ind_turn,
                            'acc':corr_neuronsRI_ind_acc,
                            'curv':corr_neuronsRI_ind_curv,
                            'Back':corr_neuronsRI_ind_Back,
                            'Forward':corr_neuronsRI_ind_Forward,
                            'head':corr_neuronsRI_ind_head,
                            'ForwardFreq':corr_neuronsRI_ind_ForwardFreq,
                            'BackFreq':corr_neuronsRI_ind_BackFreq,
                            'neurons':corr_neuronsRI_ind
                        }

    return results_correlation, results


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
