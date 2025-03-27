import pandas as pd
import os
print(os.getcwd())
import Functions.AllFunctions as AllF
import copy
import numpy as np

def make_Dataset(activity_bins,time_bins,behavior_bins,odor_bins,b,except_data=[100],sm=[1,1,1,1,1,1,1,1,1],bins=True):
    wrmNum = activity_bins.shape[0]
    filtered_range = [i for i in range(wrmNum) if i not in except_data]
    shifted_behavior = copy.deepcopy(behavior_bins)
    shifted_behavior2 = copy.deepcopy(behavior_bins)
    shifted_behavior3 = copy.deepcopy(behavior_bins)
    if bins:
        shifted_behavior[:,:-1] = shifted_behavior[:,1:]
        shifted_behavior[:,-1:] = 0

        shifted_behavior2[:,:-2] = shifted_behavior2[:,2:]
        shifted_behavior2[:,-2:] = 0

        shifted_behavior3[:,:-3] = shifted_behavior3[:,3:]
        shifted_behavior3[:,-3:] = 0

    else:
        for d in filtered_range:
            print('##########d:   '+str(d))
            shifted_behavior[d,:-sm[d]] = shifted_behavior[d,sm[d]:]
            shifted_behavior[d,-sm[d]:b[d]] = 0

            shifted_behavior2[d,:-(2*sm[d])] = shifted_behavior2[d,(2*sm[d]):]
            shifted_behavior2[d,-(2*sm[d]):b[d]] = 0
            print(shifted_behavior2[d,:])
            shifted_behavior3[d,:-(3*sm[d])] = shifted_behavior3[d,(3*sm[d]):]
            shifted_behavior3[d,-(3*sm[d]):b[d]] = 0

    dataAll = {}
    for d in filtered_range:
        if d==1:
            data1 = pd.DataFrame({
                'x0': (activity_bins[d,0,:b[d]]),
                'x1': (activity_bins[d,1,:b[d]]),
                'x2': (activity_bins[d,2,:b[d]]),
                'x3': (activity_bins[d,3,:b[d]]),
                'x4': (activity_bins[d,4,:b[d]]),
                'x5': (activity_bins[d,5,:b[d]]),
                'x6': (activity_bins[d,6,:b[d]]),
                'x7': (activity_bins[d,7,:b[d]]),
                'x8': (activity_bins[d,8,:b[d]]),
                'x27': (activity_bins[d,9,:b[d]]),
                'x28': (activity_bins[d,10,:b[d]]),
                'x39': (activity_bins[d,11,:b[d]]),
                'x40': (activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]#
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data1'] = data1
        if d==2:
            data2 = pd.DataFrame({
                'x0': (activity_bins[d,0,:b[d]]),
                'x1': (activity_bins[d,1,:b[d]]),
                'x2': (activity_bins[d,2,:b[d]]),
                'x3': (activity_bins[d,3,:b[d]]),
                'x4': (activity_bins[d,4,:b[d]]),
                'x5': (activity_bins[d,0,:b[d]]),
                'x6': (activity_bins[d,6,:b[d]]),
                'x7': (activity_bins[d,7,:b[d]]),
                'x8': (activity_bins[d,8,:b[d]]),
                'x27': (activity_bins[d,9,:b[d]]),
                'x28': (activity_bins[d,10,:b[d]]),
                'x39': (activity_bins[d,11,:b[d]]),
                'x40': (activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data2'] = data2
        if d==3:
            data3 = pd.DataFrame({
                'x0': (activity_bins[d,0,:b[d]]),
                'x1': (activity_bins[d,1,:b[d]]),
                'x2': (activity_bins[d,3,:b[d]]),
                'x3': (activity_bins[d,2,:b[d]]),
                'x4': (activity_bins[d,5,:b[d]]),
                'x5': (activity_bins[d,4,:b[d]]),
                'x6': (activity_bins[d,6,:b[d]]),
                'x7': (activity_bins[d,7,:b[d]]),
                'x8': (activity_bins[d,8,:b[d]]),
                'x27': (activity_bins[d,9,:b[d]]),
                'x28': (activity_bins[d,10,:b[d]]),
                'x39': (activity_bins[d,11,:b[d]]),
                'x40': (activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data3'] = data3
        if d==4:
            data4 = pd.DataFrame({
                'x0': (activity_bins[d,0,:b[d]]),
                'x1': (activity_bins[d,1,:b[d]]),
                'x2': (activity_bins[d,2,:b[d]]),
                'x3': (activity_bins[d,3,:b[d]]),
                'x4': (activity_bins[d,4,:b[d]]),
                'x5': (activity_bins[d,5,:b[d]]),
                'x6': (activity_bins[d,6,:b[d]]),
                'x7': (activity_bins[d,7,:b[d]]),
                'x8': (activity_bins[d,8,:b[d]]),
                'x27': (activity_bins[d,9,:b[d]]),
                'x28': (activity_bins[d,10,:b[d]]),
                'x39': (activity_bins[d,11,:b[d]]),
                'x40': (activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data4'] = data4
        if d==5:
            data5 = pd.DataFrame({
                'x0': (activity_bins[d,0,:b[d]]),
                'x1': (activity_bins[d,1,:b[d]]),
                'x2': (activity_bins[d,2,:b[d]]),
                'x3': (activity_bins[d,3,:b[d]]),
                'x4': (activity_bins[d,4,:b[d]]),
                'x5': (activity_bins[d,5,:b[d]]),
                'x6': (activity_bins[d,6,:b[d]]),
                'x7': (activity_bins[d,7,:b[d]]),
                'x8': (activity_bins[d,8,:b[d]]),
                'x27': (activity_bins[d,9,:b[d]]),
                'x28': (activity_bins[d,10,:b[d]]),
                'x39': (activity_bins[d,11,:b[d]]),
                'x40': (activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data5'] = data5

        if d==6:
            print('************d6')
            data6 = pd.DataFrame({
                'x0': (activity_bins[d,0,:b[d]]),
                'x1': (activity_bins[d,1,:b[d]]),
                'x2': (activity_bins[d,2,:b[d]]),
                'x3': (activity_bins[d,3,:b[d]]),
                'x4': (activity_bins[d,4,:b[d]]),
                'x5': (activity_bins[d,5,:b[d]]),
                'x6': (activity_bins[d,6,:b[d]]),
                'x7': (activity_bins[d,7,:b[d]]),
                'x8': (activity_bins[d,8,:b[d]]),
                'x27': (activity_bins[d,9,:b[d]]),
                'x28': (activity_bins[d,10,:b[d]]),
                'x39': (activity_bins[d,11,:b[d]]),
                'x40': (activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh':(shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data6'] = data6
    return dataAll


def make_Dataset_ZscoreAll(activity_bins,time_bins,behavior_bins,odor_bins,b,except_data=[100],sm=[1,1,1,1,1,1,1,1,1],bins=True):
    wrmNum = activity_bins.shape[0]
    filtered_range = [i for i in range(wrmNum) if i not in except_data]
    shifted_behavior = copy.deepcopy(behavior_bins)
    shifted_behavior2 = copy.deepcopy(behavior_bins)
    shifted_behavior3 = copy.deepcopy(behavior_bins)
    if bins:
        shifted_behavior[:,:-1] = shifted_behavior[:,1:]
        shifted_behavior[:,-1:] = 0

        shifted_behavior2[:,:-2] = shifted_behavior2[:,2:]
        shifted_behavior2[:,-2:] = 0

        shifted_behavior3[:,:-3] = shifted_behavior3[:,3:]
        shifted_behavior3[:,-3:] = 0

    else:
        for d in filtered_range:
            print('##########d:   '+str(d))
            shifted_behavior[d,:-sm[d]] = shifted_behavior[d,sm[d]:]
            shifted_behavior[d,-sm[d]:b[d]] = 0

            shifted_behavior2[d,:-(2*sm[d])] = shifted_behavior2[d,(2*sm[d]):]
            shifted_behavior2[d,-(2*sm[d]):b[d]] = 0
            print(shifted_behavior2[d,:])
            shifted_behavior3[d,:-(3*sm[d])] = shifted_behavior3[d,(3*sm[d]):]
            shifted_behavior3[d,-(3*sm[d]):b[d]] = 0

    dataAll = {}
    for d in filtered_range:
        if d==1:
            data1 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]]),
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]]),
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]]),
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]]),
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]]),
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]]),
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]]),
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]]),
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]]),
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]]),
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]]),
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]]),
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': AllF.compute_vector_z_score(behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]#
                'y_sh': AllF.compute_vector_z_score(shifted_behavior[d,:b[d]]),
                'y_sh2': AllF.compute_vector_z_score(shifted_behavior2[d,:b[d]]),
                'y_sh3': AllF.compute_vector_z_score(shifted_behavior3[d,:b[d]])
                })
            dataAll['data1'] = data1
        if d==2:
            data2 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]]),
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]]),
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]]),
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]]),
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]]),
                'x5': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]]),
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]]),
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]]),
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]]),
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]]),
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]]),
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]]),
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': AllF.compute_vector_z_score(behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': AllF.compute_vector_z_score(shifted_behavior[d,:b[d]]),
                'y_sh2': AllF.compute_vector_z_score(shifted_behavior2[d,:b[d]]),
                'y_sh3': AllF.compute_vector_z_score(shifted_behavior3[d,:b[d]])
                })
            dataAll['data2'] = data2
        if d==3:
            data3 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]]),
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]]),
                'x2': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]]),
                'x3': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]]),
                'x4': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]]),
                'x5': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]]),
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]]),
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]]),
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]]),
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]]),
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]]),
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]]),
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': AllF.compute_vector_z_score(behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': AllF.compute_vector_z_score(shifted_behavior[d,:b[d]]),
                'y_sh2': AllF.compute_vector_z_score(shifted_behavior2[d,:b[d]]),
                'y_sh3': AllF.compute_vector_z_score(shifted_behavior3[d,:b[d]])
                })
            dataAll['data3'] = data3
        if d==4:
            data4 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]]),
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]]),
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]]),
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]]),
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]]),
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]]),
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]]),
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]]),
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]]),
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]]),
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]]),
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]]),
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': AllF.compute_vector_z_score(behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': AllF.compute_vector_z_score(shifted_behavior[d,:b[d]]),
                'y_sh2': AllF.compute_vector_z_score(shifted_behavior2[d,:b[d]]),
                'y_sh3': AllF.compute_vector_z_score(shifted_behavior3[d,:b[d]])
                })
            dataAll['data4'] = data4
        if d==5:
            data5 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]]),
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]]),
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]]),
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]]),
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]]),
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]]),
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]]),
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]]),
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]]),
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]]),
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]]),
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]]),
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': AllF.compute_vector_z_score(behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': AllF.compute_vector_z_score(shifted_behavior[d,:b[d]]),
                'y_sh2': AllF.compute_vector_z_score(shifted_behavior2[d,:b[d]]),
                'y_sh3': AllF.compute_vector_z_score(shifted_behavior3[d,:b[d]])
                })
            dataAll['data5'] = data5

        if d==6:
            print('************d6')
            data6 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]]),
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]]),
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]]),
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]]),
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]]),
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]]),
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]]),
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]]),
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]]),
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]]),
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]]),
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]]),
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]]),
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': AllF.compute_vector_z_score(behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': AllF.compute_vector_z_score(shifted_behavior[d,:b[d]]),
                'y_sh2': AllF.compute_vector_z_score(shifted_behavior2[d,:b[d]]),
                'y_sh3': AllF.compute_vector_z_score(shifted_behavior3[d,:b[d]])
                })
            dataAll['data6'] = data6
    return dataAll

def make_Dataset_ZscoreActivity(activity_bins,time_bins,behavior_bins,odor_bins,b,except_data=[100],sm=[1,1,1,1,1,1,1,1,1],bins=True,logistic=False):
    wrmNum = activity_bins.shape[0]
    filtered_range = [i for i in range(wrmNum) if i not in except_data]
    shifted_behavior = copy.deepcopy(behavior_bins)
    shifted_behavior2 = copy.deepcopy(behavior_bins)
    shifted_behavior3 = copy.deepcopy(behavior_bins)
    if bins:
        shifted_behavior[:,:-1] = shifted_behavior[:,1:]
        shifted_behavior[:,-1:] = 0

        shifted_behavior2[:,:-2] = shifted_behavior2[:,2:]
        shifted_behavior2[:,-2:] = 0

        shifted_behavior3[:,:-3] = shifted_behavior3[:,3:]
        shifted_behavior3[:,-3:] = 0
    else:
        for d in filtered_range:
            print('##########d:   '+str(d))
            shifted_behavior[d,:-sm[d]] = shifted_behavior[d,sm[d]:]
            shifted_behavior[d,-sm[d]:] = 0

            shifted_behavior2[d,:-(2*sm[d])] = shifted_behavior2[d,(2*sm[d]):]
            shifted_behavior2[d,-(2*sm[d]):] = 0
            print(shifted_behavior2[d,300:])
            shifted_behavior3[d,:-(3*sm[d])] = shifted_behavior3[d,(3*sm[d]):]
            shifted_behavior3[d,-(3*sm[d]):] = 0

    scale =1
    dataAll = {}
    for d in filtered_range:
        if d==1:
            if not logistic:
                scale = np.nanpercentile(np.abs(behavior_bins[d,:b[d]]), 90)
            data1 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]])*scale,
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]])*scale,
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]])*scale,
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]])*scale,
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]])*scale,
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]])*scale,
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]])*scale,
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]])*scale,
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]])*scale,
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]])*scale,
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]])*scale,
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]])*scale,
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]])*scale,
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]#
                'y_sh':(shifted_behavior[d,:b[d]]),
                'y_sh2':(shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data1'] = data1
        if d==2:
            if not logistic:
                scale = np.nanpercentile(np.abs(behavior_bins[d,:b[d]]), 90)
            data2 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]])*scale,
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]])*scale,
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]])*scale,
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]])*scale,
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]])*scale,
                'x5': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]])*scale,
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]])*scale,
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]])*scale,
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]])*scale,
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]])*scale,
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]])*scale,
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]])*scale,
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]])*scale,
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data2'] = data2
        if d==3:
            if not logistic:
                scale = np.nanpercentile(np.abs(behavior_bins[d,:b[d]]), 90)
            data3 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]])*scale,
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]])*scale,
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]])*scale,
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]])*scale,
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]])*scale,
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]])*scale,
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]])*scale,
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]])*scale,
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]])*scale,
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]])*scale,
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]])*scale,
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]])*scale,
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]])*scale,
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data3'] = data3
        if d==4:
            if not logistic:
                scale = np.nanpercentile(np.abs(behavior_bins[d,:b[d]]), 90)
            data4 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]])*scale,
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]])*scale,
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]])*scale,
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]])*scale,
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]])*scale,
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]])*scale,
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]])*scale,
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]])*scale,
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]])*scale,
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]])*scale,
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]])*scale,
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]])*scale,
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]])*scale,
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data4'] = data4
        if d==5:
            if not logistic:
                scale = np.nanpercentile(np.abs(behavior_bins[d,:b[d]]), 90)
            data5 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]])*scale,
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]])*scale,
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]])*scale,
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]])*scale,
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]])*scale,
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]])*scale,
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]])*scale,
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]])*scale,
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]])*scale,
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]])*scale,
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]])*scale,
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]])*scale,
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]])*scale,
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data5'] = data5

        if d==6:
            if not logistic:
                scale = np.nanpercentile(np.abs(behavior_bins[d,:b[d]]), 90)
            data6 = pd.DataFrame({
                'x0': AllF.compute_vector_z_score(activity_bins[d,0,:b[d]])*scale,
                'x1': AllF.compute_vector_z_score(activity_bins[d,1,:b[d]])*scale,
                'x2': AllF.compute_vector_z_score(activity_bins[d,2,:b[d]])*scale,
                'x3': AllF.compute_vector_z_score(activity_bins[d,3,:b[d]])*scale,
                'x4': AllF.compute_vector_z_score(activity_bins[d,4,:b[d]])*scale,
                'x5': AllF.compute_vector_z_score(activity_bins[d,5,:b[d]])*scale,
                'x6': AllF.compute_vector_z_score(activity_bins[d,6,:b[d]])*scale,
                'x7': AllF.compute_vector_z_score(activity_bins[d,7,:b[d]])*scale,
                'x8': AllF.compute_vector_z_score(activity_bins[d,8,:b[d]])*scale,
                'x27': AllF.compute_vector_z_score(activity_bins[d,9,:b[d]])*scale,
                'x28': AllF.compute_vector_z_score(activity_bins[d,10,:b[d]])*scale,
                'x39': AllF.compute_vector_z_score(activity_bins[d,11,:b[d]])*scale,
                'x40': AllF.compute_vector_z_score(activity_bins[d,12,:b[d]])*scale,
                'Odor': odor_bins[d,:b[d]],
                'time': time_bins[d,:b[d]],
                'y': (behavior_bins[d,:b[d]]),#ForwVelocity_bin[d,:b[d]]#speed_bins[d,:b[d]]#Velocity_bin[d,:b[d]]
                'y_sh': (shifted_behavior[d,:b[d]]),
                'y_sh2': (shifted_behavior2[d,:b[d]]),
                'y_sh3': (shifted_behavior3[d,:b[d]])
                })
            dataAll['data6'] = data6
    return dataAll


def Dataset_filtered(dataAll,nonzero_columns=[ 'x2', 'x3', 'x4', 'x5', 'y'],exceptional2=True):
    filtered_dataAll = {}
    for key in dataAll.keys():
        data = dataAll[key]
        if key == 'data2':
            mask = (data[[ 'x2', 'x3', 'x4', 'x0', 'y']] != 0).all(axis=1)
        else:
            mask = (data[nonzero_columns] != 0).all(axis=1)
        filtered_data = data.loc[mask]
        filtered_dataAll[key] = filtered_data

    return filtered_dataAll

def Dataset_filtered_0andNans(dataAll, nonzero_columns=['x2', 'x3', 'x4', 'x5', 'y'], exceptional2=True):
    filtered_dataAll = {}
    for key in dataAll.keys():

        data = dataAll[key]
        if key == 'data2':
            # Exclude rows with zeros or NaNs in the specified columns
            mask = (data[['x2', 'x3', 'x4', 'x0', 'y']] != 0).all(axis=1) & (~data[['x2', 'x3', 'x4', 'x0', 'y']].isna().any(axis=1))
        else:
            # Exclude rows with zeros or NaNs in the specified columns
            mask = (data[nonzero_columns] != 0).all(axis=1) & (~data[nonzero_columns].isna().any(axis=1))

        # Filter the data using the mask
        filtered_data = data.loc[mask]
        filtered_dataAll[key] = filtered_data

    return filtered_dataAll
