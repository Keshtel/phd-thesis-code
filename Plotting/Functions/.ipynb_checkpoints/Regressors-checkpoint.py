import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import itertools

def regress_single_neuron(filtered_data,crossVal=1,indep_var = 1):
    num_datasets = len(filtered_data)    
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    column_names = tempData.columns
    print(column_names)
    combinations_of_4 = list(itertools.combinations(column_names, indep_var))  
    Rsqrt = np.zeros((len(combinations_of_4),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combinations_of_4),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combinations_of_4),num_datasets,crossVal,indep_var+1))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = train_test_split(data_temp,data_temp['y'], test_size=0.1)
            X_train0[key], X_test0[key], y_train0[key], y_test0[key] = X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp
        
        #train regressor on each worm 
        combcount=0
        for idx, combination in enumerate(combinations_of_4): 
            d_count = 0
            for key1 in list(filtered_dataAll.keys()): 
                X_train_temp = X_train0[key1]
                y_train_temp = y_train0[key1]
                X_test_temp = X_test0[key1]
                y_test_temp = y_test0[key1]
                print(combination)
                # Select the columns corresponding to the current combination
                X_train = X_train_temp[list(combination)]
                y_train = y_train_temp

                modelc = LinearRegression()
                modelc.fit(X_train1, y_train1)
                coeff[combcount,d_count,i,:indep_var]= modelc.coef_

                coeff[combcount,d_count,i,indep_var]= modelc.intercept_
                test_count = 0
                for key2 in list(filtered_dataAll.keys()):
                    X_train_temp = X_train0[key2]
                    y_train_temp = y_train0[key2]
                    # Select the columns corresponding to the current combination
                    X_train2 = X_train_temp[list(combination)]
                    y_train2 = y_train_temp
                
                    y_pred12 = modelc.predict(X_train2)
                    Rsqrt[combcount,d_count,test_count,i] = modelc.score(X_train2, y_train2)
                    msqrE[combcount,d_count,test_count,i] = mean_squared_error(y_pred12, y_train2)
                    test_count = test_count + 1 
                d_count = d_count + 1
            combcount = combcount +1
    return Rsqrt, msqrE, coeff,combinations_of_4