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
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

def compute_r_squared(y, yhat):
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - yhat) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def plot_2BarsBars(y_train20,y_pred0,time,Rsq,ylab,msqE='msqE',w=5,od=0,odorbin=[0,1,0,1]):
    plt.figure(figsize=(10, 3))
    y_pred = [float(y) for y in y_pred0]
    y_train2 = [float(y) for y in y_train20]
    plt.bar(np.array(time),np.array(y_train2),width=w)#, label='Actual')
    plt.bar(np.array(time),np.array(y_pred), alpha=0.6,width=w,label='predict')
    if od:
        plt.bar(np.array(time),np.array(odorbin), alpha=0.6,width=w,label='Odor')
    #plt.xlabel(msqE)
    plt.ylabel(ylab,fontsize=15)
    plt.title('Rsq: '+str(Rsq))
    plt.legend()
    plt.show()


def regress_single_neuron(filtered_dataAll,crossVal=1,indep_var = 1,dep='y',excluded_columns = ['odor', 'time']):
    num_datasets = len(filtered_dataAll)
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    column_names = tempData.drop(columns=excluded_columns).columns
    combinations_of_4 = list(itertools.combinations(column_names, indep_var))
    Rsqrt = np.zeros((len(combinations_of_4),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combinations_of_4),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combinations_of_4),num_datasets,crossVal,indep_var+1))
    print(list(filtered_dataAll.keys()))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = train_test_split(data_temp,data_temp[dep], test_size=0.1)
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
                # Select the columns corresponding to the current combination
                X_train1 = X_train_temp[list(combination)]
                y_train1 = y_train_temp

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

def regress_neuron_knowncombo(filtered_dataAll,knowncombo,crossVal=1,dep='y',plotFig=False,testsize = 0.1):
    num_datasets = len(filtered_dataAll)
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    combo = list(knowncombo)
    length = [len(c) for c in knowncombo]
    indep_var = np.max(length)
    print("indep_var: "+str(indep_var ))
    print(list(filtered_dataAll.keys()))
    Rsqrt = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combo),num_datasets,crossVal,indep_var+1))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            if testsize>0:
                X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = train_test_split(data_temp,data_temp[dep], test_size=testsize)

            else:
                X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = data_temp, data_temp, data_temp[dep], data_temp[dep]
            X_train0[key], X_test0[key], y_train0[key], y_test0[key] = X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp

        #train regressor on each worm
        combcount=0
        if plotFig and i == int(crossVal-1):
            fig, axes = plt.subplots(len(length), 1, figsize=(5, 5*len(length)))
        for ind , combination in enumerate(combo):
            d_count = 0
            for key1 in list(filtered_dataAll.keys()):
                X_train_temp = X_train0[key1]
                y_train_temp = y_train0[key1]
                X_test_temp = X_test0[key1]
                y_test_temp = y_test0[key1]
                # Select the columns corresponding to the current combination
                X_train1 = X_train_temp[list(combination)]
                y_train1 = y_train_temp

                modelc = LinearRegression()
                modelc.fit(X_train1, y_train1)
                coeff[combcount,d_count,i,:length[ind]]= modelc.coef_

                coeff[combcount,d_count,i,length[ind]]= modelc.intercept_
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
            if i == int(crossVal-1):
                print(combination)
                if plotFig:
                    ax = axes[ind]
                    X_test1 = X_test_temp[list(combination)]
                    y_test1 = y_test_temp
                    titre = ''
                    for t in range(len(combination)):
                        titre = titre+combination[t]
                    ax = plot_scatter_regression(modelc,X_train1, y_train1,X_test1, y_test1,ax,title=titre)
        if plotFig and i == int(crossVal-1):
            plt.tight_layout()
            plt.show()
    return Rsqrt, msqrE, coeff,combo


def regress_neuron_knowncombo_testeval(filtered_dataAll,knowncombo,crossVal=1,dep='y',plotFig=False,testsize=0.1):
    num_datasets = len(filtered_dataAll)
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    combo = list(knowncombo)
    length = [len(c) for c in knowncombo]
    indep_var = np.max(length)
    print("indep_var: "+str(indep_var ))
    Rsqrt = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combo),num_datasets,crossVal,indep_var+1))
    print(list(filtered_dataAll.keys()))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = train_test_split(data_temp,data_temp[dep], test_size=testsize)
            X_train0[key], X_test0[key], y_train0[key], y_test0[key] = X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp

        #train regressor on each worm
        combcount=0
        if plotFig and i == int(crossVal-1):
            fig, axes = plt.subplots(len(length), 1, figsize=(5, 5*len(length)))
        for ind , combination in enumerate(combo):
            if False:#i==5:
                print('Combination')
                print(combination)
                print(combcount)
            d_count = 0
            for key1 in list(filtered_dataAll.keys()):

                X_train_temp = X_train0[key1]
                y_train_temp = y_train0[key1]
                X_test_temp = X_test0[key1]
                y_test_temp = y_test0[key1]
                # Select the columns corresponding to the current combination
                X_train1 = X_train_temp[list(combination)]
                y_train1 = y_train_temp

                modelc = LinearRegression()
                modelc.fit(X_train1, y_train1)
                coeff[combcount,d_count,i,:length[ind]]= modelc.coef_

                coeff[combcount,d_count,i,length[ind]]= modelc.intercept_
                test_count = 0
                for key2 in list(filtered_dataAll.keys()):

                    X_train_temp = X_test0[key2]# these two lines are differe
                    y_train_temp = y_test0[key2]
                    # Select the columns corresponding to the current combination
                    X_train2 = X_train_temp[list(combination)]
                    y_train2 = y_train_temp

                    y_pred12 = modelc.predict(X_train2)
                    Rsqrt[combcount,d_count,test_count,i] = modelc.score(X_train2, y_train2)
                    msqrE[combcount,d_count,test_count,i] = mean_squared_error(y_pred12, y_train2)
                    test_count = test_count + 1
                d_count = d_count + 1
            combcount = combcount +1
            if i == int(crossVal-1):
                print(combination)
                if plotFig:
                    ax = axes[ind]
                    X_test1 = X_test_temp [list(combination)]
                    y_test1 = y_test_temp
                    titre = ''
                    for t in range(len(combination)):
                        titre = titre+combination[t]
                    ax = plot_scatter_regression(modelc,X_train1, y_train1,X_test1, y_test1,ax,title=titre)
        if plotFig and i == int(crossVal-1):
            plt.tight_layout()
            plt.show()
    return Rsqrt, msqrE, coeff,combo


def regress_neuron_knowncombo_testeval_ordered(filtered_dataAll,knowncombo,crossVal=1,dep='y',plotFig=False,testsize=0.1,logis=False,RF=False):
    num_datasets = len(filtered_dataAll)
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    combo = list(knowncombo)
    length = [len(c) for c in knowncombo]
    indep_var = np.max(length)
    print("indep_var: "+str(indep_var ))
    Rsqrt = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combo),num_datasets,crossVal,indep_var+1))
    print(list(filtered_dataAll.keys()))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            n_samples = int(data_temp.shape[0])
            testlength = int(testsize*n_samples)
            start_idx = np.random.randint(0, n_samples - testlength + 1)
            end_idx = start_idx + testlength
            # Slice the training and testing sets
            X_test0_temp = data_temp.iloc[start_idx:end_idx]  # Test set features
            y_test0_temp = data_temp[dep].iloc[start_idx:end_idx]   # Test set target

            X_train0_temp = pd.concat([data_temp.iloc[:start_idx], data_temp.iloc[end_idx:]])
            y_train0_temp = pd.concat([data_temp[dep].iloc[:start_idx], data_temp[dep].iloc[end_idx:]])

            X_train0[key], X_test0[key], y_train0[key], y_test0[key] = X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp

        #train regressor on each worm
        combcount=0
        if plotFig and i == int(crossVal-1):
            fig, axes = plt.subplots(len(length), 1, figsize=(5, 5*len(length)))
        for ind , combination in enumerate(combo):
            if i==5:
                print('Combination')
                print(combination)
                print(combcount)
            d_count = 0
            for key1 in list(filtered_dataAll.keys()):
                X_train_temp = X_train0[key1]
                y_train_temp = y_train0[key1]
                X_test_temp = X_test0[key1]
                y_test_temp = y_test0[key1]
                # Select the columns corresponding to the current combination
                X_train1 = X_train_temp[list(combination)]
                y_train1 = y_train_temp

                if RF:
                    modelc = RandomForestClassifier(n_estimators=100, random_state=42)
                    modelc.fit(X_train1, y_train1)
                    coeff[combcount,d_count,i,:length[ind]]= modelc.feature_importances_
                elif logis:
                    print("applying logistic regression")
                    modelc =  LogisticRegression()
                    modelc.fit(X_train1, y_train1)
                    coeff[combcount,d_count,i,:length[ind]]= modelc.coef_
                    coeff[combcount,d_count,i,length[ind]]= modelc.intercept_
                else:
                    print("applying linear regression")
                    modelc = LinearRegression()
                    modelc.fit(X_train1, y_train1)
                    coeff[combcount,d_count,i,:length[ind]]= modelc.coef_
                    coeff[combcount,d_count,i,length[ind]]= modelc.intercept_

                test_count = 0
                for key2 in list(filtered_dataAll.keys()):
                    X_train_temp = X_test0[key2]# these two lines are differe
                    y_train_temp = y_test0[key2]
                    # Select the columns corresponding to the current combination
                    X_train2 = X_train_temp[list(combination)]
                    y_train2 = y_train_temp

                    y_pred12 = modelc.predict(X_train2)
                    if logis or RF:
                        Rsqrt[combcount,d_count,test_count,i] = accuracy_score( y_train2,y_pred12)
                    else:
                        print("error linear")
                        Rsqrt[combcount,d_count,test_count,i] = modelc.score(X_train2, y_train2)
                        msqrE[combcount,d_count,test_count,i] = mean_squared_error(y_pred12, y_train2)
                    test_count = test_count + 1
                d_count = d_count + 1
            combcount = combcount +1
            if i == int(crossVal-1):
                print(combination)
                if plotFig:
                    ax = axes[ind]
                    X_test1 = X_test_temp [list(combination)]
                    y_test1 = y_test_temp
                    titre = ''
                    for t in range(len(combination)):
                        titre = titre+combination[t]
                    ax = plot_scatter_regression(modelc,X_train1, y_train1,X_test1, y_test1,ax,title=titre)
        if plotFig and i == int(crossVal-1):
            plt.tight_layout()
            plt.show()
    return Rsqrt, msqrE, coeff,combo


def Regress_with_coeff(X_line,all_means):
    param_num = X_line.shape[1]
    coefficients = all_means[:param_num]
    intercept = all_means[param_num]
    y_line = np.zeros(np.shape(X_line)[0])
    y_line = np.dot(X_line,coefficients) + intercept
    return y_line

def plot_scatter_regression(model,X_train1, y_train1,X_test1, y_test1,ax,title='Regression test'):
    y_pred1 = model.predict(X_train1)
    x_line = np.linspace(min(y_train1.values), max(y_train1.values), 100)
    y_line = (x_line.reshape(-1, 1))

    y_predtest1 = model.predict(X_test1)
    x_linetest = np.linspace(min(y_test1.values), max(y_test1.values), 100)
    y_linetest = (x_linetest.reshape(-1, 1))
    # Plot the training and test data points
    ax.scatter(y_train1,y_pred1,  label='pedict_train', color='blue', alpha=0.7)
    ax.scatter(y_test1,y_predtest1,  label='Test Data', color='green', alpha=0.7)

    # Plot the regression line
    ax.plot(x_line, y_line, label='Regression Line', color='red')

    # Customize each subplot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('ground truth', fontsize=12)
    ax.set_ylabel('predict', fontsize=12)
    ax.legend(fontsize=10)
    return ax


################333Logistic regression
def regress_neuron_knowncombo_logistic(filtered_dataAll,knowncombo,crossVal=1,dep='y',plotFig=False,testsize = 0.1,RF=False):
    num_datasets = len(filtered_dataAll)
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    combo = list(knowncombo)
    length = [len(c) for c in knowncombo]
    indep_var = np.max(length)
    print("indep_var: "+str(indep_var ))
    print(list(filtered_dataAll.keys()))
    Rsqrt = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combo),num_datasets,crossVal,indep_var+1))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            if testsize>0:
                X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = train_test_split(data_temp,data_temp[dep], test_size=testsize)

            else:
                X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = data_temp, data_temp, data_temp[dep], data_temp[dep]
            X_train0[key], X_test0[key], y_train0[key], y_test0[key] = X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp

        #train regressor on each worm
        combcount=0
        if plotFig and i == int(crossVal-1):
            fig, axes = plt.subplots(len(length), 1, figsize=(5, 5*len(length)))
        for ind , combination in enumerate(combo):
            d_count = 0
            for key1 in list(filtered_dataAll.keys()):
                X_train_temp = X_train0[key1]
                y_train_temp = y_train0[key1]
                X_test_temp = X_test0[key1]
                y_test_temp = y_test0[key1]
                # Select the columns corresponding to the current combination
                X_train1 = X_train_temp[list(combination)]
                y_train1 = y_train_temp


                if RF:
                    modelc = RandomForestClassifier(n_estimators=100, random_state=42)#LogisticRegression()
                    modelc.fit(X_train1, y_train1)
                    coeff[combcount,d_count,i,:length[ind]]= modelc.feature_importances_#modelc.coef_
                else:
                    print("applying logistic regression")
                    modelc =  LogisticRegression()
                    modelc.fit(X_train1, y_train1)
                    coeff[combcount,d_count,i,:length[ind]]= modelc.coef_
                    coeff[combcount,d_count,i,length[ind]]= modelc.intercept_

                test_count = 0
                for key2 in list(filtered_dataAll.keys()):
                    X_train_temp = X_train0[key2]
                    y_train_temp = y_train0[key2]
                    # Select the columns corresponding to the current combination
                    X_train2 = X_train_temp[list(combination)]
                    y_train2 = y_train_temp

                    y_pred12 = modelc.predict(X_train2)
                    Rsqrt[combcount,d_count,test_count,i] = accuracy_score( y_train2,y_pred12)
                    #msqrE[combcount,d_count,test_count,i] =  classification_report(y_train2,y_pred12)
                    test_count = test_count + 1
                d_count = d_count + 1
            combcount = combcount +1
            if i == int(crossVal-1):
                print(combination)
                if plotFig:
                    ax = axes[ind]
                    X_test1 = X_test_temp[list(combination)]
                    y_test1 = y_test_temp
                    titre = ''
                    for t in range(len(combination)):
                        titre = titre+combination[t]
                    ax = plot_scatter_regression(modelc,X_train1, y_train1,X_test1, y_test1,ax,title=titre)
        if plotFig and i == int(crossVal-1):
            plt.tight_layout()
            plt.show()
    return Rsqrt, msqrE, coeff,combo


def regress_neuron_knowncombo_testeval_logistic(filtered_dataAll,knowncombo,crossVal=1,dep='y',plotFig=False,testsize=0.1,RF=False):
    num_datasets = len(filtered_dataAll)
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    combo = list(knowncombo)
    length = [len(c) for c in knowncombo]
    indep_var = np.max(length)
    print("indep_var: "+str(indep_var ))
    Rsqrt = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combo),num_datasets,crossVal,indep_var+1))
    print(list(filtered_dataAll.keys()))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = train_test_split(data_temp,data_temp[dep], test_size=testsize)
            X_train0[key], X_test0[key], y_train0[key], y_test0[key] = X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp

        #train regressor on each worm
        combcount=0
        if plotFig and i == int(crossVal-1):
            fig, axes = plt.subplots(len(length), 1, figsize=(5, 5*len(length)))
        for ind , combination in enumerate(combo):

            d_count = 0
            for key1 in list(filtered_dataAll.keys()):
                X_train_temp = X_train0[key1]
                y_train_temp = y_train0[key1]
                X_test_temp = X_test0[key1]
                y_test_temp = y_test0[key1]
                # Select the columns corresponding to the current combination
                X_train1 = X_train_temp[list(combination)]
                y_train1 = y_train_temp
                if RF:
                    modelc = RandomForestClassifier(n_estimators=100, random_state=42)#LogisticRegression()
                    modelc.fit(X_train1, y_train1)
                    coeff[combcount,d_count,i,:length[ind]]= modelc.feature_importances_#modelc.coef_
                else:
                    print("applying logistic regression")
                    modelc = LogisticRegression()
                    modelc.fit(X_train1, y_train1)
                    coeff[combcount,d_count,i,:length[ind]]= modelc.coef_
                    coeff[combcount,d_count,i,length[ind]]= modelc.intercept_
                test_count = 0
                for key2 in list(filtered_dataAll.keys()):
                    X_train_temp = X_test0[key2]# these two lines are differe
                    y_train_temp = y_test0[key2]
                    # Select the columns corresponding to the current combination
                    X_train2 = X_train_temp[list(combination)]
                    y_train2 = y_train_temp

                    y_pred12 = modelc.predict(X_train2)
                    Rsqrt[combcount,d_count,test_count,i] = accuracy_score(y_train2,y_pred12)
                    #msqrE[combcount,d_count,test_count,i] =  classification_report( y_train2,y_pred12)
                    test_count = test_count + 1
                d_count = d_count + 1
            combcount = combcount +1
            if i == int(crossVal-1):
                print(combination)
                if plotFig:
                    ax = axes[ind]
                    X_test1 = X_test_temp [list(combination)]
                    y_test1 = y_test_temp
                    titre = ''
                    for t in range(len(combination)):
                        titre = titre+combination[t]
                    ax = plot_scatter_regression(modelc,X_train1, y_train1,X_test1, y_test1,ax,title=titre)
        if plotFig and i == int(crossVal-1):
            plt.tight_layout()
            plt.show()
    return Rsqrt, msqrE, coeff,combo


def regress_neuron_knowncombo_testeval_xgboost(filtered_dataAll,knowncombo,crossVal=1,dep='y',plotFig=False,testsize=0.1):
    num_datasets = len(filtered_dataAll)
    X_train0 = {}
    X_test0 = {}
    y_train0 = {}
    y_test0 = {}
    A = list(filtered_dataAll.keys())
    tempData = filtered_dataAll[A[0]]
    combo = list(knowncombo)
    length = [len(c) for c in knowncombo]
    indep_var = np.max(length)
    print("indep_var: "+str(indep_var ))
    Rsqrt = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    msqrE = np.zeros((len(combo),num_datasets,num_datasets,crossVal))
    coeff = np.zeros((len(combo),num_datasets,crossVal,indep_var+1))
    print(list(filtered_dataAll.keys()))
    for i in range(crossVal):
        #make train and test set for each worm
        for key in list(filtered_dataAll.keys()):
            data_temp = filtered_dataAll[key]
            X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp = train_test_split(data_temp,data_temp[dep], test_size=testsize)
            X_train0[key], X_test0[key], y_train0[key], y_test0[key] = X_train0_temp, X_test0_temp, y_train0_temp, y_test0_temp

        #train regressor on each worm
        combcount=0
        if plotFig and i == int(crossVal-1):
            fig, axes = plt.subplots(len(length), 1, figsize=(5, 5*len(length)))
        for ind , combination in enumerate(combo):

            d_count = 0
            for key1 in list(filtered_dataAll.keys()):
                X_train_temp = X_train0[key1]
                y_train_temp = y_train0[key1]
                X_test_temp = X_test0[key1]
                y_test_temp = y_test0[key1]
                # Select the columns corresponding to the current combination
                X_train1 = X_train_temp[list(combination)]
                y_train1 = y_train_temp

                dtrain = xgb.DMatrix(X_train1, label=y_train1)
                dtest = xgb.DMatrix(X_test_temp, label=y_test_temp)
                params = {
                'objective': 'binary:logistic',  # Binary classification
                'max_depth': 5,                 # Depth of each tree
                'eta': 0.1,                     # Learning rate (step size)
                'eval_metric': 'logloss',       # Evaluation metric
                'seed': 42                      # For reproducibility
                    }
                modelc = xgb.train(params, dtrain, num_boost_round=100)  # 100 boosting rounds

                #modelc =  model = RandomForestClassifier(n_estimators=100, random_state=42)#LogisticRegression()
                #modelc.fit(X_train1, y_train1)
                coeff[combcount,d_count,i,:length[ind]]= modelc.feature_importances_#modelc.coef_

                #coeff[combcount,d_count,i,length[ind]]= modelc.intercept_
                test_count = 0
                for key2 in list(filtered_dataAll.keys()):
                    X_train_temp = X_test0[key2]# these two lines are differe
                    y_train_temp = y_test0[key2]
                    # Select the columns corresponding to the current combination
                    X_train2 = X_train_temp[list(combination)]
                    y_train2 = y_train_temp
                    dtest = xgb.DMatrix(X_train2, label=y_train2)

                    y_pred120 = modelc.predict(dtest)
                    y_pred12 = (y_pred120 > 0.5).astype(int)
                    Rsqrt[combcount,d_count,test_count,i] = accuracy_score(y_train2,y_pred12)
                    #msqrE[combcount,d_count,test_count,i] =  classification_report( y_train2,y_pred12)
                    test_count = test_count + 1
                d_count = d_count + 1
            combcount = combcount +1
            if i == int(crossVal-1):
                print(combination)
                if plotFig:
                    ax = axes[ind]
                    X_test1 = X_test_temp [list(combination)]
                    y_test1 = y_test_temp
                    titre = ''
                    for t in range(len(combination)):
                        titre = titre+combination[t]
                    ax = plot_scatter_regression(modelc,X_train1, y_train1,X_test1, y_test1,ax,title=titre)
        if plotFig and i == int(crossVal-1):
            plt.tight_layout()
            plt.show()
    return Rsqrt, msqrE, coeff,combo
