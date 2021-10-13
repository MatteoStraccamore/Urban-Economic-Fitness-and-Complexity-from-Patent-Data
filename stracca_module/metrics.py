from scipy.io import mmread
from scipy import sparse
from itertools import repeat
import pandas as pd
import math
import random
import os
import numpy as np
import scipy.sparse as sps
import csv
import sympy
import sklearn.datasets
import sklearn.utils
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier


#Funzione per caricare le matrici V ed M dall'azienda più diversificata alla meno diversificata, considerando codici a 4 digit;
#N.B. Ricordiamo che le aziende che fanno almeno 2 codici tecnologici dal 2000 al 2009 sono 197944;
def importazione_M(num_az, Y_start = 1980, Y_end = 2014, method = 'HD', binarized = True, T = 0):
        
    #Il method definisce il tipo di diversificazione per le aziende:
    #HD = dalla 0 alla num_az più diversificata
    #MD = dalla 20000 alla 20000 + num_az più diversificata
    #LD = dalla 42225 - num_az alla 42225 più diversificata
    if method == None or method not in ('all','HD','MD','LD'):
        error = """Error; define method: 'all' = all cities 
        'HD' = High Diversification cities;
        'MD' = Medium Diversification cities;
        'LD' = Low Diversification cities."""
        print(error)
        return
    
    elif method == 'all':
        print("all cities")
    elif method == 'HD':
        print("HD cities")
    elif method == 'MD':
        print("MD cities")
    elif method == 'LD':
        print("LD cities")
        
    M = dict()
    V=dict()
    num_tec = 650
    l = [i for i in range(1980,2014+1)]
    if (Y_start != 1980) or (Y_end != 2014):
        l = l[(int(np.where(np.array(l)==Y_start)[0])):(int(np.where(np.array(l)==Y_end)[0]+1))]
    
    for j in l:
        A = mmread('city_tec_mtx_4digit_anno%i.mtx'%j)
        if method == 'all':
            V[j] = A.todense()
        elif method == 'HD':
            V[j] = A.todense()[0:num_az,:]
        elif method == 'MD':
            V[j] = A.todense()[20000:20000+num_az,:]
        elif method == 'LD':
            V[j] = A.todense()[42225-num_az:42225,:]
            
        if not binarized:
            M[j] = np.zeros((V[j].shape[0],V[j].shape[1]))
            M[j]=V[j]
        
        if binarized:
            M[j] = np.zeros((V[j].shape[0],V[j].shape[1]))
            if T == 0:
                M[j][V[j] != 0] = 1
            else:
                M[j][V[j] >= T] = 1
                M[j][V[j] < T] = 0
    return(M)



def activations(num_az_act, method = 'HD', T = 0):
    
    list_T = (0.00, 0.25, 0.50, 0.75, 1.00)
    
    if T not in list_T:
        error = '''Error, T can be = 0.00, 0.25, 0.50, 0.75, 1.00'''
        print(error)
        return
    if method == None or method not in ('all','HD','MD','LD'):
        error = """Error; define method: 'all' = all cities 
        'HD' = High Diversification cities;
        'MD' = Medium Diversification cities;
        'LD' = Low Diversification cities;"""
        print(error)
        return
    
    MACT_TOT = np.loadtxt("matrice_activations_2000_2009_4digits_T%.2f.txt" %T)

    if method == "all":
        MACT = MACT_TOT
    elif method == "HD":
        MACT = MACT_TOT[0:num_az_act,:]
    elif method == "MD":
        MACT = MACT_TOT[100000:100000+num_az_act,:]
    elif method == "LD":
        MACT = MACT_TOT[197944-num_az_act:197944,:]

    return MACT


    
def full_predictionsTOT(X, Y ,TEST = 0, method = 'RF', anno_test = 2009, alberi = 10, n_jobs = 4, n_splits = 4,
                        min_samples_leaf = 1, max_depth = None, criterion = 'gini', max_features = 'auto', printe = True):
    #Per RF i parametri ottimali sono
    #min_samples_leaf = 4, max_depth = 20, criterion = 'entropy', alberi = 50
    #Per CTS
    #min_samples_leaf = 10, max_depth = 20, criterion='gini', alberi = 10
    """La seguente funzione da in output full_predictionsTOT per metodo RF (quindi dimensioni pari ad MTEST) o CTS
     (quindi dimensioni pari a [9(anni) X num_az]X[tecnologie]]"""
    if method == None or method not in ('RF','RF_CTS'):
        print("Error; define method: 'RF' = simple Random Forest; 'CTS' = Continuous Technology Space")
        return
    
    elif method == 'RF':
        print("RF method")
    elif method == 'RF_CTS':
        print("RF_CTS method")
        
    if (method == 'RF'):
        
        #RF semplice
        forest = RandomForestClassifier(n_jobs = n_jobs, n_estimators=alberi, min_samples_leaf = min_samples_leaf,
                                        max_depth = max_depth, criterion=criterion, max_features = max_features)
        
        if MX[2000].shape != MY[2002].shape:
            a = '''Errore, MX ed MY devono avere le stesse dimensioni e devono avere le stesse aziende.
            MX non deve essere binarizzata ed MY sì.'''
            print(a)
            return
        
        anno_partenza1 = 2000
        anno_partenza2 = 2002
        anno_arrivo1 = 2007
        #Non può superare il 2009 altrimenti diamo informazioni aggiuntive
        anno_arrivo2 = 2009
        
        #concateno per le matrici X (NON BINARIZZATE
        for i in range (anno_partenza1, anno_arrivo1+1):
            if i == 2000:
                TOT_X = X[i]
            else:
                TOT_X = np.concatenate((TOT_X, X[i]), axis=0)

        #concateno per le matrici M_Y (BINARIZZATE)
        for i in range (anno_partenza2, anno_arrivo2+1):
            if i == 2002:
                TOT_Y = Y[i]
            else:
                TOT_Y = np.concatenate((TOT_Y, Y[i]), axis=0)
                
        TOT_Y = np.squeeze(np.asarray(TOT_Y))

        #una tecnologia alla volta
        for j in range (0, X[2000].shape[1]):
            forest = forest.fit(TOT_X, TOT_Y[:,j])
            #calcolo le prob a partire dal 2009 per l'anno 2011
            full_predictions = forest.predict_proba(TEST[anno_test])
            full_predictions = np.asmatrix(full_predictions)
            full_predictions = full_predictions[:,0]
            #ordino
            if j == 0:
                full_predictionsTOT = full_predictions
            else:
                full_predictionsTOT = np.concatenate((full_predictionsTOT, full_predictions), axis=1)
            if printe:
                print(j)
            
        return full_predictionsTOT
            
    else:
        
        #CTS
        #CV K-FOLD
        if X[2000].shape != Y[2002].shape:
            a = '''Errore, X ed Y devono avere le stesse dimensioni e devono avere le stesse aziende.
            MX non deve essere binarizzata ed Y sì.'''
            print(a)
            return
        
        from sklearn.model_selection import KFold
        
        forest = RandomForestClassifier(n_jobs = n_jobs, n_estimators=alberi, min_samples_leaf = min_samples_leaf,
                                        max_depth = max_depth, criterion='gini', max_features = max_features)
            
        n_splits = n_splits
        kf = KFold(n_splits=n_splits, shuffle = True, random_state=100)

        TOT_X_TRAIN = dict()
        TOT_X_TEST = dict()
        TOT_Y = dict()
        train = dict()
        test = dict()

        anno_partenza1 = 2000
        anno_partenza2 = 2002
        anno_arrivo1 = 2009
        anno_arrivo2 = 2011

        full_predictionsTOT = np.zeros((X[2000].shape[0]*((anno_arrivo1%10)+1), X[2000].shape[1]))
        full_predictionsTOT = np.asmatrix(full_predictionsTOT)

        times=0
        for train_index, test_index in kf.split(X[2000]):
            train[times] = train_index
            test[times] = test_index

            TOT_X_TRAIN[times] = X[2000][train_index]
            TOT_X_TEST[times] = X[2000][test_index]
            TOT_Y[times] = Y[2002][train_index]
            TOT_Y[times] = np.squeeze(np.asarray(TOT_Y[times]))
            
            #concateno le matrici
            for i in range (anno_partenza1+1, anno_arrivo1+1):
                TOT_X_TRAIN[times] = np.concatenate((TOT_X_TRAIN[times], X[i][train_index]), axis=0)
                TOT_X_TEST[times] = np.concatenate((TOT_X_TEST[times], X[i][test_index]), axis=0)
            for i in range (anno_partenza2+1, anno_arrivo2+1):
                TOT_Y[times] = np.concatenate((TOT_Y[times], MY[i][train_index]), axis=0)
            TOT_Y[times]=np.squeeze(np.asarray(TOT_Y[times]))
            times = times+1
            
        #una tecnologia alla volta
        for j in range (0, X[2000].shape[1]):
            #CV
            for k in range (0, n_splits):
                forest = forest.fit(TOT_X_TRAIN[k], TOT_Y[k][:,j])
                full_predictions = forest.predict_proba(TOT_X_TEST[k])
                full_predictions = np.asmatrix(full_predictions)
                full_predictions = full_predictions[:,0]

                #DEVO RIMETTERE IN ORDINE LE AZIENDE
                jj = 0
                for m in range (0, full_predictionsTOT.shape[0], X[2000].shape[0]):
                    full_predictionsTOT[test[k].reshape(len(test[k]),1) + m,j] = full_predictions[jj:jj+len(test[k])]
                    jj = jj + len(test[k])
                    
            if printe:
                print(j)
    
        return full_predictionsTOT


def best_F1_act(TRUE,PREDICTIONS):
#best-F1 activations
    best_F1acc = 0
    max_tacc = 0
    precision_recall_curve = sklearn.metrics.precision_recall_curve(
        TRUE.flatten(),
        PREDICTIONS.flatten())
    precision = precision_recall_curve[0][np.arange(precision_recall_curve[0].size - 1)]
    recall = precision_recall_curve[1][np.arange(precision_recall_curve[1].size - 1)]
    F1 = 2*(precision*recall)/(precision+recall)
    best_F1acc = np.nanmax(F1)
    return(best_F1acc)

def ROC_AUC_act(TRUE,PREDICTIONS):
    ueueue = TRUE.flatten()
    ueueueue = PREDICTIONS.flatten()
    AUCacc = sklearn.metrics.roc_auc_score(ueueue, ueueueue)
    return(AUCacc)

def Precision_K(true, predict, K):
    if K == -1:
        K = np.shape(true)[0]
    if np.shape(true)[0] != np.shape(predict)[0]:
        print("ERROR! True and Predicted have not the same dimensions ({} and {})".format(np.shape(true)[0],
                                                                                          np.shape(predicted)[0]))
    sorting_array = np.array([true, predict]).transpose()
    sorting_array = np.array(sorted(sorting_array, key = lambda x: x[1], reverse = True))
    predict_K = np.ones(K)
    true_K = sorting_array[:K,0]
    precisionK = precision_score(true_K, predict_K)
    return precisionK

def Kernel_G(tsne_obj, BETA):
    MACHINE_EPSILON = np.finfo(np.double).eps
    W = np.zeros((tsne_obj.shape[0], tsne_obj.shape[0]))
    dist = W
    dista = sklearn.metrics.pairwise_distances(tsne_obj, squared=True).astype(np.float32, copy=False)
    #calcolo gli scores dal kernel gaussiano
    W = np.zeros((tsne_obj.shape[0],tsne_obj.shape[0]))
    logW = W
    norm = np.zeros((tsne_obj.shape[0]))
    for i in range (0,tsne_obj.shape[0]):
        for j in range (i,tsne_obj.shape[0]):
            logW[i,j] = -dista[i,j]*BETA[i]
            logW[j,i] = logW[i,j]
    W = np.exp(logW)
    norm = np.abs(W).sum(axis=1)
    W = np.maximum(W/norm.reshape(W.shape[0],1), MACHINE_EPSILON)
    return W