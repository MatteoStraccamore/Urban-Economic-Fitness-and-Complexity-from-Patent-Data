import pandas as pd
import numpy as np
from scipy.stats import spearmanr





# Computation of Fitness and Complexity
def Fitn_Comp(biadjm):
    FFQQ_ERR = 10 ** -4
    spe_value = 10**-3
    bam = np.array(biadjm)
    c_len, p_len = bam.shape
    ff1 = np.ones(c_len)
    qq1 = np.ones(p_len)
    ff0 = np.sum(bam, axis=1)
    ff0 = ff0 / np.mean(ff0)
    qq0 = 1. / np.sum(bam, axis=0)
    qq0 = qq0 / np.mean(qq0)

    ff0 = ff1
    qq0 = qq1
    ff1 = np.dot(bam, qq0)
    qq1 = 1./(np.dot(bam.T, 1. / ff0))
    ff1 /= np.mean(ff1)
    qq1 /= np.mean(qq1)
    coef = spearmanr(ff0, ff1)[0]

    coef = 0.
    i=0
    while np.sum(abs(ff1 - ff0)) > FFQQ_ERR and np.sum(abs(qq1 - qq0)) > FFQQ_ERR and 1-abs(coef)>spe_value:
        i+=1
        print(i)
        ff0 = ff1
        qq0 = qq1
        ff1 = np.dot(bam, qq0)
        qq1 = 1./(np.dot(bam.T, 1. / ff0))
        ff1 /= np.mean(ff1)
        qq1 /= np.mean(qq1)
        coef = spearmanr(ff0, ff1)[0]
    return (ff0, qq0)


# Computation of Coherent Diversification
def coherence(biadjm,B_network):
    bam = np.array(biadjm)
    B = np.array(B_network)
    div = np.sum(bam,axis=1)
    gamma = np.nan_to_num(np.dot(B,bam.T).T)
    GAMMA = bam * gamma
    return np.nan_to_num(np.sum(GAMMA,axis=1)/div)