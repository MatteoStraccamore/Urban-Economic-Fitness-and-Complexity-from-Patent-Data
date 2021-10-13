from scipy.io import mmread
from scipy.io import mmwrite
from scipy import sparse
from itertools import repeat
import pandas as pd
import math
import random
import os
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps
import matplotlib.pyplot as pltl
import csv
import sympy
import sklearn.datasets
import sklearn.utils
from scipy.sparse import coo_matrix


def co_occorrence(M):
    B = np.zeros((M.shape[1],M.shape[1]))
    B = np.dot(np.transpose(M), M)
    B = np.nan_to_num(B)
    return B

def technology_space(M):
    ut = np.sum(M,axis=0)
    B = np.dot(np.transpose(M), M)/np.maximum(ut.T,ut)
    B = np.nan_to_num(B)
    return(B)

def taxonomy(M):
    ut = np.sum(M,axis=0)
    df = np.sum(M,axis=1)
    M = M/np.sqrt(df)
    M = np.nan_to_num(M)
    B = np.dot(np.transpose(M), M)/np.maximum(ut.T,ut)
    B = np.nan_to_num(B)
    return(B)

def micro_partial(M):
    B = np.dot(np.transpose(M), M)
    B = np.nan_to_num(B)
    ut = np.sum(M,axis=0)
    K = M.shape[0]
    mu = np.dot(np.transpose(ut), ut)/K
    mu = np.nan_to_num(mu)
    sigma = np.dot((K-np.transpose(ut)),(K-ut))/(K*(K-1))
    sigma = np.nan_to_num(sigma)
    sigma = sigma*mu
    sigma = np.nan_to_num(sigma)
    sigma = np.sqrt(sigma)
    sigma = np.nan_to_num(sigma)
    B = (B-mu)/sigma
    B = np.nan_to_num(B)
    return(B)