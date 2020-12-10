'''
This script lists down all the helper functions which are required for processing raw data
'''

from pickle import load
from pickle import dump
import numpy as np


# Function to Save data to pickle form
def save_clean_data(data,filename):
    dump(data,open(filename,'wb'))
    print('Saved: %s' % filename)

# Function to load pickle data from disk
def load_files(filename):
    return load(open(filename,'rb'))

# Function to find cosine similarity of vectors
def cosineSim(val1,val2):
    numerator = sum(val1*val2)
    denom1 = np.sqrt(np.matmul(val1,val1.T))
    denom2 = np.sqrt(np.matmul(val2,val2.T))
    sim = numerator / (denom1 * denom2)
    return sim

# Function to make the similarity matrix symmetric

def simMatsym(simSubset):
    for i in range(len(simSubset)):
        simSubset[i:, i] = simSubset[i, i:]
    return simSubset





