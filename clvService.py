'''
This is the script for different clv related services
'''

import os.path
from os import path
import argparse
from utils import Conf
from Data import DataProcessor
from utils import helperFunctions
from Processes import RecoProcess,clvCalculator
import pandas as pd
from trainModule import MatrixCreate
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from os import walk
from config import configfile as configpy

# Get the customer personalization pickle file
configfilepath = configpy.configfile

# Load the configuration file
conf = Conf(configfilepath)

# Load the personalization details
custDic = helperFunctions.load_files(conf["custDict_update"])

# Function to calculate quantile ranges

def quantRanges(custDic):
    # Take the collection of all CLV values
    clvAll = []
    for i in range(len(custDic)):
        clvAll.append(custDic[i]['clv'])
    Q1 = []
    Q2 = []
    Q3 = []
    Q4 = []
    # Find the quantile means
    [Q1.append(item['clv']) for item in custDic if item["clvQuant"] == "Q1"]
    [Q2.append(item['clv']) for item in custDic if item["clvQuant"] == "Q2"]
    [Q3.append(item['clv']) for item in custDic if item["clvQuant"] == "Q3"]
    [Q4.append(item['clv']) for item in custDic if item["clvQuant"] == "Q4"]
    # Find the quantile values
    clvqt = np.quantile(clvAll, [0.25, 0.50, 0.75,0.96])
    clvmin = min(clvAll)
    clvmax = max(clvAll)
    # Storing all values in a dictionary
    allMet = {}
    allMet['clvqt'] = clvqt
    allMet['clvmin'] = clvmin
    allMet['clvmax'] = clvmax
    allMet['Q1'] = int(np.mean(Q1))
    allMet['Q2'] = int(np.mean(Q2))
    allMet['Q3'] = int(np.mean(Q3))
    allMet['Q4'] = int(np.mean(Q4))

    return allMet



# Function to get clv details

def getClv(custID, custDic):
    # Get the quntile ranges
    #clvqt, clvmin, clvmax = quantRanges(custDic)
    # Get the relevant customer record
    custRec = next(item for item in custDic if item["Customer_ID"] == custID)
    # Get the customer CLV quantile
    clvQuant = custRec['clvQuant']
    # Get the performance with respect to potential
    clvPerformance = ((custRec['clv'] / custRec['clv_segment']) * 100).round(2)
   # Get the
    allMet = quantRanges(custDic)
    # Overall performance
    clvOverall = ((custRec['clv'] / int(allMet['clvqt'][-1])) * 100).round(2)

    return clvQuant, clvPerformance,clvOverall,allMet

# Get the customer clv details

clvQuant, clvPotential,clvOverall,allMet = getClv(conf["custID"], custDic)
print(clvQuant)
print(clvPotential)
print(clvOverall)

#allMet= quantRanges(custDic)
print(allMet)


