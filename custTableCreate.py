'''
THis is the script for controlling all the inference processes.

This process looks at the matrices which were created during the training phase and then execute different processes to store
the product related values to a customer data base. The processes that will be executed are the following

1. Recommended products for a customer - Based on neighbour hood methods
2. Associated products for the recommended products : This is for bundling strategies
3. Segmentation where the customer is mapped to. This should be based on the buying behaviour and some significant clusters
generated based on buying behaviour
4. Product propensity. This will give recommendations of products a customer is going to buy and real time offers and associated
products. This is a real time recommendation based on markov property
5. Predictions on what the customer is going to buy in the next few weeks.

References
https://www.visiture.com/blog/customer-lifetime-value-why-it-matters-and-how-to-calculate-it/

https://towardsdatascience.com/data-driven-growth-with-python-part-2-customer-segmentation-5c019d150444

https://towardsdatascience.com/predicting-customer-lifetime-value-with-buy-til-you-die-probabilistic-models-in-python-f5cac78758d9
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
from pymongo import MongoClient

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c','--conf',required=True,help='Path to the configuration file')
args = vars(ap.parse_args())

# Load the configuration file
conf = Conf(args['conf'])

# Defining the mongo db credentials
client = MongoClient(port=27017)
db = client.customer_personalization

# Process 1 : Get the list of customers

# Load the customer propensity matrix
propLists = os.listdir(conf["propensity_mats"])

# INitiate a list for loading customer ids

print("[INFO] Loading the propensity matrix")
for proplst in propLists:
    filename = conf["propensity_mats"] + proplst
    # Load the product propensity matrix
    prodSub = helperFunctions.load_files(filename)
    # From the product propensity matrix get the list of customers
    custAll = prodSub.index.tolist()

# Load the similarity matrix also from the disk
print("[INFO] Loading the Similarity matrix")

simLists = os.listdir(conf["similarity_mats"])

for simList in simLists:
    filename = conf["similarity_mats"] + simList
    # Load the similarity matrix
    simSubset = helperFunctions.load_files(filename)

print("[INFO] Loading the Customer details")
# Load the customer details from disk
filename = conf["custDetails"]
custDetails = helperFunctions.load_files(filename)

# Starting the recommendation process

# Initialise the recommendation class

rp = RecoProcess(conf,prodSub,simSubset,custDetails)

if os.path.exists(conf["prodBaskets"]):
    print("[INFO] Loading the product baskets")
    # Load the product baskets
    prodBaskets = helperFunctions.load_files(conf["prodBaskets"])
else:
    # Create the product association matrix creation process
    # get the list of all orders
    print("[INFO] Creating product association lists")
    ordDetails = custDetails[conf["order_id"]].unique().tolist()
    prodBaskets = []
    for i,oid in enumerate(ordDetails):
        # First filter on an order Id and convert the products into a list and append to the prodBaskets
        prodBaskets.append(custDetails[custDetails[conf["order_id"]] == oid][conf["product_id"]].values.tolist())
        if i % 5000 == 0:
            print(i)
    print(len(prodBaskets))
    print("[INFO] Saving the product baskets")
    filename = conf["prodBaskets"]
    helperFunctions.save_clean_data(prodBaskets,filename)

print("[INFO] Starting the product association creation process")

if os.path.exists(conf["association_dic"]):
    print("[INFO] Product association dictionary exists")
else:
    # Create list of all products
    prodList = custDetails[conf["product_id"]].unique().tolist()
    print("length of products",len(prodList))

    # Create the association baskets for all the products
    for i in range(0, len(prodList),1000):
        prods = tqdm(prodList[i:i + 1000])
        # Get the associated dictionary
        rp.bundlingProds(prods, prodBaskets)
        if i % 2000 == 0:
            print("[INFO] Saving product association for {} products".format(i))

print("[INFO] Starting the recommendation process")

# Get the list of users
if os.path.exists(conf["users"]):
    print("[INFO] Loading customer list")
    # Load the list of customers
    custAll = helperFunctions.load_files(conf["users"])

# Load the associated dictionary
if os.path.exists(conf["association_dic"]):
    print("[INFO] Loading Association dictionary ")
    # Load the list of customers
    assocDic = helperFunctions.load_files(conf["association_dic"])



# Get teh parameters for neighbourhood, neighbourhood products and number of reccomendations

nbr = conf["neighbors"]
nbrProd = conf["neighbor_prods"]
recom = conf["prod_reco"]

# Initializing the CLV class

cl = clvCalculator(conf,custDetails)

if os.path.exists(conf["custDict"]):
    print("[INFO] Loading personalisation dictionary ")
    # Load the list of customers
    custDic = helperFunctions.load_files(conf["custDict"])

    # Find the quantile values of clv values
    custQuant = cl.clvQuant(custDic)
    # Finding the CLV quantile of the customer
    for i in range(len(custDic)):
        custDic[i]["clvQuant"] = cl.custQuantile(custDic[i]["clv"])
        custDic[i]["neighClvquant"] = cl.neighQuant(custDic[i]["neighbours"])

    helperFunctions.save_clean_data(custDic, conf["custDict_update"])

    # Dumping details into personalization tables in MongoDB
    print('[INFO] dumping the details in MongoDB')
    db.personalization.insert(custDic)


else:
    # Creating a list for consolidating values
    allCustRec = []
    # Starting a for loop to store all the customer dictionary values
    for i,custID in enumerate(custAll):
        if np.isnan(custID):
            print("Found a NAN value in customer ID")
            continue
        # Start a new dictionary for storing customer values
        custRec = {}
        # Get the recommended products and customer list
        prodReco, custList,custProds = rp.recoProds_simple(str(custID),nbr,nbrProd)
        # Get the associated products for the top n recommended products
        myDict = {k: v for k, v in assocDic.items() if k in prodReco[:recom]}
        # Get teh associated products for the customers normal products
        basketDict = {k: v for k, v in assocDic.items() if k in custProds}
        # Get customer life time value for the customer
        clv = cl.clvCalc(custID)
        # Find the CLV of the segment of customers
        clvSeg = cl.clvSegment(custList)
        # Store all details
        custRec["Customer_ID"] = custID
        custRec["recoProds"] = myDict
        custRec["myProducts"] = basketDict
        custRec["neighbours"] = custList
        custRec["clv"] = clv
        custRec["clv_segment"] = clvSeg
        # Update the list
        allCustRec.append(custRec)
        if i % 1000 == 0:
            print(i)

    # Saving the details
    helperFunctions.save_clean_data(allCustRec, conf["custDict"])



















