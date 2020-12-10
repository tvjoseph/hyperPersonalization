'''
1-Dec-2020
This is the script for creating all the data matrices required for hyperpersonalization
'''

# Importing the required packages
import os.path
from os import path
import argparse
from utils import Conf
from Data import DataProcessor
from utils import helperFunctions
from trainModule import MatrixCreate
from joblib import Parallel, delayed
from tqdm import tqdm


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c','--conf',required=True,help='Path to the configuration file')
args = vars(ap.parse_args())

# Load the configuration file
conf = Conf(args['conf'])

print("[INFO] loading the raw files")
dl = DataProcessor(conf)
# Creating the consolidated data set
# Check if the file exists, if so, skip the consolidation activity
if path.exists(conf["custDetails"]):
    print("Customer details file already exists")
    # Let us load the customer Details
    custDetails = helperFunctions.load_files(conf["custDetails"])
    # Get list of unique customers
    users = helperFunctions.load_files(conf["users"])
    print('Shape of customer data {}'.format(custDetails.shape[0]))
    print(len(users))
else:
    print("[INFO] Customer details getting consolidated")
    '''
    we need to work on making the customer file consolidation generic based on the attributes defined in the configuration file
    '''
    #dl.custdataConsolidate(75000)  # Specifying the user ID below which we want the data
    custDetails = dl.gvCreator()
    print(custDetails.shape)
    print("[INFO] saving the customer details as pickle file")

    filename = conf["output"] + "/" + "custDetails" + ".pkl"
    helperFunctions.save_clean_data(custDetails, filename)

    # Saving the users details also
    users = list(custDetails[conf["customer_id"]].unique())
    filename = conf["output"] + "/" + "users" + ".pkl"
    helperFunctions.save_clean_data(users, filename)
    print(custDetails.head())
    print(len(users))


# Starting the product wise buying matrix
# Instantiating the matrixCreators class
mc = MatrixCreate(conf,dl)

# Check if the matrices were created, if not create the matrices
matLists = os.listdir(conf["matrixPath"])

if len(matLists) == 0:
    for i in range(0,len(users),1000):
        inputs = tqdm(users[i:i+1000])
        if __name__ == "__main__":
            processed_list = Parallel(n_jobs=4)(delayed(mc.prodConsolidator)(j) for j in inputs)
        print("Saving the data of processed list",len(processed_list))
        mc.custData(processed_list,i,len(users))
    # Getting the list of files which were saved
    matLists = os.listdir(conf["matrixPath"])

# list down all the propensity matrices
propLists = os.listdir(conf["propensity_mats"])

# Check if the propensity list is less than the matList
if len(propLists) < len(matLists):
    print("[INFO] Creating the propensity lists")
    inlist = []
    # Taking only the second part of propLists name as this is the format of matlists
    for plist in propLists:
        inlist.append(plist.split("_")[1])
    # Finding those files in Matlists for which propensity list is not created
    deltaList = [i for i in matLists + inlist if i not in matLists or i not in inlist]
    # Creating propensity list for those files left out
    if len(deltaList) > 0:
        mc.propensityMat(deltaList)


# Start the Similarity matrix creation
print("[INFO] Starting creation of similarity matrices")
for lst in propLists:
    # Load the file
    filename = conf["propensity_mats"] + lst
    prodSub = helperFunctions.load_files(filename)
    print(prodSub.shape)
    # Start the similarity matrix creator
    mc.simmatCreate(prodSub,lst)

