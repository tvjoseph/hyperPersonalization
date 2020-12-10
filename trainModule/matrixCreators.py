'''

This is the script for creating different matrices required for customer personalization
'''

import sys
sys.path.append('/media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/bizportfolio')
from utils import helperFunctions
import pandas as pd
import numpy as np

class MatrixCreate:

    def __init__(self,conf,dl):
        '''
        :param conf: This is the configuration file
        :param dl: This is the data processor class
        '''
        # Getting the attribute mapping
        self.product_id = conf["product_id"]
        self.order_id = conf["order_id"]
        self.product = conf["product"]
        self.prod_qnty = conf["prod_qnty"]
        self.order_date = conf["order_date"]
        self.unit_price = conf["unit_price"]
        self.customer_id = conf["customer_id"]
        # This is the configuration file required for all methods
        self.conf = conf
        # Load the custDetails also
        self.custDetails = helperFunctions.load_files(conf["custDetails"])
        # Create the base product data frame
        #_, ordProducts, _ = dl.dataLoader()
        self.baseDf = pd.DataFrame(self.custDetails[self.product_id].unique())
        self.baseDf.columns = ['Index']
        # This is the dataframe for consolidating the customer data
        self.consDf = self.baseDf

    # Function to consolidate each customer details to the base data frame

    def prodConsolidator(self,custId):
        # Get all the Order details for the customer
        custDetails = self.custDetails[self.custDetails[self.customer_id].eq(custId)]
        # Getting the count of the products
        cust1 = pd.DataFrame(custDetails[self.product_id].value_counts())
        #print(cust1.head())
        # adding the index to the file
        cust1['Index'] = cust1.index
        # Adding the customer ID as a column name
        cust1.columns = [str(custId), 'Index']
        # Merging with the product data frame
        baseDf = self.baseDf
        prodDf = pd.merge(baseDf, cust1, on=['Index'], how='left')
        #print(prodDf.shape)
        return prodDf

    def custData(self,processed_list,iter,saveLen):
        '''

        :param processed_list: This is the list of
        :param iter: Iterations which is in a batch of 1000 each
        :saveLen : This is the value of the consolidated matrix when the pickle file has to be created
        :return: Saves the consolidated list as
        '''
        # Initialise the aggregation dataframe
        conDf1 = self.baseDf
        # Loop through the processed list to find the details
        iter = iter
        for i, list in enumerate(processed_list):
            conDf1 = pd.merge(conDf1, list, on=['Index'], how='left')
            # Consolidate the main data frame every 1000 records
            if i >= (len(processed_list) - 1):
                conDf1 = conDf1.drop(['Index'], axis=1)
                self.consDf = pd.concat([self.consDf, conDf1], axis=1)
                # Initialise the aggregation dataframe
                conDf1 = self.baseDf
                print(self.consDf.shape)
                if self.consDf.shape[1] > saveLen:
                    #print(iter-len(processed_list))
                    filename = self.conf["matrixPath"]  + "consDf" + str(iter + len(processed_list)) +".pkl"
                    print("[INFO] Saving the consolidated data frame as pickle file")
                    helperFunctions.save_clean_data(self.consDf,filename)
                    self.consDf = self.baseDf
                    #iter += self.consDf.shape[1]


    # The next method is the method for creating the propensity matrix

    def propensityMat(self,matLists):
        '''
        :param matLists: This is the list of all the customer product buying basket
        :return: Saved propensity list
        '''
        # Loop through each of the lists
        for lst in matLists:
            filename = self.conf["matrixPath"] + lst
            custProp = helperFunctions.load_files(filename)
            # Resetting the index to the product names
            custProp = custProp.set_index(custProp['Index'], drop=True)
            # Remove the index column
            custProp = custProp.drop(['Index'], axis=1)
            # Getting the column wise sum of products
            custProds = pd.DataFrame(custProp.sum(axis=0)).T
            # Getting row wise division
            custProp = custProp.div(custProds.iloc[0])
            # Save the customer propensity matrix as pickle file
            filename = self.conf["propensity_mats"] + "custProp" + "_" + lst
            print("[INFO] Saving customer propensity matrices as pickle file",filename)
            helperFunctions.save_clean_data(custProp.T, filename)

    # Method for creating similarity matrices

    def normalise(self,A):
        lengths = pd.DataFrame(np.sqrt((A ** 2).sum(axis=1)))
        return A.div(lengths[0], axis=0)

    def simmatCreate(self,prodSub,lst):
        '''
        :param prodSub: This is the propensity file
        :param st: Start of the index for iteration
        :param en: End of the index for iteration
        :lst : This is the name of the file
        :return: Saved similarity matrix
        '''
        # Fill nas with zeros
        prodSub = prodSub.fillna(0)
        # Normalize the data frame
        prodNorm = self.normalise(prodSub)
        # Find similarity matrix by taking dot products of the normalized matrix
        simMat = np.dot(prodNorm, prodNorm.T)
        # Create a pandas data frame of similarity
        simDf = pd.DataFrame(simMat,columns=prodSub.index,index=prodSub.index)
        # Save the similarity matrix
        print("[INFO] Saving the similarity matrix")
        filename = self.conf['similarity_mats'] + "similarity_" + lst
        helperFunctions.save_clean_data(simMat, filename)
        # Saving the data frame similarity
        filename = self.conf['similarity_mats'] + "simDf_" + lst
        helperFunctions.save_clean_data(simDf, filename)




