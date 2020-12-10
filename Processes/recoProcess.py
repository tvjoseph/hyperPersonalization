'''

THese are the scripts for different processes in the recommendation cycle

'''

import sys
sys.path.append('/media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/bizportfolio/Hyperpersonalization')
import os
import pandas as pd
import pickle
import numpy as np
import random
from utils import helperFunctions
from collections import Counter
import operator



class RecoProcess:
    def __init__(self,conf,prodSub,simSubset,custDetails):
        # Load the required product propensity matrix and similarity matrices
        self.prodSub = prodSub
        # Load the similarity data frame
        self.simSubset = simSubset
        self.custDetails = custDetails
        self.assocDic = {}
        self.conf = conf

    def prodSorter(self,prods, custList,i):
        # Create a product list based on the products recommendations
        prodSort = pd.DataFrame(prods, index=prods)
        prodSort.columns = ['prodID']
        # loop through all the neighbour hood customers
        for lst in custList:
            # Store the customer id in a seperate variable for later use
            cid = lst
            # Convert the customer ID to an index
            lst = list(self.prodSub.index).index(str(lst))
            # First find the intersection of the current customers list and the recommended products
            custInter = [x for x in self.prodSub.iloc[lst][self.prodSub.iloc[lst].notna()].index.tolist() if x in prods]
            # First check if the number of products for the neighbour hood customer and the recommended products intersect
            if len(custInter) > 0:
                # Get the product list of the neigbourhood customers which were recommended to the current customer
                prodList = self.prodSub.iloc[lst][self.prodSub.iloc[lst].notna()].loc[custInter]
                # Getting the scores of the products for the customer and converting into a data frame
                custProds = pd.DataFrame(prodList)
                custProds.columns = [str(cid)]
                custProds['prodID'] = custProds.index
                # Merge this data frame with customer products
                prodSort = pd.merge(prodSort, custProds,how='left', on=['prodID'])

        # Fill the nan with 0
        prodSort = prodSort.fillna(0)
        # Reindex the prod sort to make the index as the products
        prodSort.index = prodSort['prodID']
        # Drop the product id column
        prodSort = prodSort.drop('prodID', axis=1)
        # Get the customer similarity scores for all the neighbourhood customers
        simNeigh = pd.DataFrame(self.simSubset.iloc[i][list(prodSort.columns)])
        # Do the dot product of the product sort with the similarity neighbour hood scores, sort and then get the index values which are the
        # list of the most preffered products
        prods = prodSort.dot(simNeigh).sort_values(list(simNeigh.columns), ascending=False).index.tolist()
        # Get the customer scores for all the neighbour hood customers in a array format ready for matrix multiplication
        ##custScores = np.reshape(self.simSubset[i, list(prodSort.columns[1:])],(self.simSubset[i, list(prodSort.columns[1:])].shape[0], 1))
        # Do matrix multiplication to get weighted scores
        #weighted_prods = np.matmul(np.array(prodSort.iloc[:, 1:]), custScores)
        # Get the indexes of the products with highest scores
        #prodIndex = list(pd.DataFrame(weighted_prods).sort_values(list(pd.DataFrame(weighted_prods).columns), ascending=False).index)
        # Sort the products based on the scores
        #prods = [prods[x] for x in prodIndex]
        return prods

    def recoProds_simple(self,custID, usr, prods):
        # Let us now get into the recommendation process
        # Get the index of the customer ID
        i = list(self.prodSub.index).index(custID)
        # Get the list of users similar to this custID
        simCusts = self.simSubset.iloc[i]
        # Insert a big number for the customer in  picture as it will always be 1
        ##simCusts[i] = 100
        # Get the top 20 users and Remove the first customer as this will be the same customer
        ##custList = simCusts.argsort()[-usr:][::-1][1:]
        custList = pd.DataFrame(simCusts).sort_values([str(custID)], ascending=[False]).index.tolist()[1:usr]
        prodBasket = []
        for lst in custList:
            # Store the customer ID
            cid = lst
            # Get the index of the customer
            lst = list(self.prodSub.index).index(str(lst))
            # Get the list of products for the customer which are not NA and append it with
            prodBasket.append(pd.DataFrame(self.prodSub.iloc[lst][self.prodSub.iloc[lst].notna()]).sort_values([str(cid)], ascending=[
                False]).index.tolist()[:prods])
            # Get the product list for this custID which are not NA
            ##prodList = self.prodSub.iloc[lst][self.prodSub.iloc[lst].notna()]
            # Get the top 10 products from all the top customers similar to the customer
            ##prodBasket.append(list(pd.DataFrame(prodList).sort_values(list(pd.DataFrame(prodList).columns), ascending=False).index[:prods]))
            # Get the individual products
        #prodBasket = list(set(prodBasket[0]))
        prodBasket = list(set([x for Basket in prodBasket for x in Basket]))
        # Find the list of customer products baskets
        custProds = self.prodSub.iloc[i][self.prodSub.iloc[i].notna()].index.tolist()
        # Take the list of products in prodBasket not with customer basket
        prodReco = [x for x in prodBasket if x not in custProds]
        # Sort the products
        prodReco = self.prodSorter(prodReco, custList,i)

        # Return the recommended products

        return prodReco, custList,custProds

    # Method for getting product baskets for a list of customers

    def prodBasket(self,custList):
        # Subset the product baskets for these customers
        custTrans = self.custDetails[self.custDetails["user_id"].isin(custList)]
        # Get the individual order IDs for these customer details
        orders = list(set(custTrans['order_id']))
        # Make product basket for the given orders
        prodBaskets_ids = []
        for oid in orders:
            # First filter on an order Id and convert the products into a list and append to the prodBaskets
            prodBaskets_ids.append(list(custTrans[custTrans['order_id'] == oid]['product_id'].values))

        return prodBaskets_ids

    # This is the function to calculate the total number of transactions for the associated products.
    # This function is used inside the assoCalculations functions
    def assoprodCounter(self,assoDf, prodBaskets):
        # Create an empty list to store all the lengths
        prodsLen = []
        # start a for loop to go through all associated products
        for prods in list(assoDf[0]):
            # Create an empty list to store all associated products baskets
            associateprods = []
            # Calculate length of each of the associated product baskets and append in the list of lengths
            prodsLen.append(len([associateprods.append(inlist) for inlist in prodBaskets if prods in inlist]))
        # Keep the lengths as the next column in the assoDf data frame
        assoDf['allcount'] = prodsLen
        return assoDf

    # This is the function to calculate the details like support,confidence and lift. This function is used inside
    # associationMaker() function
    def assoCalculations(self,prodBaskets, prodFreq, prod, sorted_res):
        # Take the list which does not contain the product
        assocProds = [inlist for inlist in sorted_res if prod not in inlist]
        # Convert into a data frame
        assoDf = pd.DataFrame(assocProds)
        # Calculate the support
        assoDf['support'] = assoDf[1] / len(prodBaskets)
        # Calculate the confidence
        assoDf['confidence'] = assoDf[1] / prodFreq
        # Calculate the length of the products
        assoDf = self.assoprodCounter(assoDf, prodBaskets)
        # Calculate all the supports of the associated products
        assoDf['allSupport'] = assoDf['allcount'] / len(prodBaskets)
        # Calculate the lifts of the products
        assoDf['Lift'] = assoDf['support'] / (assoDf['allSupport'] * (prodFreq / len(prodBaskets)))
        return assoDf

    # main function to return the associated products for a product ID
    def associationMaker(self,prod, prodBaskets):
        # Initialize an empty list for storing values of association
        associatedBasket = []
        # Find all baskets where the said product exists
        [associatedBasket.append(inlist) for inlist in prodBaskets if prod in inlist]
        # Check if the basket has products
        if len(associatedBasket) > 0:
            # Find the frequency of the product
            prodFreq = len(associatedBasket)
            # Find the count of all the products in the baskets
            res = dict(Counter(i for sub in associatedBasket for i in set(sub)))
            # Sort the count of products in ascending order
            sorted_res = sorted(res.items(), key=operator.itemgetter(1))
            # Sort in descending order
            sorted_res.reverse()
            # Get all the association rules as a dataframe
            ##assoDf = self.assoCalculations(prodBaskets, prodFreq, prod, sorted_res)
            return sorted_res
        else:
            assoDf = []
            return assoDf

    def bundlingProds(self,prods, prodBaskets):
        # Create an empty dictionary to store the associated products
        assocDict = {}
        for prod in prods:
            # print(prod)
            # Get the associated product basket for one of the sorted products
            assoDf = self.associationMaker(prod, prodBaskets)
            # Check if we are returning an empty basket
            if len(assoDf) == 0:
                continue
            # Sort the associations based on the values
            ##assoDf = assoDf.sort_values(['support', 'confidence', 'Lift'], ascending=False)
            # Take top five associations
            ##topFive = list(assoDf[0][0:5])
            topTen = [lst[0] for lst in assoDf[1:11]]
            # Store the associated products in a dictionary
            self.assocDic[prod] = topTen
            if len(self.assocDic[prod]) % 2000 == 0:
                helperFunctions.save_clean_data(self.assocDic, self.conf["association_dic"])













