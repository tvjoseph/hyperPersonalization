'''
This is the script for calculating the customer life time values
'''

import sys
sys.path.append("/media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/bizportfolio/Hyperpersonalization")
from utils import helperFunctions
import pandas as pd
import numpy as np

class clvCalculator:
    def __init__(self,conf,custDetails):
        self.conf = conf
        self.custDetails = custDetails



    def clvCalc(self,custID):
        '''
        :param custID: This is the customer ID for whom we need to calculate the CLV
        :return:
        '''
        # Making a subset of the data for the customer
        custDet_cust = self.custDetails[self.custDetails[self.conf["customer_id"]] == float(custID)]
        #  # Find Total value per month
        #print(custID)
        custMonthlyVal = pd.DataFrame(custDet_cust.groupby(['year_month'])['grossValue'].agg('sum'))['grossValue'].agg('mean').round(3)
        # Find the average order value
        custOrdVal = pd.DataFrame(custDet_cust.groupby([self.conf["order_id"]])['grossValue'].agg('sum'))['grossValue'].agg('mean').round(3)
        # Find life span of the customer
        monthSpan = (custDet_cust.Parse_date.max().year - custDet_cust.Parse_date.min().year) * 12 + (custDet_cust.Parse_date.max().month - custDet_cust.Parse_date.min().month)
        # Calculate the Customer Life time value
        CLV = (custMonthlyVal * custOrdVal * monthSpan).round(2)

        return CLV

    def clvSegment(self,custList,mn=True):
        # Make a list to store all life time values
        allClv = []
        # Loop through each of the customer IDs
        for custID in custList:
            allClv.append(self.clvCalc(custID))
        if mn:
            # Find the mean value of the segment and return the value
            return np.mean(allClv)
        else:
            # Return the list of CLV values for the neighbours
            return allClv

    def clvQuant(self,custDic):
        clvAll = []
        for i in range(len(custDic)):
            clvAll.append(custDic[i]['clv'])
        return np.quantile(clvAll,[0.25,0.50,0.75])

    def custQuantile(self,clv):
        if clv <= 0:
            return 'Q1'
        elif clv > 0 and clv <= 133542:
            return 'Q2'
        elif clv > 133542 and clv <= 754114:
            return 'Q3'
        else:
            return 'Q4'
    def neighQuant(self,custList):
        # Get the Clv values for all the nighbours
        neighClv = self.clvSegment(custList,mn=False)
        # Create a list to store the quantile values
        nQuant = []
        for i in range(len(neighClv)):
            nQuant.append(self.custQuantile(neighClv[i]))
        return(nQuant)




