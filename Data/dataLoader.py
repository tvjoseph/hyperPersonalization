'''
This is the scrip for loading data for hyperparametrization application

'''
import sys
sys.path.append('/media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/bizportfolio')
import os
import pandas as pd
import pickle
import numpy as np
#from sklearn.model_selection import train_test_split
import random
from utils import helperFunctions
from datetime import datetime, timedelta,date
from dateutil.parser import parse

class DataProcessor:
    def __init__(self,configfile):
        # This is the first method in the DataProcessor class
        self.config = configfile

    def dataLoader(self):
        # THis is the method to load data from the input files
        #orderPath = self.config.get('DataFiles', 'orderData')
        orderPath = self.config["orderData"]
        orders = pd.read_csv(orderPath,encoding = "ISO-8859-1")
        return orders
    # This is the process for parsing dates
    def dateParser(self):
        custDetails = self.dataLoader()
        #Parsing  the date
        custDetails['Parse_date'] = custDetails[self.config["order_date"]].apply(lambda x: parse(x))
        # Parsing the weekdaty
        custDetails['Weekday'] = custDetails['Parse_date'].apply(lambda x: x.weekday())
        # Parsing the Day
        custDetails['Day'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%A"))
        # Parsing the Month
        custDetails['Month'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%B"))
        # Getting the year
        custDetails['Year'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%Y"))
        # Getting year and month together as one feature
        custDetails['year_month'] = custDetails['Year'] + "_" +custDetails['Month']

        return custDetails

    def gvCreator(self):
        custDetails = self.dateParser()
        # Creating gross value column
        custDetails['grossValue'] = custDetails[self.config["prod_qnty"]] * custDetails[self.config["unit_price"]]

        return custDetails


    # This is the process for getting the data for predicting for the customer

    def testDataloader(self):
        # Getting the paths for the x and y data sets
        xPath = self.config.get('modelDatasets', 'Xtrain')
        yPath = self.config.get('modelDatasets', 'ytrain')
        # Loading the data from pickle files
        pickle_in_x = open(xPath,"rb")
        pickle_in_y = open(yPath,"rb")
        xTrain = pickle.load(pickle_in_x)
        yTrain = pickle.load(pickle_in_y)
        yTrain = np.reshape(yTrain, (yTrain.shape[0], 1))
        # Splitting data into train and test sets and getting the test sets
        _, X_test, _, y_test = train_test_split(xTrain, yTrain, test_size=0.3, random_state=123)
        # Pick a random number within range from length of the test data
        testInt = random.randint(0, (len(X_test)-1))
        # Get the sequence
        seq = X_test[testInt:testInt+1, :, :]
        return seq

    # This is the process for consolidating the customer data related to orders and products into one data base

    #### Action to be done ###########3
    '''
    This custdataConsolidate function has to be modified to factor for general case of when there are multiple files and 
    also case when the data comes from a single file. The process should be first reading the data from each of the individual files and 
    checking if the attributes which we want are present in the file and then progressively checking them and the consolidating the required
    data
    '''

    def custdataConsolidate(self,ordNo):
        # First get the individual data from the earlier function
        orders, ordProducts,_ = self.dataLoader()
        # Taking only a subset of orders
        orders = orders[orders["user_id"] < ordNo]
        # Merge both the data sets based on order_id
        custDetails = pd.merge(orders, ordProducts, on=['order_id'])
        # Print the shape
        print('Shape of the consolidated customer details',custDetails.shape)
        # Save the customer details as a pickle file
        print("[INFO] saving the customer details as pickle file")
        filename = self.config["output"] + "/" + "custDetails" + ".pkl"
        helperFunctions.save_clean_data(custDetails, filename)





