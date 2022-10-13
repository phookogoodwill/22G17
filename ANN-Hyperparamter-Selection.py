import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}, Used to get rid of warnings from tensorflow

# Dependencies
from ast import Delete
from email.utils import collapse_rfc2231_value
from unicodedata import numeric
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import StandardScaler
from dateutil import parser
import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt
import datetime
import random

#------------------------------------------------------------------------------------------
# read the csv and convert it to a data frame.
#------------------------------------------------------------------------------------------
input_list = pd.read_csv("diamond.csv")

#------------------------------------------------------------------------------------------
# Function that checks for rows with a null and deletes them from the dataframe
#------------------------------------------------------------------------------------------
def DeleteNull (DataFrame):
    NoNulldf=DataFrame
    NoNulldf = NoNulldf.dropna()
    return NoNulldf

#print(DeleteNull(input_list))

#------------------------------------------------------------------------------------------
# Function that stores ALL the column names and their entries in a dictionary
#------------------------------------------------------------------------------------------
def DataframeDictionary (Dataframe):
    columns_dictionary = Dataframe.to_dict('list')
    return columns_dictionary

#print (DataframeDictionary(input_list))
#-------------------------------------------------------------------------------------------
#Function that identifies whether a column is numeric or not
#-------------------------------------------------------------------------------------------
def Check_numeric(column):  
    entry=column[0]
    try:
        float(entry)
        return True
    except ValueError:
        return False 


#-----------------------------------------------------------------------------------------
#Function that checks if a column is a time stamp entry
#-----------------------------------------------------------------------------------------

def CheckTimestamp (column):
    entry= column[0]
    try:
        parser.parse(entry).timestamp()
        Bool=True
    except:
        Bool=False
    return Bool

#-------------------------------------------------------------------------------------
#FUNCTION that transforms entries of a timestamp column to an epoch
#-------------------------------------------------------------------------------------
def ConvertTimestamp (column):
        newColumn = []

        for entry in column:
             newColumn.append(parser.parse(entry).timestamp())
                
        return newColumn

#-------------------------------------------------------------------------------------
#FUNCTION that transforms entries of a string column to integers
#-------------------------------------------------------------------------------------

def ConvertString(Column):#input is a dictionary of columns to be converted
    
    map={}
    index=1    
    newColumn=[]
    for entry in Column:
        if entry not in map:
            map[entry] = index
            index += 1
        
    newColumn = list([map[entry] for entry in Column])
        
    # function returns the dictionary with the converted target columns, list_CSV is a dictionary
    return newColumn


#-------------------------------------------------------------------------------------
# FUNCTION that converts all the columns from the data frame (stored in a dictionary) to integers (if string) or to epoch (if timestamp)
#-------------------------------------------------------------------------------------
def ConvertColumns(ColumnsDictionary):
    ColumnTitles=ColumnsDictionary.keys()
    _ColumnsDictionary={}
    #for every column title, access the value and save it as a list or an array
    for title in ColumnTitles:
        Column= list(ColumnsDictionary.get(title))
        if Check_numeric(Column) == True:
            _ColumnsDictionary[title]= Column
        else:
             if CheckTimestamp(Column)==True:
                _ColumnsDictionary[title]=ConvertTimestamp(Column)
             else:
                _ColumnsDictionary[title]=ConvertString(Column)

    return _ColumnsDictionary

#print(ConvertColumns((DataframeDictionary(DeleteNull((input_list))))))

#-------------------------------------------------------------------------------------------------
#Function that writes that converts the converted dictionary to a dataframe
#-------------------------------------------------------------------------------------------------

def dict_dataframe (DictUpdated):
    DictUpdated2=DictUpdated
    df = pd.DataFrame.from_dict(DictUpdated2)
    return df

#print(dict_dataframe(ConvertColumns((DataframeDictionary(DeleteNull((input_list)))))))

#------------------------------------------------------------------------------------------------
#Scaling Function; scales all columns in the dataframe
#-------------------------------------------------------------------------------------------------

def Scaler (DataFrame):
    ScaledDf=DataFrame
    output= "Price"
    scaler = StandardScaler()
    Columns= ScaledDf.columns

    for x in Columns:
        if x != output:
            ScaledDf[[x]] = scaler.fit_transform(ScaledDf[[x]])

    return ScaledDf

#print(Scaler(dict_dataframe(ConvertColumns((DataframeDictionary(DeleteNull((input_list))))))))
#------------------------------------------------------------------------------------------------
#Shuffling function; function that shuffles the entries in each column
#------------------------------------------------------------------------------------------------
def Shuffle (ScaledDataFrame):
   shuffledDf= ScaledDataFrame.sample(frac=1)
   return shuffledDf

#print(Shuffle(Scaler(dict_dataframe(ConvertColumns((DataframeDictionary(DeleteNull((input_list)))))))))

#####################################################################################################################
## BUILDING A NEURAL NETWORK (copy and paste from project planning code :) )
#####################################################################################################################

converted_df= Shuffle(Scaler(dict_dataframe(ConvertColumns((DataframeDictionary(DeleteNull((input_list))))))))
#converted_df=converted_df.drop(columns=["Price"])
#print(converted_df)

print(converted_df)
def BuildNetwork (inputs, regress=False):

        # define our MLP network
    model = keras.Sequential()
    model.add(keras.layers.Dense(512, input_dim=inputs, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(keras.layers.Dense(1, activation="linear"))
    # return our model
    return model

attribute_df = converted_df.drop(['Price'], axis=1)
trainAttributes_df = attribute_df.head(int(0.7*attribute_df.shape[0]))
trainPredictions_df = converted_df.loc[:,"Price"].head(int(0.7*converted_df.shape[0]))
testAttributes_df = attribute_df.tail(int(0.3*attribute_df.shape[0]))
testPredictions_df = converted_df.loc[:,"Price"].tail(int(0.3*converted_df.shape[0]))

print(trainAttributes_df)

inputNodes=7

model = BuildNetwork(inputNodes, True)
model.compile(loss="mean_absolute_error", optimizer='adam')

monitor = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1, 
        patience=10, verbose=1, mode='auto',
        restore_best_weights=True)
model.fit(trainAttributes_df, trainPredictions_df, validation_data=(testAttributes_df, testPredictions_df), callbacks=[monitor], verbose=1,epochs=1000)#batch_size=batchSize, epochs=epochs)

predictions = model.predict(testAttributes_df).flatten()[0:100]
actual = list(testPredictions_df.to_numpy())[0:100]

plt.figure()
plt.scatter(range(len(predictions)), predictions)
plt.ylim(0,80000)
plt.title('Predicted Diamond Prices')
plt.ylabel('Prices (R)')
plt.xlabel('Diamond ID')
plt.savefig("Prediction.png", dpi=400)

plt.figure()
plt.scatter(range(len(actual)), actual)
plt.ylim(0,80000)
plt.title('Actual Diamond Prices')
plt.ylabel('Prices (R)')
plt.xlabel('Diamond ID')
plt.savefig("Actual.png", dpi=400)
