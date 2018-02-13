"""
@author: Hiten, Prajay, Maulik.
"""

"""
This file is a structured file for the Maleware classification of the Microsoft data.
The processing in the file goes as below directions: 
1. Download Byte-Lable files from the sources from the list of the text files. 
2. Clean the text: Remove '??' from the Byte files and remove all lines with having all 0s.
3. Split the terms and create N-grams for the each term. 
4. Create Data Dictionary for generated terms.
5. Train the Naive-Bayes model and predict the output.
6. For small records: As the test result is already available, compare the test results with the
predicted results and create a confusion matrix.
"""

from pyspark.sql.functions import concat, col, lit
from pyspark import SparkContext,SparkConf

from pyspark import SparkContext,SparkConf

from pyspark.sql.types  import *
from pyspark.sql.functions import udf,col,split
from pyspark.sql import SQLContext


from pyspark.ml.feature import *
from pyspark.ml.classification import *

import requests
import os
import sys
import argparse
import re
import pyspark


"""
This function fetches the byte files data given the text file name.
TODO: The path is right now static, which should be converted to a broadcast variable.
@param file_row is a RDD row of the format(zip index, (byte file name, lable)).
@return byte_data is the whole byte file as text.
"""
def fetch_url(file_row):
    #Get the file name from the RDD row.
    file_name = file_row[1][0]
    #Path variable is the path to the google storage location of the byte files.
    path = 'https://storage.googleapis.com/uga-dsp/project2/data/bytes'
    #Set the byte file url.
    fetch_url = path+"/"+ file_name +".bytes"
    #Get the file data and convert into text.
    byte_data = requests.get(fetch_url).text
    return byte_data


"""
This function cleans the byte data. Removes headers, unwanted sequences of data and special characters.
@param data_row is on byte file text for training.
@return data_row is the cleaned file.
@ref https://stackoverflow.com/questions/43260538/how-to-delete-words-with-more-than-six-letters-in-notepad
"""
def clean_data(data_row):
    #Remove linefeed and new line.
    data_row = re.sub('\\r\\n',' ',data_row)
    #Remove question marks.
    data_row = re.sub('\??','',data_row)
    #Remove the rows with all zeros. There are rows where there is a sequence of 8 zeroes.
    data_row = re.sub('00 00 00 00 00 00 00 00','',data_row)
    #Remove the rows with all Cs. There are rows where there is a sequence of 8 zeroes.
    data_row = re.sub('CC CC CC CC CC CC CC CC','',data_row)
    #Remove the headers. Words larger than 2 characters.
    data_row = re.sub(r'\b[A-Z|0-9]{3,}\b','',data_row)
    #Strip the text
    data_row = data_row.strip()
    return data_row




#Create the Spark Config and Context.
conf = SparkConf().setAppName('P2MalewareClassification')

conf =conf.set('spark.driver.memory', '6g')
conf =conf.set('spark.executor.memory', '6g')
conf =conf.set('spark.driver.cores', '2')
conf =conf.set('spark.executor.cores', '2')
conf =conf.set('spark.python.worker.memory', '4g')
conf =conf.set('spark.yarn.am.memoryOverhead', '1g')
conf =conf.set('spark.yarn.driver.memoryOverhead', '2g')
conf =conf.set('spark.yarn.executor.memoryOverhead', '2g')

sc = SparkContext.getOrCreate(conf=conf)
#Create SQL context.
sqlContext = SQLContext(sc)


parser = argparse.ArgumentParser(description='Welcome to Maleware Classification. - Team Emma.')
parser.add_argument('-a','--train_x', type=str,
                    help='training x set')
parser.add_argument( '-b','--train_y' ,help='training y set')
parser.add_argument('-c','--test_x', type=str,
                    help='testing x set')
parser.add_argument('-d','--test_y', type=str,
                    help='testing y set')
parser.add_argument('-e','--path', type=str,
                    help='path to folder')

args = vars(parser.parse_args())
#print(args)

#Get the training file names and make zip index as key.
rdd_train_x = sc.textFile(args['train_x']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Get the training lables and make zip index as key.
rdd_train_y = sc.textFile(args['train_y']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Get the testing file names and make zip index as key.
rdd_test_x = sc.textFile(args['test_x']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Get the testing lables and make zip index as key.
rdd_test_y = sc.textFile(args['test_y']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Join training by index to create a merged set.
rdd_train = rdd_train_x.join(rdd_train_y)
#Join by zip-index to create a merged set. 
rdd_test = rdd_test_x.join(rdd_test_y)

#Get the  file list files. 
#Fetch the data.
#Clean the Byte file of redundant information.
#Zip with index everything. 
#Join with the File-list, lable file. 
#Keep the labels and byte file only. 
rdd_train_text = rdd_train.map(fetch_url).map(clean_data).zipWithIndex().map(lambda l: (l[1],l[0])).join(rdd_train).map(lambda l: (l[1][0],l[1][1][1]))
rdd_test_text = rdd_test.map(fetch_url).map(clean_data).zipWithIndex().map(lambda l: (l[1],l[0])).join(rdd_test).map(lambda l: (l[1][0],l[1][1][1]))

rdd_train_text = rdd_train_text.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
print("Download and Clean complete.")

#Create data-frames from the RDD
df_train_original = sqlContext.createDataFrame(rdd_train_text,schema=["text","label"])
df_test_original = sqlContext.createDataFrame(rdd_test_text,schema=["text","label"])


#Tokenize the document by each word and transform.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
print("1.Tokenize")
#Using the tokenized word find 4-gram words and transform.
ngram = NGram(n=4, inputCol=tokenizer.getOutputCol(), outputCol="nGrams")
print("2.Ngram.")
#Create the hashing function from the tokens and find features.
hashingTF = HashingTF(inputCol=ngram.getOutputCol(), outputCol="features",numFeatures=10000)
print("3.Hashing.")
#Train the naive bayes model.
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
print("4.Naive Bayes")
#Pipeline.
pipeline = Pipeline(stages=[tokenizer,ngram, hashingTF, nb])
#Fit the model.
model = pipeline.fit(df_train_original)
print("5.Model done.")
#Predict the output.
prediction = model.transform(df_test_original)
print("Prediction Done.")

#Find accuracy from the correctly identified results.
prediction_rdd = prediction.rdd
prediction_accuracy = prediction_rdd.filter(lambda l : int(l[1]) == int(l[7])).count()/prediction_rdd.count()
print("Prediction Accuracy:")
print(prediction_accuracy)


#Save Results on the server.
prediction_rdd = prediction_rdd.map(lambda l: int(l[7]))
prediction_rdd.coalesce(1).saveAsTextFile('results')
