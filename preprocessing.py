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

from pyspark import SparkContext,SparkConf

from pyspark.sql.types  import *
from pyspark.sql.functions import udf,col,split
from pyspark.sql import SQLContext

from pyspark.mllib.classification import *
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.feature import *

import requests
import os
import sys
import argparse




def merge_col(*x):
	temp = list();
	for i in x:
		temp.extend(i)
	return temp



def tfidf_processor(df,inputCol="text",outputCol="tfidf_vector"):
	hashingTF = HashingTF(inputCol=inputCol, outputCol="raw_features", numFeatures=400)
	tf = hashingTF.transform(df)
	idf = IDF(inputCol="raw_features", outputCol=outputCol,minDocFreq=3)
	idfModel = idf.fit(tf)
	df = idfModel.transform(tf)
	return df.drop(col("raw_features"));


def count_vectorizer_processor(df,inputCol="merge_text_array",outputCol="features"):
	cv_train = CountVectorizer(inputCol=inputCol, outputCol=outputCol, vocabSize=3, minDF=2.0)
	model = cv_train.fit(df)
	df = model.transform(df)
	return df




def word2ved_processor(df):
	wv_train =  Word2Vec(inputCol="text",outputCol="vector").setVectorSize(20)
	model = wv_train.fit(df)
	df = model.transform(df)
	return df


def ngram_processor(df,n_count=3):
	cols = list();
	for i in range(2,n_count+1):
		cols.append("text_ngram_"+str(i))
		ngram = NGram(n=i, inputCol="text", outputCol="text_ngram_"+str(i))
		df = ngram.transform(df)

	merge_udf = udf(merge_col)
	df = df.withColumn("merge_text",merge_udf("text","text_ngram_2","text_ngram_3"))
	df = df.withColumn("merge_text_array",split(col("merge_text"), ",")).drop(col("merge_text"))

	return df


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



def open_row(x):
	entries = [i for i in x[0].split(' ')]
	entries.append(x[1])
	return entries
	

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
sc = SparkContext.getOrCreate(conf=conf)
#Create SQL context.
sqlContext = SQLContext(sc)


parser = argparse.ArgumentParser(description='Welcome to Team Emma.')
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

#Get the training file names.
rdd_train_x = sc.textFile(args['train_x']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Get the training lables.
rdd_train_y = sc.textFile(args['train_y']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Get the testing file names.
rdd_test_x = sc.textFile(args['test_x']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Get the testing lables.
rdd_test_y = sc.textFile(args['test_y']).zipWithIndex().map(lambda l:(l[1],l[0]))
#Join training by index to create a merged set.
rdd_train = rdd_train_x.join(rdd_train_y)
#Join by zip-index to create a merged set. 
rdd_test = rdd_test_x.join(rdd_test_y)


#Get the data. Download and clean.
rdd_train_text = rdd_train.map(fetch_url).map(clean_data)
rdd_test_text = rdd_test.map(fetch_url).map(clean_data)
print("Download and Clean complete.")
#original dataframe


df_train_orignal = sqlContext.createDataFrame(rdd_train,schema=["text","class_label"])
df_test_orignal = sqlContext.createDataFrame(rdd_test,schema=["text","class_label"])


df_train_orignal.printSchema()

df_train_orignal.cache()
df_test_orignal.cache()

## word2vec code

#df_train_word2vec = word2ved_processor(df_train_orignal)
#df_test_word2vec = word2ved_processor(df_test_orignal)
#df_train_word2vec.show()
#df_train_word2vec.show()


##ngram code 
#df_train_ngram = ngram_processor(df_train_orignal,n_count=3);
#df_test_ngram = ngram_processor(df_test_orignal,n_count=3)

#df_train_ngram.show()
#df_test_ngram.show();

#count vectorizer + ngram
#df_train_vectorizer = count_vectorizer_processor(df_train_ngram,"merge_text_array")
#df_test_vectorizer = count_vectorizer_processor(df_test_ngram,"merge_text_array")

df_tfidf_train = tfidf_processor(df_train_orignal,"text","tfidf_vector");
print("now processing tf-idf");
df_tfidf_train.count();
df_tfidf_test = tfidf_processor(df_test_orignal,"text","tfidf_vector");
df_tfidf_test.count();
df_tfidf_test = df_tfidf_test.rdd.map(lambda l:LabeledPoint(l[1],l[-1].toArray()))
df_tfidf_train= df_tfidf_train.rdd.map(lambda l:LabeledPoint(l[1],l[-1].toArray()))
print("processing complete");

#print(df_tfidf_test.take(20)[-1][-1])
#df_train_vectorizer.show();
#df_test_vectorizer.show();
#df_tfidf_train.show()
#df_tfidf_test.show()


# Split data approximately into training (60%) and test (40%)
training=df_tfidf_train
test=df_tfidf_test

# Train a naive Bayes model.
model = NaiveBayes.train(training, 0.8)

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
#print(predictionAndLabel.map(lambda x:x[0]).collect())
print('model accuracy {}'.format(accuracy))


#print(df_tfidf_test.take(20)[-1][-1])
#df_train_vectorizer.show();
#df_test_vectorizer.show();
#df_tfidf_train.show()
#df_tfidf_test.show()
