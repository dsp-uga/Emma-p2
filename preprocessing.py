from pyspark.sql import SQLContext


from pyspark import SparkContext
import argparse

import requests
import os;
import sys
PATH = "https://storage.googleapis.com/uga-dsp/project2/data/bytes"

sc = SparkContext("local[*]",'pyspark tuitorial')
sqlcontext = SQLContext(sc)

def fetch_url(x,path):
	fetch_url = path+"/"+x+".bytes"
	return fetch_url;
	#text = requests.get(fetch_url).text

	#entries = text.split(os.linesep)
	#entries = [i.strip().replace("'","") for i in  entries]
	#return text;

def open_row(x):
	entries = x.split(os.linesep)
	entries = [i.strip().replace("'","") for i in  entries]


parser = argparse.ArgumentParser(description='Welcome to Team Emma.')
parser.add_argument('--train_x', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--train_y',type=str,
                    help='sum the integers (default: find the max)')
parser.add_argument('--test_x', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--test_y', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args)

rdd_train_x = sc.textFile(args.train_x)
rdd_train_y = sc.textFile(args.train_y).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_test_x = sc.textFile(args.test_x)
rdd_test_y = sc.textFile(args.test_y)
rdd_train_x= rdd_train_x.map(lambda l:fetch_url(l,PATH)).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_test_x = rdd_test_x.map(lambda l:fetch_url(l,PATH))
rdd_train = rdd_train_x.join(rdd_train_y)
print(rdd_train.take(10))
sys.exit(-1);



df_train_x = sqlContext.createDataFrame(rdd_train, ['headers','byte_1','byte_2','byte_3','byte_4','byte_5','byte_6','byte_7','byte_8','byte_9','byte_10','byte_11','byte_12','byte_13','byte_14','byte_15','byte_16'])
df_test_x = sqlContext.createDataFrame(rdd_test_x, ['headers','byte_1','byte_2','byte_3','byte_4','byte_5','byte_6','byte_7','byte_8','byte_9','byte_10','byte_11','byte_12','byte_13','byte_14','byte_15','byte_16'])
df_train_y = sqlcontext.createDataFrame()
print(rdd_train.take(10))






## Process dataset for loading 
