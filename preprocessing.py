from pyspark.sql import SQLContext

from pyspark.ml.feature import *
from pyspark import SparkContext
import argparse

import requests
import os;
import sys
PATH = "https://storage.googleapis.com/uga-dsp/project2/data/bytes"

sc = SparkContext("local[*]",'pyspark tuitorial')
sqlContext = SQLContext(sc)

def fetch_url(x,path):
	class_label = x[1][1]
	url = x[1][0]
	fetch_url = path+"/"+url+".bytes"
	text = requests.get(fetch_url).text
	entries = text.split(os.linesep)
	entries = [(i.strip().replace("'",""),class_label) for i in  entries]
	return entries;

def open_row(x):
	entries = [i for i in x[0].split(' ')]
	entries.append(x[1])
	return entries
	

def clean(x):
	temp = x[0].split(' ')[1:]
	text = ' '.join(temp)
	return (text,x[1])


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

rdd_train_x = sc.textFile(args.train_x).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_train_y = sc.textFile(args.train_y).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_test_x = sc.textFile(args.test_x).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_test_y = sc.textFile(args.test_y).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_train = rdd_train_x.join(rdd_train_y)
rdd_test = rdd_test_x.join(rdd_test_y)
rdd_train = rdd_train.flatMap(lambda l :fetch_url(l,PATH)).map(lambda l:clean(l));
rdd_test = rdd_test.flatMap(lambda l :fetch_url(l,PATH)).map(lambda l:clean(l))


print(rdd_train.take(1))

df_train = sqlContext.createDataFrame(rdd_train)
df_test = sqlContext.createDataFrame(rdd_test)
print(df_train.show())
print(df_test.show())








## Process dataset for loading 
