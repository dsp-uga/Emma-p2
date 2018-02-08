from pyspark.sql import SQLContext

from pyspark.ml.feature import *
from pyspark import SparkContext
from pyspark.sql.functions import udf,col,split
import argparse

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.classification import *

import requests
import os;
from pyspark.sql.types  import *
import sys
PATH = "https://storage.googleapis.com/uga-dsp/project2/data/bytes"

sc = SparkContext("local[*]",'pyspark tuitorial')
sqlContext = SQLContext(sc)

def merge_col(*x):
	temp = list();
	for i in x:
		temp.extend(i)
	return temp



def tfidf_processor(df,inputCol="text",outputCol="tfidf_vector"):
	hashingTF = HashingTF(inputCol=inputCol, outputCol="raw_features", numFeatures=300)
	tf = hashingTF.transform(df)
	idf = IDF(inputCol="raw_features", outputCol=outputCol,minDocFreq=2)
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
	return (temp,x[1])



parser = argparse.ArgumentParser(description='Welcome to Team Emma.')
parser.add_argument('--train_x', type=str,
                    help='training x set')
parser.add_argument( '--train_y' ,help='training y set')
parser.add_argument('--test_x', type=str,
                    help='testing x set')
parser.add_argument('--test_y', type=str,
                    help='testing y set')
parser.add_argument('--path', type=str,
                    help='path to folder')

args = parser.parse_args()
print(args)
rdd_train_x = sc.textFile(args.train_x).zipWithIndex().map(lambda l:(l[1]-1,l[0]))
rdd_train_y = sc.textFile(args.train_y).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_test_x = sc.textFile(args.test_x).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_test_y = sc.textFile(args.test_y).zipWithIndex().map(lambda l:(l[1],l[0]));
rdd_train = rdd_train_x.join(rdd_train_y)
rdd_test = rdd_test_x.join(rdd_test_y)
#take 30 due to gc overhead
rdd_train = rdd_train.flatMap(lambda l :fetch_url(l,args.path)).map(lambda l:clean(l))
#rdd_train = sc.parallelize(rdd_train)

rdd_test= rdd_test.flatMap(lambda l :fetch_url(l,args.path)).map(lambda l:clean(l))
#rdd_test = sc.parallelize(rdd_test)

#original dataframe


df_train_orignal = sqlContext.createDataFrame(rdd_train,schema=["text","class_label"])
df_test_orignal = sqlContext.createDataFrame(rdd_test,schema=["text","class_label"])
df_train_orignal ,df_train_orignal_validate =df_train_orignal.randomSplit([0.7,0.3])


df_train_orignal.printSchema()

df_train_orignal.cache()
df_test_orignal.cache()

## word2vec code

df_train_word2vec = word2ved_processor(df_train_orignal)
df_test_word2vec = word2ved_processor(df_test_orignal)
df_train_word2vec.show()
df_train_word2vec.show()


##ngram code 
df_train_ngram = ngram_processor(df_train_orignal,n_count=3);
df_test_ngram = ngram_processor(df_test_orignal,n_count=3)

df_train_ngram.show()
df_test_ngram.show();

#count vectorizer + ngram
df_train_vectorizer = count_vectorizer_processor(df_train_ngram,"merge_text_array")
df_test_vectorizer = count_vectorizer_processor(df_test_ngram,"merge_text_array")

df_tfidf_train = tfidf_processor(df_train_orignal,"text","tfidf_vector");
df_tfidf_test = tfidf_processor(df_test_orignal,"text","tfidf_vector");
df_tfidf_test = df_tfidf_test.rdd.map(lambda l:LabeledPoint(l[1],l[-1].toArray()))
df_tfidf_train= df_tfidf_train.rdd.map(lambda l:LabeledPoint(l[1],l[-1].toArray()))
model =  LogisticRegressionWithLBFGS.train(df_tfidf_train, iterations=100,numClasses=8)
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))


#print(df_tfidf_test.take(20)[-1][-1])
#df_train_vectorizer.show();
#df_test_vectorizer.show();
#df_tfidf_train.show()
#df_tfidf_test.show()