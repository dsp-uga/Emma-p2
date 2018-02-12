from pyspark.sql import SQLContext
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

from threading import Thread

from pyspark.ml.feature import *
from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import udf,col,split,lit

import argparse
import pyspark
from pyspark.ml.linalg import Vectors, VectorUDT


import requests
import os;
from pyspark.sql.types  import *
import sys

#PATH = "https://storage.googleapis.com/uga-dsp/project2/data/bytes"

#"local[*]",'pyspark tuitorial'

#https://medium.com/@rbahaguejr/threaded-tasks-in-pyspark-jobs-d5279844dac0

#Spark configuration in
conf = SparkConf()
conf = conf.setMaster("local[*]")
conf =conf.set("spark.driver.memory", "40G")
con = conf.set('spark.scheduler.mode','FAIR')
model_list =  dict();
#conf.set("spark.driver.cores", 4)

sc = SparkContext('local[*]','Team Emma',conf=conf)

sqlContext = SQLContext(sc)

current_class = 0

#.set("spark.executor.memory", "4g")


layers = [100, 50,25,12,10,8,4, 2]


split_train = 0.3
limit =80000


def boosting(dataset,labelCol,iteration=20):
		
		model = MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol=labelCol, blockSize=1, seed=123)

		for i in range(0,iteration):
			training=dataset.sample(False,split_train,seed=42).limit(limit)
			model_history =model.fit(training) #we need model history
			model = MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol=labelCol, blockSize=1, seed=123)
			model.setInitialWeights(model_history.weights)
		model_list[labelCol] = model_history

#calculate per col

	
def build_labels(df_tfidf_train):
	current_class = 0;
	df_tfidf_train = df_tfidf_train.withColumn("zero",one_vs_all_udf(df_tfidf_train.class_label,lit(0)))
	current_class = 1
	df_tfidf_train = df_tfidf_train.withColumn("one",one_vs_all_udf(df_tfidf_train.class_label,lit(1)))
	current_class = 2
	df_tfidf_train = df_tfidf_train.withColumn("two",one_vs_all_udf(df_tfidf_train.class_label,lit(2)))
	current_class = 3
	df_tfidf_train = df_tfidf_train.withColumn("three",one_vs_all_udf(df_tfidf_train.class_label,lit(3)))
	current_class = 4
	df_tfidf_train = df_tfidf_train.withColumn("four",one_vs_all_udf(df_tfidf_train.class_label,lit(4)))
	current_class = 5
	df_tfidf_train = df_tfidf_train.withColumn("five",one_vs_all_udf(df_tfidf_train.class_label,lit(5)))
	current_class = 6
	df_tfidf_train = df_tfidf_train.withColumn("six",one_vs_all_udf(df_tfidf_train.class_label,lit(6)))
	current_class = 7
	df_tfidf_train = df_tfidf_train.withColumn("seven",one_vs_all_udf(df_tfidf_train.class_label,lit(7)))
	current_class = 8
	df_tfidf_train = df_tfidf_train.withColumn("eight",one_vs_all_udf(df_tfidf_train.class_label,lit(8)))
	return df_tfidf_train

def merge_col(*x):
	temp = list();
	for i in x:
		temp.extend(i)
	return temp


def prediction_and_label(model,dataset,class_label):

	predictions = model.transform(dataset)
	evaluator = MulticlassClassificationEvaluator(
	labelCol=class_label, predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Class: "+ class_label + " Accuracy: "+str(accuracy))
	return accuracy,predictions



def tfidf_processor(df,inputCol="text",outputCol="tfidf_vector"):
	hashingTF = HashingTF(inputCol=inputCol, outputCol=outputCol, numFeatures=100)
	df = hashingTF.transform(df)
	#idf = IDF(inputCol="raw_features", outputCol=outputCol,minDocFreq=3)
	#idfModel = idf.fit(tf)
	#df = idfModel.transform(tf)
	return df

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
	key = x[0]
	class_label = x[1][1]
	url = x[1][0]
	fetch_url = path+"/"+url+".bytes"
	text = requests.get(fetch_url).text
	entries = text.split(os.linesep)
	entries = [(i.strip().replace("'",""),class_label,key) for i in  entries]
	return entries;

def open_row(x):
	entries = [i for i in x[0].split(' ')]
	entries.append(x[1])
	return entries
	

def clean(x):
	label = x[1]
	key = x[2]
	count = 0
	class_label = x[2]
	temp = x[0].split(' ')[1:]
	for i in range(0,len(temp)):
		if temp[i] == '00' or temp[i] == '??':
			count = count + 1
	if count == len(temp):
		return None
	
	
	
	return (temp,label,key)


def one_vs_all(x,y):
		if float(x[0]) ==y:
			return float(1)
		else:
			return float(0)



one_vs_all_udf = udf(one_vs_all,FloatType())



parser = argparse.ArgumentParser(description='Welcome to Team Emma.')
parser.add_argument('-a','--train_x', type=str,
                    help='training x set')
parser.add_argument( '-b','--train_y' ,help='training y set')
parser.add_argument('-c','--test_x', type=str,
                    help='testing x set')
#parser.add_argument('-d','--test_y', type=str,
#                    help='testing y set')
parser.add_argument('-e','--path', type=str,
                    help='path to folder')

args = vars(parser.parse_args())
#print(args)
rdd_train_x = sc.textFile(args['train_x']).zipWithIndex().map(lambda l:(l[1],l[0]))
rdd_train_y = sc.textFile(args['train_y']).zipWithIndex().map(lambda l:(float(l[1]),l[0]));
#rdd_test_x = sc.textFile(args['test_x']).zipWithIndex().map(lambda l:(l[1],l[0]));
#rdd_test_y = sc.textFile(args['test_y']).zipWithIndex().map(lambda l:(float(l[1]-1),l[0]));
rdd_train = rdd_train_x.join(rdd_train_y).randomSplit([0.4,0.6])[0];
#rdd_test = rdd_test_x.join(rdd_test_y)
#take 30 due to gc overhead


rdd_train = rdd_train.flatMap(lambda l :fetch_url(l,args['path'])).map(lambda l:clean(l)).filter(lambda l:l !=None)

#rdd_train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
#rdd_train = sc.parallelize(rdd_train)
print("Download complete");
#rdd_test= rdd_test.flatMap(lambda l :fetch_url(l,args['path'])).map(lambda l:clean(l)).filter(lambda l: l!=None)
#rdd_test = sc.parallelize(rdd_test)
#print("Test Zeros" + str(rdd_test.count()));


print("Download complete")



df_train_original = sqlContext.createDataFrame(rdd_train,schema=["text","class_label","key"])
training , testing = df_train_original.randomSplit([0.4,0.6])
training = training.limit(800000).repartition(10000)


#df_test_original = sqlContext.createDataFrame(rdd_test,schema=["text","class_label"])
#df_train_original = df_train_original.repartition(10000)
	#df_test_original = df_test_original.repartition(30000) 

	#df_train_orignal ,df_train_orignal_validate =df_train_orignal.randomSplit([0.7,0.3])


#df_train_original.printSchema()
 
#df_test_original.cache()

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

print("now processing tf-idf");
training= tfidf_processor(training,"text","features").repartition(1000)
testing= tfidf_processor(testing,"text","features").repartition(1000)
training = build_labels(training)
testing = build_labels(testing)

training.count()
testing.count()
training.cache()
testing.cache();
print("loading completee")

#df_tfidf_test = tfidf_processor(df_test_original,"text","tfidf_vector");
#df_tfidf_test = df_tfidf_test.rdd.map(lambda l:[l[1],l[-1].toArray()])
#df_tfidf_train= df_tfidf_train.rdd.map(lambda l:[l[1],l[-1].toArray()]).repartition(10000)
#df_tfidf_train = df_tfidf_train.randomSplit([0.05,05,0.99])[0:]
#print("processing complete");

#print(df_tfidf_test.take(20)[-1][-1])
#df_train_vectorizer.show();
#df_test_vectorizer.show();
#df_tfidf_train.show()
#df_tfidf_test.show()


# Split data approximately into training (60%) and test (40%)"""

#https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size






print("Model Sequence one started for model 0")

model_0= Thread(target=boosting, args=(training,"zero"))


model_1 = Thread(target=boosting, args=(training,"one"))

model_2 = Thread(target=boosting, args=(training,"two"))

model_3 = Thread(target=boosting, args=(training,"three"))

model_4 = Thread(target=boosting, args=(training,"four"))

model_5 = Thread(target=boosting, args=(training,"five"))

model_6 = Thread(target=boosting, args=(training,"six"))

model_7 = Thread(target=boosting, args=(training,"seven"))


model_0.start()
model_1.start()
model_2.start()
model_3.start()
model_4.start()
model_5.start()
model_6.start()
model_7.start()


model_0.join()
model_1.join()
model_2.join()
model_3.join()
model_4.join()
model_5.join()
model_6.join()
model_7.join()

print(model_list)
"""
model_list[0] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="zero")
print("Model Sequence one started for model 1")
model_list[1] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="one")
print("Model Sequence one started for model 2")
model_list[2] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="two")
print("Model Sequence one started for model 3")
model_list[3] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="three")
print("Model Sequence one started for model 4")
model_list[4] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="four")
print("Model Sequence one started for model 5")
model_list[5] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="five")
print("Model Sequence one started for model 6")
model_list[6] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="six")
print("Model Sequence one started for model 7")
model_list[7] = boosting(training.sample(False,split_train,seed=42).limit(limit*4),labelCol="seven")
"""


#training = training.repartition(5000)
#testing = testing.repartition(50000)


accuracy_1 ,prediction_1  = prediction_and_label(model_list["zero"],testing,"zero")
accuracy_2 ,prediction_2  = prediction_and_label(model_list["one"],testing,"one")
accuracy_3 ,prediction_3 = prediction_and_label(model_list["two"],testing,"two")
accuracy_4 ,prediction_4 = prediction_and_label(model_list["three"],testing,"three")
accuracy_5 ,prediction_5  = prediction_and_label(model_list["four"],testing,"four")
accuracy_6 ,prediction_6  = prediction_and_label(model_list["five"],testing,"five")
accuracy_7 ,prediction_7  = prediction_and_label(model_list["six"],testing,"six")
accuracy_8 ,prediction_8  = prediction_and_label(model_list["seven"],testing,"seven")

prediction_1  = prediction_1.withColumn("predictions", col("prediction").cast("int"))
prediction_2  = prediction_2.withColumn("predictions", col("prediction").cast("int"))
prediction_3  = prediction_3.withColumn("predictions", col("prediction").cast("int"))
prediction_4  = prediction_4.withColumn("predictions", col("prediction").cast("int"))
prediction_5  = prediction_5.withColumn("predictions", col("prediction").cast("int"))
prediction_6  = prediction_6.withColumn("predictions", col("prediction").cast("int"))
prediction_7  = prediction_7.withColumn("predictions", col("prediction").cast("int"))
prediction_8  = prediction_8.withColumn("predictions", col("prediction").cast("int"))





prediction_1 = prediction_1.groupBy("key").sum("predictions").alias("predictions_0")
prediction_2 = prediction_2.groupBy("key").sum("predictions").alias("predictions_1")
prediction_3 = prediction_3.groupBy("key").sum("predictions").alias("predictions_2")
prediction_4 = prediction_4.groupBy("key").sum("predictions").alias("predictions_3")
prediction_5 = prediction_5.groupBy("key").sum("predictions").alias("predictions_4")
prediction_6 = prediction_6.groupBy("key").sum("predictions").alias("predictions_5")
prediction_7 = prediction_7.groupBy("key").sum("predictions").alias("predictions_6")
prediction_8 = prediction_8.groupBy("key").sum("predictions").alias("predictions_7")

prediction_1.show();
prediction_2.show();
prediction_3.show();
prediction_4.show();
prediction_5.show();
prediction_6.show();
prediction_7.show();
prediction_8.show();




print("Program execution complete")
