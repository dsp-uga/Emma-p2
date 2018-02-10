
from pyspark.sql import SQLContext
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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

#Spark configuration in
conf = SparkConf()
conf = conf.setMaster("local[*]")
conf =conf.set("spark.driver.memory", "45G")
#conf.set("spark.driver.cores", 4)

sc = SparkContext('local[*]','Team Emma',conf=conf)

sqlContext = SQLContext(sc)

current_class = 0

#.set("spark.executor.memory", "4g")



def intialize_model():

	model = list();

	model_0 = MultilayerPerceptronClassifier(maxIter=100,labelCol="zero", layers=[100, 2, 2], blockSize=1, seed=123)
	model_1 = MultilayerPerceptronClassifier(maxIter=100,labelCol="one", layers=[100, 2, 2], blockSize=1, seed=123)
	model_2= MultilayerPerceptronClassifier(maxIter=100,labelCol="two", layers=[100, 2, 2], blockSize=1, seed=123)
	model_3= MultilayerPerceptronClassifier(maxIter=100,labelCol="three", layers=[100, 2, 2], blockSize=1, seed=123)
	model_4= MultilayerPerceptronClassifier(maxIter=100,labelCol="four", layers=[100, 2, 2], blockSize=1, seed=123)
	model_5= MultilayerPerceptronClassifier(maxIter=100,labelCol="five", layers=[100, 2, 2], blockSize=1, seed=123)
	model_6= MultilayerPerceptronClassifier(maxIter=100,labelCol="six", layers=[100, 2, 2], blockSize=1, seed=123)
	model_7= MultilayerPerceptronClassifier(maxIter=100,labelCol="seven", layers=[100, 2, 2], blockSize=1, seed=123)	

	model.append(model_0)
	model.append(model_1)
	model.append(model_2)
	model.append(model_3)
	model.append(model_4)
	model.append(model_5)
	model.append(model_6)
	model.append(model_7)


	return model


def update_model(old_model):
	model = list()

	model_0 = MultilayerPerceptronClassifier(maxIter=100, layers=[100, 2, 2],labelCol="zero", blockSize=1, seed=123)
	model_0.setInitialWeights(old_model[0].weights)
	model_1 = MultilayerPerceptronClassifier(maxIter=100, layers=[100, 2, 2],labelCol="one", blockSize=1, seed=123)
	model_1.setInitialWeights(old_model[1].weights)
	model_2= MultilayerPerceptronClassifier(maxIter=100, layers=[100, 2, 2],labelCol="two", blockSize=1, seed=123)
	model_2.setInitialWeights(old_model[2].weights)
	model_3= MultilayerPerceptronClassifier(maxIter=100, layers=[100,2, 2,],labelCol="three", blockSize=1, seed=123)
	model_3.setInitialWeights(old_model[3].weights)
	model_4= MultilayerPerceptronClassifier(maxIter=100, layers=[100,2, 2],labelCol="four", blockSize=1, seed=123)
	model_4.setInitialWeights(old_model[4].weights)
	model_5= MultilayerPerceptronClassifier(maxIter=100, layers=[100,2, 2],labelCol="five", blockSize=1, seed=123)
	model_5.setInitialWeights(old_model[5].weights)
	model_6= MultilayerPerceptronClassifier(maxIter=100, layers=[100,2, 2],labelCol="six", blockSize=1, seed=123)
	model_6.setInitialWeights(old_model[6].weights)
	model_7= MultilayerPerceptronClassifier(maxIter=100, layers=[100,2, 2],labelCol="seven", blockSize=1, seed=123)	
	model_7.setInitialWeights(old_model[7].weights)

	model.append(model_0)
	model.append(model_1)
	model.append(model_2)
	model.append(model_3)
	model.append(model_4)
	model.append(model_5)
	model.append(model_6)
	model.append(model_7)


	return model




split_train = 0.05
split_test = 0.05
limit = 100
waste = 1 - split_train + split_test



def boosting(df_tfidf_train,model):
		accuracy = [0 for i in range(0,8)]

		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[0] = model[0].fit(training)


		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[1]  = model[1].fit(training)

		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[2]  = model[2].fit(training)


		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[3] = model[3].fit(training)

		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[4] = model[4].fit(training)
		
		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[5]  = model[5].fit(training)
	
		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[6] = model[6].fit(training)

		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		model[7]  = model[7].fit(training)
		
		model = update_model(model)
		return model



	
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


def prediction_and_label(model,dataset):
	prediction= model.transform(dataset)
	predictionAndLabels = prediction.map(lambda lp: (float(model.predict(lp.features)), lp.label))

	metrics = MulticlassMetrics(predictionAndLabels)
	# Overall statistics
	precision = metrics.precision()
	recall = metrics.recall()
	f1Score = metrics.fMeasure()
	print("Summary Stats")
	print("Precision = %s" % precision)
	print("Recall = %s" % recall)
	print("F1 Score = %s" % f1Score)



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

rdd_train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
#rdd_train = sc.parallelize(rdd_train)
print("Download complete");
#rdd_test= rdd_test.flatMap(lambda l :fetch_url(l,args['path'])).map(lambda l:clean(l)).filter(lambda l: l!=None)
#rdd_test = sc.parallelize(rdd_test)
#print("Test Zeros" + str(rdd_test.count()));


print("Download complete")



df_train_original = sqlContext.createDataFrame(rdd_train,schema=["text","class_label","key"])

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

df_tfidf_train = tfidf_processor(df_train_original,"text","tfidf_vector")
print("now processing tf-idf");
df_tfidf_train = build_labels(df_tfidf_train)

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

splits = [6.0,9.0,10.0][0]


training , testing = df_tfidf_train.randomSplit([0.7,0.3])

training = training.repartition(5000)
testing = testing.repartition(50000)
model_list = intialize_model()
for i in range(0,20):
	model_list = boosting(training,model_list)

prediction_and_label(model_list[0],testing)
prediction_and_label(model_list[1],testing)
prediction_and_label(model_list[2],testing)
prediction_and_label(model_list[3],testing)
prediction_and_label(model_list[4],testing)
prediction_and_label(model_list[5],testing)
prediction_and_label(model_list[6],testing)
prediction_and_label(model_list[7],testing)