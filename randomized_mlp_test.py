
from pyspark.sql import SQLContext
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

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
conf =conf.set("spark.driver.memory", "12G")
conf =conf.set("spark.executor.memory", "2G")
#conf.set("spark.driver.cores", 4)

sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

current_class = 0

#.set("spark.executor.memory", "4g")


layers = [100, 50,25,12,10,8,4, 2]
def intialize_model():

	model = list();

	model_0 = MultilayerPerceptronClassifier(maxIter=100,labelCol="one", layers=layers, blockSize=1, seed=123)
	model_1 = MultilayerPerceptronClassifier(maxIter=100,labelCol="two", layers=layers, blockSize=1, seed=123)
	model_2= MultilayerPerceptronClassifier(maxIter=100,labelCol="three", layers=layers, blockSize=1, seed=123)
	model_3= MultilayerPerceptronClassifier(maxIter=100,labelCol="four", layers=layers, blockSize=1, seed=123)
	model_4= MultilayerPerceptronClassifier(maxIter=100,labelCol="five", layers=layers, blockSize=1, seed=123)
	model_5= MultilayerPerceptronClassifier(maxIter=100,labelCol="six", layers=layers, blockSize=1, seed=123)
	model_6= MultilayerPerceptronClassifier(maxIter=100,labelCol="seven", layers=layers, blockSize=1, seed=123)
	model_7= MultilayerPerceptronClassifier(maxIter=100,labelCol="eight", layers=layers, blockSize=1, seed=123)	
	model_8= MultilayerPerceptronClassifier(maxIter=100,labelCol="nine", layers=layers, blockSize=1, seed=123)	

	model.append(model_0)
	model.append(model_1)
	model.append(model_2)
	model.append(model_3)
	model.append(model_4)
	model.append(model_5)
	model.append(model_6)
	model.append(model_7)
	model.append(model_8)



	return model


def update_model(old_model):
	model = list()
	model_0 = MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="one", blockSize=1, seed=123)
	model_0.setInitialWeights(old_model[0].weights)
	model_1 = MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="two", blockSize=1, seed=123)
	model_1.setInitialWeights(old_model[1].weights)
	model_2= MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="three", blockSize=1, seed=123)
	model_2.setInitialWeights(old_model[2].weights)
	model_3= MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="four", blockSize=1, seed=123)
	model_3.setInitialWeights(old_model[3].weights)
	model_4= MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="five", blockSize=1, seed=123)
	model_4.setInitialWeights(old_model[4].weights)
	model_5= MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="six", blockSize=1, seed=123)
	model_5.setInitialWeights(old_model[5].weights)
	model_6= MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="seven", blockSize=1, seed=123)
	model_6.setInitialWeights(old_model[6].weights)
	model_7= MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="eight", blockSize=1, seed=123)	
	model_7.setInitialWeights(old_model[7].weights)
	model_8= MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol="nine", blockSize=1, seed=123)	
	model_8.setInitialWeights(old_model[7].weights)

	model.append(model_0)
	model.append(model_1)
	model.append(model_2)
	model.append(model_3)
	model.append(model_4)
	model.append(model_5)
	model.append(model_6)
	model.append(model_7)
	model.append(model_8)


	return model




split_train = 0.100
limit =200000





def boosting(df_tfidf_train,model,labelCol):
		accuracy = [0 for i in range(0,8)]

		print("performing boosting")
		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		temp = model.fit(training)
	
		model = MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol=labelCol, blockSize=1, seed=123)
		model.setInitialWeights(temp.weights)
		print("iteration 2")
		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		temp = model.fit(training)
		
		model = MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol=labelCol, blockSize=1, seed=123)
		model.setInitialWeights(temp.weights)
		print("iteratin 3")

		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		temp = model.fit(training)
		print("iteration 4")
		model = MultilayerPerceptronClassifier(maxIter=100, layers=layers,labelCol=labelCol, blockSize=1, seed=123)
		model.setInitialWeights(temp.weights)
		training=df_tfidf_train.sample(False,split_train,seed=42).limit(limit)
		temp = model.fit(training)
		

		return temp
#calculate per col

	
def build_labels(df_tfidf_train):
	current_class = 1;
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
	current_class = 9
	df_tfidf_train = df_tfidf_train.withColumn("nine",one_vs_all_udf(df_tfidf_train.class_label,lit(9)))
	return df_tfidf_train

def merge_col(*x):
	temp = list();
	for i in x:
		temp.extend(i)
	return temp


def prediction_and_label(model,dataset,class_label,validation=False):

	predictions = model.transform(dataset)

	if validation:
		evaluator = MulticlassClassificationEvaluator(
		labelCol=class_label, predictionCol="prediction", metricName="accuracy")
		accuracy = evaluator.evaluate(predictions)
		print("Class: "+ class_label + " Accuracy: "+str(accuracy))
		return accuracy,predictions.repartition(3000)
	else:
		return predictions.repartition(3000)



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


def clean_test(x):
	label = 0
	key = x[2]
	count = 0
	class_label = x[2]
	temp = x[0].split(' ')[1:]
	for i in range(0,len(temp)):
		if temp[i] == '00' or temp[i] == '??':
			count = count + 1
	if count == len(temp):
		return None
	
	
	
	return (temp,key)


#https://stackoverflow.com/questions/31898964/how-to-write-the-resulting-rdd-to-a-csv-file-in-spark-python
def toCSVLine(data):
  return ','.join(str(d) for d in data)


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
rdd_test = sc.textFile(args['test_x']).zipWithIndex().map(lambda l:(l[1],l[0]))#.take(2);
#rdd_test_y = sc.textFile(args['test_y']).zipWithIndex().map(lambda l:(float(l[1]-1),l[0]));
rdd_train = rdd_train_x.join(rdd_train_y)#.take(2)

#rdd_test = sc.parallelize(rdd_test)
#rdd_train = sc.parallelize(rdd_train)
#print("Test Zeros" + str(rdd_test.count()));
#rdd_test = rdd_test_x.join(rdd_test_y)
#take 30 due to gc overhead


rdd_train = rdd_train.flatMap(lambda l :fetch_url(l,args['path'])).map(lambda l:clean(l)).filter(lambda l:l !=None).repartition(10000)


#piigy back the rdd intoder to main the flow same

rdd_test = rdd_test.flatMap(lambda l :fetch_url(l,args['path'])).map(lambda l:clean_test(l)).filter(lambda l:l !=None).repartition(10000)



rdd_train = rdd_train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

rdd_test = rdd_train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)


#rdd_test= rdd_test.flatMap(lambda l :fetch_url(l,args['path'])).map(lambda l:clean(l)).filter(lambda l: l!=None)
#rdd_test = sc.parallelize(rdd_test)
#print("Test Zeros" + str(rdd_test.count()));




df_train_original = sqlContext.createDataFrame(rdd_train,schema=["text","class_label","key"])


testing = sqlContext.createDataFrame(rdd_train,schema=["text","key"])

training = df_train_original.randomSplit([0.4,0.6])[0]
training = training.limit(800000)


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

training = tfidf_processor(training,"text","features")
testing = tfidf_processor(testing,"text","features")
print("now processing tf-idf");
training = build_labels(training)


training = training.repartition(5000)
testing= testing.repartition(5000)
#df_tfidf_test = tfidf_processor(df_test_original,"text","tfidf_vector");
#df_tfidf_test = df_tfidf_test.rdd.map(lambda l:[l[1],l[-1].toArray()])
#df_tfidf_train= df_tfidf_train.rdd.map(lambda l:[l[1],l[-1].toArray()]).repartition(10000)
#df_tfidf_train = df_tfidf_train.randomSplit([0.05,05,0.99])[0:]
#print("processing complete");

#print(df_tfidf_test.take(20)[-1][-1])
#df_train_vectorizer.show();
#df_test_vectorizer.show();p
#df_tfidf_train.show()
#df_tfidf_test.show()


# Split data approximately into training (60%) and test (40%)"""




model_list = intialize_model()

print("Model 1")
model_list[0]= boosting(training,model_list[0],"one")
prediction_1  =prediction_and_label(model_list[0],testing,"one")

print("Model 2")
model_list[1]= boosting(training,model_list[1],"two")
prediction_2  = prediction_and_label(model_list[1],testing,"two")

print("Model 3")
model_list[2]= boosting(training,model_list[2],"three")
prediction_3 = prediction_and_label(model_list[2],testing,"three")

print("Model 4")
model_list[3]= boosting(training,model_list[3],"four")
prediction_4  = prediction_and_label(model_list[3],testing,"four")

print("Model 5")
model_list[4]= boosting(training,model_list[4],"five")
prediction_5  = prediction_and_label(model_list[4],testing,"five")
print("Model 6")
model_list[5]= boosting(training,model_list[5],"six")
prediction_6  = prediction_and_label(model_list[5],testing,"six")
print("Model 7")
model_list[6]= boosting(training,model_list[6],"seven")
prediction_7  = prediction_and_label(model_list[6],testing,"seven")
print("Model 8")
model_list[7]= boosting(training,model_list[7],"eight")
prediction_8 = prediction_and_label(model_list[7],testing,"eight")
print("Model 9")
model_list[8]= boosting(training,model_list[8],"nine")
prediction_9 = prediction_and_label(model_list[8],testing,"nine")

training.unpersist();


prediction_1  = prediction_1.withColumn("predictions", col("prediction").cast("int"))
prediction_2  = prediction_2.withColumn("predictions", col("prediction").cast("int"))
prediction_3  = prediction_3.withColumn("predictions", col("prediction").cast("int"))
prediction_4  = prediction_4.withColumn("predictions", col("prediction").cast("int"))
prediction_5  = prediction_5.withColumn("predictions", col("prediction").cast("int"))
prediction_6  = prediction_6.withColumn("predictions", col("prediction").cast("int"))
prediction_7  = prediction_7.withColumn("predictions", col("prediction").cast("int"))
prediction_8  = prediction_8.withColumn("predictions", col("prediction").cast("int"))
prediction_9  = prediction_9.withColumn("predictions", col("prediction").cast("int"))


prediction_1 = prediction_1.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_1')
prediction_2 = prediction_2.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_2')
prediction_3 = prediction_3.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_3')
prediction_4 = prediction_4.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_4')
prediction_5 = prediction_5.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_5')
prediction_6 = prediction_6.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_6')
prediction_7 = prediction_7.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_7')
prediction_8 = prediction_8.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_8')
prediction_9 = prediction_9.groupBy("key").sum("predictions").withColumnRenamed('sum(predictions)', 'prediction_9')



prediction = prediction_1.join(prediction_2,prediction_1.key==prediction_2.key).drop(prediction_2.key)
prediction = prediction.join(prediction_3,prediction.key==prediction_3.key).drop(prediction_3.key)
prediction = prediction.join(prediction_4,prediction.key==prediction_4.key).drop(prediction_4.key)
prediction = prediction.join(prediction_5,prediction.key==prediction_5.key).drop(prediction_5.key)
prediction = prediction.join(prediction_6,prediction.key==prediction_6.key).drop(prediction_6.key)
prediction = prediction.join(prediction_7,prediction.key==prediction_7.key).drop(prediction_7.key)
prediction = prediction.join(prediction_8,prediction.key==prediction_8.key).drop(prediction_8.key)
prediction = prediction.join(prediction_9,prediction.key==prediction_9.key).drop(prediction_9.key)
prediction.show();


output = prediction.rdd.coalesce(1).map(toCSVLine)
output.saveAsTextFile('gs://dspproject2/result_large.txt')

#prediction = prediction.join(prediction_3,prediction.key==prediction_3.key).select(prediction["key"],"prediction_1","prediction_2","prediction_3")

