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
from random import shuffle

#PATH = "https://storage.googleapis.com/uga-dsp/project2/data/bytes"

#"local[*]",'pyspark tuitorial'

#Spark configuration in
conf = SparkConf()
conf =conf.set("spark.driver.memory", "90G")
conf = conf.set("spark.yarn.executor.memoryOverhead","5G")
conf = conf.set("spark.yarn.driver.memoryOverhea","5G")
#conf.set("spark.driver.cores", 4)


sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

current_class = 0

#.set("spark.executor.memory", "4g")


layers = [100,80,60,40,20,10,5,2]

def chain_sampler(dataset,limit):
	training_sample_1=dataset.filter(df_tfidf_train.class_label==1).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_2=dataset.filter(df_tfidf_train.class_label==2).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_3=dataset.filter(df_tfidf_train.class_label==3).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_4=dataset.filter(df_tfidf_train.class_label==4).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_5=dataset.filter(df_tfidf_train.class_label==5).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_6=dataset.filter(df_tfidf_train.class_label==6).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_7=dataset.filter(df_tfidf_train.class_label==7).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_8=dataset.filter(df_tfidf_train.class_label==8).randomSplit([0.7,0.3])[0].limit(limit)
	training_sample_9=dataset.filter(df_tfidf_train.class_label==9).randomSplit([0.7,0.3])[0].limit(limit)

	training = training_sample_1.union(training_sample_2)
	training = training.union(training_sample_3)
	training = training.union(training_sample_4)
	training = training.union(training_sample_5)
	training = training.union(training_sample_6)
	training = training.union(training_sample_7)
	training = training.union(training_sample_8)
	training = training.union(training_sample_9)
	return training







def intialize_model():

	model = list();

	model_0 = MultilayerPerceptronClassifier(maxIter=5000,labelCol="one", layers=layers, blockSize=2, seed=123)
	model_1 = MultilayerPerceptronClassifier(maxIter=5000,labelCol="two", layers=layers, blockSize=2, seed=123)
	model_2= MultilayerPerceptronClassifier(maxIter=5000,labelCol="three", layers=layers, blockSize=2, seed=123)
	model_3= MultilayerPerceptronClassifier(maxIter=5000,labelCol="four", layers=layers, blockSize=2, seed=123)
	model_4= MultilayerPerceptronClassifier(maxIter=5000,labelCol="five", layers=layers, blockSize=2, seed=123)
	model_5= MultilayerPerceptronClassifier(maxIter=5000,labelCol="six", layers=layers, blockSize=2, seed=123)
	model_6= MultilayerPerceptronClassifier(maxIter=5000,labelCol="seven", layers=layers, blockSize=2, seed=123)
	model_7= MultilayerPerceptronClassifier(maxIter=5000,labelCol="eight", layers=layers, blockSize=2, seed=123)	
	model_8= MultilayerPerceptronClassifier(maxIter=5000,labelCol="nine", layers=layers, blockSize=2, seed=123)	

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






split_train = 0.900
limit =60000





def boosting(df_tfidf_train,model,labelCol,index):
		accuracy = [0 for i in range(0,8)]

		print("performing boosting")
		training_sample=df_tfidf_train.filter(df_tfidf_train.class_label==index).randomSplit([0.7,0.3])[0].limit(50000)
		training_sample_1=df_tfidf_train.filter(df_tfidf_train.class_label!=index).randomSplit([0.7,0.3])[0].limit(50000)
		#.sample(True,split_train,seed=42).limit(10000)
		training_sample=training_sample.union(training_sample_1)
		temp = model.fit(training_sample)
		
		model = MultilayerPerceptronClassifier(maxIter=5000, layers=layers,labelCol=labelCol, blockSize=128, seed=123)
		model.setInitialWeights(temp.weights)
		print("performing boosting-0")
		training_sample=df_tfidf_train.filter(df_tfidf_train.class_label==index).randomSplit([0.7,0.3])[0].limit(50000)
		training_sample_1=df_tfidf_train.filter(df_tfidf_train.class_label!=index).randomSplit([0.7,0.3])[0].limit(50000)
		#.sample(True,split_train,seed=42).limit(10000)
		training_sample=training_sample.union(training_sample_1)
		temp = model.fit(training_sample)

		model = MultilayerPerceptronClassifier(maxIter=5000, layers=layers,labelCol=labelCol, blockSize=128, seed=123)
		model.setInitialWeights(temp.weights)
		print("performing boosting-1")
		training_sample=df_tfidf_train.filter(df_tfidf_train.class_label==index).randomSplit([0.7,0.3])[0].limit(50000)
		training_sample_1=df_tfidf_train.filter(df_tfidf_train.class_label!=index).randomSplit([0.7,0.3])[0].limit(50000)
		#.sample(True,split_train,seed=42).limit(10000)
		training_sample=training_sample.union(training_sample_1)
		temp = model.fit(training_sample)

		model = MultilayerPerceptronClassifier(maxIter=5000, layers=layers,labelCol=labelCol, blockSize=128, seed=123)
		model.setInitialWeights(temp.weights)
		print("performing boosting-2")
		training_sample=df_tfidf_train.filter(df_tfidf_train.class_label==index).randomSplit([0.7,0.3])[0].limit(50000)
		training_sample_1=df_tfidf_train.filter(df_tfidf_train.class_label!=index).randomSplit([0.7,0.3])[0].limit(50000)
		#.sample(True,split_train,seed=42).limit(10000)
		training_sample=training_sample.union(training_sample_1)
		temp = model.fit(training_sample)
		
		model = MultilayerPerceptronClassifier(maxIter=5000, layers=layers,labelCol=labelCol, blockSize=128, seed=123)
		model.setInitialWeights(temp.weights)
		print("performing boosting-3")
		training_sample=df_tfidf_train.filter(df_tfidf_train.class_label==index).randomSplit([0.7,0.3])[0].limit(50000)
		training_sample_1=df_tfidf_train.filter(df_tfidf_train.class_label!=index).randomSplit([0.7,0.3])[0].limit(50000)
		#.sample(True,split_train,seed=42).limit(10000)
		training_sample=training_sample.union(training_sample_1)
		temp = model.fit(training_sample)
		
		model = MultilayerPerceptronClassifier(maxIter=5000, layers=layers,labelCol=labelCol, blockSize=128, seed=123)
		model.setInitialWeights(temp.weights)
		print("performing boosting-4")
		training_sample=df_tfidf_train.filter(df_tfidf_train.class_label==index).randomSplit([0.7,0.3])[0].limit(50000)
		training_sample_1=df_tfidf_train.filter(df_tfidf_train.class_label!=index).randomSplit([0.7,0.3])[0].limit(50000)
		#.sample(True,split_train,seed=42).limit(10000)
		training_sample=training_sample.union(training_sample_1)
		temp = model.fit(training_sample)
		
		

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
		print("------------------")
		print("Class: "+ class_label + " Accuracy: "+str(accuracy))
		print("------------------")
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




def word2ved_processor(df,text,features):
	wv_train =  Word2Vec(inputCol=text,outputCol=features).setVectorSize(30)
	print("building wor2vec model")
	model = wv_train.fit(df)
	print("word2vec transformation complete")
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




def fetch_url_testing(x,path,testing=False):
	key = x[0]
	class_label = x[1][1]
	url = x[1][0]
	fetch_url = path+"/"+url+".bytes"
	text = requests.get(fetch_url).text
	entries = text.split(os.linesep)

	shuffle(entries)

	entries = [i.strip().replace("'","") for i in  entries]
	entries = [i for i in entries if '?' not in entries]
	return (entries,key)



def fetch_url(x,path):
	key = x[0]
	class_label = x[1][1]
	url = x[1][0]
	fetch_url = path+"/"+url+".bytes"
	text = requests.get(fetch_url).text
	entries = text.split(os.linesep)

	shuffle(entries)

	entries = [i.strip().replace("'","") for i in  entries]
	entries = [i for i in entries if '?' not in entries]


	return (entries,class_label,key)

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
	key = x[1]
	count = 0
	class_label = x[0]
	temp = x[0].split(' ')[1:]
	for i in range(0,len(temp)):
		if temp[i] == '00' or temp[i] == '??':
			count = count + 1
	if count == len(temp):
		return None
	
	return (key,temp)


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
parser.add_argument('-d','--test_y', type=str,
                    #help='testing y set')
parser.add_argument('-e','--path', type=str,
                    help='path to folder')
parser.add_argument('-o','--output', type=str,
                    help='path to folder')

args = vars(parser.parse_args())
output_path = args['output']
#print(args)
rdd_train_x = sc.textFile(args['train_x']).zipWithIndex().map(lambda l:(l[1],l[0]))
rdd_train_y = sc.textFile(args['train_y']).zipWithIndex().map(lambda l:(float(l[1]),l[0]));
rdd_test = sc.textFile(args['test_x']).zipWithIndex().map(lambda l:(l[1],l[0]))
#rdd_test_y = sc.textFile(args['test_y']).zipWithIndex().map(lambda l:(float(l[1]),l[0]));
rdd_train = rdd_train_x.join(rdd_train_y)




#rdd_test = sc.parallelize(rdd_test)
#rdd_test_y = sc.textFile(args['test_y']).zipWithIndex().map(lambda l:(float(l[1]-1),l[0]));
#rdd_train = rdd_train_x.join(rdd_train_y)#.take(2)

#print("Test Zeros" + str(rdd_test.count()));
#rdd_test = rdd_test_x.join(rdd_test_y)
#take 30 due to gc overhead


rdd_train = rdd_train.flatMap(lambda l :fetch_url(l,args['path'])).filter(lambda l:l !=None).repartition(1000)

rdd_test = rdd_test.flatMap(lambda l :fetch_url_testing(l,args['path'])).map(lambda l:clean_test(l)).filter(lambda l:l !=None).repartition(1000)


#piigy back the rdd intoder to main the flow same

#rdd_test = rdd_test.flatMap(lambda l :fetch_url_testing(l,args['path'])).map(lambda l:clean_test(l)).filter(lambda l:l !=None).repartition(1000)



rdd_train = rdd_train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

rdd_test = rdd_test.persist(pyspark.StorageLevel.MEMORY_AND_DISK)


#rdd_test= rdd_test.flatMap(lambda l :fetch_url(l,args['path'])).map(lambda l:clean(l)).filter(lambda l: l!=None)
#rdd_test = sc.parallelize(rdd_test)
#print("Test Zeros" + str(rdd_test.count()));



training = sqlContext.createDataFrame(rdd_train,schema=["text","class_label","key"]).randomSplit([0.8,0.2])[0]
testing = sqlContext.createDataFrame(rdd_train,schema=["text","key"])



#testing = sqlContext.createDataFrame(rdd_test,schema=["key","text"]).randomSplit([0.7,0.3])[0]


#print(training.count())


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


#ngram code 
#df_train_ngram = ngram_processor(training,n_count=2);
#df_test_ngram = ngram_processor(testing,n_count=2)

#df_train_ngram.show()
#df_test_ngram.show();

#count vectorizer + ngram
#df_train_vectorizer = count_vectorizer_processor(df_train_ngram,"merge_text_array")
#df_test_vectorizer = count_vectorizer_processor(df_test_ngram,"merge_text_array")



#testing = word2ved_processor(testing,"text","features")
#training = word2vec_processor(testing,"text","features")

training = tfidf_processor(training,"text","features")
testing = tfidf_processor(testing,"text","features")

model = NaiveBayes.train(training, 1.0)
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print(predictionAndLabel.map(lambda x:x[1]).collect())
print('model accuracy {}'.format(accuracy))

