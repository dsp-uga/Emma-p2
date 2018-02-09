from pyspark.sql import SQLContext


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import *
from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import udf,col,split
import argparse
import pyspark
from pyspark.ml.linalg import Vectors, VectorUDT



import requests
import os;
from pyspark.sql.types  import *
import sys
PATH = "https://storage.googleapis.com/uga-dsp/project2/data/bytes"

#"local[*]",'pyspark tuitorial'
conf = SparkConf()
conf = conf.setMaster("local[*]")
conf =conf.set("spark.driver.memory", "45G")
#conf.set("spark.driver.cores", 4)

sc = SparkContext('local[*]','Team Emma',conf=conf)

sqlContext = SQLContext(sc)

current_class = 0

#.set("spark.executor.memory", "4g")

def build_labels(df_tfidf_train):
	current_class = 0;
	df_tfidf_train = df_tfidf_train.withColumn("zero",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 1
	df_tfidf_train = df_tfidf_train.withColumn("one",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 2
	df_tfidf_train = df_tfidf_train.withColumn("two",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 3
	df_tfidf_train = df_tfidf_train.withColumn("three",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 4
	df_tfidf_train = df_tfidf_train.withColumn("four",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 5
	df_tfidf_train = df_tfidf_train.withColumn("five",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 6
	df_tfidf_train = df_tfidf_train.withColumn("six",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 7
	df_tfidf_train = df_tfidf_train.withColumn("seven",one_vs_all_udf(df_tfidf_train.class_label))
	current_class = 8
	df_tfidf_train = df_tfidf_train.withColumn("eight",one_vs_all_udf(df_tfidf_train.class_label))
	return df_tfidf_train

def merge_col(*x):
	temp = list();
	for i in x:
		temp.extend(i)
	return temp


def prediction_and_label(model,dataset,label):
	predictions = model.transform(dataset)
	predictions.printSchema()
	evaluator = MulticlassClassificationEvaluator(
    labelCol=label, predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	return (accuracy);


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
	count = 0
	temp = x[0].split(' ')[1:]
	for i in range(0,len(temp)):
		if temp[i] == '00' or temp[i] == '??':
			count = count + 1
	if count == len(temp):
		return None
	
	
	
	return (temp,x[1])


def one_vs_all(x):
		if x[0] ==current_class:
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
rdd_train_y = sc.textFile(args['train_y']).zipWithIndex().map(lambda l:(float(l[1]-1),l[0]));
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



df_train_original = sqlContext.createDataFrame(rdd_train,schema=["text","class_label"])
#df_test_original = sqlContext.createDataFrame(rdd_test,schema=["text","class_label"])
#df_train_original = df_train_original.repartition(10000)
	#df_test_original = df_test_original.repartition(30000) 

	#df_train_orignal ,df_train_orignal_validate =df_train_orignal.randomSplit([0.7,0.3])


df_train_original.printSchema()

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

training_0,testing_0=df_tfidf_train.randomSplit([0.6,0.4])
training_1,testing_1=df_tfidf_train.randomSplit([0.6,0.4])

training_2,testing_2=df_tfidf_train.randomSplit([0.6,0.4])

training_3,testing_3=df_tfidf_train.randomSplit([0.6,0.4])

training_4,testing_4=df_tfidf_train.randomSplit([0.6,0.4])

training_5,testing_5=df_tfidf_train.randomSplit([0.6,0.4])

training_6,testing_6=df_tfidf_train.randomSplit([0.6,0.4])
training_7,testing_7=df_tfidf_train.randomSplit([0.6,0.4])


model_0 = LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="zero", featuresCol="tfidf_vector").fit(training_0)
model_1 = LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="one", featuresCol="tfidf_vector").fit(training_1)
model_2= LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="two", featuresCol="tfidf_vector").fit(training_2)
model_3= LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="three", featuresCol="tfidf_vector").fit(training_3)
model_4= LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="four", featuresCol="tfidf_vector").fit(training_4)
model_5= LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="five", featuresCol="tfidf_vector").fit(training_5)
model_6= LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="six", featuresCol="tfidf_vector").fit(training_6)
model_7= LogisticRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8,labelCol="seven", featuresCol="tfidf_vector").fit(training_7)

value_0 = prediction_and_label(model_0,training_0,"zero")
value_1 = prediction_and_label(model_1,training_1,"one")
value_2  = prediction_and_label(model_2,training_2,"two")
value_3  = prediction_and_label(model_3,training_3,"three")
value_4  = prediction_and_label(model_4,training_4,"four")
value_5  = prediction_and_label(model_5,training_5,"five")
value_6  = prediction_and_label(model_6,training_6,"six")
value_7  = prediction_and_label(model_7,training_7,"seven")

value_01= prediction_and_label(model_0,testing_0,"zero")
value_11 = prediction_and_label(model_1,testing_1,"one")
value_21  = prediction_and_label(model_2,testing_2,"two")
value_31  = prediction_and_label(model_3,testing_3,"three")
value_41  = prediction_and_label(model_4,testing_4,"four")
value_51  = prediction_and_label(model_5,training_5,"five")
value_61  = prediction_and_label(model_6,training_6,"six")
value_71  = prediction_and_label(model_7,training_7,"seven")


print(str(value_0)+":" + str(value_1) + ":" + str(value_2) +";"+  str(value_3) + ";"+ str(value_4) +  ";"+ str(value_5) + ";"+ str(value_6) + ";"+ str(value_7))
print(str(value_01)+":" + str(value_11) + ":" + str(value_21) +";"+  str(value_31) + ";"+ str(value_41)  + ";"+ str(value_51) + ";"+ str(value_61) + ";"+ str(value_71))
