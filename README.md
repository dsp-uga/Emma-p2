# Malware-Classification
# Team-Emma
# Authors: Prajay Shetty, Hiten Nirmal, Maulik Shah

Project Definition:
The end goal of this project is to classify Malewares into 9 different categories given nearly half a terabyte of uncompressed data from the Microsoft Malware Classification Challenge. The data is made up of around 8500 byte files containing only hexadecimal codes. The challange is to crate a model that can classify around 2700 test unlabled data files. For a bit of help, there is a subset of the data available to test the code on the local machines. 

All the training for the large data sets are done on the google cloud Dataproc APIs, which are available through education funds given for this class only.


# Methodology 
There are few approaches we used for creating a model, which are Naive Bayes, Random Forest,Deep Neural Networks and Logistic regression,etc. 
1. Pre-processing: Overall the task is to clean the data by removing extra 00s and CCs, remove new lines, return feeds and extra white spaces. There are two different approaches we took for preprocessing the data:
  a. File based approach: Treat file as a document(NLP term) and process it likewise.
  b. Line based approach : Devide the files into separate lines, clea it,and label each line from it's parent file. 
 Apart from that, we used N-grams for creating word features. The hashed TF vector worked for all our approaches. 
 TODO: The n-gram model used in the pipeline was just for a particular level of n-grams. E.g. the 4-gram models only contains group of 4 words, the 1,2 or 3 word combinations are not there. Due to lack of clearity in how to implement it in the pipeline, the approach haven't got implemented. 
  
2. Sampling: There are different sampling we used for different approaches, which are as below:
   a. For the Naive-Bayes, Random-Forest and Linear Regression, the 60-40 split of training and validation is made.
   b. For the Deep Learning network, we took random samples of random size to train each perceptron. 
   
3. Models: The models and their performance is as below:
   a. Naive Bayes: After a lot of hardwork, we were able to train 60% of training data in 30 minutes on google cloud, but the problem with this approach is lack of accuracy. 
   Potential Bugs: The model predicts everything as label 3, which is the largest class in the corpus. 
   b. Deep Neural Nets: TODO: Prajay to update this.
   c. Ramdom Forest: TODO: Hiten to update this.
   d. Logistic Regression: TODO: Prajay to update this.
   
4. Performance Tuning: The biggest headache for the project was performance tuning. Some measures we took for performance tuning are: 
a. Set the Pyspark properties to use optimal resources of the clusters: Setting the driver and executer memory, and processor cores were very much helpful. Maximum output size set at the end helped to get rid of training the large data problem.
b. Use of Collect: collecting the data at the right time helped a lot to reduce the executor memory overload issues. 
c. Storing the data on the memory and disk: This option helped to process the data offline.
d. Repartitioning: Repartitioning RDDs were the final weapon, which helped us to remove all the memory overhead issues. Repartitioning RDD to 500 partition, helped to cope up with frequent memory failiours. 
 

# CodeBase:
1. Datafiles: There are two sets of data files.
  a. FileNames and Labels: Each line in this files contain either a file name or a lable. They are being put in the 'small' folder for reference. For fetching it into the program, any web locaiton can be passed in the arguments. 
  b. NaiveBayes: The naiveBayes.py file contains consolidated code for preprocessing and modeling for this approach. The file can be run as : 
 --spark-submit PATH_TO_FILE/naiveBayes.py -aPATH_TO_FILE/X_train.txt -bPATH_TO_FILE/y_train.txt -cPATH_TO_FILE/X_test.txt -dPATH_TO_FILE/y_test.txt -ehttps://storage.googleapis.com/uga-dsp/project2/data/bytes 


NOTE: Here, because we didn't want to change the code for all the datasets, we created y_test.txt files with all '1's for the number of files in X_test.txt file.
  
#Using Google CLoud

Data Storage- where main python file is saved along with sample data

DataProc- Creating a cluster and submitting a job

Before creating a cluster make sure a billing account is added to that project. Open Google Cloud consloe--Billing--add billing details.

Create a cluster-gcloud dataproc clusters create cluster-name Manually set master and worker configuration by using GCP console.

Setting up a Job: gcloud dataproc jobs submit spark --cluster cluster-name -mainpthonfile.py-arguments

# Contributors:
Prajay - 
Cross Validation: Implemented random based Cross validator 
MLP :Implemented random mlp  and randomized Logistic regression
Hashing TF: Used hashing tf for feature extraction
Line wise prediction: Predicted the classificaiton via line not based on files and then converted to file space (Embedding)

All the code is available under mlp branch

Hiten - Implementation of Naive-Bayes, Random Forest in DataFrame /Pipeline, Analysis, processing of data, Google CLoud

Maulik - Implementation of the Naive-Bayes,preprocessing,performance tuning and documentation. 

# Future Works:
Implement N-grams in efficient way into the pipeline to update efficiency. Check if the Naive-bayes is bug free or not.
Implement full version of code and make it bug free and retest on large dataset


# References
http://spark.apache.org/docs/2.1.0/api/python/pyspark.html
https://spark.apache.org/docs/2.2.0/ml-features.html
https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes
https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#
https://spark.apache.org/docs/latest/ml-guide.html
https://spark.apache.org/docs/latest/configuration.html

https://web.stanford.edu/class/cs124/lec/naivebayes.pdf


https://www.hackingnote.com/en/spark/trouble-shooting/total-size-is-bigger-than-maxResultSize/
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/3783546674231736/4413065072037724/latest.html
https://jaceklaskowski.gitbooks.io/mastering-apache-spark/spark-webui-StagePage.html
http://blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-2/
https://stackoverflow.com/questions/32336915/pyspark-java-lang-outofmemoryerror-java-heap-space
https://community.hortonworks.com/articles/80301/spark-configuration-and-best-practice-advice.html
