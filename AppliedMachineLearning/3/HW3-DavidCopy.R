#------------------------------------- SETUP WORK --------------------------------------
#clear the workspace and console
rm(list=ls())
cat("\014") #code to send ctrl+L to the console and therefore clear the screen

#import libraries
library(klaR)
library(caret)
library(stringr)
library(randomForest)

#----------------------------------------retrieve data ----------------------------------------
#setwd("~/Dropbox/3")
setwd('~/../Dropbox/School/Spring 2016/Machine Learning/Homeworks/3')

#retrieve all the data (each row is 147 items long: 1 label followed by 2x73 feature vectors)
data <- read.table('pubfig_dev_50000_pairs_no_header.txt',sep='\t',header=F);

#grab the labels
labels <- data[,1];
#grab the features (2x 73 features vectors)
features <- matrix(NA,nrow=nrow(data),ncol=(ncol(data)-1)/2);
#features <- data[,-1]; #accuracy: .53

#pre-process the features to yield a single 73 item feature vector that is the scaled euclidian distance between each respective feature
features <- scale(abs(as.matrix(data[,2:74]) - as.matrix(data[,75:ncol(data)]))); #accuracy: 0.7684

#features <- abs(scale(as.matrix(data[,2:74]) - as.matrix(data[,75:ncol(data)]))); #accuracy: .79,  0.7758

#features <- scale(as.matrix(data[,2:74]) - as.matrix(data[,75:ncol(data)]))^2; #accuracy: .759, 0.7584


#-------------------------split up the data for testing and training------------------------------
#there is too much data for rapid iteration, use this to scale down how big the initial pool is during prototyping
useDataIndices <- createDataPartition(y=labels, p=.5, list=FALSE); 
testDataIndices <- createDataPartition(y=labels[useDataIndices], p=.2, list=FALSE);
trainingLabels <- labels[useDataIndices]; trainingLabels <- trainingLabels[-testDataIndices];
testLabels <- labels[useDataIndices]; testLabels <- testLabels[testDataIndices];
trainingFeatures <- features[useDataIndices,]; trainingFeatures <- trainingFeatures[-testDataIndices,];
testFeatures <- features[useDataIndices,]; testFeatures <- testFeatures[testDataIndices,];


########################
# Part 1: try various basic classifiers ---------------------------------------------------------
########################

#run an SVM
svm <- svmlight(trainingFeatures,trainingLabels) #,pathsvm='/Users/Cybelle/Documents/grad_school/school/classes/2015-2016_spring/cs498daf/svm_light')
predictedLabels<-predict(svm, testFeatures) 
foo<-predictedLabels$class #"foo" = class labels (1 or 0) for each item in test set
#get classification accuracy:
accuracy<-sum(foo==testLabels)/(sum(foo==testLabels)+sum(!(foo==testLabels)))

#run Naive Bayes
model<-train(trainingFeatures, as.factor(trainingLabels), 'nb', trControl=trainControl(method='cv', number=10))
teclasses<-predict(model,newdata=testFeatures)
cm<-confusionMatrix(data=teclasses, testLabels)
accuracy<-cm$overall[1] #accuracy on 15% of dataset w/ abs diff : .7318

#run Random Forest, accuracy = .7831 (abs then scale)
faceforest.allvals <- randomForest(x=trainingFeatures,y=trainingLabels,
                                   xtest=testFeatures,ytest=testLabels);
predictedLabels <- faceforest.allvals$test$predicted > .5
predictedLabels[predictedLabels] = 1;
predictedLabels[!predictedLabels] = 0;
foo <- predictedLabels;
accuracy<-sum(foo==testLabels)/(sum(foo==testLabels)+sum(!(foo==testLabels)))



#######################
#Part 2: Approximate Nearest Neighbors
#######################
# -have a reference dictionary of people (names) and many images of that person's face represented as attribute vectors
# -want to determine if two feature vectors represent face images of the same person... 
# -will use the reference dictionary as a lookup table, and simply find the nearest neighbor 
#   (in this case exact same feature vector) from the dictionary for both of the example's 
#   feature vectors. Compare the labels (names) of each found neighbor and see if they're the same name.
# -expect 100% accuracy here with a nearest neighbors classifier, but how close can we get to 100% with an 
#   approximate nearest neighbors package? Answer: basically 100%. 
# - Purpose: to demonstrate that approximate nearest neighbors can work just about the same as nearest neighbors

#install.packages("RANN");
require(RANN);

#retrieve the reference dictionary of names and associated face images
namedata <- read.table("pubfig_attributes.txt",header=F,sep="\t");
namedata.att <- namedata[,-c(1:2)];

n <- 10000;
face1.nn <- nn2(namedata.att,data[1:n,2:74],k=1);
face2.nn <- nn2(namedata.att,data[1:n,75:ncol(data)],k=1);

predicted <- namedata[face1.nn$nn.idx,1] == namedata[face2.nn$nn.idx,1]; #creates a boolean vector
predicted[predicted] = 1; #turn that boolean vector into 0's and 1's
accuracy = sum(data[1:n,1] == predicted) / length(predicted);
print(accuracy)

#perfecto!



#######################
#Part 3a: K Means Clustering + SVMLight
#######################
# Training:
#   -run kmeans clustering on the training data to create k clusters
#   -for each cluster
#     -train a separate svm classifier using the training data in that cluster
#
# Classify a test example:
#   -determine which cluster is closest to the example point
#   -use the svm from that chosen cluster to classify the example point

#accuracy at k = 5, 50% of dataset used, 20% test 80% train was 0.7614

#install.packages("stats")
require(stats)
require(svmlight)


k = 5;  #number of clusters to use... recall that there are only two classes
kmeans.train.output <- kmeans(trainingFeatures,centers=k); 

#the trainingLabels are provided as 0 and 1, but the svm wants them to map to -1 and 1... so convert 0's to -1
trainingLabels[trainingLabels==0] = -1;
testLabels[testLabels==0] = -1;

#train an SVM for each cluster
svm_list <- list();
for (i in 1:k){
  svm_list[[i]]<-svmlight(trainingFeatures[kmeans.train.output$cluster==i,], trainingLabels[kmeans.train.output$cluster==i], pathsvm='/Users/Cybelle/Documents/grad_school/school/classes/2015-2016_spring/cs498daf/svm_light')
}

#select the nearest cluster center for each test example
nearest_cluster_center <- nn2(kmeans.train.output$centers,testFeatures,k=1);


#create a structure to hold the predicted labels of any example near each SVM
predictedLabelsList = vector(mode = "list", length = k);
for(i in 1:k){
  if(sum(nearest_cluster_center$nn.idx==i) > 0){
    predictedLabelsList[[i]] = predict(svm_list[[i]], testFeatures[nearest_cluster_center$nn.idx==i,]) 
  }
}

#predict the label of the test examples using the svm associated with the nearest cluster to that test example
predictedLabels = rep(NA,nrow(testFeatures));
for(i in 1:k){
  if(sum(nearest_cluster_center$nn.idx==i) > 0 & !is.null(predictedLabelsList[[i]]$class)){
    predictedLabels[nearest_cluster_center$nn.idx==i] = as.numeric(predictedLabelsList[[i]]$class);
  }
}

predictedLabels[predictedLabels==1] = -1;
predictedLabels[predictedLabels==2] = 1;

accuracy = sum(predictedLabels == testLabels)/length(predictedLabels);
print(accuracy)

#######################
#Part 3b: Soft K Means Clustering + SVMLight
#######################
# Training:
#   -run kmeans clustering on the training data to create k clusters
#   -for each cluster
#     -train a separate svm classifier using the training data in that cluster
#
# Classify a test example:
#   -determine which cluster is closest to the example point
#   -use the svm from that chosen cluster to classify the example point

#accuracy at k = 5, 50% of dataset used, 20% test 80% train was 0.7614

#install.packages("cluster")
require(cluster)

k = 5;  #number of clusters to use... recall that there are only two classes
kmeans.train.output <- fanny(trainingFeatures[1:5000,],k,memb.exp=1.5); #not getting clear clusters

#the trainingLabels are provided as 0 and 1, but the svm wants them to map to -1 and 1... so convert 0's to -1
trainingLabels[trainingLabels==0] = -1;
testLabels[testLabels==0] = -1;

#train an SVM for each cluster
svm_list <- list();
for (i in 1:k){
  svm_list[[i]]<-svmlight(trainingFeatures[kmeans.train.output$cluster==i,], trainingLabels[kmeans.train.output$cluster==i], pathsvm='/Users/Cybelle/Documents/grad_school/school/classes/2015-2016_spring/cs498daf/svm_light')
}

#select the nearest cluster center for each test example
nearest_cluster_center <- nn2(kmeans.train.output$centers,testFeatures,k=1);


#create a structure to hold the predicted labels of any example near each SVM
predictedLabelsList = vector(mode = "list", length = k);
for(i in 1:k){
  if(sum(nearest_cluster_center$nn.idx==i) > 0){
    predictedLabelsList[[i]] = predict(svm_list[[i]], testFeatures[nearest_cluster_center$nn.idx==i,]) 
  }
}

#predict the label of the test examples using the svm associated with the nearest cluster to that test example
predictedLabels = rep(NA,nrow(testFeatures));
for(i in 1:k){
  if(sum(nearest_cluster_center$nn.idx==i) > 0 & !is.null(predictedLabelsList[[i]]$class)){
    predictedLabels[nearest_cluster_center$nn.idx==i] = as.numeric(predictedLabelsList[[i]]$class);
  }
}

predictedLabels[predictedLabels==1] = -1;
predictedLabels[predictedLabels==2] = 1;

accuracy = sum(predictedLabels == testLabels)/length(predictedLabels);
print(accuracy)

#######################
#Part 3c: Radial SVM with SVMLight
#######################

#run a radial basis kernal SVM (in theory, should bx like either linear or polynomial)
svm <- svmlight(trainingFeatures,trainingLabels, svm.options = "-t 2 -g .03")#,pathsvm='/Users/Cybelle/Documents/grad_school/school/classes/2015-2016_spring/cs498daf/svm_light')
predictedLabels<-predict(svm, testFeatures) 
foo<-predictedLabels$class #"foo" = class labels (1 or 0) for each item in test set
#get classification accuracy:
accuracy<-sum(foo==testLabels)/(sum(foo==testLabels)+sum(!(foo==testLabels))) #g = .1: .6624, 1: .5004, .01: 0.7956, .001: .7862, .015:




