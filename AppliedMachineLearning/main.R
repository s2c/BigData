#------------------------------------- SETUP WORK --------------------------------------
#clear the workspace and console
rm(list=ls())
cat("\014") #code to send ctrl+L to the console and therefore clear the screen

#setup working directory 
#setwd('~/../Dropbox/School/Spring 2016/Machine Learning/Homeworks/2')
#setwd('/Users/Cybelle/Documents/grad_school/school/classes/2015-2016_spring/cs498daf/hw/hw2/hw2_code_local_copy')

setwd('/Users/Cybelle/Dropbox/2/main.R');

#import libraries to help with data splitting/partitioning, 
#cross validation and easy classifier construction
library(klaR)
library(caret)




#------------------------------------- Acquire and Pre-Process Data------------------------------

#read all the data into a single table
allData <- read.csv('adult.txt', header = FALSE) 

#grab the labels from the main data file... use as.factor to make 
#the format comptabile with future functions to be used
labels <- as.factor(allData[,15])


#grab the features from the main data file, removing the labels 
#assume no data is missing... ie: ignore missing values without noting them as NA
allFeatures <- allData[,-c(15)]

#the continous features are in cols 1,3,5,11,12,13
continuousFeatures <- allFeatures[,c(1,3,5,11,12,13)]

#there are no ? (missing feature) for any of the continuous feature, so this modification is irrelevant
#adjust the features such that a 0 is reported as "NA"
#for (f in c(1,2,3,4,5,6))
#{
#  #determine which examples had a 0 for this feature (ie: unknown value)
#  examplesMissingFeatureF <- continuousFeatures[, f] == ?
#  #replace all these missing values with an NA
#  continuousFeatures[examplesMissingFeatureF, f] = NA
#}
#remove any example with an NA
#continuousFeatures[complete.cases(continuousFeatures),]

#normalize the features (mean center and scale so that variance = 1 i.e. convert to z scores):

continuousFeatures<-scale(continuousFeatures)

#convert labels into 1 or -1
labels.n = rep(0,length(labels));
labels.n[labels==" <=50K"] = -1;
labels.n[labels==">50K"] = 1;
labels = labels.n;
rm(labels.n);


#Separate the resulting dataset randomly
#break off 80% for training examples
trainingData <- createDataPartition(y=labels, p=.8, list=FALSE)
trainingFeatures <- continuousFeatures[trainingData,]
trainingLabels <- labels[trainingData]


#Of the remaining 20%, half become testing exmaples and half become validation examples
remainingLabels <- labels[-trainingData]
remainingFeatures <- continuousFeatures[-trainingData,]

testingData <- createDataPartition(y=remainingLabels, p=.5, list=FALSE)
testingLabels <- remainingLabels[testingData]
testingFeatures <- remainingFeatures[testingData,]

validationLabels <- remainingLabels[-testingData]
validationFeatures <- remainingFeatures[-testingData,]





#------------------------------------- SETUP Classifier --------------------------------------
numEpochs = 50;
numStepsPerEpoch = 300;

#set a and b to 0 initially:
a = rep(0,ncol(continuousFeatures));
b = 0;

c1 = 1 #set "a" for determining steplength
c2 = 1 #set "b" for determining steplength


for (e in 1:numEpochs){
  
  #select validation set for epoch:
  
  etestingData <- createDataPartition(y=testingData, p=.9, list=FALSE)
  etestingLabels <- remainingLabels[etestingData]
  etestingFeatures <- remainingFeatures[etestingData,]
  
  #set the steplength (eta)
  steplength = 1 / (e*c1 + c2);
    
  for (step in 1:numStepsPerEpoch){
    
    
    
  }
  
}




