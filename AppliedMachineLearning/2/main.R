#------------------------------------- SETUP WORK --------------------------------------
#clear the workspace and console
rm(list=ls())
cat("\014") #code to send ctrl+L to the console and therefore clear the screen

#setup working directory 
setwd('~/../Dropbox/School/Spring 2016/Machine Learning/Homeworks/2')
#setwd('/Users/Cybelle/Documents/grad_school/school/classes/2015-2016_spring/cs498daf/hw/hw2/hw2_code_local_copy')
#setwd('/Users/Cybelle/Dropbox/2/main.R');

#import libraries to help with data splitting/partitioning, 
#cross validation and easy classifier construction
library(klaR)
library(caret)
library(stringr)



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
labels.n[labels==" >50K"] = 1;
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



#------------------------------------- DEFINE AN ACCURACY MEASURE ----------------------------
getAccuracy <- function(a,b,features,labels){
  estFxn = features %*% a + b;
  predictedLabels = rep(0,length(labels));
  predictedLabels [estFxn < 0] = -1 ;
  predictedLabels [estFxn >= 0] = 1 ;
  
  return(sum(predictedLabels == labels) / length(labels))
}



#------------------------------------- SETUP Classifier --------------------------------------
numEpochs = 100;
numStepsPerEpoch = 500;
nStepsPerPlot = 30;


evalidationSetSize = 50;

c1 = 0.01 #set "a" for determining steplength
c2 = 50 #set "b" for determining steplength
lambda_vals = c(0.001, 0.01, 0.1, 1);
bestAccuracy = 0;
accMat <- matrix(NA, nrow = (numStepsPerEpoch/nStepsPerPlot)*numEpochs+1, ncol = length(lambda_vals)); #vector for storing accuracy for each epoch
accMatv <- matrix(NA, nrow = (numStepsPerEpoch/nStepsPerPlot)*numEpochs+1, ncol = length(lambda_vals)); #accuracy on validation set (not epoch validation set)

for(i in 1:4){
  lambda = lambda_vals[i];
  accMatRow = 1;
  accMatCol = i; #changes with the lambda
  
  #set a and b to 0 initially:
  a = rep(0,ncol(continuousFeatures));
  b = 0;
  
  stepIndex = 0;
  
  for (e in 1:numEpochs){
    
    #divide into training and validation set for epoch (validation set size = evalidationSetSize -> 50 datapoints):
    etrainingData <- createDataPartition(y=trainingLabels, p=(1 - evalidationSetSize/length(trainingLabels)), list=FALSE)
    
    etrainingFeatures <- trainingFeatures[etrainingData,]
    etrainingLabels <- trainingLabels[etrainingData]
    
    evalidationFeatures <- trainingFeatures[-etrainingData,]
    evalidationLabels <- trainingLabels[-etrainingData]
    
    #set the steplength (eta)
    steplength = 1 / (e*c1 + c2);
      
    for (step in 1:numStepsPerEpoch){
      stepIndex = stepIndex+1;
      index = sample.int(nrow(etrainingFeatures),1);
      xk = etrainingFeatures[index,];
      yk = etrainingLabels[index];
      
      costfxn = yk * (a %*% xk + b); #not actually the cost function!
      
      if(costfxn >= 1){
        
        a_dir = lambda * a;
        a = a - steplength * a_dir;
        #b = b
        
      } else {
        
        a_dir = (lambda * a) - (yk * xk);
        a = a - steplength * a_dir;
        b_dir = -yk
        b = b - (steplength * b_dir);
        
      }
      
      
      #need to add in a logged step before 30... because our graph starts at ~70% accuracy... 
      #whereas the first step really has accuracy ~50%... but our grpah starts at step 30.
      if (stepIndex %% nStepsPerPlot == 1){#30){
        accMat[accMatRow,accMatCol] = getAccuracy(a,b,evalidationFeatures,evalidationLabels);
        accMatv[accMatRow,accMatCol] = getAccuracy(a,b,validationFeatures,validationLabels);
        accMatRow = accMatRow + 1;
      }
      
    }
    
  }
  
  #View(accMat)  #starting at ~>70% accuracy... because we only start recording at the 30th value
  #accMat[500,1]
  #stepValues = seq(1,15000,length=500)
  #plot(stepValues,accMat[1:500,1], type = "b",xlim=c(0, 15000), ylim=c(0, 1))
  #lines(accMat[1:500,1])
  tempAccuracy = getAccuracy(a,b,validationFeatures,validationLabels)
  print(str_c("tempAcc = ", tempAccuracy," and bestAcc = ", bestAccuracy) )
  if(tempAccuracy > bestAccuracy){
    bestAccuracy = tempAccuracy
    best_a = a;
    best_b = b;
    best_lambdaIndex = i;
  }
  
}


getAccuracy(best_a,best_b, testingFeatures, testingLabels)

colors = c("red","blue","green","black");
xaxislabel = "Step";
yaxislabels = c("Accuracy on Randomized Epoch Validation Set","Accuracy on Validation Set");
title="Accuracy as a Function of Step and Lambda";
ylims=c(0,1);

stepValues = seq(1,15000,length=500)

mats =  list(accMat,accMatv);


for(j in 1:length(mats)){
  
  mat = mats[[j]];
  
  for(i in 1:4){
    
   if(i == 1){
     plot(stepValues, mat[1:500,i], type = "l",xlim=c(0, 15000), ylim=ylims,
         col=colors[i],xlab=xaxislabel,ylab=yaxislabels[j],main=title)
    } else{
      lines(stepValues, mat[1:500,i], type = "l",xlim=c(0, 15000), ylim=ylims,
          col=colors[i],xlab=xaxislabel,ylab=yaxislabels[j],main=title)
    }
   Sys.sleep(1);
  }
  legend(x=10000,y=.5,legend=c("lambda=.001","lambda=.01","lambda=.1","lambda=1"),fill=colors);

}

