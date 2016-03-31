#Libraries and WorkDirectory
rm(list=ls())
ptm <- proc.time()
setwd("C:\\Users\\rock-\\Dropbox\\School\\BigData_MachineLearning\\AppliedMachineLearning\\HW1")

mydata <- read.table("pima-indians-diabetes.data.txt",header=FALSE,sep=",") #read the entire dataset
library(klaR)
library(caret)

#setting up initial
rounds = 10 # number of rounds to train for 
traScore <- array(dim = rounds) #training score
testScore <- array(dim = rounds) #testing score
#setting up
features <- mydata[,-c(9)] #all except the last column
labels <- mydata[,9]

for( i in c(3,5,6,7))
{
  not_ava <-features[,i]==0
  features[not_ava,i]=NA
}

for (round in (1:rounds))
{ #setup
  trData <- createDataPartition(y=labels, p = .8 , list = FALSE)
  trFeatures<-features[trData, ]
  trLabels<-labels[trData]
  trposflag<-trLabels>0
  pEgs<-trFeatures[trLabels==1, ]
  nEgs<-trFeatures[!trLabels==1,]
  teFeatures <- features[-trData,]
  teLabels <- labels[-trData]
  #Training
  pEgsTrMean <- sapply(pEgs,mean, na.rm=TRUE)
  nEgsTrMean <- sapply(nEgs,mean, na.rm=TRUE)
  pEgsTrSd <- sapply(pEgs,sd, na.rm=TRUE)
  nEgsTrSd <- sapply(nEgs,sd, na.rm=TRUE)
  pEgsTrOffset <- t(t(trFeatures)-pEgsTrMean)
  pEgsTrScale <- t(t(pEgsTrOffset)/pEgsTrSd)
  pEgsTrLogs <- -(1/2)*rowSums(apply(pEgsTrScale,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(pEgsTrSd))
  nEgsTrOffset <- t(t(trFeatures)-nEgsTrMean)
  nEgsTrScale <- t(t(nEgsTrOffset)/nEgsTrSd)
  nEgsTrLogs <- -(1/2)*rowSums(apply(nEgsTrScale,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(nEgsTrSd))
  lvtwr <- pEgsTrLogs > nEgsTrLogs
  rights <- lvtwr==trLabels
  traScore[round] <- sum(rights)/(sum(rights)+sum(!rights))
  #Testing
  pEgsTeOffset <- t(t(teFeatures)-pEgsTrMean)
  pEgsTeScale <- t(t(pEgsTeOffset)/pEgsTrSd)
  pEgsTeLogs <- -(1/2)*rowSums(apply(pEgsTeScale,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(pEgsTrSd))
  nEgsTeOffset <- t(t(teFeatures)-nEgsTrMean)
  nEgsTeScale <- t(t(nEgsTeOffset)/nEgsTrSd)
  nEgsTeLogs <- -(1/2)*rowSums(apply(nEgsTeScale,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(nEgsTrSd))
  lvte <- pEgsTeLogs > nEgsTeLogs
  rightsTest <- lvte==teLabels
  testScore[round] <- sum(rightsTest)/(sum(rightsTest)+sum(!rightsTest))
  plot(seq(1,10),testScore,"h")
}
cat("Average Test Accuracy: ", mean(testScore, na.rm=TRUE) ," Standard div: " , sd(testScore,na.rm=TRUE))
proc.time() - ptm