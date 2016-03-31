#Libraries and WorkDirectory
rm(list=ls())
ptm <- proc.time()
setwd("C:\\Users\\rock-\\Dropbox\\School\\BigData_MachineLearning\\AppliedMachineLearning\\HW1")
mydata <- read.table("pima-indians-diabetes.data.txt",header=FALSE,sep=",") #read the entire dataset
library(klaR)
library(caret)
#partitions = 10 # number of rounds to train for
features <- mydata[,-c(9)] 
labels <- as.factor(mydata[,9])
score <- array(dim=5)
#setting up

for (i in 1:5)
{ 
  trData <- createDataPartition(y=labels, p = .8 , list = FALSE)
  trFeatures<-features[trData, ]
  trLabels<- labels[trData]
  teFeatures <- features[-trData,]
  teLabels <- labels[-trData]
  svm<-svmlight(trFeatures, trLabels, pathsvm='F:\\svm_light_windows64')
  predi<-predict(svm, teFeatures)
  foo<-predi$class
  score[i]=sum(foo==teLabels)/(sum(foo==teLabels)+sum(!(foo==teLabels)))
}
cat("Average Accuracy: ", mean(score, na.rm=TRUE) ," Standard div: " , sd(score,na.rm=TRUE))
plot(seq(1,5),score,type = "h")
proc.time() - ptm