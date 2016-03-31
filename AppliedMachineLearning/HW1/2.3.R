#Libraries and WorkDirectory
rm(list=ls())
ptm <- proc.time()
setwd("C:\\Users\\rock-\\Dropbox\\School\\BigData_MachineLearning\\AppliedMachineLearning\\HW1")
mydata <- read.table("pima-indians-diabetes.data.txt",header=FALSE,sep=",") #read the entire dataset
library(klaR)
library(caret)
#partitions = 10 # number of rounds to train for 
score <- array(dim=5)
#setting up

for (i in 1:5)
{
features <- mydata[,-c(9)] 
labels <- as.factor(mydata[,9])
trData <- createDataPartition(y=labels, p = .8 , list = FALSE)
trFeatures<-features[trData, ]
trLabels<- labels[trData]
teFeatures <- features[-trData,]
teLabels <- labels[-trData]
#Training/Testing
model <- train(trFeatures,trLabels,'nb',trControl = trainControl(method= 'cv',number=10))
test <- predict(model, newdata = teFeatures)
cm <- confusionMatrix(data=test,teLabels)
curr <- cm$overall['Accuracy']
score[i] <- curr[1]
}

cat("Average Accuracy: ", mean(score, na.rm=TRUE) ," Standard div: " , sd(score,na.rm=TRUE))
plot(seq(1,5),score,type = "h")
proc.time() - ptm