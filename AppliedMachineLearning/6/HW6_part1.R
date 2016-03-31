#clear the workspace and console
rm(list=ls())
#cat("\014") #code to send ctrl+L to the console and therefore clear the screen

#setup working directory 
#setwd('C:\\Users\\rock-\\Dropbox\\School\\BigData_MachineLearning\\AppliedMachineLearning\\6\\data')
setwd('/Users/Cybelle/Dropbox/6/data');

#install.packages("stats")
#install.packages("glmnet")
#install.packages("MASS")
#install.packages("DAAG")
library('stats')
library('glmnet')
library("DAAG")

require("graphics")
require("lattice")
require("MASS")
require("stringr")
#-------------------------------------------------------------------------
# QUESTION 1.1
#-------------------------------------------------------------------------
defaultData = read.csv("default_plus_chromatic_features_1059_tracks.txt",header=FALSE);

#1, part 1: 

#longitude regression
defaultDFLong <- data.frame(y = defaultData[,ncol(defaultData)],x =  defaultData[,-ncol(defaultData):-(ncol(defaultData)-1)])
defaultModLong <- lm(y~.,data= defaultDFLong) #Dont want to include latitude 
summary(defaultModLong) #R^2 = 0.3182 

#plot predicted values of y against actual values of y
plot(y ~ fitted(defaultModLong), data=defaultDFLong)

#plot residuals against actual values of y
plot(y ~ resid(defaultModLong), data=defaultDFLong)

#lattitude regression
defaultDFLat <- data.frame(y = defaultData[,(ncol(defaultData)-1)],x =  defaultData[,-ncol(defaultData):-(ncol(defaultData)-1)])
defaultModLat <- lm(y~.,data= defaultDFLat)
summary(defaultModLat) #R^2 = 0.2412

#plot predicted values of y against actual values of y
plot(y ~ fitted(defaultModLat), data=defaultDFLat)

#plot residuals against actual values of y
plot(y ~ resid(defaultModLat), data=defaultDFLat)

#clearly issues with the residuals - they are correlated with the value of y

#
minlat = min(defaultDFLat$y);
minlong = min(defaultDFLong$y);
defaultDFLat$y = defaultDFLat$y - min(defaultDFLat$y) + .01;
defaultDFLong$y = defaultDFLong$y - min(defaultDFLong$y) + .01;

range(defaultDFLat$y)
range(defaultDFLong$y)

#boxcox transformation

#first, refit the models on the range adjusted data

#latitude
defaultModLat <- lm(y~.,data= defaultDFLat)
summary(defaultModLat) #R^2 = 0.29

#plot predicted values of y against actual values of y
plot(y ~ fitted(defaultModLat), data=defaultDFLat)

#plot residuals against actual values of y
plot(y ~ resid(defaultModLat), data=defaultDFLat)

#longitude
defaultModLong <- lm(y~.,data= defaultDFLong) #Dont want to include latitude 
summary(defaultModLong)

#plot predicted values of y against actual values of y
plot(y ~ fitted(defaultModLong), data=defaultDFLong)

#plot residuals against actual values of y
plot(y ~ resid(defaultModLong), data=defaultDFLong)

bc.output.long = boxcox(defaultModLong,lambda=seq(-2,2,1/10),plotit=TRUE,eps=1/50,xlab=expression(lambda),
       ylab="log-likelihood")

lambda.ml.long = bc.output.long$x[bc.output.long$y == max(bc.output.long$y)]; #0.8686869

bc.output.lat = boxcox(defaultModLat,lambda=seq(-2,2,1/10),plotit=TRUE,eps=1/50,xlab=expression(lambda),
                   ylab="log-likelihood")

lambda.ml.lat = bc.output.lat$x[bc.output.lat$y == max(bc.output.lat$y)]; #1.474747

defaultDFLong.bc = defaultDFLong;
defaultDFLat.bc = defaultDFLat;

defaultDFLong.bc$y = defaultDFLong.bc$y^lambda.ml.long;
defaultDFLat.bc$y = defaultDFLat.bc$y^lambda.ml.lat;



#latitude
defaultModLat.bc <- lm(y ~.,data= defaultDFLat.bc)
summary(defaultModLat.bc) #R^2 = 0.2546 (before box cox: 0.2412)

#plot predicted values of y against actual values of y
plot(y ~ fitted(defaultModLat.bc), data=defaultDFLat.bc)

#plot residuals against actual values of y
plot(y ~ resid(defaultModLat.bc), data=defaultDFLat.bc)


#longitude
defaultModLong.bc <- lm(y~.,data= defaultDFLong.bc) #Dont want to include latitude 
summary(defaultModLong.bc) #R^2 = 0.3159 (vs. .3182 without boxcox)

#plot predicted values of y against actual values of y
plot(y ~ fitted(defaultModLong.bc), data=defaultDFLong.bc)

#plot residuals against actual values of y
plot(y ~ resid(defaultModLong.bc), data=defaultDFLong.bc)

#boxcox not helping much -> don't use it

#ridge regression
#latitude
lat.m.ridge = cv.glmnet(x=as.matrix(defaultDFLat[,-1]), y=as.matrix(defaultDFLat[,1]),family="gaussian",alpha=0,nfolds=10);
summary(lat.m.ridge)
lat.m.ridge$lambda.min #lambda that gives minimum cross validated error
plot(lat.m.ridge, xvar="lambda")
#longitude
long.m.ridge = cv.glmnet(x=as.matrix(defaultDFLong[,-1]), y=as.matrix(defaultDFLong[,1]),family="gaussian",alpha=0,nfolds=10);
summary(long.m.ridge)
long.m.ridge$lambda.min #lambda that gives minimum cross validated error
plot(long.m.ridge, xvar="lambda")


#mse without cross validation
mse_lat = mean(resid(defaultModLat)^2)
mse_long = mean(resid(defaultModLong)^2)

#refit with cross-validation
#latitude
defaultModLat.cv = CVlm(data=defaultDFLat,form.lm=formula(y~.),m=10);
mse_lat.cv = mean((defaultModLat.cv$cvpred - defaultModLat.cv$y)^2);
#longitude
defaultModLong.cv = CVlm(data=defaultDFLong,form.lm=formula(y~.),m=10);
mse_long.cv = mean((defaultModLong.cv$cvpred - defaultModLong.cv$y)^2);

#for non-regularized cross-validated regression, 
#cross validated MSE for latitude, longitude are:
mse_lat.cv #550
mse_long.cv #3934
#with ridge regression, cross validated MSE for latitude, longitude are:
lat.m.ridge$cvm[lat.m.ridge$lambda == lat.m.ridge$lambda.min] #280
long.m.ridge$cvm[long.m.ridge$lambda == long.m.ridge$lambda.min] #1872
#so the MSE goes down by about 50% when you apply ridge regression as
#compared to non-regularized cross-validated regression

#lasso regression: same but now estimate number of variables in model as well
lat.m.lasso = cv.glmnet(x=as.matrix(defaultDFLat[,-1]), y=as.matrix(defaultDFLat[,1]),family="gaussian",alpha=1,nfolds=10);
long.m.lasso = cv.glmnet(x=as.matrix(defaultDFLong[,-1]), y=as.matrix(defaultDFLong[,1]),family="gaussian",alpha=1,nfolds=10);

lat.m.lasso$lambda.min #0.503
long.m.lasso$lambda.min #0.325

plot(lat.m.lasso, xvar="lambda")
plot(long.m.lasso, xvar="lambda")

#with lasso, cross validated MSE for latitude, longitude are:
lat.m.lasso$cvm[lat.m.lasso$lambda == lat.m.lasso$lambda.min] #278
long.m.lasso$cvm[long.m.lasso$lambda == long.m.lasso$lambda.min] #1887

#so, again, improved compared to non-regularized cross-validated regression
#but only latitude slightly better than with ridge regression,
#longitude did slightly worse, suggesting the two response variables
#may ultimately be best modeled with different alpha coefficients

#number of non-zero coefficients at best lambda:
lat.m.lasso$nzero[lat.m.lasso$lambda == lat.m.lasso$lambda.min] #21 coeffs
long.m.lasso$nzero[lat.m.lasso$lambda == lat.m.lasso$lambda.min] #39 coeffs


#elastic net (and also, ridge and lasso, for comparison)
sink("output.txt");
alphas = seq(0,1,.1);
for(alpha in alphas){
  cat(str_c("alpha: ",alpha));
  cat("\n")

  lat.m.en = cv.glmnet(x=as.matrix(defaultDFLat[,-1]), y=as.matrix(defaultDFLat[,1]),family="gaussian",alpha=alpha,nfolds=10);
  long.m.en = cv.glmnet(x=as.matrix(defaultDFLong[,-1]), y=as.matrix(defaultDFLong[,1]),family="gaussian",alpha=alpha,nfolds=10);
  cat(str_c("latitude -- lambda.min: ",lat.m.en$lambda.min))
  cat("\n")
  cat(str_c("longitude -- lambda.min: ",long.m.en$lambda.min ))
  cat("\n")
  cat(str_c("latitude -- ncoeffs: ",lat.m.en$nzero[lat.m.en$lambda == lat.m.en$lambda.min]))
  cat("\n")
  cat(str_c("longitude -- ncoeffs: ",long.m.en$nzero[lat.m.en$lambda == lat.m.en$lambda.min]))
  cat("\n")
  
  #cross validated MSE for latitude, longitude are:
  cat(str_c("latitude -- cross-validated MSE: ",lat.m.en$cvm[lat.m.en$lambda == lat.m.en$lambda.min] ))
  cat("\n")
  cat(str_c("longitude -- cross-validated MSE: ",long.m.en$cvm[long.m.en$lambda == long.m.en$lambda.min]))
  cat("\n")
   
  plot(lat.m.en, xvar="lambda")
  plot(long.m.en, xvar="lambda")
 
}

sink();

#part 2:



