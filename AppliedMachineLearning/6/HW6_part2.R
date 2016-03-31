require("caret");

rm(list=ls())

setwd('/Users/Cybelle/Dropbox/6/submission/data');

d <- read.csv("ccard.csv",header=T);

d$SEX = as.factor(d$SEX);
d$EDUCATION = as.factor(d$EDUCATION);
d$MARRIAGE = as.factor(d$MARRIAGE);

m1 <- glm(default_payment_next_month ~ .,data=d,family=binomial(link="logit"));

plot(m1$residuals ~ d$default_payment_next_month)
plot(m1$fitted ~ d$default_payment_next_month)

#anova(m1)
summary(m1);
#AIC = 27889

d_sub = d[,c(1:2,4:8,12:13,18:24)];
m2 <- glm(default_payment_next_month ~ .,data=d_sub,family=binomial(link="logit"));
summary(m2)
#AIC = 27942

#convert categorical variables to dummy variables
d <- read.csv("ccard.csv",header=T);
labels = d[,ncol(d)];
d = d[,-ncol(d)];
d$SEX = d$SEX - 1;
d$edu1 = d$EDUCATION == 1;
d$edu2 = d$EDUCATION == 2;
d$edu3 = d$EDUCATION == 3;
d$edu4 = d$EDUCATION == 4;
d$edu5 = d$EDUCATION == 5;
d$edu6 = d$EDUCATION == 6;
d = d[,-which(names(d)=="EDUCATION")];
d$mar1 = d$MARRIAGE == 1;
d$mar2 = d$MARRIAGE == 2;
d$mar3 = d$MARRIAGE == 3;
d = d[,-which(names(d)=="MARRIAGE")];

#elastic net (and also, ridge and lasso, for comparison)
sink("output_part2.txt");
alphas = seq(0,1,.1);
for(alpha in alphas){
  cat(str_c("alpha: ",alpha));
  cat("\n")
  
  m = cv.glmnet(x=as.matrix(d), y=as.matrix(labels),
                family="binomial",alpha=alpha,nfolds=10,
                type.measure="class");
  
  cat(str_c("lambda.min: ",m$lambda.min))
  cat("\n")
  cat(str_c("ncoeffs: ",m$nzero[m$lambda == m$lambda.min]))
  cat("\n")
  
  #cross validated MSE:
  cat(str_c("cross-validated MSE: ",m$cvm[m$lambda == m$lambda.min] ))
  cat("\n")
  
  plot(m, xvar="lambda")
}

sink();

#alpha = .3 did the best. (cross validated MSE = 0.189), it had 27 coefficients



trainindices = createDataPartition(y=labels,p=.8,list=FALSE);

alpha = .3
trainingData = as.matrix(d[trainindices,]);
trainingLabels = as.matrix(labels[trainindices])
testingData = as.matrix(d[-trainindices,])
testingLabels = as.matrix(labels[-trainindices])

m = cv.glmnet(x=trainingData, y=trainingLabels,
              family="binomial",alpha=alpha,nfolds=10,
              type.measure="class");

predict.out = predict(m,newx=testingData,s="lambda.min",type="class");

sum(predict.out == testingLabels)/length(testingLabels);
#accuracy:  0.809
coef(m, s=m$lambda.min)


