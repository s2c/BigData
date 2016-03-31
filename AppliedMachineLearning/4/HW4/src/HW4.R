#clear the workspace and console
rm(list=ls())
cat("\014") #code to send ctrl+L to the console and therefore clear the screen

#setup working directory 
setwd('~/../Dropbox/School/Spring 2016/Machine Learning/Homeworks/4/data')

#install.packages("plsdepot")

library('lattice')
library('plsdepot')

irisdat <- read.csv("iris.data.txt",header=F);

#get numeric columns only
numiris=irisdat[,c(1,2,3,4)]
iris.lab = irisdat[,c(5)]

#open new eps file
postscript("irisscatterplot.eps")

speciesnames<- c('setosa','versicolor','virginica')

#point type for each type of flower
pchr<- c(1,2,3)
#colors for plot
colr<- c('red','green','blue','yellow','orange')

#create a dataframe with factor column "species", three rows 1 to 3
ss<- expand.grid(species=1:3)


parset<- with(ss,simpleTheme(pch=pchr[species],
                            col=colr[species]))

splom(irisdat[,c(1:4)],groups=irisdat$V5,
      par.settings=parset,
      varnames=c('Sepal\nLength','Sepal\nWidth',
                  'Petal\nLength','Petal\nWidth'),
      key=list(text=list(speciesnames),
               points=list(pch=pchr),columns=3))
dev.off()


#3.4b

#use a different package to handle the projection of the data onto the first two principal components
iris.pc = princomp(scale(numiris, center=TRUE,scale=TRUE),scores=T)
#plot(iris.pc)
#screeplot(iris.pc)
#screeplot(iris.pc,type="lines")

#M = iris.pc$loading[,1:2]  #first two principal components
#t(M) %*% M   #check, make sure this equals the identity (2x2) matrix

#plot the data projected onto the first two principal components
plot(iris.pc$scores[,1],iris.pc$scores[,2],pch=".")

#create the same plot as above, but color the points by their class
#use wine.lab to decide the color for each of the 178 exs
DF <- data.frame(projDataPC1 = iris.pc$scores[,1], projDataPC2 = iris.pc$scores[,2], label = iris.lab);
attach(DF); 
plot(projDataPC1, projDataPC2, col=c("red","green","blue")[label], xlab = "Principal Component 1", ylab = "Principal Component 2", main = "Data Projection onto First 2 PCs"); 
detach(DF);
legend(x ="topright", legend = c('setosa','virginica','versicolor'), col = c("red","blue","green"), pch=1);


#ALTERNATE PLOTTING FOR 3.4B
#NIPALS is one option for this.

#code for PCA adapted from:
#http://www.r-bloggers.com/computing-and-visualizing-pca-in-r/

log.ir <- log(numiris) #not necessary for assignment

ir.species <- irisdat[, 5]

#this is what assignment is asking for:
#conduct pca on the z-score (mean center and scale each vector to unit variance)
ir.pca <- prcomp(numiris,
                 center = TRUE,
                 scale. = TRUE)

#can also take the log transform and take the z-score of that
logir.pca <- prcomp(log.ir,
                    center = TRUE,
                     scale. = TRUE) 

predict(ir.pca, newdata=numiris)

library(devtools)
install_github("vqv/ggbiplot")

library(ggbiplot)
g <- ggbiplot(ir.pca, obs.scale = 1, var.scale = 1, 
              groups = ir.species, ellipse = TRUE, 
              circle = TRUE)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)

g <- ggbiplot(logir.pca, obs.scale = 1, var.scale = 1, 
              groups = ir.species, ellipse = TRUE, 
              circle = TRUE)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)


#distortions?

#3.4c
#PLS1:

ir.species.n <- as.numeric(ir.species)
ir.species.col <- colr[ir.species.n]

pls1 = plsreg1(numiris,ir.species.n)
#The data is scaled to standardized values (mean=0, variance=1).

plot(pls1,what="observations",col.points=ir.species.col,
     main="Iris Data Projected onto\nFirst Two Discriminative Directions",
     ylab="Second Discriminative Direction", xlab="First Discriminative Direction")

#data in each cluster appears closer to the cluster center, less scattered, more 
#separation between clusters, fewer outliers


###################################
#3.5
###################################

wine.df <- read.csv("wine.data",header=F);

wine.lab <- wine.df[,1];
wine.feat <- wine.df[,-1];

wine.pca <- prcomp(wine.feat,
                 center = TRUE,
                 scale. = TRUE)

plot((wine.pca$sdev)^2,type="b",main="Sorted Eigenvalues of Principle Components\nfor Wine Dataset",
     xlab="Index of Principle Component",ylab="Eigenvalue")


#data(wine)


#matlab style stem plot, code adapted from:
#http://www.r-bloggers.com/matlab-style-stem-plot-with-r/
stem <- function(x,y,pch=16,linecol=1,clinecol=1,...){
  if (missing(y)){
    y = x
    x = 1:length(x) }
  #plot(x,y,pch=pch,ylim=c(-.5,.5),xlab="Feature",ylab="Weight",main="Feature Weights on First Principle Component",...)
  #plot(x,y,pch=pch,ylim=c(-.4,.8),xlab="Feature",ylab="Weight",main="Feature Weights on Second Principle Component",...)
  plot(x,y,pch=pch,ylim=c(-.4,.8),xlab="Feature",ylab="Weight",main="Feature Weights on Third Principle Component",...)
  for (i in 1:length(x)){
    lines(c(x[i],x[i]), c(0,y[i]),col=linecol)
    
  }
  lines(c(x[1]-2,x[length(x)]+2), c(0,0),col=clinecol)
}



#stem(wine.pca$rotation[,1])
#text(x=1:length(wine.pca$rotation[,1]),y=wine.pca$rotation[,1]+.05,labels=names(wine))

#stem(wine.pca$rotation[,2])
#text(x=1:length(wine.pca$rotation[,2]),y=wine.pca$rotation[,2]+.1,labels=names(wine))



stem(wine.pca$rotation[,3])
text(x=1:length(wine.pca$rotation[,3]),y=wine.pca$rotation[,3]+.1,labels=names(wine))


#retrieve the first 2 principal components
firstTwoPCs <- wine.pca$rotation[,1:2];
plot(pc$scores[,1],pc$scores[,2],pch=".");


#use a different package to handle the projection of the data onto the first two principal components
pc = princomp(scale(wine.feat, center=TRUE,scale=TRUE),scores=T)
plot(pc)
screeplot(pc)
screeplot(pc,type="lines")

#M = pc$loading[,1:2]  #first two principal components
#t(M) %*% M   #check, make sure this equals the identity (2x2) matrix

#plot the data projected onto the first two principal components
plot(pc$scores[,1],pc$scores[,2],pch=".")

#create the same plot as above, but color the points by their class
#use wine.lab to decide the color for each of the 178 exs
DF <- data.frame(projDataPC1 = pc$scores[,1], projDataPC2 = pc$scores[,2], label = wine.lab);
attach(DF); 
plot(projDataPC1, projDataPC2, col=c("red","blue","green")[label], xlab = "Principal Component 1", ylab = "Principal Component 2", main = "Data Projection onto First 2 PCs"); 
detach(DF);
legend(x ="topright", legend = c(1,2,3), col = c("red","blue","green"), pch=1);


################
#Problem 3.7:   -----------------------------------------------------
################
wdbc.data <- read.csv("wdbc.data",header=F);
wdbc.id = wdbc.data[,c(1)];
wdbc.class = wdbc.data[,c(2)]
wdbc.feat = wdbc.data[,c(3:ncol(wdbc.data))]


#wdbc.pca <- prcomp(wdbc.feat, center = TRUE, scale = TRUE);

#use a different package to handle the projection of the data onto the first two principal components
wdbc.pc = princomp(scale(wdbc.feat, center=TRUE,scale=TRUE),scores=T)
#plot(wdbc.pc)
#screeplot(wdbc.pc)
#screeplot(wdbc.pc,type="lines")

#M = wdbc.pc$loading[,1:2]  #first two principal components
#t(M) %*% M   #check, make sure this equals the identity (2x2) matrix

#plot the data projected onto the first two principal components
#install.packages("rgl")
library(rgl)
plot3d(wdbc.pc$scores[,1],wdbc.pc$scores[,2], wdbc.pc$scores[,3])

#create the same plot as above, but color the points by their class
#use wdbc.lab to decide the color for each of the exs
DF <- data.frame(projDataPC1 = wdbc.pc$scores[,1], projDataPC2 = wdbc.pc$scores[,2], projDataPC3 = wdbc.pc$scores[,3], label = wdbc.class);

attach(DF);
plot3d(projDataPC1, projDataPC2, projDataPC3, col=c("red","blue")[label], xlab = "Principal Component 1", ylab = "Principal Component 2", zlab = "Principal Component 3", main = "Data Projection onto First 3 PCs"); 
detach(DF);


#install.packages("Rcmdr")
library(Rcmdr)
attach(DF);
scatter3d(projDataPC1, projDataPC2, projDataPC3, col=c("red","blue")[label], xlab = "Principal Component 1", ylab = "Principal Component 2", zlab = "Principal Component 3", main = "Data Projection onto First 3 PCs"); 
detach(DF);



#3.7b
library(rgl)
library('plsdepot')
wdbc.class.n <- as.numeric(wdbc.class)
colr<- c('red','green','blue','yellow','orange')
wdbc.class.col <- colr[wdbc.class.n]

pls1 = plsreg1(wdbc.feat,wdbc.class.n, comps = 3)

#if it had been 2 discriminative directions from pls1
#plot(pls1,what="observations",col.points=wdbc.class.col,
#     main="Iris Data Projected onto\nFirst Two Discriminative Directions",
#     ylab="Second Discriminative Direction", xlab="First Discriminative Direction")
#or...
#plot(pls1$x.scores[,c(1)],pls1$x.scores[,c(2)]);

#instead we have 3 directions, so plot the projections along all 3

DF <- data.frame(projDataDir1 = pls1$x.scores[,c(1)], projDataDir2 = pls1$x.scores[,c(2)],projDataDir3 = pls1$x.scores[,c(3)], label = wdbc.class);

attach(DF);
plot3d(projDataDir1, projDataDir2, projDataDir3, col=c("red","blue")[label], xlab = "Discriminative Dir 1", ylab = "Discriminative Dir 2", zlab = "Discriminative Dir 3", main = "Data Projection onto First 3 Discriminative Directions"); 
detach(DF);