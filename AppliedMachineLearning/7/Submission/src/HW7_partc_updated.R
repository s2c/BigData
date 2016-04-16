#SEE COMMENTS FROM FILE: HW7_partb_updated.R for more details, this file has not been fully commented 

rm(list=ls())
require("fields")
require("graphics")
require("lattice")
require("MASS")
require("stringr")
require("glmnet")
require("base")
require('caret')
setwd('/Users/Cybelle/Dropbox/7/Data')
#setwd('C:\\Users\\rock-\\Dropbox\\School\\bigdata\\AppliedMachineLearning\\7\\data\\')
#-------------------------------------------------------------------------
# QUESTION 1.1
#-------------------------------------------------------------------------
LocData = read.csv("Locations.txt",header=TRUE, sep = " ");
TempData = read.csv("Oregon_Met_Data.txt",header=TRUE, sep = " ")
LocData = LocData[,c(1,7:8)] 	#only concerned with the UTM units for lattitude and longitude
TempData = TempData[,c(1,5,6)]	#only concerned with the min temperature and the day of the year

data = merge(TempData,LocData); 
data = data[data$Tmin_deg_C < 200 & !is.na(data$Tmin_deg_C),] #remove any missing or impossible temperatures

full_data = data; 

agg_data = aggregate(data$Tmin_deg_C,by=list(data$SID,data$East_UTM,data$North_UTM),mean)
names(agg_data) = c("SID","East_UTM","North_UTM","Tmin_deg_C");
rm(data);

xmat <- as.matrix( LocData [, c(2,3)] );	

spaces <- dist ( xmat , method = "euclidean" , diag = FALSE, upper = FALSE)  
msp <- as.matrix( spaces )
mean_dist_btn_pts = mean(msp);

scales = c(.25,.5,1,1.5,2,3)*mean_dist_btn_pts;

setwd('/Users/Cybelle/Dropbox/7/img')


wmat <- exp(-msp^2/(2*scales[1]^2 ) ) 

for ( i in 2 : length(scales) ){
  grammmat <- exp(-msp^2/(2*scales[i]^2 ) )
  wmat <- cbind(wmat , grammmat )
}

wmat.df = data.frame(wmat);
wmat.df$SID = 1:ncol(msp); 

trainData = agg_data;

trainData <- merge(trainData,wmat.df); 

ndat <- dim( trainData ) [ 1 ]

alphas = c(0,.25,.5,.75,1);

wmods = vector(mode="list", length=length(alphas));

for (alpha_idx in 1:length(alphas)){
  
  alpha = alphas[alpha_idx];
  

  wmod<-cv.glmnet (as.matrix(trainData[,5:ncol(trainData)]) , as.vector ( trainData [ , 4 ] ) , alpha=alpha)
  
  #create a 2D grid of evenly spaes points held in pmat (points matrix) in the region housing all of the stations
  #1st: determine the min and max boundaries defining the region containing all the stations
  xmin<-min( xmat [ , 1 ] )
  xmax<-max( xmat [ , 1 ] )
  ymin<-min( xmat [ , 2 ] )
  ymax<-max( xmat [ , 2 ] )
  #2nd: split that region up into 100x100 evenly spaced grid
  xvec<-seq ( xmin , xmax , length=100)
  yvec<-seq ( ymin , ymax , length=100)
  #3rd: populate pmat as a 10000x2 matrix where each row is one of the evenly spaced coordinates (ie: the cols are latitude and longitude)
  #and the rows are Major Ordered representing the 100x100 matrix.
  pmat<-matrix (0 , nrow=100*100 , ncol=2) #pmat: a grid of values evenly spaced in longitude and latitude
  ptr<-1
  for ( i in 1: 100)
  {
    for ( j in 1: 100)
    {
      pmat [ ptr , 1 ]<-xvec [ i ]
      pmat [ ptr , 2 ]<-yvec [ j ]
      ptr<-ptr+1
    }
  }
  
  #Create gaussian kernels for the even grid of base points... 
  diffij <- function ( i , j ) sqrt ( rowSums ( ( pmat [ i , ]-xmat [ j , ])^2 ) ) #define a helper function
  distsampletopts <- outer( seq_len (10000) , seq_len (dim( xmat ) [ 1 ] ) , diffij ) #squared differences
  #The actual kernels... distsampletopts has squared distances, srange[i] controls the size.
  wmat<-exp(-distsampletopts^2 /(2*scales[1]^2 ) ) #The actual kernels...these will be the features for these new evenly spaced grid points
  for ( i in 2 : length(scales) )
  {
    grammmat<-exp(-distsampletopts^2 /(2*scales[ i ]^2 ) )
    wmat<-cbind(wmat , grammmat )
  }
  
  preds <-predict.cv.glmnet(wmod, wmat, s='lambda.min' )
 
  pred_means = rowMeans(preds,na.rm=T);
  
  zmat<-matrix (0 , nrow=100, ncol=100)  # think of the final image as a heightmap or heatmap and the z coordinate is the predicted value (min temp)
  ptr<-1
  for ( i in 1: 100)
  { 
    for ( j in 1: 100)
    {
      zmat [i,j ]<-pred_means [ ptr ]
      ptr<-ptr+1
    }
  }
  
 
  png(str_c("part_c_scale=all6_alpha=",alpha));
  image.plot(xvec,yvec, t(zmat),
             xlab='East' , ylab='North', col=heat.colors(12),
             useRaster=TRUE,main=str_c("part c scale = all 6, alpha=",alpha))
  
  dev.off()
  
  png(str_c("part_c_scale=all6_alpha=",alpha,"_mse_vs_lambda"))
  plot(wmod)
  dev.off()
  
  wmods[[alpha_idx]] = wmod;
  #wmod$nzero[wmod$lambda == wmod$lambda.min]

  print(str_c("alpha = ",alpha))
  print(str_c("lambda.min = ",wmod$lambda.min))
}

save(list=c("wmods"),file="partc_wmods")





