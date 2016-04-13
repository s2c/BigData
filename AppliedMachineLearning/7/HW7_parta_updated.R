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

#cross validate to find optimal single h (scale contains possible h's)

agg_data = aggregate(data$Tmin_deg_C,by=list(data$SID,data$East_UTM,data$North_UTM),mean)
names(agg_data) = c("SID","East_UTM","North_UTM","Tmin_deg_C");
rm(data);

k = 8;
folds = sample(rep(1:k,each=nrow(agg_data)/k),nrow(agg_data));



#scale <- c ( 0.1 , 0.15 , 0.2 , 0.25 , 0.3 , 0.35 , 0.4)
#scales <- c(0.1,1,100,10000,100000,1000000);	#the range of scales to be used

#for(scale in scales){

#wmods = vector(mode="list", length=max(data$Julian));

xmat <- as.matrix( LocData [, c(2,3)] );	#the latitude and longitude of each station

#calculate the distance between every pair of stations
spaces <- dist ( xmat , method = "euclidean" , diag = FALSE, upper = FALSE)  
msp <- as.matrix( spaces )
mean_dist_btn_pts = mean(msp);

scales = c(.25,.5,1,1.5,2,3)*mean_dist_btn_pts;
  
mse = matrix(rep(NA,k*length(scales)),nrow=length(scales),ncol=k);

setwd('/Users/Cybelle/Dropbox/7/img')

for (scale_idx in 1:length(scales)){
    scale = scales[scale_idx];
    
    for (fold in folds){
      
      #blending weights... such that one can know the weight of any bump (one centered at each station) for the current station
      wmat <- exp(-msp^2/(2*scale^2 ) ) 
      
      #for ( i in 2 : length(scale) ){
      #  grammmat <- exp(-msp^2/(2*scale[i]^2 ) )
      #  wmat <- cbind(wmat , grammmat )
      #}
      
      wmat.df = data.frame(wmat);
      wmat.df$SID = 1:ncol(msp); #add SID column to enable merging with x matrix below
      
      
      trainData = agg_data[folds!=fold,];
      testData = agg_data[folds==fold,];
      
      trainData <- merge(trainData,wmat.df); #add features from wmat.df (merge based on SID) - repeat features for same station
      testData <- merge(testData,wmat.df);
    
      ndat <- dim( trainData ) [ 1 ]
      
      #regress to find a model... the features used are the blend weights
      wmod<-glmnet (as.matrix(trainData[,5:ncol(trainData)]) , as.vector ( trainData [ , 4 ] ) , lambda=0)
      
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
      wmat<-exp(-distsampletopts^2 /(2*scale[1]^2 ) ) #The actual kernels...these will be the features for these new evenly spaced grid points
     
       #for ( i in 2 : length(scale) )
     # {grammmat<-exp(-distsampletopts^2 /(2*scale [ i ]^2 ) )
     # wmat<-cbind(wmat , grammmat )
     # }
   
      validation_preds = predict.glmnet(wmod, as.matrix(testData[,5:ncol(testData)]), s='lambda.min' )
      
      mse[scale_idx,fold] = mean((testData[,4] - validation_preds)^2);
      
      #predict a min temperature for this day for each of the evenly spaced grid points... 
      preds <-predict.glmnet(wmod, wmat, s='lambda.min' )
        
      #write.csv(preds,str_c("preds_part_a_annual_scale=",scale,"fold=",fold,".csv"),row.names=F);
      #save(list=c("wmod","preds"),file=str_c("part_a_wmods_preds_scale=",scale," fold=",fold));
      
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
      
      wscale=max(abs (min( pred_means ) ) , abs (max( pred_means ) ) )
      
      png(str_c("part_a_scale=",round(scale,0),"_fold=",fold))
      
      image.plot(xvec,yvec, t(zmat),
           xlab='East' , ylab='North', col=heat.colors(12),
           #grey( seq (0 , 1 , length=256)) ,
           useRaster=TRUE,main=str_c("part a scale=",round(scale,0)," fold=",fold))
           #legend.lab = seq(min( pred_means ),max( pred_means ),length=6))
      
      dev.off()

    }

}

save(list=c("mse"),file="part_a_mse");

ave_mse = rowMeans(mse);
ave_mse.df = data.frame(scales,ave_mse);
write.csv(ave_mse.df,"average_mse_across_folds_for_each_scale.csv",row.names=F)
View(ave_mse.df)



