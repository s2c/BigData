rm(list=ls())
require("graphics")
require("lattice")
require("MASS")
require("stringr")
require("glmnet")
require("base")
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

#cross validate to find optimal single h (s_range contains possible h's)




#s_range <- c ( 0.1 , 0.15 , 0.2 , 0.25 , 0.3 , 0.35 , 0.4)
s_ranges <- c(0.1,1,100,10000,100000,1000000);	#the range of scales to be used

for(s_range in s_ranges){
  
wmods = vector(mode="list", length=max(data$Julian));

xmat <- as.matrix( LocData [, c(2,3)] );	#the latitude and longitude of each station
#calculate the distance between every pair of stations
spaces <- dist ( xmat , method = "euclidean" , diag = FALSE, upper = FALSE)  
msp <- as.matrix( spaces )
#blending weights... such that one can know the weight of any bump (one centered at each station) for the current station
wmat <- exp(-msp/(2*s_range[1]^2 ) ) 
#for ( i in 2 : length(s_range) ){
#  grammmat <- exp(-msp/(2*s_range[i]^2 ) )
#  wmat <- cbind(wmat , grammmat )
#}

wmat.df = data.frame(wmat);
wmat.df$SID = 1:112; #add SID column to enable merging with x matrix below


#create 366 models for min temperature as a function of 2D position (lat & long)... 
#one model for each day of the year... 
#Later all the models can be used to predict each days temperature at a 2D coordinate, 
#then the 366 resulting predictions can be averaged to yield the mean anual min temp.


start <- Sys.time ()
preds = matrix(rep(NA,10000*max(data$Julian)),nrow=10000,ncol=max(data$Julian));

for (day in 1:max(data$Julian)){  #3){
  
  print(day)
  
  #day = 1; #only for testing if the above line is commented out
  
  day.df <- data[data$Julian == day,]; #matrix of all data points for that day... 
  day.df <- merge(day.df,wmat.df); #add features from wmat.df (merge based on SID) - repeat features for same station
  ndat <- dim( day.df ) [ 1 ]
  
  
  
  
  
  
  #regress to find a model... the features used are the blend weights that were added to day.df in cols 6 and up.
  wmod<-glmnet (as.matrix(day.df[,6:ncol(day.df)]) , as.vector ( day.df [ , 3 ] ) , lambda=0)
  
  
  
  
  
  
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
  wmat<-exp(-distsampletopts /(2*s_range[1]^2 ) ) #The actual kernels...these will be the features for these new evenly spaced grid points
  #for ( i in 2 : length(s_range) )
 # {grammmat<-exp(-distsampletopts /(2*s_range [ i ]^2 ) )
 # wmat<-cbind(wmat , grammmat )
 # }
  
  #predict a min temperature for this day for each of the evenly spaced grid points... 
  preds[,day]<-predict.glmnet(wmod, wmat, s='lambda.min' )
  
  #plot (wmod)
  # and t h i s t he r e g u l a r i z a t i o n r e s u l t s
  
  wmods[[day]] = wmod;
  
}

write.csv(preds,str_c("preds_part_a_annual_s_range=",s_range,".csv"),row.names=F);
save(list=c("wmods","preds"),file=str_c("part_a_wmods_preds_s_range=",s_range));
# 
# pred_means = rowMeans(preds,na.rm=T);
# 
# zmat<-matrix (0 , nrow=100, ncol=100)  # think of the final image as a heightmap or heatmap and the z coordinate is the predicted value (min temp)
# ptr<-1
# for ( i in 1: 100)
# { 
#   for ( j in 1: 100)
#   {
#     zmat [j, i ]<-pred_means [ ptr ]
#     ptr<-ptr+1
#   }
# }
# wscale=max(abs (min( pred_means ) ) , abs (max( pred_means ) ) )

#image(xvec,yvec, ( t ( zmat)+wscale )/(2*wscale ) ,
#      xlab='Longitude' , ylab='Latitude' ,
#      col=heat.colors(12),#grey( seq (0 , 1 , length=256)) ,
#      useRaster=TRUE)
# t h i s g e t s t he heat map

#}


#Sys.time () - start 

}



