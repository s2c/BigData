require('fields')
require("graphics")
require("lattice")
require("MASS")
require("stringr")
require("glmnet")
require("base")

rm(list=ls())

setwd('/Users/Cybelle/Dropbox/7/Data')
LocData = read.csv("Locations.txt",header=TRUE, sep = " ");
TempData = read.csv("Oregon_Met_Data.txt",header=TRUE, sep = " ")
LocData = LocData[,c(1,7:8)] 	#only concerned with the UTM units for lattitude and longitude
TempData = TempData[,c(1,5,6)]	#only concerned with the min temperature and the day of the year

data = merge(TempData,LocData); 
data = data[data$Tmin_deg_C < 200 & !is.na(data$Tmin_deg_C),] #remove any missing or impossible temperatures


setwd('/Users/Cybelle/Dropbox/7/Varun_run_copy/Data')

#part a

filenames = c('part_a_wmods_preds_s_range=0.1',
              'part_a_wmods_preds_s_range=1',
              'part_a_wmods_preds_s_range=1e+05',
              'part_a_wmods_preds_s_range=1e+06',
              'part_a_wmods_preds_s_range=100',
              'part_a_wmods_preds_s_range=10000');

for (f in filenames){
  
  load(f)
  
  xmat <- as.matrix( LocData [, c(2,3)] );	#the latitude and longitude of each station
  
  #create a 2D grid of evenly spaes points held in pmat (points matrix) in the region housing all of the stations
  #1st: determine the min and max boundaries defining the region containing all the stations
  xmin<-min( xmat [ , 1 ] )
  xmax<-max( xmat [ , 1 ] )
  ymin<-min( xmat [ , 2 ] )
  ymax<-max( xmat [ , 2 ] )
  #2nd: split that region up into 100x100 evenly spaced grid
  xvec<-seq ( xmin , xmax , length=100)
  yvec<-seq ( ymin , ymax , length=100)
  
  pred_means = rowMeans(preds,na.rm=T);
  
  zmat<-matrix (0 , nrow=100, ncol=100)  # think of the final image as a heightmap or heatmap and the z coordinate is the predicted value (min temp)
  ptr<-1
  for ( i in 1: 100)
  { 
    for ( j in 1: 100)
    {
      zmat [j, i ]<-pred_means [ ptr ]
      ptr<-ptr+1
    }
  }
  wscale=max(abs (min( pred_means ) ) , abs (max( pred_means ) ) )
  
  image.plot(xvec,yvec, ( t ( zmat)+wscale )/(2*wscale ) ,
       xlab='Longitude' , ylab='Latitude' ,
       col=heat.colors(12),#grey( seq (0 , 1 , length=256)) ,
       useRaster=TRUE,main='')

  rm(preds,wmods)
}
