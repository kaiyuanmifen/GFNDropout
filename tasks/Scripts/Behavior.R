
#Result visualizaiton
library(ggplot2)
library(viridis)
library(ggridges)
data=read.csv("../Results/masks/cifar10c_snow_1_bottomup_False_False_False_False_mask.csv",header = T)
head(data)
data=data[,2:ncol(data)]
names(data)
dim(data)

names(data)=c(1:1024,"Repeat")
tail(names(data))
data$DataPoint=1:(nrow(data)/20)

VecAll=NULL

for (i in (1:1024)){
  Vec=data[,c(i,1025,1026)]
  Vec$UnitIndex=i
  names(Vec)[1]="Mask"
  VecAll=rbind(VecAll,Vec)
}
Data=VecAll

dim(Data)
head(Data)

VecPlot=Data[Data$DataPoint==2,]
VecPlot=Data
unique(VecPlot$Repeat)
dim(VecPlot)
VecPlot$Repeat=factor(VecPlot$Repeat)
#VecPlot=VecPlot[VecPlot$Repeat%in%c(1,10),]

VecPlot=VecPlot[VecPlot$Mask==0,]

VecPlot=VecPlot[VecPlot$Repeat==1,]

VecPlot$DataPoint=factor(VecPlot$DataPoint)

ggplot(VecPlot, aes(x = UnitIndex,fill=DataPoint)) +
  geom_density(alpha=0.4,bw=10)

ggplot(VecPlot, aes(x = UnitIndex, y = DataPoint,fill=DataPoint)) +
  geom_density_ridges_gradient(scale = 4, rel_min_height = 0.001, gradient_lwd = 1.,bandwidth = 30) +
  labs(title = 'Dropout density',
       subtitle = '') +
  theme_ridges(font_size = 13, grid = TRUE) + theme(axis.title.y = element_blank())+
  xlab("Neuron unit")+
  ylab("repeat")+
  guides(fill=guide_legend(title="Data point"))





####distances among different point
Intra=NULL
Inter=NULL
for (augmenation in c("none","snow","frost","gaussian_noise")){#c("snow","frost","gaussian_noise")
  
  for (severity in 1:5){
    
    for (Method in c("random","topdown","bottomup")){
      
    
    if (augmenation=="none"){
      dataname="cifar10"
      severity=0
    } else {dataname="cifar10c"}
    File=paste0("../Results/masks/",dataname,"_",augmenation,"_",severity,"_",Method,"_False_False_False_False_0_mask.csv")
  
    print(File)
    
    
    data=read.csv(File,header = T)
    data=data[,2:ncol(data)]
    data=data[1:(128*20),]#only use the first batch
    data$DataPoint=1:128
    
    ###intra-data point distance
    DIS=NULL
    for (DataPoint in unique(data$DataPoint) ){
      VecPlot=data[data$DataPoint==DataPoint ,c(1:1024)]
      DIS=c(DIS,max(dist(VecPlot,method="manhattan")))
      
    
    }
    DIS=data.frame(Method=Method,Dist=DIS,augmenation=augmenation,severity=severity)
    Intra=rbind(Intra,DIS)
    

    
    ###inter-data point distance
    
    VecPlot=NULL
    for (DataPoint in unique(data$DataPoint)){
      Vec=data[data$DataPoint==DataPoint ,c(1:1024)]
      VecPlot=rbind(VecPlot,colMeans(Vec))

    }
    DIS=max(c(dist(VecPlot,method="manhattan")))
    DIS=data.frame(Method=Method,Dist=DIS,augmenation=augmenation,severity=severity)
    Inter=rbind(Inter,DIS)
    }
    
  }
  
}

head(Inter)
head(Intra)


#  geom_point()+geom_errorbar(aes(ymin=Accuracy-se, ymax=Accuracy+se), width=.2,
#position=position_dodge(0.05))+

#inta distance

Data_name="CIFAR10"

Intra$severity=factor(Intra$severity)
Inter$severity=factor(Inter$severity)

AllData=list(Intra,Inter)
names(AllData)=c("Intra","Inter")

for (ExperimentName in c("Intra","Inter")) {
  for (augmenation in c("none","snow","frost","gaussian_noise")){
  
  Vec=AllData[ExperimentName][[1]]
  VecPlot=Vec[Vec$augmenation==augmenation,] 
  
  if (augmenation=="none"){
    VecPlot$severity=1#there is no augmentaion
  }
  
  VecPlot$augmenation[VecPlot$augmenation=="none"]="no"
  
  VecPlot$Method[VecPlot$Method=="random"]="Random"
  VecPlot$Method[VecPlot$Method=="bottomup"]="GFlowOut"
  VecPlot$Method[VecPlot$Method=="topdown"]="ID-GFlowOut"
  
  
  VecPlot=aggregate(Dist~Method,data=VecPlot,mean)
  
  VecPlot$Method=factor(VecPlot$Method,levels =c("Random","ID-GFlowOut","GFlowOut") )
  #VecPlot$severity=factor(VecPlot$severity )
  #VecPlot=VecPlot[VecPlot$severity==1,]
  
  
  VecPlot=unique(VecPlot)
  
  

  
  P=ggplot(data=VecPlot, aes(x=Method, y=Dist, fill=Method)) +
    geom_bar(stat="identity")+
    xlab("Amount of augmentation") +
    ylab("Diveristy(pairwise distance)") + ggtitle(paste(Data_name,"with",augmenation,'augmentation',"(",ExperimentName,"sample )"))+
    theme(plot.title = element_text(hjust = 0.5),
          panel.background = element_blank(),legend.position = "bottom",
          panel.grid.minor = element_line(colour="grey", size=0.5),
          text = element_text(size=20),
          legend.text=element_text(size=20))+
    coord_cartesian(ylim=c(min(VecPlot$Dist)-2,max(VecPlot$Dist)+2))
  
  P
 
  ggsave(plot = P,filename = paste0("../images/Diversity_",Data_name,"_",augmenation,"_",ExperimentName,".pdf"),device = "pdf",width=10,height=6)
  
  }
}

