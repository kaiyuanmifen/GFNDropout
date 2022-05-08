

AllFiles=list.files('Results/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){

  Vec=read.csv(paste0("Results/",File))
  names(Vec)[1]="Epoch"

  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  Vec$p=as.numeric(Infor[length(Infor)-5])
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  
  Vec$N_units=as.integer(Infor[length(Infor)-6])
  
  Vec$Data=Infor[length(Infor)-7]
  
  
  
  if (grepl(pattern = "GFFN",Infor[2])){
  Vec$Method=paste0(Infor[1],"_",Infor[2],"_",Infor[7])
  } else {
    Vec$Method=paste0(Infor[1],"_",Infor[2])
  }
  
  
  Vec$RewardType=as.integer(Infor[length(Infor)-3])
  
  Vec$beta=as.numeric(Infor[length(Infor)-4])
  
  Vec$DataRatio=as.numeric(Infor[length(Infor)-2])
  
  Vec$Model=Infor[1]
  
  
  Data=rbind(Data,Vec)
  
  
  #}
  
  }

tail(Data,100)
unique(Data$Method)
unique(Data$Model)


#####task

library(ggplot2)
names(Data)
head(Data)
#Data=Data[Data$Method!="MLP_SVD",]


DataNames="CIFAR10"
DataRatio=1
Model="MLP"

Exp=paste0(Model,"_",DataNames,"_",N_units,"_",p,"_",RewardType,"_",DataRatio)
          
#VecPlot1=Data[(Data$Data==DataNames)&(Data$Method%in%paste0(Model,c("_GFNDB","_GFNFM")))&(Data$N_units==N_units)&(Data$p==p)&(Data$OODReward==OODReward),]
head(VecPlot)

VecPlot=Data[(Data$DataRatio==DataRatio)&(Data$Data==DataNames)&(Data$Model==Model),]

unique(VecPlot$Method)

#VecPlot=VecPlot[grepl(pattern = "MLP_nodropout",x=VecPlot$Method),]

Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Method)) +geom_smooth()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images/',Exp,"_testacc.png"),scale=3)


Plot <- ggplot(VecPlot[VecPlot$Epoch>25,], aes(x=Method, y=test_acc,color=Method)) +geom_boxplot()+
  ggtitle(DataNames)

ggsave(plot = Plot,paste0('images/',Exp,"_testaccBox.png"),scale=3)


head(VecPlot)
Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc_OOD,color=Method)) +geom_line()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images/',Exp,"_testaccOOD.png"),scale=3)

Plot <- ggplot(VecPlot[VecPlot$Epoch>25,], aes(x=Method, y=test_acc_OOD,color=Method)) +geom_boxplot()+
  ggtitle(DataNames)

ggsave(plot = Plot,paste0('images/',Exp,"_testaccOODBox.png"),scale=3)


