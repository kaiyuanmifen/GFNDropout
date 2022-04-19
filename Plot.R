

AllFiles=list.files('Results/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){

  Vec=read.csv(paste0("Results/",File))
  names(Vec)[1]="Epoch"

  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  Vec$p=as.numeric(Infor[length(Infor)-4])
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  
  Vec$N_units=as.integer(Infor[length(Infor)-5])
  
  Vec$Data=Infor[length(Infor)-6]
  
  Vec$Method=paste0(Infor[1:(length(Infor)-7)],collapse = "_")
  Vec$OODReward=as.integer(Infor[length(Infor)-2])
  
  Vec$beta=as.numeric(Infor[length(Infor)-3])
  
  
  Data=rbind(Data,Vec)
  
  
  #}
  
  }





#####task

library(ggplot2)
names(Data)
head(Data)
#Data=Data[Data$Method!="MLP_SVD",]

for (Model in c("CNN","MLP")){
  for (DataNames in c("SVHN", "CIFAR10","MNIST")){
    for (N_units in c(20,40,80)){
      for (p in c(0.1,0.2,0.5,0.7,0.9)){
        for (OODReward in c(1,0)){

Exp=paste0(Model,"_",DataNames,"_",N_units,"_",p,"_",OODReward)
          
VecPlot1=Data[(Data$Data==DataNames)&(Data$Method%in%paste0(Model,c("_GFNDB","_GFNFM")))&(Data$N_units==N_units)&(Data$p==p)&(Data$OODReward==OODReward),]
VecPlot2=Data[(Data$Data==DataNames)&(Data$Method%in%paste0(Model,c("_dropout","_Standout")))&(Data$N_units==N_units)&(Data$p==p),]
VecPlot3=Data[(Data$Data==DataNames)&(Data$Method%in%paste0(Model,c( "_nodropout","_SVD")))&(Data$N_units==N_units),]


# VecPlot1=Data[(Data$Data==DataNames)&(Data$Method%in%c("CNN_GFNDB","CNN_GFNFM"))&(Data$N_units==N_units)&(Data$p==p)&(Data$OODReward==OODReward),]
# VecPlot2=Data[(Data$Data==DataNames)&(Data$Method%in%c("CNN_dropout","CNN_Standout"))&(Data$N_units==N_units)&(Data$p==p),]
# VecPlot3=Data[(Data$Data==DataNames)&(Data$Method%in%c( "CNN_nodropout","CNN_SVD"))&(Data$N_units==N_units),]


VecPlot=rbind(VecPlot1,VecPlot2,VecPlot3)

head(VecPlot)
Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Method)) +geom_line()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images/',Exp,"_testacc.png"),scale=3)


head(VecPlot)
Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc_OOD,color=Method)) +geom_line()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images/',Exp,"_testaccOOD.png"),scale=3)
        }

      }
    }
  }
}
# 
# for (DataNames in c("MNIST","CIFAR10")){
# 
#   for (p in c(0.2,0.8,0.5)){
# 
#     for (N_units in unique(Data$N_units)){
# 
#       NAME=paste0(DataNames,"_p_",p,"_NUnits_",N_units)
#       print(NAME)
#     
#       VecPlot=Data[(Data$Data==DataNames)&(Data$Epoch<200)&(Data$p==p)&(Data$N_units==N_units),]
#       p <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Method)) +geom_smooth()+
#           ggtitle(NAME)
#     
#     
#       ggsave(plot = p,paste0('images/',NAME,"_testacc.png"),scale=3)
#    }
#   }
# }

# 
# 
