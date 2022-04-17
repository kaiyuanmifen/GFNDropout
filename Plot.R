

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

DataNames="MNIST"#CIFAR10 MNIST
N_units=200
p=0.2
OODReward=0
VecPlot1=Data[(Data$Data==DataNames)&(Data$Method%in%c("MLP_GFNDB","MLP_GFNFM"))&(Data$N_units==N_units)&(Data$p==p)&(Data$OODReward==OODReward),]
VecPlot2=Data[(Data$Data==DataNames)&(Data$Method%in%c("MLP_dropout","MLP_Standout"))&(Data$N_units==N_units)&(Data$p==p),]
VecPlot3=Data[(Data$Data==DataNames)&(Data$Method%in%c( "MLP_nodropout","MLP_SVD"))&(Data$N_units==N_units),]

VecPlot=rbind(VecPlot1,VecPlot2,VecPlot3)

head(VecPlot)
p <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Method)) +geom_line()+
  ggtitle(DataNames)
p

ggsave(plot = p,paste0('images/',DataNames,"_testacc.png"),scale=3)


head(VecPlot)
p <- ggplot(VecPlot, aes(x=Epoch, y=test_acc_OOD,color=Method)) +geom_line()+
  ggtitle(DataNames)
p

ggsave(plot = p,paste0('images/',DataNames,"_testaccOOD.png"),scale=3)




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
