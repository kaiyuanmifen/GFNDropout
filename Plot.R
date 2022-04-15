

AllFiles=list.files('Results/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){

  Vec=read.csv(paste0("Results/",File))
  names(Vec)[1]="Epoch"

  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  Vec$p=as.numeric(Infor[length(Infor)-2])
  
  if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  
  Vec$N_units=as.integer(Infor[length(Infor)-3])
  
  Vec$Data=Infor[length(Infor)-4]
  
  Vec$Method=paste0(Infor[1:(length(Infor)-5)],collapse = "_")
  
  
  Data=rbind(Data,Vec)
  
  
  }
  
  }





#####task

library(ggplot2)
names(Data)
#Data=Data[Data$Method!="MLP_SVD",]

DataNames="MNIST"
VecPlot=Data[(Data$Data==DataNames)&(Data$Epoch<200)&(Data$N_units==10)&(Data$p==0.2),]
head(VecPlot)
p <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Method)) +geom_smooth()+
  ggtitle(DataNames)
p

ggsave(plot = p,paste0('images/',DataNames,"_testacc.png"),scale=3)


DataNames="CIFAR10"
VecPlot=Data[(Data$Data==DataNames)&(Data$Epoch<200)&(Data$N_units==30)&(Data$p==0.2),]
head(VecPlot)
p <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Method)) +geom_smooth()+
  ggtitle(DataNames)
p

ggsave(plot = p,paste0('images/',DataNames,"_testacc.png"),scale=3)


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
