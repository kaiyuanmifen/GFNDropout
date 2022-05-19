

AllFiles=list.files('Results_All_Best/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){

  Vec=read.csv(paste0("Results_All_Best/",File))
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

for (Model in c("RESNET")){
  for (DataNames in c("SVHN", "CIFAR10","MNIST")){
    for (N_units in c(1024)){
      for (p in c(0.1,0.2,0.5,0.7,0.9)){
          for (beta in c(1.0)){
        for (OODReward in c(1,0)){
          for (seed in c(1)){
#Task_name=args.Method+"_"+args.Data+"_"+str(args.Hidden_dim)+"_"+str(args.p)+"_"+str(args.beta)+"_"+str(args.RewardType)+"_"+str(args.DataRatio)+"_"+str(args.seed)

Exp=paste0(Model,"_",DataNames,"_",N_units,"_",p,"_",beta,"_2_1.0","_",seed)
          
VecPlot1=Data[(Data$Data==DataNames)&(Data$Method%in%paste0(Model,c("_GFNDB","_GFNFM","_GFFN")))&(Data$N_units==N_units)&(Data$p==p)&(Data$OODReward==OODReward),]
VecPlot2=Data[(Data$Data==DataNames)&(Data$Method%in%paste0(Model,c("_dropout","_Standout")))&(Data$N_units==N_units)&(Data$p==p),]
VecPlot3=Data[(Data$Data==DataNames)&(Data$Method%in%paste0(Model,c( "_nodropout","_SVD")))&(Data$N_units==N_units),]


# VecPlot1=Data[(Data$Data==DataNames)&(Data$Method%in%c("CNN_GFNDB","CNN_GFNFM"))&(Data$N_units==N_units)&(Data$p==p)&(Data$OODReward==OODReward),]
# VecPlot2=Data[(Data$Data==DataNames)&(Data$Method%in%c("CNN_dropout","CNN_Standout"))&(Data$N_units==N_units)&(Data$p==p),]
# VecPlot3=Data[(Data$Data==DataNames)&(Data$Method%in%c( "CNN_nodropout","CNN_SVD"))&(Data$N_units==N_units),]


VecPlot=rbind(VecPlot1,VecPlot2,VecPlot3)

head(VecPlot)
Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Method)) +geom_line()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images_All_Best_101/',Exp,"_testacc.png"),scale=3)


head(VecPlot)
Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc_OOD,color=Method)) +geom_line()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images_All_Best_101/',Exp,"_testaccOOD.png"),scale=3)
        }

      }
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
