#Result visualizaiton
library(ggplot2)

#CIFAT 10C and 100C

Data_name="CIFAR-10"
data=read.csv("../Results/cifar10c_testresult.csv",header = TRUE)
#data=data[,2:ncol(data)]



names(data)=c("load_file","dataset","corruption_name","corruption_severity",
            "Method","BNN","use_pretrained","Tune_last_layer_only",
            "augment_test","seed","val_loss","val_accuracy", "base_aic0",
            "base_aic1","base_aic2", "up0",
              "up1","up2", "elbo", "ece")

head(data)

#data=data[data$Method!="topdown",]
#unique(data$Method)
#write.csv(data,file = "../Results/cifar10c_testresult.csv")
head(data)

unique(data$Method)


data$Method[grepl(data$load_file,pattern = "Contextual")]="Contextual"
data$Method[grepl(data$load_file,pattern = "Concrete")]="Concrete"



data$Method[data$Method=="random"]="Random"

data$Method[data$Method=="topdown"]="ID-GFlowOut"

data$Method[data$Method=="bottomup"]="GFlowOut"

unique(data$Method)


data$Method=factor(data$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)


tail(data)

data=data[data$Method!="Random",]
#data$corruption_severity=factor(data$corruption_severity)


se <- function(x, ...) sqrt(var(x, ...)/length(x))

###all corrupation
VecPlot=NULL

for (corruption_severity in unique(data$corruption_severity)){
  
  for (Method in unique(data$Method)){
    Vec=data[(data$corruption_severity==corruption_severity)&(data$Method==Method),]
    Vec=data.frame(corruption=corruption,corruption_severity=corruption_severity,
                   Method=Method,Accuracy=mean(Vec$val_accuracy),se=se(Vec$val_accuracy))
    
    VecPlot=rbind(VecPlot,Vec)
  }  
  
}
VecPlot$Method=factor(VecPlot$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)

P=ggplot(data=VecPlot, aes(x=corruption_severity, y=Accuracy, color=Method)) +
  geom_line(size=1)+
  geom_point()+geom_errorbar(aes(ymin=Accuracy-se, ymax=Accuracy+se), width=.2,
                             position=position_dodge(0.05))+
  xlab("Amount of deformation") +
  ylab("Accuracy") + ggtitle(paste(Data_name,"with","all",'deformations'))+
  theme(plot.title = element_text(hjust = 0.5),
        panel.background = element_blank(),legend.position = "bottom",
        panel.grid.minor = element_line(colour="grey", size=0.5),
        text = element_text(size=20),
        legend.text=element_text(size=20))+
  scale_color_brewer(palette="Spectral")

P
ggsave(plot = P,filename = paste0("../images/",Data_name,"_","all",".pdf"),device = "pdf",width=10,height=6)


for (corruption in unique(data$corruption_name)){
  VecPlot=NULL
  
  for (corruption_severity in unique(data$corruption_severity)){
    
    for (Method in unique(data$Method)){
          Vec=data[(data$corruption_name==corruption)&(data$corruption_severity==corruption_severity)&(data$Method==Method),]
          Vec=data.frame(corruption=corruption,corruption_severity=corruption_severity,
                                Method=Method,Accuracy=mean(Vec$val_accuracy),se=se(Vec$val_accuracy))
    
          VecPlot=rbind(VecPlot,Vec)
          }  
    
  }
 
  VecPlot$Method=factor(VecPlot$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)
  
  P=ggplot(data=VecPlot, aes(x=corruption_severity, y=Accuracy, color=Method)) +
    geom_line(size=1)+
    geom_point()+geom_errorbar(aes(ymin=Accuracy-se, ymax=Accuracy+se), width=.2,
                                position=position_dodge(0.05))+
    xlab("Amount of deformation") +
    ylab("Accuracy") + ggtitle(paste(Data_name,"with",toupper(corruption),'deformation'))+
    theme(plot.title = element_text(hjust = 0.5),
          panel.background = element_blank(),legend.position = "bottom",
          panel.grid.minor = element_line(colour="grey", size=0.5),
          text = element_text(size=20),
          legend.text=element_text(size=20))+
    scale_color_brewer(palette="Spectral")
  
  
  ggsave(plot = P,filename = paste0("../images/",Data_name,"_",corruption,".pdf"),device = "pdf",width=10,height=6)
}




#####CIFAR100

Data_name="CIFAR-100"
data=read.csv("../Results/cifar100c_testresult.csv",header = TRUE)
#data=data[,2:ncol(data)]

head(data)

names(data)=c("load_file","dataset","corruption_name","corruption_severity",
              "Method","BNN","use_pretrained","Tune_last_layer_only",
              "augment_test","seed","val_loss","val_accuracy", "base_aic0",
              "base_aic1","base_aic2", "up0",
              "up1","up2", "elbo", "ece")

head(data)
# 
# data=data[data$Method!="topdown",]
# unique(data$Method)
# write.csv(data,file = "../Results/cifar100c_testresult.csv")
head(data)

unique(data$Method)


data$Method[grepl(data$load_file,pattern = "Contextual")]="Contextual"
data$Method[grepl(data$load_file,pattern = "Concrete")]="Concrete"



data$Method[data$Method=="random"]="Random"

data$Method[data$Method=="topdown"]="ID-GFlowOut"

data$Method[data$Method=="bottomup"]="GFlowOut"

unique(data$Method)


data$Method=factor(data$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)


tail(data)

data=data[data$Method!="Random",]
#data$corruption_severity=factor(data$corruption_severity)


se <- function(x, ...) sqrt(var(x, ...)/length(x))

###all corrupation
VecPlot=NULL

for (corruption_severity in unique(data$corruption_severity)){
  
  for (Method in unique(data$Method)){
    Vec=data[(data$corruption_severity==corruption_severity)&(data$Method==Method),]
    Vec=data.frame(corruption=corruption,corruption_severity=corruption_severity,
                   Method=Method,Accuracy=mean(Vec$val_accuracy),se=se(Vec$val_accuracy))
    
    VecPlot=rbind(VecPlot,Vec)
  }  
  
}
VecPlot$Method=factor(VecPlot$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)

P=ggplot(data=VecPlot, aes(x=corruption_severity, y=Accuracy, color=Method)) +
  geom_line(size=1)+
  geom_point()+geom_errorbar(aes(ymin=Accuracy-se, ymax=Accuracy+se), width=.2,
                             position=position_dodge(0.05))+
  xlab("Amount of deformation") +
  ylab("Accuracy") + ggtitle(paste(Data_name,"with","all",'deformations'))+
  theme(plot.title = element_text(hjust = 0.5),
        panel.background = element_blank(),legend.position = "bottom",
        panel.grid.minor = element_line(colour="grey", size=0.5),
        text = element_text(size=20),
        legend.text=element_text(size=20))+
  scale_color_brewer(palette="Spectral")

P
ggsave(plot = P,filename = paste0("../images/",Data_name,"_","all",".pdf"),device = "pdf",width=10,height=6)


for (corruption in unique(data$corruption_name)){
  VecPlot=NULL
  
  for (corruption_severity in unique(data$corruption_severity)){
    
    for (Method in unique(data$Method)){
      Vec=data[(data$corruption_name==corruption)&(data$corruption_severity==corruption_severity)&(data$Method==Method),]
      Vec=data.frame(corruption=corruption,corruption_severity=corruption_severity,
                     Method=Method,Accuracy=mean(Vec$val_accuracy),se=se(Vec$val_accuracy))
      
      VecPlot=rbind(VecPlot,Vec)
    }  
    
  }
  
  VecPlot$Method=factor(VecPlot$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)
  
  P=ggplot(data=VecPlot, aes(x=corruption_severity, y=Accuracy, color=Method)) +
    geom_line(size=1)+
    geom_point()+geom_errorbar(aes(ymin=Accuracy-se, ymax=Accuracy+se), width=.2,
                               position=position_dodge(0.05))+
    xlab("Amount of deformation") +
    ylab("Accuracy") + ggtitle(paste(Data_name,"with",toupper(corruption),'deformation'))+
    theme(plot.title = element_text(hjust = 0.5),
          panel.background = element_blank(),legend.position = "bottom",
          panel.grid.minor = element_line(colour="grey", size=0.5),
          text = element_text(size=20),
          legend.text=element_text(size=20))+
    scale_color_brewer(palette="Spectral")
  
  
  ggsave(plot = P,filename = paste0("../images/",Data_name,"_",corruption,".pdf"),device = "pdf",width=10,height=6)
}















####transferlearning 


###cifar10


Data_name="CIFAR-10"
data=read.csv("../Results/transfer_cifar10_testresult.csv",header = F)



names(data)=c("load_file","dataset","corruption_name","corruption_severity",
              "Method","BNN","use_pretrained","Tune_last_layer_only",
              "augment_test","seed","val_loss","Accuracy", "base_aic0",
              "base_aic1","base_aic2", "up0",
              "up1","up2", "elbo", "ece")


head(data)

unique(data$Method)


data$Method[grepl(data$load_file,pattern = "Contextual")]="Contextual"
data$Method[grepl(data$load_file,pattern = "Concrete")]="Concrete"



data$Method[data$Method=="random"]="Random"

data$Method[data$Method=="topdown"]="ID-GFlowOut"

data$Method[data$Method=="bottomup"]="GFlowOut"




data$Method=factor(data$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)


data$N_samples=as.integer(unlist(lapply(strsplit(data$load_file,"_"),FUN = function(x){x[[6]]})))



data=data[data$Method!="Random",]
#data$corruption_severity=factor(data$corruption_severity)

head(data)

VecPlot=data

VecPlot$Method=factor(VecPlot$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)

P=ggplot(data=VecPlot, aes(x=N_samples, y=Accuracy, color=Method)) +
  geom_line(size=1)+
  geom_point()+
  xlab("Number of data points") +
  ylab("Accuracy") + ggtitle(paste(Data_name,'transfer learning'))+
  theme(plot.title = element_text(hjust = 0.5),
        panel.background = element_blank(),
        legend.position = "bottom",
        panel.grid.minor = element_line(colour="grey", size=0.5),
        text = element_text(size=30),
        legend.text=element_text(size=30))+
  scale_color_brewer(palette="Spectral")

P
ggsave(plot = P,filename = paste0("../images/transfer",Data_name,"_","all",".pdf"),
       device = "pdf",width=14,height=8)


###cifar-100


Data_name="CIFAR-100"
data=read.csv("../Results/transfer_cifar100_testresult.csv",header = F)



names(data)=c("load_file","dataset","corruption_name","corruption_severity",
              "Method","BNN","use_pretrained","Tune_last_layer_only",
              "augment_test","seed","val_loss","Accuracy", "base_aic0",
              "base_aic1","base_aic2", "up0",
              "up1","up2", "elbo", "ece")


head(data)

unique(data$Method)


data$Method[grepl(data$load_file,pattern = "Contextual")]="Contextual"
data$Method[grepl(data$load_file,pattern = "Concrete")]="Concrete"



data$Method[data$Method=="random"]="Random"

data$Method[data$Method=="topdown"]="ID-GFlowOut"

data$Method[data$Method=="bottomup"]="GFlowOut"




data$Method=factor(data$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)


data$N_samples=as.integer(unlist(lapply(strsplit(data$load_file,"_"),FUN = function(x){x[[6]]})))



data=data[data$Method!="Random",]
#data$corruption_severity=factor(data$corruption_severity)

head(data)

VecPlot=data

VecPlot$Method=factor(VecPlot$Method,levels=c("Random","Concrete","Contextual","ID-GFlowOut","GFlowOut"),ordered = T)

P=ggplot(data=VecPlot, aes(x=N_samples, y=Accuracy, color=Method)) +
  geom_line(size=1)+
  geom_point()+
  xlab("Number of data points") +
  ylab("Accuracy") + ggtitle(paste(Data_name,'transfer learning'))+
  theme(plot.title = element_text(hjust = 0.5),
        panel.background = element_blank(),
        legend.position = "bottom",
        panel.grid.minor = element_line(colour="grey", size=0.5),
        text = element_text(size=30),
        legend.text=element_text(size=30))+
  scale_color_brewer(palette="Spectral")

P
ggsave(plot = P,filename = paste0("../images/transfer",Data_name,"_","all",".pdf"),
       device = "pdf",width=14,height=8)





