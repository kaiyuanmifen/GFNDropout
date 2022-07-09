

AllFiles=list.files('Results/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){

  Vec=read.csv(paste0("Results/",File))
  names(Vec)[1]="Epoch"
  Vec$Info=File

 
  
  Data=rbind(Data,Vec)
  
  
  #}
  
}

Data$Task=Reduce(strsplit(Data$Info,split = "_"),f = rbind)[,4]
Data$Dropout=Reduce(strsplit(Data$Info,split = "_"),f = rbind)[,5]
Data$Repeat=Reduce(strsplit(Data$Info,split = "_"),f = rbind)[,6]
Data$Data=Reduce(strsplit(Data$Info,split = "_"),f = rbind)[,1]



tail(Data,10)


#####task

library(ggplot2)


for (DataName in unique(Data$Data)){
for (task in unique(Data$Task)){
PLOT=ggplot(Data[(Data$Task==task)&(Data$Data==DataName),], aes(x=Epoch, y=test_acc,color=Dropout)) +geom_smooth()+ggtitle(label = paste(DataName,task))

ggsave(PLOT,filename = paste0("images/",DataName,task,"performance.png"),scale = 4)

PLOT=ggplot(Data[(Data$Dropout=="GFN")&(Data$Task==task)&(Data$Data==DataName),], aes(x=Epoch, y=actual_dropout_rates,color=Dropout)) +geom_line()+ggtitle(label = paste(DataName,task))
ggsave(PLOT,filename = paste0("images/",DataName,task,"Dropoutrate.png"))

}}
