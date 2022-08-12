####visualization reuslts


library(ggplot2)


Dir="logs/mnist-gfn_20220812174445/"

AllExp=list.files(Dir)
AllExp=AllExp[grepl(AllExp,pattern = ".csv")]


for (Name in AllExp){

Data=read.csv(paste0(Dir,Name),header = F)
Data=Data[,c(2,3)]
names(Data)=c("step","value")
# 
# if (grepl(Name,pattern = "loss")){
#   B=ggplot(data=Data, aes(x=step, y=log(value), )) +
#     geom_path()+
#     geom_point()+
#     geom_smooth()+
#     ggtitle(Name)
# } else{

plotname=strsplit(Name,"[.]")[[1]][1]
B=ggplot(data=Data, aes(x=step, y=value, )) +
  geom_path()+
  geom_point()+
  geom_smooth()+
  ggtitle(plotname)
#}

ggsave(B,filename =paste0("images/",plotname,".png"),width = 5,height = 5 )
}



####analyse test output

data=read.csv("images/testmasked_testResults.csv")
head(data)

names(data)[1]="Index"

B=ggplot(data=data, aes(x=LogRmu, y=LogPF)) +
  geom_point()+
  geom_smooth(method="glm")+
  ggtitle("LogPF vs LogRmu")


ggsave(B,filename =paste0("images/","maskedLogPF_mu vs Rmu",".png"),width = 5,height = 5  )





B=ggplot(data=data, aes(x=LogZmu, y=LogPF)) +
  geom_point()+
  geom_smooth()+
  ggtitle("LogPF vs Rmu")


ggsave(B,filename =paste0("images/","LogZmuvsRmu",".png") )



B=ggplot(data=data, aes(x=LogPF, y=LogRmu)) +
  geom_point()+
  geom_smooth()+
  ggtitle("LogPF vs LogRmu")



B=ggplot(data=data, aes(x=LogPF, y=LogZmu+LogPF-LogRmu)) +
  geom_point()+
  geom_smooth()+
  ggtitle("LogPF vs LogPF-LogRmu")



ggplot(data, aes(x=(LogZmu-LogRmu))) + geom_histogram()




ggplot(data, aes(x=(LogPF-LogRmu))) + geom_histogram()



ggplot(data, aes(x=(LogZmu+LogPF-LogRmu))) + geom_histogram()


ggplot(data, aes(x=(LogZmu+LogPF-LogRmu)^2)) + geom_histogram()
