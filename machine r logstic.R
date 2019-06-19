rm(list = ls(all=TRUE))


#reading of data 
data<-read.csv("Machine_data.csv")
#removing of unwanted variables
data<-data[-1]
set.seed(1)

#data<-data.frame(data)
#replacing "-1" with na's
data[data==-1]<-NA
#to check no.of na's
sum(is.na(data))
library(DMwR)
data[,-23]<-centralImputation(data[,-23]) 
sum(is.na(data))
#to omit no.of na's
#data<-na.omit(data)
#sum(is.na(data))
str(data)

# scaling of data
data[-23]=scale(data[-23])

# spliting of data into train and test

library(caTools)
split=sample.split(data,SplitRatio = 0.75)
split
train<-subset(data,split=="TRUE")
test<-subset(data,split=="FALSE")

#PCA
library(caret)

#pre1<-preProcess(train[,setdiff(colnames(train),"rating")])
#train_scale<-predict(pre1,train[,setdiff(colnames(train),"rating")])
#test_scale<-predict(pre1,test[,setdiff(colnames(test),"rating")])
#dim(train_scale)
#dim(test_scale)

#prin_comp<-princomp(train_scale)
#plot(prin_comp)

library(e1071)

pca=preProcess(x=train[-23],method='pca',pcaComp =10)
train=predict(pca,train)
test=predict(pca,test)

#model building

model<-glm(Breakdown~ . -PC3 -PC6,train,family="binomial")
summary(model)



#step AIC
#library(MASS)
#STEP<-stepAIC(model,direction = "both")
#summary(STEP)

#predicting train 
res<-predict(model,train,type="response")

#plot(Breakdown~.,train , col="red4")
#lines(train$Breakdown,predicted,col="green",lwd=2)  

#confussion matrix of train data
b<-(table(Actualvalue=train$Breakdown,predictedvalues=res>0.5))
b
accurcytrain<-sum(diag(b))/sum(b)
accurcytrain
#predicting test 
res1<-predict(model,test,type="response")
#confussion matrix of train data
a<-(table(Actualvalue=test$Breakdown,predictedvalues=res1>0.3))
a
accurcytest<-sum(diag(a))/sum(a)
accurcytest
# to know the threshold value
library(ROCR)
library(pROC)
ROCRAPred<-prediction(res,train$Breakdown)
as.nu
rocrpref<-performance(ROCRAPred,"tpr","fpr")

plot(rocrpref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))


