rm(list=ls(all=TRUE))

#Read the data into R
data<-read.csv("Machine_data.csv")
dim(data) #136399 * 24

#Checking for NAs
data[ data == '-1']<- NA
sum(is.na(data))
colSums(is.na(data))  #V1 - 1541, V3-76, V5-45, V14 - 48, V15-3, v18-162, V21-2938
sum(!complete.cases(data)) #- 3204 rows have NAs
sum(!complete.cases(data))/nrow(data) #0.023 (its 2 percent of data)
#We can omit NA rows since only 2 percent of records have NA in it.
data<-na.omit(data)
dim(data) #133195 * 24

#Checking for duplicate records
data<-data[!duplicated(data),]
dim(data) #133195 * 24

#Understanding Data
str(data)
summary(data)
names(data)
apply(data,2,function(x){length(unique(x))})

#id is not needed removing it
data$id <- NULL
#data$Breakdown <- as.factor(as.character(data$Breakdown))
data$Breakdown <- as.factor(as.character(data$Breakdown))

#Binning V19 and v20 as they have more levels
data$V19 <- ifelse(data$V19 <= 0.5, 0.5, 
                   (ifelse(data$V19 <= 1, 1, 
                           (ifelse(data$V19 <= 1.5, 1.5, 
                                   (ifelse(data$V19 <= 2 , 2, 
                                           (ifelse(data$V19 <=2.5, 2.5, 
                                                   (ifelse(data$V19 <=3, 3, 
                                                           (ifelse(data$V19 <= 3.5, 3.5, 4)))))))))))))


data$V20 <- ifelse(data$V20 <= 1, 1, 
                   (ifelse(data$V20 <=1.5 , 1.5, 
                           (ifelse(data$V20 <= 2 , 2, 
                                   (ifelse(data$V20 <=2.5, 2.5, 
                                           (ifelse(data$V20 <=3, 3, 
                                                   (ifelse(data$V20 <= 3.5, 3.5, 4)))))))))))

#Observed outliers 

boxplot(data$V22)
boxplot(data$V19)
boxplot(data$V18)
boxplot(data$V17)
boxplot(data$V16) # 4, 5, 7
dim(data)  #133195   *  23

data<- data[data$V22 < 16,]
dim(data)  #132549   *  23

data<- data[data$V19 <2.5,]
dim(data)  #132490   *  23

data<- data[!(data$V16 %in% c(4,5,7)),]
dim(data) #132311     23 
#136399 - intial rows 
#133195 after omiting NAs
#132490 after removing outliers
(136399-132490)/136399 #its ~3% rows removed

str(data)
apply(data,2,function(x){length(unique(x))})

#Split the dataset into test and train using stratified sampling
library('DMwR')
library('caret')
set.seed(785)
rows<-createDataPartition(data$Breakdown,times=1,p=0.7,list = F)
train<-data[rows,]
test<-data[-rows,]
train <- SMOTE(Breakdown ~.,train, perc.over = 100, k = 5, perc.under = 200)

#checking Breakdown ratio in train and test
dim(train) #93238 * 23
dim(test) #39957 * 23 
table(train$Breakdown)
14520/93238 # 15.57
table(test$Breakdown)
6222/39957  # 15.57
table(train$Breakdown)
table(test$Breakdown)
# build the classification model using Adaboost
#install.packages("ada")
library(ada) 
model = ada(Breakdown ~ ., iter = 20,data = train, loss="logistic") 
# iter = 20 Iterations 
model

# predict the values using model on test data sets. 
pred = predict(model, test);
table(pred)

confusionMatrix(pred,test$Breakdown,positive = "1")



#*********************
# Constructing the Dense matrix on the train and test data
library(vegan)
library(dummies)
#install.packages("xgboost")
library(xgboost)
train$Breakdown <- as.numeric(train$Breakdown)-1
test$Breakdown <- as.numeric(test$Breakdown)-1
ind_Attr <- setdiff(names(train), "Breakdown")
str(train$Breakdown)
table(train$Breakdown)
dtrain = xgb.DMatrix(data = as.matrix(train[,ind_Attr]),
                     label = train$Breakdown)
dtrain
print(dtrain)
dtest = xgb.DMatrix(data = as.matrix(test[,ind_Attr]),
                    label = test$Breakdown)

# fit the model
model = xgboost(data = dtrain, max.depth = 4, 
                eta = 0.4, nthread = 2, nround = 40, 
                objective = "binary:logistic", verbose = 1)

#watchlist = list(train=dtrain, test=dtest)


model = xgb.train(data=dtrain, max.depth=4,
                  eta=0.3, nthread = 2, nround=20, 
                  watchlist=watchlist,
                  eval.metric = "error", 
                  objective = "binary:logistic", verbose = 1)
# eval.metric allows us to monitor two new metrics for each round, logloss and error.

importance <- xgb.importance(feature_names = ind_Attr, model = model)
print(importance)
xgb.plot.importance(importance_matrix = importance)

# save model to binary local file
xgb.save(model, "xgboost.model")
rm(model)

# load binary model to R
model <- xgb.load("xgboost.model")

# prediction on test data
pred <- predict(model, as.matrix(test[,ind_Attr]))

# size of the prediction vector
print(length(pred))

# limit display of predictions to the first 10
print(head(pred))

# The numbers we get are probabilities that a datum will be classified as 1. 
# Therefore, we will set the rule that if this probability for a 
# specific datum is > 0.5 then the observation is classified as 1 (or 0 otherwise).

prediction <- ifelse(pred > 0.5,1,0)
print(head(prediction))
table(prediction)
prediction <- as.factor(as.character(prediction))

confusionMatrix(prediction, test$Breakdown, positive = '1')



