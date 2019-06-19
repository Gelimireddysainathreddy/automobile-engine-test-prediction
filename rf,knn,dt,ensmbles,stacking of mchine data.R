
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
table(data$Breakdown)
(20575/133195)
#id is not needed removing it
data$id <- NULL
#data$Breakdown <- as.factor(as.character(data$Breakdown))
data$Breakdown<-factor(data$Breakdown,levels=c(0,1))

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

# Standardize all the real valued variables in the dataset 
#as some models we use might be impacted due to non standardized variables

std_method <- preProcess(train, method = c("center", "scale"))
train <- predict(std_method, train)
test <- predict(std_method, test)
train <- SMOTE(Breakdown ~.,train, perc.over = 100, k = 5, perc.under = 200)


install.packages("randomForest")
library(randomForest)
model_rf <- randomForest(Breakdown ~ . , data,ntree = 50,mtry = 15)
#look at variable importance from the built model using the importance() function and visualise it using the varImpPlot() funcion

importance(model_rf)
varImpPlot(model_rf)

# Store predictions from the model
preds_rf <- predict(model_rf, test)
table(preds_rf)
confusionMatrix(preds_rf, test$Breakdown, positive = "1")
dim(test)
table(test$Breakdown)
# Predict on the train data
preds_train_rf <- predict(model_rf, train)
confusionMatrix(preds_train_rf, train$Breakdown)
table(preds_train_rf, train$Breakdown)


# We'll build our KNN model, using the knn3() function from the caret package

model_knn <- knn3(Breakdown ~ . , train, k = 5)

preds_k <- predict(model_knn, test)
head(preds_k)
# * The predict function on the knn model returns probabilities for each of the two classes in the target variable, so we'll get to the class labels using the ifelse() function
preds_knn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
head(preds_knn)
confusionMatrix(preds_knn, test$Breakdown,positive = "1")

# * Store the predictions on the train data

preds_train_k <- predict(model_knn, train)

preds_train_knn <- ifelse(preds_train_k[, 1] > preds_train_k[, 2], 0, 1)

confusionMatrix(preds_train_knn, train$Breakdown, positive =  "1")
unique(preds_train_k)
# Decision Trees
library(rpart)

model_dt <- rpart(Breakdown ~ . , train)

# * The predictions here too are probabilities for each of the two classes in the target variable

preds_dt <- predict(model_dt, test)

preds_tree <- ifelse(preds_dt[, 1] > preds_dt[, 2], 0, 1)

confusionMatrix(preds_tree, test$Breakdown, positive ="1")

# * Store the predictions on the train data
preds_train_dt <- predict(model_dt)
preds_train_tree <- ifelse(preds_train_dt[, 1] > preds_train_dt[, 2], 0, 1)
confusionMatrix(preds_train_tree, train$Breakdown)

library(ipred)

set.seed(1234)
dim(train)
model_tree_bag <- bagging(Breakdown ~ . , data=train,nbagg = 10,
                          control = rpart.control(cp = 0.01, xval = 10))

# * Test the model on the validation data and store the predictions on both the test and validation data

preds_tree_bag <- predict(model_tree_bag, test)
unique(preds_tree_bag)
confusionMatrix(preds_tree_bag, test$Breakdown, positive = "1")

preds_train_tree_bag <- predict(model_tree_bag)
confusionMatrix(preds_train_tree_bag, train$Breakdown)

# Building a Stacked Ensemble
# * Before building a stacked ensemble model, we have to coallate all the predictions on the train and validation datasets into a dataframe
# Getting all the predictions on the train data into a dataframe

train_preds_df <- data.frame(rf = preds_train_rf, knn = preds_train_knn,
                             tree = preds_train_tree, tree_bag = preds_train_tree_bag,
                             Breakdown = train$Breakdown)

# convert the target variable into a factor
train_preds_df$Breakdown <- as.factor(as.character(train_preds_df$Breakdown))

# * Now, since the outputs of the various models are extremely correlated let's use PCA to reduce the dimensionality of the dataset
# * Use the sapply() function to convert all the variables other than the target variable into a numeric type

numeric_st_df <- sapply(train_preds_df[, !(names(train_preds_df) %in% "Breakdown")], 
                        function(x) as.numeric(as.character(x)))

cor(numeric_st_df)
pca_stack <- prcomp(numeric_st_df, scale = F)
summary(pca_stack)


# Transform the data into the principal components using the predict() fucntion and keep only 3 of the original components

predicted_stack <- as.data.frame(predict(pca_stack, numeric_st_df))[1:2]

# Now, add those columns to the target variable (Cancer) and convert it to a data frame
stacked_df <- data.frame(predicted_stack, Breakdown = train_preds_df$Breakdown)

# * We will be building a logistic regression on the dataset to predict the final target variable

stacked_model <- glm(Breakdown ~ . , data = stacked_df,family = "binomial")

# Getting all the predictions from the validation data into a dataframe

stack_df_test <- data.frame(rf = preds_rf, knn = preds_knn,
                            tree = preds_tree, tree_bag = preds_tree_bag,
                            Breakdown = test$Breakdown)

# Convert the target variable into a factor
stack_df_test$Breakdown <- as.factor(stack_df_test$Breakdown)

# Convert all other variables into numeric
numeric_st_df_test <- sapply(stack_df_test[, !(names(stack_df_test) %in% "Breakdown")],
                             function(x) as.numeric(as.character(x)))

# Apply dimensionality reduction on the numeric attributes

predicted_stack_test <- as.data.frame(predict(pca_stack, numeric_st_df_test))[1:2]

# Combine the target variable along with the reduced dataset
stacked_df_test <- data.frame(predicted_stack_test, Breakdown = stack_df_test$Breakdown)

# * Now, apply the stacked model on the above dataframe

preds_st_test <-  predict(stacked_model, stacked_df_test,type = "response")
preds_st_test <- ifelse(preds_st_test > 0.5,"1","0")


# * Use the confusionMatrix() function from the caret package to get the evaluation metrics on the test data for the various models built today

# Random Forest

confusionMatrix(preds_rf, test$Breakdown)

# KNN

confusionMatrix(preds_knn, test$Breakdown)

# CART Tree

confusionMatrix(preds_tree, test$Breakdown)

# Bagged CART Trees

confusionMatrix(preds_tree_bag, test$Breakdown)

# Stacked Model

confusionMatrix(preds_st_test, stacked_df_test$Breakdown)


