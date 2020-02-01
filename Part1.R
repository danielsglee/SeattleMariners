library(tidyverse)
library(rvest)
library(stringr)
library(randomForest)
library(regclass)
library(caret)
library(grDevices)
library(openxlsx)



# 1 - Importing and Preprocessing -----------------------------------------

setwd("C:/Users/Daniel Lee/Desktop/R Projects/Seattle Mariners R&D")

traindf <- read.csv("2020-train.csv",na.strings = c("","NA"))
testdf <- read.csv("2020-test.csv",na.strings = c("","NA"))

traindf <- traindf %>% 
  filter(pitch_call != "InPlay") %>%
  filter(pitch_call != "FoulBall") %>%
  filter(pitch_call != "StrikeSwinging") %>%
  filter(pitch_call != "HitByPitch") %>%
  filter(pitch_call != "BallIntentional")

completeFun <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])
  return(data[completeVec, ])
}                

traindf <- na.omit(traindf) 
testdf <- completeFun(testdf,c("plate_height","plate_side"))
testdf <- testdf %>%
  filter(balls != 4)
testdf <- testdf %>% select (-is_strike)

traindf <- subset(traindf, select = c("pitcher_id","batter_id","pitcher_side","batter_side","catcher_id","umpire_id","balls","strikes","release_speed","rel_height","rel_side","vert_break","horz_break","plate_height","plate_side","pitch_type","pitch_call"))
testdf <- subset(testdf, select = c("pitcher_id","batter_id","pitcher_side","batter_side","catcher_id","umpire_id","balls","strikes","release_speed","rel_height","rel_side","vert_break","horz_break","plate_height","plate_side","pitch_type"))

# 2 - Feature Engineering and Transformation ------------------------------

#Recategorize Pitch Types
traindf$pitch_type <- as.character(traindf$pitch_type)
traindf <- traindf %>% filter(pitch_type != "")
traindf$pitch_type <- gsub(pattern = "FA",x = traindf$pitch_type, replacement = "Fastball")
traindf$pitch_type <- gsub(pattern = "SL", x = traindf$pitch_type, replacement = "OffSpeed")
traindf$pitch_type <- gsub(pattern = "CU", x = traindf$pitch_type, replacement = "OffSpeed")
traindf$pitch_type <- gsub(pattern = "CH", x = traindf$pitch_type, replacement = "OffSpeed")
traindf$pitch_type <- gsub(pattern = "KN", x = traindf$pitch_type, replacement = "OffSpeed")
traindf$pitch_type <- gsub(pattern = "XX", x = traindf$pitch_type, replacement = "OffSpeed")
traindf$pitch_type <- as.factor(traindf$pitch_type)
testdf$pitch_type <- as.character(testdf$pitch_type)
testdf$pitch_type <- gsub(pattern = "FA",x = testdf$pitch_type, replacement = "Fastball")
testdf$pitch_type <- gsub(pattern = "SL", x = testdf$pitch_type, replacement = "OffSpeed")
testdf$pitch_type <- gsub(pattern = "CU", x = testdf$pitch_type, replacement = "OffSpeed")
testdf$pitch_type <- gsub(pattern = "CH", x = testdf$pitch_type, replacement = "OffSpeed")
testdf$pitch_type <- gsub(pattern = "KN", x = testdf$pitch_type, replacement = "OffSpeed")
testdf$pitch_type <- gsub(pattern = "XX", x = testdf$pitch_type, replacement = "OffSpeed")
testdf$pitch_type <- as.factor(testdf$pitch_type)

#Bin the release speed by 2
traindf$release_speed <- cut(traindf$release_speed, breaks = seq(40,106,by = 2))
testdf$release_speed <- cut(testdf$release_speed, breaks = seq(40,106,by = 2))

#Exploring Relationship between variables
traindf %>% 
  ggplot(aes(x = release_speed, colour = pitch_call)) +
  geom_density()

#Assigning Categorical Variable for Balls and Strikes Combination
traindf <- transform(traindf, BallCount = paste(balls,strikes,sep =''))
testdf <- transform(testdf, BallCount = paste(balls,strikes,sep =''))

#Detect Outliers and Getting Rid of Erroneous Pitches
 StrDf <- traindf %>% 
   filter(pitch_call == "StrikeCalled")
 
 H <- StrDf$plate_height
 S <- StrDf$plate_side
 
 HOutliers <- H[which(H %in% boxplot.stats(H)$out)]
 SOutliers <- S[which(S %in% boxplot.stats(S)$out)]
 
 StrDf <- StrDf[!StrDf$plate_height %in% HOutliers,]
 StrDf <- StrDf[!StrDf$plate_side %in% SOutliers,]

 # StrikeZone Constructing
 sample <- StrDf %>%
   group_by(umpire_id,batter_side) %>%
   summarize(StrZoneYD = min(plate_height), StrZoneYU = max(plate_height), StrZoneXL = min(plate_side), StrZoneXR = max(plate_side))

 traindf <- merge(traindf,sample,by = c("umpire_id","batter_side"))
 
 testdf <- merge(testdf,sample,by = c("umpire_id","batter_side"))
 
 traindf$BorderInclusion <- NA
 testdf$BorderInclusion <- NA

 #Assign Yes and No to included/excluded pitches
 for(i in 1:302319)(
       if(traindf$StrZoneYD[i] <= traindf$plate_height[i] & traindf$plate_height[i] <= traindf$StrZoneYU[i] & traindf$StrZoneXL[i] <= traindf$plate_side[i] & traindf$plate_side[i] <= traindf$StrZoneXR[i]){
           traindf$BorderInclusion[i] <- "Yes"
         }
       else{
           traindf$BorderInclusion[i] <- "No"
         }
     )
 for(i in 1:143788)(
      if(testdf$StrZoneYD[i] <= testdf$plate_height[i] & testdf$plate_height[i] <= testdf$StrZoneYU[i] & testdf$StrZoneXL[i] <= testdf$plate_side[i] & testdf$plate_side[i] <= testdf$StrZoneXR[i]){
          testdf$BorderInclusion[i] <- "Yes"
        }
      else{
          testdf$BorderInclusion[i] <- "No"
        }
    )
 
 traindf$BorderInclusion <- as.factor(traindf$BorderInclusion)
 testdf$BorderInclusion <- as.factor(testdf$BorderInclusion)
 
 #Feature Importance Check
 newdf<- subset(traindf, select = -c(balls,strikes,catcher_id,pitcher_id,batter_id,umpire_id, pitcher_side))
 set.seed(2020)
 newdf <- sample_n(newdf,50000)
 newdf <- na.omit(newdf)
 newdf$BorderInclusion <- as.factor(newdf$BorderInclusion)
 newdf$pitch_call <- factor(newdf$pitch_call)
 RF <- randomForest(pitch_call~.,data = newdf,ntree = 500)
 
 NAdf <- testdf[rowSums(is.na(testdf))!=0,]
 testdf <- na.omit(testdf)
 traindf$pitch_call <- gsub(pattern = "StrikeCalled", x = traindf$pitch_call, replacement = "1")
 traindf$pitch_call <- gsub(pattern = "BallCalled", x = traindf$pitch_call, replacement = "0")
 traindf$pitch_call <- factor(traindf$pitch_call)
 traindf <- traindf %>%
   filter(catcher_id != "1813cf1c")
 testdf$is_strike <- NA
 
 SmallTrDf <- traindf %>%
   select(-c(umpire_id,catcher_id,release_speed,pitcher_id,batter_id,batter_side,pitcher_side,balls,strikes,pitch_type))
 LargeTrDf <- traindf %>%
   select(-c(umpire_id,pitcher_id,batter_id,batter_side,pitcher_side,balls,strikes,pitch_type))
 SmallTeDf <- testdf %>%
   select(-c(umpire_id,catcher_id,release_speed,pitcher_id,batter_id,batter_side,pitcher_side,balls,strikes,pitch_type))
 LargeTeDf <- testdf %>%
   select(-c(umpire_id,pitcher_id,batter_id,batter_side,pitcher_side,balls,strikes,pitch_type))

#-------Logistic Regression
training_rows_logit <- SmallTrDf %>%
   select(pitch_call) %>%
   unlist() %>%
   createDataPartition(p = 0.7, list = F)
 
X_train_logit <- SmallTrDf %>%
      select(-pitch_call) %>%
      slice(training_rows_logit) %>%
      data.frame()
Y_train_logit <- SmallTrDf %>%
      select(pitch_call) %>%
      slice(training_rows_logit) %>%
      unlist()
X_test_logit <- SmallTrDf %>%
      select(-pitch_call) %>%
      slice(-training_rows_logit) %>%
      data.frame()
Y_test_logit <- SmallTrDf %>%
      select(pitch_call) %>%
      slice(-training_rows_logit) %>%
      unlist()
X_test <- SmallTeDf
Y_test <- SmallTeDf %>%
       select(is_strike) %>%
       unlist()

glm_model <- train(x = X_train_logit, y = Y_train_logit, method = "glm", family = "binomial")

test_pred <- predict(glm_model,X_test_logit)
 
actual_test_pred <- as.data.frame(predict(glm_model,X_test))
  
model_quality <- confusionMatrix(data = test_pred, reference = Y_test_logit)

#-------Random Forest
set.seed(2020)
RFtraindf <- LargeTrDf %>% select(-catcher_id)
RFtraindf <- RFtraindf %>% sample_n(50000)

RFmodel <- randomForest(pitch_call ~ ., data = RFtraindf, ntree = 1000, nodesize = 5)

#Tune RF - ntrees: 500 or 1000, although we see tiny decrease of OOB error at 1000
oob.error.rate <- data.frame(RFmodel$err.rate)
oob.error.rate$Trees <- c(1:1000)
oob.error.plot <- oob.error.rate %>%
  ggplot(aes(x = Trees, y = X1)) +
  geom_line()
oob.error.plot

#Tune RF -mtry: 3 is the lowest
set.seed(2020)
RFtraindf <- LargeTrDf %>% select(-catcher_id)
RFtraindf <- RFtraindf %>% sample_n(5000)                                   
oob.values <- vector(length=10)

for(i in 1:10){
  temp.model <- randomForest(pitch_call ~ ., data = RFtraindf, mtry = i, ntree = 1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}

#Predict 
actual_test_pred$RF <- predict(RFmodel,LargeTeDf)

# 3 - Boosted Logistic Regression -----------------------------------------

library(mltools)
library(data.table)
library(xgboost)

labels <- traindf$pitch_call
ts_label <- testdf$is_strike

#one-hot encoding
new_tr <- LargeTrDf %>% select(-pitch_call)
new_tr <- as.matrix(one_hot(as.data.table(new_tr)))
new_ts <- LargeTeDf %>% select(-is_strike)
new_ts <- as.matrix(one_hot(as.data.table(new_ts)))

#convert factor to numeric
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

xgb1 <- xgb.train(params = params, data = dtrain, nrounds = 100, watchlist = list(val=dtest,train=dtrain), print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")

#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#Final Output (1,0) to LargeTeDf
LargeTeDf$is_strike <- xgbpred[,1]

#Handling NA Data - predicting with missing categories
newdf<- subset(traindf, select = -c(balls,strikes,catcher_id,pitcher_id,batter_id,umpire_id, pitcher_side,BorderInclusion,StrZoneYD,StrZoneYU,StrZoneXL,StrZoneXR))

set.seed(2020)
newdf <- sample_n(newdf,50000)
newdf <- na.omit(newdf)
newdf$pitch_call <- factor(newdf$pitch_call)
RF <- randomForest(pitch_call~.,data = newdf,ntree = 500)

#NAdata <- Data with Erroneous IDs, or any data somehow got skipped <Code Omitted>
#NAdata2 <- Data with NAs in the predictors<Code Omitted>

#Predict the NA data pitch call
dummy_pred <- as.data.frame(predict(RF,NAdata))
dummy_pred2 <- as.data.frame(predict(RFmodel,NAdata2))

#Drop Columns, Match columns for the final output and rbind... <Code Omitted>

#Replace 1s to Strike and 0s to Balls
LargeTeDf$is_strike <- gsub(pattern = "0",x = LargeTeDf$is_strike, replacement = "Ball")
LargeTeDf$is_strike <- gsub(pattern = "1",x = LargeTeDf$is_strike, replacement = "Strike")

#write CSV
write.csv2(LargeTeDf,file = "StrikeCallPrediction.csv")
write.xlsx(LargeTeDf,file = "StrikeCallPrediction.xlsx")
