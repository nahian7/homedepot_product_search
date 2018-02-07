library(gbm)
library(Metrics)
library(readr)
library(SnowballC)
library(xgboost)
cat("Reading data\n")
train <- read_csv('../input/train.csv')
test <- read_csv('../input/test.csv')
desc <- read_csv('../input/product_descriptions.csv')

train <- merge(train,desc, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)
test <- merge(test,desc, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)



t <- Sys.time()
word_match <- function(words,title,desc){
  n_title <- 0
  n_desc <- 0
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  for(i in 1:length(words)){
    pattern <- paste("(^| )",words[i],"($| )",sep="")
    n_title <- n_title + grepl(pattern,title,perl=TRUE,ignore.case=TRUE)
    n_desc <- n_desc + grepl(pattern,desc,perl=TRUE,ignore.case=TRUE)
  }
  return(c(n_title,nwords,n_desc))
}
word_match2 <- function(words,title,desc){
  n_title <- 0
  n_desc <- 0
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  for(i in 1:length(words)){
    pattern <- words[i]
    n_title <- n_title + grepl(pattern,title,ignore.case=TRUE)
    n_desc <- n_desc + grepl(pattern,desc,ignore.case=TRUE)
  }
  return(c(n_title,nwords,n_desc))
}
word_stem <- function(words){
 
  i <- 1
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  pattern <- wordStem(words[i], language = "porter")
  for(i in 2:length(words)){
    pattern <- paste(pattern,wordStem(words[i], language = "porter"),sep=" ")
  }
  return(pattern)
}







train_words <- as.data.frame(t(mapply(word_match,train$search_term,train$product_title,train$product_description)))
train$nmatch_title <- train_words[,1]
train$nwords <- train_words[,2]
train$nmatch_desc <- train_words[,3]


cat("Get number of words and word matching title in test\n")
test_words <- as.data.frame(t(mapply(word_match,test$search_term,test$product_title,test$product_description)))
test$nmatch_title <- test_words[,1]
test$nwords <- test_words[,2]
test$nmatch_desc <- test_words[,3]

rm(train_words,test_words)

train$search_term2 <- sapply(train$search_term,word_stem)
train_words <- as.data.frame(t(mapply(word_match2,train$search_term2,train$product_title,train$product_description)))
train$nmatch_title2 <- train_words[,1]
train$nmatch_desc2 <- train_words[,3]
#train$search_termlength <- sapply(train$search_term,length)
str(train)
train$search_term2 <-NULL

cat("Get number of words and word matching title in tes with porter stem\n")
test$search_term2 <- sapply(test$search_term,word_stem)
train_words <- as.data.frame(t(mapply(word_match2,test$search_term2,test$product_title,test$product_description)))
test$nmatch_title2 <- train_words[,1]
test$nmatch_desc2 <- train_words[,3]
test$search_term2 <-NULL
rm(test_words)


h<-sample(nrow(train),5000)
dval<-xgb.DMatrix(data=data.matrix(train[h,7:12]),label=train[h,5])
dtrain<-xgb.DMatrix(data=data.matrix(train[-h,7:12]),label=train[-h,5])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.025, 
                max_depth           = 6, 
                subsample           = 0.7, 
                colsample_bytree    = 0.9, 
                eval_metric         = "rmse",
                min_child_weight    = 6
)



clf <- xgb.train(data = dtrain, 
                 params               = param, 
                 nrounds              = 500, #300
                 verbose              = 1,#2
                 watchlist            = watchlist,
                 early.stop.round     = 50, 
                 print.every.n        = 1
)
clf$bestScore

test_relevance <- predict(clf,data.matrix(test[,6:11]),ntreelimit =clf$bestInd)
summary(test_relevance)
test_relevance <- ifelse(test_relevance>3,3,test_relevance)
test_relevance <- ifelse(test_relevance<1,1,test_relevance)

submission <- data.frame(id=test$id,relevance=test_relevance)
write_csv(submission,"xgb_submission.csv")
