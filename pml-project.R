setwd("C:/Users/PDAUSER/Desktop/Practical Machine Learning")

library(caret)

# Answers need to be a character vector such as #
# rep("A", 20) if each answer was A #

# Function that will turn the answers into files for submission #

pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

# Get the training data #
url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url = url, destfile = "pml-training.csv")
training <- read.csv("pml-training.csv")

# Get the testing data #
url2 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url = url2, destfile = "pml-testing.csv")
finaltesting <- read.csv("pml-testing.csv")

# Create data partitions #
inTrain <- createDataPartition(training$classe, p = 0.6, list = FALSE)

# Training set with 60% of the data #
tr <- training[inTrain, ]
te <- training[-inTrain, ]

# Remove near zero variance predictors #
nzv <- nearZeroVar(tr, saveMetrics = TRUE)
nearzerofilter <- rownames(nzv[nzv$nzv == FALSE, ])
tr <- tr[, nearzerofilter]

# ... correlated predictors using a .85 cutoff #
tr <- tr[, 6:ncol(tr)]
trainCor <- cor(tr[,1:(ncol(tr)-1)], use = "complete.obs")
corFilter <- findCorrelation(trainCor, cutoff = .85)
tr <- tr[, -corFilter]

# ... and variables with greater than 85% NA #
tr <- tr[, colMeans(is.na(tr)) <= .85]


# Random forests model #
modelFit <- train(classe ~.,
                  data = tr,
                  method = "rf",
                  trControl = trainControl(method = "cv", number = 5),
                  preProcess = c("center", "scale")
                  )

# Validation and test sets with 20% each #
inValidation <- createDataPartition(te$classe, p = .5, list = FALSE)
validation <- te[inValidation, names(te) %in% names(tr) ]
te <- te[-inValidation, names(te) %in% names(tr)]

# Create predictions and confusion matrix for test sample #
pred <- predict(modelFit, newdata = te)
confusionMatrix(te$classe, pred)

# Create predictions and confusion matrix for validation #
pred2 <- predict(modelFit, newdata = validation)
confusionMatrix(validation$classe, pred2)[[5]]

# Create predictions for final submission #
finalPred <- predict (modelFit, newdata = finaltesting)
finalPred <- as.character(finalPred)
pml_write_files(finalPred)

# Cache option
```{r setup, include=FALSE}
opts_chunk$set(cache=TRUE)
```
