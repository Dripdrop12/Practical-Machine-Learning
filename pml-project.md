Human Activities Recognition Using Random Forests
========================================================
author: Jonathan Hill
date: Tue May 19 16:08:59 2015
transition: rotate
transition-speed: slow

Original Authors 
========================================================
This dataset is licensed under the Creative Commons license (CC BY-SA).

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3abHmT200

Goal
========================================================

* Using raw data from devices such as Jawbone Up, Nike FuelBand, and Fitbit 
+ Predict when someone is performing bicep curls with poor form and classify what they are doing wrong (classes B, C, D, and E)

Introduction
========================================================
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:

Class | Description
------------- | -------------
A | exactly according to the specification
B | throwing the elbows to the front
C | lifting the dumbbell only halfway
D | lowering the dumbbell only halfway
E | throwing the hips to the front

Sensor Locations
========================================================
<img src="http://groupware.les.inf.puc-rio.br/static/WLE/on-body-sensing-schema.png" width="35%" height="66%">


Downloading the Data
========================================================


```r
# The training data #
url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url = url, destfile = "pml-training.csv")

# The testing data #
url2 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url = url2, destfile = "pml-testing.csv")
```

```r
finaltesting <- read.csv("pml-testing.csv")
training <- read.csv("pml-training.csv")
```

Required R Packages
========================================================

```r
library(caret)
library(randomForest)
```

Partitions for Cross Validation
========================================================

In order to train the model, I created a partition with 60% of the data called "tr." 

With the remaining 40%, I created a partition with 20% to test the model ("te") and another with the remaining 20% to validate the model ("validation").

```r
inTrain <- createDataPartition(training$classe, p = 0.6, list = FALSE)
tr <- training[inTrain, ]
te <- training[-inTrain, ]
```

Pre-processing
========================================================

```r
# Remove near zero variance predictors #
nzv <- nearZeroVar(tr, saveMetrics = TRUE)
nearzerofilter <- rownames(nzv[nzv$nzv == FALSE, ])
tr <- tr[, nearzerofilter]

# ... correlated predictors using a .85 cutoff #
tr <- tr[, 6:ncol(tr)]
trainCor <- cor(tr[,1:(ncol(tr)-1)], use = "complete.obs")
corFilter <- findCorrelation(trainCor, cutoff = .85)
tr <- tr[, -corFilter]

# ... and variables with greater than 85% NA
tr <- tr[, colMeans(is.na(tr)) <= .85]
```

Model Selection
========================================================
Because the decision points of the model do not need to be interpreted and the continuous variables in the data are not very interpretable, random forests is a good option for this problem.

Centering and scaling continuous variables prevents large values from biasing the model.

There was very little difference in accuracy among models using 3, 5 and 10-fold cross-validated resampling, but processing time was much longer for the model using 10-fold cross-validated resampling.  Therefore, the final model uses 5-fold cross-validated resampling.

Model
========================================================

```r
# Random forests #
modelFit <- train(classe ~.,
                  data = tr,
                  method = "rf",
                  trControl = trainControl(method = "cv", number = 5),
                  preProcess = c("center", "scale")
                  )
```

Model (cont.)
========================================================
```
Random Forest

11776 samples
46 predictors
5 classes: A, B, C, D, E

Pre-processing: center, scale
Resampling: Cross-Validated ( 5 fold)
```

```
  mtry  Accuracy     Kappa  AccuracySD
1    2 0.9893012 0.9864630 0.003763127
2   23 0.9946505 0.9932330 0.001425489
3   45 0.9847997 0.9807731 0.001320220
```

Test and Validation Partitions
========================================================

```r
# 20% each #
inValidation <- createDataPartition(te$classe, p = .5, list = FALSE)
validation <- te[inValidation, names(te) %in% names(tr) ]
te <- te[-inValidation, names(te) %in% names(tr)]
```
These two partitions will help estimate the out-of-sample error of the model because they represent 40% of the data. They were not used during the model design and data cleaning steps, and they will help show how well the model could perform in the real world.

Test Confusion Matrix
========================================================

```r
pred <- predict(modelFit, newdata = te)
```

```
          Reference
Prediction    A    B    C    D    E
         A 1115    1    0    0    0
         B    0  758    1    0    0
         C    0    1  683    0    0
         D    0    0   10  632    1
         E    0    0    0    1  720
```
The mean out-of-sample accuracy in the test sample is 0.9954 with a 95% confidence interval of 0.9928 to 0.9973.

Validation Confusion Matrix
========================================================

```r
pred2 <- predict(modelFit, newdata = validation)
```


```
          Reference
Prediction    A    B    C    D    E
         A 1116    0    0    0    0
         B    0  755    3    1    0
         C    0    1  683    0    0
         D    0    0    5  638    0
         E    0    0    0    0  721
```
The mean out-of-sample accuracy for the validation sample is 0.9967 with a 95% confidence interval of 0.9943 to 0.9982.

Final Predictions
========================================================
Because these out-of-sample accuracy estimates are very good, the final test is to predict the class of 20 unclassified activities.



The final predictions had the following classes: B, A, B, A, A, E, D, B, A, A, B, C, B, A, E, E, A, B, B, B. And these were 100% accurate.  

However, taking the upper and lower estimates for the model's out-of-sample accuracy, its accuracy will probably fluxuate between 99.28% and 99.82% at a 95% confidence level.


