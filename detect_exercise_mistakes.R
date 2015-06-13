setwd("~/GitHub/detect-exercise-mistakes")

library(caret)
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
#                 "pml-training.csv")
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
#               "pml-testing.csv")
pml.training <- read.csv("pml-training.csv")
pml.testing <- read.csv("pml-testing.csv")

# Find columns with all NA
pml.training <- pml.training[, colSums(!is.na(pml.testing)) != 0]
pml.testing <- pml.testing[, colSums(!is.na(pml.testing)) != 0]

# Remove other variables that shouldn't impact output
drops <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
           "cvtd_timestamp", "new_window", "problem_id")
pml.training <- pml.training[, !(names(pml.training) %in% drops)]
pml.testing <- pml.testing[, !(names(pml.testing) %in% drops)]

# Check that all cases are complete
nrow(pml.training) == sum(complete.cases(pml.training))
nrow(pml.testing) == sum(complete.cases(pml.testing))


# Exploratory Data Analysis and Pre-Processing

library(rindex)
library(ellipse)

plotme <- names(pml.training)[strncmp(names(pml.training), "total", 5) == 0]
plotme <- c(plotme, "classe")

inXplore <- createDataPartition(y = pml.training$classe,
                                p = 0.01, list = FALSE)
xplore <- pml.training[inXplore, ];
xplore <- xplore[, (names(xplore) %in% plotme)]

featurePlot(x = xplore[, 1 : 4],
            y = xplore$classe,
            plot = "ellipse",
            ## Add a key at the top
            auto.key = list(columns = 5))

# Principal Components Analysis

predictors <- subset(pml.training, select = -c(classe))
ncol(predictors)
numComp <- preProcess(predictors, method = "pca", thresh = 0.9)$numComp
numComp

any(nearZeroVar(predictors, saveMetrics = TRUE)$zeroVar)
findLinearCombos(predictors)$linearCombos

predCor <- cor(predictors)
summaryPredCor <- summary(predCor[upper.tri(predCor)])
summaryPredCor

highlyCorPred <- findCorrelation(predCor, cutoff = .95)


# Create a building data set and validation set
# inBuild <- createDataPartition(y = pml.training$classe,
#                                p = 0.7, list = FALSE)
# validation <- pml.training[-inBuild, ];
# buildData <- pml.training[inBuild, ]
# inTrain <- createDataPartition(y = buildData$classe,
#                                p = 0.1, list = FALSE)
# training <- buildData[inTrain, ]
# testing <- buildData[-inTrain, ]

set.seed(1234)
inTrain <- createDataPartition(y = pml.training$classe,
                               p = 0.75, list = FALSE)
training <- pml.training[inTrain, ]
testing <- pml.training[-inTrain, ]

# Model Fitting
library(randomForest)
# startTime <- proc.time()
# modRf <- train(classe ~ ., data = training, method = "rf")
# endTime <- proc.time()
# endTime - startTime
# save(modRf, file = "modRf_0p75.RData")
load(file = "modRf_0p75.RData")
modRf
plot(modRf, main = "Figure 2: Training Parameter Tuning")
plot(modRf$finalModel, main = "Figure 3: Random Forest Error Rates")

predRf <- predict(modRf, testing)
postResample(predRf, testing$classe)
confusionMatrix(predRf, testing$classe)

# method = "rf"
# p = 0.05, Accuracy = 0.93, user (sec) = 301.89
# p = 0.1,  Accuracy = 0.97
# p = 0.2,  Accuracy = 0.988
# p = 0.5,  Accuracy = 0.9954
# p = 0.7,  Accuracy = 0.998 (?)
# p = 0.75, Accuracy = 0.9983687, user (sec) = 7995.61


# method = "gbm"
# p = 0.1, Accuracy = 0.95, user (sec) = 388.27

# Write files to be submitted

setwd("~/GitHub/detect-exercise-mistakes/submission")
submission <- predict(modRf, pml.testing)

pml_write_files = function(x) {
  n = length(x)
  for(i in 1 : n) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote=FALSE, row.names = FALSE, col.names = FALSE)
  }
}

pml_write_files(as.character(submission))
