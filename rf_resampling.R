rm(list = ls()) #clear work space

dataImport<- read.table('data/Data_E&W_group.txt')
dataImport$V15 <- factor(dataImport$V15)

library(randomForest)
library(ROSE)

itr = 10 # number of iterations
nVar = 14 # number of input variables

AccuracyTrain = numeric(itr)
AccuracyValidate = numeric(itr)
ErrorOOB = numeric(itr)
ImportanceA = matrix(, nrow = itr, ncol = nVar)
ImportanceG = matrix(, nrow = itr, ncol = nVar)
Importance1 = matrix(, nrow = itr, ncol = nVar)
Importance2 = matrix(, nrow = itr, ncol = nVar)
Importance3 = matrix(, nrow = itr, ncol = nVar)
Importance4 = matrix(, nrow = itr, ncol = nVar)


for (nFeatures in c(1:10))
{
  print("mtry = ")
  print(nFeatures)
  
  for (i in c(1:itr))
  {
    
    data <- dataImport[sample(nrow(dataImport)),] # reshuffle
    
    nData = dim(data)[1]
    nTrain = dim(data)[1]/2
    nValidate = nData-nTrain
    dataTrain = data[1:nTrain,]
    dataValidate = data[nValidate:nData,]
    
    dataTrain1 <- dataTrain[ which(dataTrain$V15=='1'), ]
    dataTrain2 <- dataTrain[ which(dataTrain$V15=='2'), ]
    dataTrain3 <- dataTrain[ which(dataTrain$V15=='3'), ]
    dataTrain4 <- dataTrain[ which(dataTrain$V15=='4'), ]
    
    classes <- table(dataTrain$V15) 
    classSize <- max(classes)
    dataTrain1.resample <- dataTrain1[sample(classes["1"],(classSize-classes["1"])),]
    dataTrain2.resample <- dataTrain2[sample(classes["2"],(classSize-classes["2"])),]
    dataTrain3.resample <- dataTrain3[sample(classes["3"],(classSize-classes["3"]),replace=T),]
    dataTrain4.resample <- dataTrain4[sample(classes["4"],(classSize-classes["4"]),replace=T),]
    
    dataTrain.resample <- rbind(dataTrain,dataTrain1.resample,dataTrain2.resample,dataTrain3.resample,dataTrain4.resample)
    
    model <- randomForest(V15 ~ ., data = dataTrain.resample, ntree = 10000, mtry = nFeatures, importance=TRUE)
    #model <- randomForest(V15 ~ ., data = dataTrain, ntree = 10000, mtry = 3,importance=TRUE)
    
    predTrain <- predict(model, newdata = dataTrain)
    correctTrain <- predTrain == dataTrain$V15
    predValidate <- predict(model, newdata = dataValidate)
    correctValidate <- predValidate == dataValidate$V15
    
    accuracyTrain <- table(correctTrain)["TRUE"]/nTrain
    accuracyValidate <- table(correctValidate)["TRUE"]/nValidate
    
    AccuracyTrain[i] <- accuracyTrain
    AccuracyValidate[i] <- accuracyValidate
    ErrorOOB[i] <- mean(model$err.rate[,1])
    
    print(i)
    
    VI_F=importance(model)
    importance1 <- VI_F[,c("1")]
    importance2 <- VI_F[,c("2")]
    importance3 <- VI_F[,c("3")]
    importance4 <- VI_F[,c("4")]
    importanceA <- VI_F[,c("MeanDecreaseAccuracy")]
    importanceG <- VI_F[,c("MeanDecreaseGini")]
    
    ImportanceA[i,] <- importanceA
    ImportanceG[i,] <- importanceG
    Importance1[i,] <- importance1
    Importance2[i,] <- importance2
    Importance3[i,] <- importance3
    Importance4[i,] <- importance4
  }
  
  output <- numeric(7)
  
  output[1] <- mean(AccuracyTrain)
  output[2] <- sd(AccuracyTrain)
  output[3] <- mean(AccuracyValidate)
  output[4] <- sd(AccuracyValidate)
  output[6] <- mean(ErrorOOB)
  output[7] <- sd(ErrorOOB)
  Output <- as.matrix(t(output))
  
  write.table(Output,"Output.csv", append=TRUE, sep=",",col.names = FALSE)
  
  library(matrixStats)
  Importance <- matrix(, nrow = 12, ncol = nVar)
  Importance[1,] <- colMeans(ImportanceA)
  Importance[2,] <- colSds(ImportanceA)
  Importance[3,] <- colMeans(ImportanceG)
  Importance[4,] <- colSds(ImportanceG)
  Importance[5,] <- colMeans(Importance1)
  Importance[6,] <- colSds(Importance1)
  Importance[7,] <- colMeans(Importance2)
  Importance[8,] <- colSds(Importance2)
  Importance[9,] <- colMeans(Importance3)
  Importance[10,] <- colSds(Importance3)
  Importance[11,] <- colMeans(Importance4)
  Importance[12,] <- colSds(Importance4)
  
  write.table(Importance,"OutputImportance.csv", append=FALSE, sep=",",col.names = FALSE)
  
}
