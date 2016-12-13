rm(list = ls()) #clear work space

dataImport<- read.table('data/Data_E&W_group.txt')
dataImport$V15 <- factor(dataImport$V15)

table(dataImport$V15)

library(randomForest)

itr = 50 # number of iterations
nVar = 10 # number of input variables

AccuracyTrain = numeric(itr)
AccuracyValidate = numeric(itr)
ErrorOOB = numeric(itr)
ImportanceA = matrix(, nrow = itr, ncol = nVar)
ImportanceG = matrix(, nrow = itr, ncol = nVar)
Importance1 = matrix(, nrow = itr, ncol = nVar)
Importance2 = matrix(, nrow = itr, ncol = nVar)
Importance3 = matrix(, nrow = itr, ncol = nVar)
Importance4 = matrix(, nrow = itr, ncol = nVar)

for (i in c(1:itr))
{
  data <- dataImport[sample(nrow(dataImport)),]
  
  nData = dim(data)[1]
  nTrain = dim(data)[1]/2
  nValidate = nData-nTrain
  dataTrain = data[1:nTrain,]
  dataValidate = data[nValidate:nData,]
  
  model <- randomForest(V15 ~ V3+V5+V7+V8+V9+V10+V11+V12+V13+V14, data = dataTrain, ntree = 10000, mtry = 3,importance=TRUE)
  
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


#for (nFeatures in c(1:10))
#{
#  print("mtry = ")
#  print(nFeatures)
#  
#}
