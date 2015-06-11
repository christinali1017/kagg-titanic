getwd()
setwd("/Users/wish/MyApp/titanic/")
ref_data <- read.csv("./data/train.csv")
test_data <- read.csv("./data/test.csv")

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

test_data$Survived <- 0
test_data$Survived <- NA
combi_data <- rbind(ref_data,test_data)
combi_data$Name <- as.character(combi_data$Name)

combi_data$Title <- sapply(combi_data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi_data$Surname <- sapply(combi_data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi_data$Title <- sub(' ', '', combi_data$Title)
combi_data$Title[combi_data$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi_data$Title[combi_data$Title %in% c('Capt', 'Col', 'Don', 'Major', 'Sir')] <- 'Sir'
combi_data$Title[combi_data$Title %in% c('Dona', 'Lady', 'Jonkheer')] <- 'Lady'

combi_data$Title <- factor(combi_data$Title)
combi_data$FamilySize <- combi_data$SibSp + combi_data$Parch + 1
combi_data$FamilyID <- paste(as.character(combi_data$FamilySize), combi_data$Surname, sep="")
combi_data$FamilyID[combi_data$FamilySize <= 2] <- 'Small'
famIDs <- data.frame(table(combi_data$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi_data$FamilyID[combi_data$FamilyID %in% famIDs$Var1] <- 'Small'
combi_data$FamilyID <- factor(combi_data$FamilyID)

ref_data <- combi_data[1:891,]
test_data <- combi_data[892:1309,]

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=ref_data, method="class")
fancyRpartPlot(fit)
Prediction <- predict(fit, test_data, type = "class")
submission <- data.frame(PassengerId = test_data$PassengerId, Survived = Prediction)
write.csv(submission, file = "submit.csv",row.names = FALSE)