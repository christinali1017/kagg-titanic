
library(ggplot2)
library(Rtsne)

set.seed(1)

train <- read.csv("/Users/wish/MyApp/disaster-kagg-titanic/train.csv")

train$Outcome     <- as.factor(ifelse(train$Survived==1, "Survived", "Died"))
train$IsMale      <- ifelse(train$Sex=="male", 1, 0)
train$Class       <- as.factor(train$Pclass)
train$AgeOrMedian <- train$Age
train$AgeOrMedian[is.na(train$Age)] <- median(train$Age, na.rm=TRUE)

tsne <- Rtsne(as.matrix(train[,c("Pclass", "AgeOrMedian", "IsMale", "SibSp", "Parch", "Fare")]),
              check_duplicates = FALSE, pca = TRUE, perplexity=30, theta=0.5, dims=2)

embedding <- as.data.frame(tsne$Y)
train$V1  <- embedding$V1
train$V2  <- embedding$V2

p <- ggplot(train, aes(x=V1, y=V2, color=Outcome, shape=Sex, alpha=Class)) +
     geom_point(size=3) +
     scale_alpha_discrete(range=c(1.0, 0.5, 0.25)) +
     theme_light(base_size=20) +
     theme(strip.background = element_blank(),
           strip.text.x     = element_blank(),
           axis.text.x      = element_blank(),
           axis.text.y      = element_blank(),
           axis.ticks       = element_blank(),
           axis.line        = element_blank(),
           panel.border     = element_blank()) +
     xlab("") + ylab("")

ggsave("/Users/wish/MyApp/disaster-kagg-titanic/tsne4.png", p, width=8, height=6, units="in")