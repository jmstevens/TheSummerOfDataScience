titanic <- read.csv(file="/Users/jrzemien/Titanic/Data/train_bob_2.csv", na.strings= c(""))
Survived <- titanic$Survived
Pclass <- titanic$Pclass
Sex <- titanic$Sex
Age <- titanic$Age
Embarked <- titanic$Embarked
SibSp <- titanic$SibSp
Parch <- titanic$Parch
Cabin <- titanic$Cabin


dumbmodel <- glm(Survived~Pclass+Sex+Age+SibSp+Parch, family= binomial(link="logit"))
summary(dumbmodel)

dumbanova <- anova(dumbmodel, test="Chisq")
summary(dumbanova)
