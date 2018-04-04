# This script is used as a learning experiance and reference
# Most code is taken from the following kernal https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic

# Load and Check Data
# Load Packages
R_Libraries <- function(packages, install_packs){
	if(install_packs == T){
        paste('Installing Packages................')
		install.packages(packages,repos = 'https://cran.mtu.edu/')
        paste('Installation complete...............')
	}
    paste('Loading Packages...................')
	lapply(packages,require,character.only = TRUE)
    paste('Load complete.....................')
}
packages <- c('ggplot2','ggthemes','scales','dplyr','mice','randomForest')
R_Libraries(packages, F)

# Load the data
train <- read.csv('Data/train.csv', stringsAsFactors = F)
test <- read.csv('Data/test.csv', stringsAsFactors = F)

full <- bind_rows(train, test) # bind training and test data into single df

# Check the data
str(full)

## Feature Engineering
# Break down passenger name into more meaningful variables

# Grab title from the passenger name
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
print(table(full$Sex, full$Title), class = TRUE)


# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess',
                'Capt', 'Col', 'Don', 'Dr', 'Major',
                'Rev', 'Sir', 'Jonkheer')
# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle'] <- 'Miss'
full$Title[full$Title == 'Ms'] <- 'Miss'
full$Title[full$Title == 'Mme'] <- 'Mrs'
full$Title[full$Title %in% rare_title] <- 'Rare Title'

# Show title counts by sex again
print(table(full$Sex, full$Title), class = TRUE)

# Grab surname from passenger name
full$Surname <- sapply(full$Name,
                    function(x) strsplit(x, split = '[,.]')[[1]][1])

# Do families sink or swim together?
