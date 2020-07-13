#datasource: https://www.kaggle.com/uciml/sms-spam-collection-dataset

# Install all required packages.
install.packages(c("ggplot2", "e1071", "caret", "quanteda", 
                   "irlba", "randomForest"))

#ggplot2-visualization; e1071-for caret; caret-this depends upon e1071; quanteda-text analytics; 
#irlba-singular value decomposition & feature extraction; randomForest-simple & powerful


# Load up the .CSV data and explore in RStudio.
spam.raw <- read.csv("spam.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-16")
View(spam.raw)
#stringsAsFactors: when you read as string; don't make it a factor; 
#R converts strings to factors  by default but we need raw text for text analytics


# Clean up the data frame and view our handiwork.
spam.raw <- spam.raw[, 1:2]
#need all rows and 1st 2 columns
names(spam.raw) <- c("Label", "Text")
#change the names of the columns
View(spam.raw)



# Check data to see if there are missing values.
complete.cases(spam.raw)
length(which(!complete.cases(spam.raw)))
#complete.cases: returns a logical vector indicating which cases dont have missing values


# Convert our class label into a factor.
spam.raw$Label <- as.factor(spam.raw$Label)
#cz ham & spam are 2 categories



# The first step, as always, is to explore the data.
# First, let's take a look at distibution of the class labels (i.e., ham vs. spam).
table(spam.raw$Label)
#table: relative raw count of hams & spams
prop.table(table(spam.raw$Label))
#prop.table: converts raw count into percentages

#looing at the distribution of labels is extremely important 
#Our classes are imbalanced

# Next up, let's get a feel for the distribution of text lengths of the SMS 
# messages by adding a new feature for the length of each message.
spam.raw$TextLength <- nchar(spam.raw$Text)
#nchar: count the number of characters
#calculate the length of each message and store it in TextLength
summary(spam.raw$TextLength)
#max has 910 characters and min 2 characters


# Visualize distribution with ggplot2, adding segmentation for ham/spam.
library(ggplot2)

ggplot(spam.raw, aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")




# At a minimum we need to split our data into a training set and a
# test set. In a true project we would want to use a three-way split 
# of training, validation, and test.
#
# As we know that our data has non-trivial class imbalance, we'll 
# use the mighty caret package to create a randomg train/test split 
# that ensures the correct ham/spam class label proportions (i.e., 
# we'll use caret for a random stratified split).
library(caret)
help(package = "caret")


# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.; to get same set of random numbers in the department
#caret comes with many splitting functions one of which is createDataPartition 
set.seed(32984)
indexes <- createDataPartition(spam.raw$Label, times = 1,
                               p = 0.7, list = FALSE)
#70% to build the model.ie. training and 30% to test how good the model is
#spliting ensures that the proportions are  maintained across all the splits
#0.7*5572=3900
#times: no of splits/iterations

indexes[1:20]
#random row numbers as it contains only 70% of data


#negative sign to get all the remaining indexes from train

# Verify proportions.
prop.table(table(train$Label))
prop.table(table(test$Label))

# Text analytics requires a lot of data exploration, data pre-processing
# and data wrangling. Let's explore some examples.

# HTML-escaped ampersand character.
train$Text[20]


# HTML-escaped '<' and '>' characters. Also note that Mallika Sherawat
# is an actual person, but we will ignore the implications of this for
# this introductory tutorial.
test$Text[16]


# A URL.
train$Text[381]

# There are many packages in the R ecosystem for performing text
# analytics. One of the newer packages in quanteda. The quanteda
# package has many useful functions for quickly and easily working
# with text data.
library(quanteda)
help(package = "quanteda")
#Quantitative Analysis of Texual Data

# Tokenize SMS text messages.
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, split_hyphens = TRUE)
#tokens: tokenize a set of text
#what=word; you can also use characters
#remove hyphens and convert it to space

# Take a look at a specific SMS message and see how it transforms.
train.tokens[[381]]


# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[381]]


# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to your problem/domain.
# stopwords = the, a, for, your, is, have
stopwords()
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
#tokens_select: selects or removes tokens from token object
train.tokens[[381]]


# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[381]]
#stemming: similar words into single frequency; run, ran, running
#credits-->credit, renewal-->renew

# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
#dfm: create sparse document feature matrix


# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
#most of them are 0's
dim(train.tokens.matrix)
#3901 rows as textmsgs don't change but number of columns increases to 5847


# Investigate the effects of stemming.
colnames(train.tokens.matrix)[1:50]

# Per best practices, we will leverage cross validation (CV) as the basis of our modeling process. Using CV we can create 
# estimates of how well our model will do in Production on new, unseen data. CV is powerful, but the downside is that it
# requires more processing and therefore more time.
#One round of cross-validation involves partitioning a sample of data into complementary subsets, 
#performing the analysis on one subset (called the training set), and validating the analysis on 
#the other subset (called the validation set or testing set). 
#To reduce variability, in most methods multiple rounds of cross-validation are performed using different partitions, 
#and the validation results are combined (e.g. averaged) over the rounds to give an estimate of the 
#model's predictive performance.

# Setup a the feature data frame with labels.
train.tokens.df <- cbind(Label = train$Label, convert(train.tokens.dfm, to = "data.frame"))
#data.frame: this will convert dfm into simple R df

#to find column number from column name
which( colnames(train.tokens.df)=="8am" )

# Often, tokenization requires some additional pre-processing)
names(train.tokens.df)[c(139, 141, 211, 213)]
#[1] "try:wal" "4txt"    "2nd"     "8am"  
#these are not valid column names; ML algorithms will throw an error stating "don't understand the term"
#so we transfer them into legitimate names



# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))
#make syntactically valid names


# Use caret to create stratified folds for 10-fold cross validation repeated 
# 3 times (i.e., create 30 random stratified samples)

set.seed(48743)
cv.folds <- createMultiFolds(train$Label, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)
#index = cv.folds: we need to specify this inorder to get stratified folds

# Our data frame is non-trivial in size. As such, CV runs will take 
# quite a long time to run. To cut down on total execution time, use
# the doSNOW package to allow for multi-core training in parallel.
#
# WARNING - The following code is configured to run on a workstation-
#           or server-class machine (i.e., 12 logical cores). Alter
#           code to suit your HW environment.
#
#install.packages("doSNOW")
library(doSNOW)


# Time the code execution
#just for performance testing; getting the current system time
start.time <- Sys.time()


# Create a cluster to work on 3 logical cores.
#To check the number of available logical cores on your system;type this in terminal: sysctl -n hw.ncpu  
#I have 4 cores so using 3; higher number = quick execution
cl <- makeCluster(3, type = "SOCK")
# type = "SOCK": socket cluster
# makeCluster: this creates multiple instances of Rstudio in background; 
# then allows caret to borrow those instances to do the processing
# building the cluster is not enough; you need to register it
registerDoSNOW(cl)
#caret will recognize this command


# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm as our first model. We will graduate to using more 
# powerful algorithms later when we perform feature extraction to shrink the size of our data.
# train: main function for actual model building
drops <- c("document")
train.tokens.df <- train.tokens.df[ , !(names(train.tokens.df) %in% drops)]

#train.tokens.df <- train.tokens.df[, !duplicated(colnames(train.tokens.df))]
rpart.cv.1 <- train(Label ~ ., data = train.tokens.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)
# method = "rpart": this tells train what kind of ML model needs to be built
# rpart is single decision tree. We can also change it to "rf" ie random forest
#Label ~ . : predict label; predicted by all other factors
#tuneLength = 7: try 7 different ways to configure rpart and find out which 1 of the  7 configurations works the best and use that 1
#this is hyperparameter tuning


# Processing is done, stop cluster.
stopCluster(cl)


# Total time of execution on workstation was approximately 4 minutes. 
total.time <- Sys.time() - start.time
total.time


# Check out our results.
rpart.cv.1

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cp = 0.01465902.
# Accuracy = 95% approx

#TFIDF enhances DTF to make it more powerful
# The use of Term Frequency-Inverse Document Frequency (TF-IDF) is a powerful technique for enhancing the information/signal contained
# within our document-frequency matrix. Specifically, the mathematics behind TF-IDF accomplish the following goals:
#    1 - The TF calculation accounts for the fact that longer documents will have higher individual term counts. 
#        Applying TF normalizes all documents in the corpus to be length independent.
#    2 - The IDF calculation accounts for the frequency of term appearance in all documents in the corpus. The intuition 
#        being that a term that appears in every document has no predictive power.
#    3 - The multiplication of TF by IDF for each cell in the matrix
#        allows for weighting of #1 and #2 for each cell in the matrix.


# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}
#if document has 10 terms then freq[1st cell]/freq[sum of all cells in 1st row]

#tf-->rows; document centric
#idf--> columns; corpus/column centric

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

#corpus.size <- length(col): for each column; calculate the num of docs; count is  same since its a matrix
#doc.count <- length(which(col > 0)): give the number of rows where column is not 0

# Our function for calculating TF-IDF.
tf.idf <- function(tf, idf) {
  tf * idf
}

View(train.tokens.matrix)

# First step, normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
#apply: R style for loop
#R can apply a function over & over against the rows or columns
#apply term.frequency function against train.tokens.matrix; 1=rows
#this will transpose the matrix; swap columns for rows
#normalize: tend to have equal length

dim(train.tokens.matrix)
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:100])
#values change cz they've been normalized

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)
#2=apply against columns


# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
#2 since the matrix was transposed
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])
#1st all the values were same cz they were normalized
#now they're rationaized; combined tf-idf value
#rationaize: terms which appear more frequently are going to be less uselful-low value


# Transpose the matrix back to document frequency
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])
#t function: transpose

#in pre processing pipeline we strip out all the punctuations, digits, etc sometimes giving us an empty string
#when we do tf-idf on empty string; we might get error from R
#so we need to Check for incopmlete cases.

# Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]



# Fix incomplete cases 
#if you try to run ML model with incomplete cases; R throws an error
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
#for those rows, replace columns with all 0's cz they might be legitimate messages and we dont want to lose them
dim(train.tokens.tfidf)
#check for incomplete cases 
sum(which(!complete.cases(train.tokens.tfidf)))


# Make a clean data frame using the same process as before.
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
View(train.tokens.tfidf.df)

# Time the code execution
start.time <- Sys.time()

# Create a cluster to work on 3 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm as our first model. We will graduate to using more 
# powerful algorithms later when we perform feature extraction to shrink
# the size of our data.
rpart.cv.2 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.2

#untill now we have considered single terms, these are unigrams or 1 gram
#but there are also bi,tri and n-grams; various combinations
#N-grams extend bag-of-words model
# adding n-grams increases the size of matrix[more than double] which might lead to curse of dimensionality in terms of space

# N-grams allow us to augment our document-term frequency matrices with word ordering. 
# This often leads to increased performance (e.g., accuracy) for machine learning models trained with more than just unigrams (i.e.,
# single terms). 
#Let's add bigrams to our training data and the TF-IDF transform the expanded featre matrix to see if accuracy improves.

# Add bigrams to our feature matrix.
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)
# tokens_ngrams: create ngrams from tokens
# n = 1:2: get uni and bigrams
# n=2: only bigrams
train.tokens[[381]]


# Transform to dfm and then a matrix.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm


# Normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)


# Calculate the IDF vector that we will use for training and test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)


# Calculate TF-IDF for our training corpus 
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, 
                             idf = train.tokens.idf)


# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)


# Fix incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))


# Make a clean data frame.
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

#by adding ngrams columns have increased significantly =29,243 features 
#(99.9% sparse): majority are 0

# Clean up unused objects in memory.
gc()
#gc: forces R to do garbage collection; freeup memory that you can

#gc hardly creates any difference. If you dont have a robust big machine with atleast 10 logical cores processing time is very large

# NOTE - The following code requires the use of command-line R to execute
#        due to the large number of features (i.e., columns) in the matrix.
#        Please consult the following link for more details if you wish
#        to run the code yourself:
#
#        https://stackoverflow.com/questions/28728774/how-to-set-max-ppsize-in-r
#
#        Also note that running the following code required approximately
#        38GB of RAM and more than 4.5 hours to execute on a 10-core 
#        workstation!
#
#but you can get AWS instance of large vm and run it

# Time the code execution
# start.time <- Sys.time()

# Leverage single decision trees to evaluate if adding bigrams improves the 
# the effectiveness of the model.
# rpart.cv.3 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
#                     trControl = cv.cntrl, tuneLength = 7)

# Total time of execution on workstation was
# total.time <- Sys.time() - start.time
# total.time

# Check out our results.
# rpart.cv.3

#
# The results of the above processing show a slight decline in rpart 
# effectiveness with a 10-fold CV repeated 3 times accuracy of 0.9457.
# As we will discuss later, while the addition of bigrams appears to 
# negatively impact a single decision tree, it helps with the mighty
# random forest!


#vector space model helps us to address above problems 
#consider columns as vectors. calculate dot product of document vectors for all docs at once..more similar docs have more product value
#dot prod=transpose[x]*x---in terms of matrix

#we can colapse terms with similar meaning into 1 high level term

#Latent semantic analysis: extract core concept from correlated term
#LSA leverages a single value decomposition: decomposing matrix ---credit,loan,debt

#LSA often removes the curse of dimensionality problem in text analytics but SVD is computationally intensive
#reduced matrices are approximations i.e some info loss
#also we need to project new data into semantic space


# We'll leverage the irlba package for our singular value 
# decomposition (SVD). The irlba package allows us to specify
# the number of the most important singular vectors we wish to
# calculate and retain for features.
library(irlba)
#irlba: truncated SVD. Captures only imp features

# Time the code execution
start.time <- Sys.time()

# Perform SVD. Specifically, reduce dimensionality down to 300 columns
# for our latent semantic analysis (LSA).
train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)
#t:transpose
#nv:number of right singular vectors to estimate;300 columns
#maxit: how many iterations of algo to run at max 
#if algo can find all 300 vectors in less than 600 iterations it will stop

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time


# Take a look at the new feature data up close.
View(train.irlba$v)
#all the v columns are single most rich feature representation of data


# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
#
#
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
train.irlba$v[1, 1:10]



#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
train.svd <- data.frame(Label = train$Label, train.irlba$v)


# Create a cluster to work on 10 logical cores.
cl <- makeCluster(10, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.4 <- train(Label ~ ., data = train.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.4




#
# NOTE - The following code takes a long time to run. Here's the math.
#        We are performing 10-fold CV repeated 3 times. That means we
#        need to build 30 models. We are also asking caret to try 7 
#        different values of the mtry parameter. Next up by default
#        a mighty random forest leverages 500 trees. Lastly, caret will
#        build 1 final model at the end of the process with the best 
#        mtry value over all the training data. Here's the number of 
#        tree we're building:
#
#             (10 * 3 * 7 * 500) + 500 = 105,500 trees!
#
# On a workstation using 10 cores the following code took 28 minutes 
# to execute.
#


# Create a cluster to work on 10 logical cores.
# cl <- makeCluster(10, type = "SOCK")
# registerDoSNOW(cl)

# Time the code execution
# start.time <- Sys.time()

# We have reduced the dimensionality of our data using SVD. Also, the 
# application of SVD allows us to use LSA to simultaneously increase the
# information density of each feature. To prove this out, leverage a 
# mighty Random Forest with the default of 500 trees. We'll also ask
# caret to try 7 different values of mtry to find the mtry value that 
# gives the best result!
# rf.cv.1 <- train(Label ~ ., data = train.svd, method = "rf", 
#                 trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
# stopCluster(cl)

# Total time of execution on workstation was 
# total.time <- Sys.time() - start.time
# total.time


# Load processing results from disk!
load("rf.cv.1.RData")

# Check out our results.
rf.cv.1

# Let's drill-down on the results.
confusionMatrix(train.svd$Label, rf.cv.1$finalModel$predicted)
