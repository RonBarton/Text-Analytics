setwd("R")
setwd("Datasets")
library(dplyr)
library(quanteda)
library(ggplot2)
library(stopwords)
library(topicmodels)
library(tidytext)
library(caTools)

# Read data
gas <- read.csv('gastext.csv',stringsAsFactors = F)

#exploration
dim(gas)
str(gas)

#convert data types
gas[,3:15]<-lapply(gas[,3:15],factor)
gas$Cust_ID <- factor(gas$Cust_ID)

#prepare data for re-ataching later @ train/split
gas_315 <- select(gas, -c(Cust_ID, Comment, Target))
head(gas_315)


# Establish the corpus and initial DFM matrix
myCorpus1 <- corpus(gas$Comment)
summary(myCorpus1)


# consider bigram?
myTokens <- tokens(myCorpus1)
bigram <- tokens_ngrams(myTokens,n=1:2)
summary(bigram)

#head(bigram[[1]], 50)
#tail(bigram[[1]], 50)
#myDfm1 <- dfm(bigram)
#View(myDfm1)

myDfm1 <- dfm(myCorpus1)

# Simple frequency analyses
tstat_freq1 <- textstat_frequency(myDfm1)
head(tstat_freq1, 20)

# Visulize the most frequent terms
myDfm1 %>% 
  textstat_frequency(n = 20) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()

#################################################################
# Remove stop words and perform stemming
dim(myDfm1)
stopwrds <-c('shower', 'point', 'productx', 'get', 'use', 'can', 'servic', 'park', 'card', 'price', 'food',
             'drink', 'price')

myDfm1 <- dfm(myCorpus1,
             remove_punc = T,
             remove = c(stopwords("english")),
             stem = T)
myDfm1 <- dfm_remove(myDfm1, stopwrds)
dim(myDfm1)
topfeatures(myDfm1,30)

textplot_wordcloud(myDfm1,max_words=100)


# Control sparse terms: to further remove some very infrequency words
myDfm1<- dfm_trim(myDfm1,min_termfreq=4, min_docfreq=2)
dim(myDfm1)

# Simple frequency analyses
tstat_freq1 <- textstat_frequency(myDfm1)
head(tstat_freq1, 20)


# Explore terms most similar to "price"
term_sim1 <- textstat_simil(myDfm1,
                           selection="price",
                           margin="feature",
                           method="correlation")
as.list(term_sim1,n=5)


# Explore terms most similar to "service"
term_sim2 <- textstat_simil(myDfm1,
                            selection="servic",
                            margin="feature",
                            method="correlation")
as.list(term_sim2,n=5)

#########################################################
#Topic Modeling
# Explore the option with 4 topics

stopwrds3 <-c('servic', 'price', 'park', 'card', 'clean', 'cleaner', 'free', 'coffe')
myDfm1 <- dfm_remove(myDfm1, stopwrds3)
dim(myDfm1)

myDfm1 <- as.matrix(myDfm1)
myDfm1 <-myDfm1[which(rowSums(myDfm1)>0),]
myDfm1 <- as.dfm(myDfm1)

myLda1 <- LDA(myDfm1,k=4,control=list(seed=101))
myLda1

# Term-topic probabilities
myLda_td1 <- tidy(myLda1)
myLda_td1

# Visulize most common terms in each topic
top_terms1 <- myLda_td1 %>%
  group_by(topic) %>%
  top_n(8, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms1 %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# View topic 8 terms in each topic
Lda_term1<-as.matrix(terms(myLda1,8))
View(Lda_term1)

# Document-topic probabilities
ap_documents1 <- tidy(myLda1, matrix = "gamma")
ap_documents1

# View document-topic probabilities in a table
#very poor results most have a .25 chance to be in each topic
Lda_document1<-as.data.frame(myLda1@gamma)
View(Lda_document1)



#########################################################
# Prepare the corpus by adding cust ID and Target
docvars(myCorpus1,"Cust_ID") <- gas$Cust_ID
docvars(myCorpus1, "Target") <- gas$Target
summary(myCorpus1)

# We will first generate SVD columns based on the entire corpus
# Pre-process the training corpus
modelDfm1 <- dfm(myCorpus1,
                remove_punc = T,
                remove=c(stopwords('english')),
                stem = T) 

modelDfm1 <- dfm_remove(modelDfm1, stopwrds)
modelDfm1 <- dfm_remove(modelDfm1, stopwrds3)

# Further remove very infrequent words 
modelDfm1 <- dfm_trim(modelDfm1,min_termfreq=4, min_docfreq = 2)

dim(modelDfm1)
topfeatures(modelDfm1,30)


# Weight the predictiv DFM by tf-idf
modelDfm_tfidf1 <- dfm_tfidf(modelDfm1)
dim(modelDfm_tfidf1)

# Perform SVD for dimension reduction
# Choose the number of reduced dimensions as 10
modelSvd1 <- textmodel_lsa(modelDfm_tfidf1, nd=10)
head(modelSvd1$docs)
View(modelSvd1)

# Add the author information as the first column
modelData1 <-cbind(docvars(myCorpus1,"Target"),as.data.frame(modelSvd1$docs))
colnames(modelData1)[1] <- "Target"
head(modelData1)

modelData2 <- cbind(modelData1, gas_315)
head(modelData2)


# Split the data into training & test
set.seed(101)
sample = sample.split(modelData2$Target, SplitRatio = 0.7)
df.train <- subset(modelData2, sample == TRUE)
df.test <- subset(modelData2, sample == FALSE)


# Build a logistic model based on the training dataset
regModel <- glm(formula=Target~.,
                family=binomial(link=logit),
                data=df.train)

# Compare model prediction with known authorships
pred <- predict(regModel, newdata=df.train, type='response')
pred.result <- ifelse(pred > 0.5,1,0)
print(table(pred.result, df.train$Target))

# Predict authorship for the test dataset
unknownPred <- predict(regModel, newdata=df.test, type='response')

###############################################################################













