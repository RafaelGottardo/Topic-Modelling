
#### Word Emedding in R ####

#### Load Packages 

# install.packages("doc2vec")
# install.packages("tokenizers.bpe")
# install.packages("udpipe")
# install.packages("word2vec")
library(text2vec)
library(doc2vec)
library(word2vec)
library(tokenizers.bpe)
library(udpipe)

data("movie_review", package = "text2vec")
df <- data.frame(#create a unique id variable for each review
                doc_id = paste0("movie_", movie_review$id),
                # use a cleaning function from the word2vec package
                text = txt_clean_word2vec(movie_review$review),
                stringsAsFactors = FALSE)


model <- paragraph2vec(x = df, type = "PV-DBOW",
                       dim = 100, iter = 10, min_count = 5,
                       lr = 0.05, threads = 4)


embedding <- as.matrix(model, which = "docs")
