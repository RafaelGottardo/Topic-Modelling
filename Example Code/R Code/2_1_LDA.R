### Topic Modelling Using LDA in R ####

# Install and load packages

# install.packages("topicmodels")
library(topicmodels)
library(tidyverse)
library(tidytext)
library(tm)
library(quanteda)

# Load data 

ON_22 <- read_csv("ON22_newspapers.csv")

# Clean data 

ON_22$text <- as.character(ON_22$text)
ON_22_corp <- corpus(ON_22, text_field = "text")

toks <- tokens(ON_22_corp,
               what = "word",
               remove_punct = TRUE,
               remove_symbols = TRUE,
               remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("english"))

dfm <- dfm(toks)

dtm <- convert(dfm, to = "topicmodels")

# Fit the LDA model

lda_model <- LDA(dtm, # specify the pre-processed data
                 k = 8, # specify the number of topic
                 control = list(seed = 1998) # seed for replicabilty
                 )

# Extract the per-topic word probabilities (beta)

topics <- tidy(lda_model, matrix = "beta")
topics %>% 
 head(., n = 10) %>% 
  kableExtra::kable("latex", digits = 6,
                    position = "H", 
                    booktabs = TRUE, 
                    linesep = "",
                    escape = F)


# extract the top terms for each topic 

top_terms <- topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  theme_bw() +
  theme(text = element_text(size = 8))

ggsave("plots/LDA_topics.png", width = 4.72441, height = 2.3622)
