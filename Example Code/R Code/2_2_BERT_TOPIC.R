#### BERT TOPIC MODELLING IN R ####

#### Load Packages

# You might need to install these packages
# install.packages("devtools")
# install.packages("reticulate")
# devtools::install_github("tpetric7/bertopicr")

library(dplyr)
library(tidyr)
library(purrr)
library(utils)
library(tibble)
library(readr)
library(tictoc)
library(htmltools)
library(reticulate)


# Might need to configure this on Mac
# bertopicr::configure_macos_homebrew_zlib()

# BERTopic is designed for Python and needs a virtual environment

setup_python_environment(
  envname = "r-bertopic",
  method = "conda"
)

use_condaenv("r-bertopic", required = TRUE)

py_config() 

# Now load the BERTopic package and install the required packages in Python
library(bertopicr)
reticulate::conda_install("bertopic", 
                          envname = "/Users/rafaelc-g/miniconda3/envs/r-bertopic") # This should be your environment name fro py_config()

# Import Python packages

py <- import_builtins()
np <- import("numpy")
umap <- import("umap")
UMAP <- umap$UMAP
hdbscan <- import("hdbscan")
HDBSCAN <- hdbscan$HDBSCAN
sklearn <- import("sklearn")
CountVectorizer <- sklearn$feature_extraction$text$CountVectorizer
bertopic <- import("bertopic")
plotly <- import("plotly")
datetime <- import("datetime")

# Import Data

ON_22 <- read_csv("ON22_newspapers.csv")

# Import Stopwords - this is needed for visualization

stopwords_path <- system.file(
  "extdata", "all_stopwords.txt",
  package = "bertopicr"
)
all_stopwords <- read_lines(stopwords_path)

# Prepare the Data
ON_22 <- ON_22 %>% 
  filter(!is.na(dates))
texts <- ON_22$text
titles <- ON_22$doc_id
timestamps <- as.list(ON_22$dates)

# You can view individual texts

texts[[1]]

#### Embeddings ####

sentence_transformers <- import("sentence_transformers")
SentenceTransformer <- sentence_transformers$SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model$encode(texts, show_progress_bar=TRUE)


#### Create Models ####
umap_model <- UMAP(n_neighbors=15L, n_components=5L, min_dist=0.0, metric='cosine', random_state=1998L)
hdbscan_model <- HDBSCAN(min_cluster_size=50L, min_samples = 20L, metric='euclidean', 
                         cluster_selection_method='eom', gen_min_span_tree=TRUE, prediction_data=TRUE, core_dist_n_jobs = 1)


# create vectors

vectorizer_model <- CountVectorizer(min_df=2L, ngram_range=tuple(1L, 3L), 
                                    max_features = 10000L, max_df = 50L,
                                    stop_words = all_stopwords)

sentence_vectors <- vectorizer_model$fit_transform(texts)
sentence_vectors_dense <- np$array(sentence_vectors)
sentence_vectors_dense <- py_to_r(sentence_vectors_dense)


### Topic Model

BERTopic <- bertopic$BERTopic

topic_model <- BERTopic(
  embedding_model = embedding_model,
  umap_model = umap_model,
  hdbscan_model = hdbscan_model,
  vectorizer_model = vectorizer_model,
  calculate_probabilities = TRUE,
  top_n_words = 200L, # number of top words to keep in model
  verbose = TRUE
)

# Fit the topic model 

fit_transform <- topic_model$fit_transform(texts, embeddings)
topics <- fit_transform[[1]]

transform_result <- topic_model$transform(texts)
probs <- transform_result[[2]] 

# Converting R Date to Python datetime

# Convert each R date object to an ISO 8601 string
timestamps <- lapply(timestamps, function(x) {
  format(x, "%Y-%m-%dT%H:%M:%S")  # ISO 8601 format
})

# Dynamic topic model

topics_over_time  <- topic_model$topics_over_time(texts, timestamps,
                                                  nr_bins=20L, global_tuning=TRUE, 
                                                  evolution_tuning=TRUE)


results <- ON_22 %>% 
  mutate(Topic = topics, 
         Probability = apply(probs, 1, max))  # Assuming the highest probability for each article

results <- results %>% 
  mutate(row_id = row_number()) %>% 
  select(row_id, everything())


document_info_df <- get_document_info_df(model = topic_model, 
                                         texts = texts, 
                                         drop_expanded_columns = TRUE)

topics_df <- get_topics_df(model = topic_model)

# Visualize the words representing the topics

visualize_barchart(model = topic_model, 
                   filename = "topics_topwords_interactive_barchart.html", 
                   open_file = FALSE) # TRUE enables output in browser

# Over Time Visualization  

visualize_topics_over_time(model = topic_model, 
                           topics_over_time_model = topics_over_time,
                           top_n_topics = 10, # default is 20
                           filename = "topics_over_time") 

# By Newspaper

classes = as.list(ON_22$newspaper) # specify grouping variable
topics_per_class = topic_model$topics_per_class(texts, classes=classes)

visualize_topics_per_class(model = topic_model, 
                           topics_per_class = topics_per_class,
                           start = 0, # default
                           end = 10, # default
                           filename = "topics_per_class", # default, html extension 
                           auto_open = FALSE) # TRUE enables output in browser
