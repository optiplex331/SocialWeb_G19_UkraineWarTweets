# Social Web Assignment - Group 19 - Ukraine War Tweets

In this project, we analyze tweets related to the Ukraine war

## Environment setup

Dependencies to run `sentiment_analysis.ipynb` and `sentiment_analysis_parallel.py`:

```text
[tool.poetry.dependencies]
python = "^3.12"
gdown = "^5.2.0"
pandas = "^2.2.3"
nltk = "^3.9.1"
matplotlib = "^3.9.3"
```


## Steps and corresponding files:

The code is organized in the following steps and files:

1. Data preprocessing
   1. Data cleaning: `data_cleaning.ipynb`
   2. Data preprocessing and merging: `data_merge.ipynb`
2. Retweet network analysis: `retweet_network.ipynb`
3. Sentiment analysis 
   1. How it is done: `sentiment_analysis.ipynb`
   2. Code that is actually run, in parallel: `sentiment_analysis_parallel.py`. See the last ⚠️**NOTE** in `sentiment_analysis.ipynb` for more details.
4. Results plotting and analysis: TODO
