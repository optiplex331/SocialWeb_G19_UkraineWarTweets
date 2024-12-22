# ## 2. Sentimental analysis overtime

# Imports

import os
import re
import gc
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# **NOTE**: run this once and then comment out, so the following won't be run in each process
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')
# The above should have been run in `sentiment_analysis.ipynb` before running this script

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Calculate sentiment score and label."""
    score = sia.polarity_scores(text)["compound"]
    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return score, label

def add_sentiment_columns(df):
    """Add sentiment analysis columns to DataFrame."""
    # Apply the sentiment function to the text column
    results = df['text'].apply(analyze_sentiment)
    # Unpack results into two new columns
    df['sentiment_score'], df['sentiment_label'] = zip(*results)
    return df

cleaned_folder = './cleaned'
file_paths = [os.path.join(cleaned_folder, file) for file in os.listdir(cleaned_folder) if file.endswith('_cleaned.csv.gz')]

# Process multiple files in parallel

# take a sample from the file, just to test the process speed, set to 0 to process all
sample_size: int = 0

def analyze_sentiment_file(file_path):
    """Load a CSV file, analyze sentiment, and save the results."""
    # add extra line breaks before and after messages, so they won't be a mess when running in parallel
    print(f"\nProcessing: {file_path}\n")
    df = pd.read_csv(file_path, compression='gzip')
    print(f'\nLoaded: {file_path}\n')
    if sample_size:
        df = df.sample(int(sample_size))

    print(f'\nAnalyzing sentiment: {file_path}\n')
    df = add_sentiment_columns(df)
    print(f'\nDone analyzing sentiment: {file_path}\n')

    # Save the DataFrame to ./sentiment folder, named with xxx_sentiment.csv
    output_path = os.path.join('./sentiment', os.path.basename(file_path).replace('_cleaned.csv.gz', '_sentiment.csv'))
    df.to_csv(output_path)
    print(f"\nSaved with sentiment analysis: {output_path}\n")
    return output_path

os.makedirs('./sentiment', exist_ok=True)

if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        processed_files = list(executor.map(analyze_sentiment_file, file_paths))
        print(f"\nProcessed files: {processed_files}\n")

