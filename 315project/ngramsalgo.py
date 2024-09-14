import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

# Load the spam.csv file into a pandas dataframe
df = pd.read_csv('spam.csv', encoding='latin-1')

# Separate the dataframe into two subsets: spam messages and ham messages
spam_df = df[df['v1'] == 'spam']
ham_df = df[df['v1'] == 'ham']

# Define a preprocessing function that will remove punctuation, lowercase the text, and filter out junk words
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    # Filter out junk words like 'the' or 'and'
    junk_words = set(['the', 'and', 'of', 'to', 'in', 'a', 'that', 'for', 'is', 'with', 'on', 'this', 'you', 'it', 'not', 'be', 'are', 'from', 'or', 'at'])
    tokens = [token for token in tokens if token not in junk_words]
    return tokens

# Apply the preprocessing function to each message in the spam and ham subsets and concatenate the resulting lists of tokens into two separate token lists
spam_tokens = spam_df['v2'].apply(preprocess).sum()
ham_tokens = ham_df['v2'].apply(preprocess).sum()

# Define a function that takes a list of tokens and an integer n as input, and returns a list of n-grams (tuples of n consecutive tokens)
def get_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Apply the n-gram function to the spam and ham token lists to get two lists of spam and ham n-grams
n = 2 # Change n here to get different n-grams
spam_ngrams = get_ngrams(spam_tokens, n)
ham_ngrams = get_ngrams(ham_tokens, n)

# Define a function that takes a list of n-grams and returns a Counter object with the counts of each n-gram
def count_ngrams(ngrams):
    return Counter(map(tuple, ngrams))

# Apply the Counter function to the spam and ham n-gram lists to get two counters of spam and ham n-gram counts
spam_counts = count_ngrams(spam_ngrams)
ham_counts = count_ngrams(ham_ngrams)

# Define a function that takes a message and a set of n-gram counters, and returns the probability that the message is spam
def spam_probability(message, n, spam_counts, ham_counts):
    tokens = preprocess(message)
    ngrams = get_ngrams(tokens, n)
    counts = count_ngrams(ngrams)
    spam_score = 0
    ham_score = 0
    for ngram, count in counts.items():
        spam_score += count * spam_counts[ngram]
        ham_score += count * ham_counts[ngram]
    total_score = spam_score + ham_score
    if total_score == 0:
        return 0.5
    return spam_score / total_score
