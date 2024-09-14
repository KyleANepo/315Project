import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv', encoding='latin-1')
spam_df = df[df['v1'] == 'spam']
ham_df = df[df['v1'] == 'ham']

def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    # Filter out junk words like 'the' or 'and'
    junk_words = set(['the', 'and', 'of', 'to', 'in', 'a', 'that', 'for', 'is', 'with', 'on', 'this', 'you', 'it', 'not', 'be', 'are', 'from', 'or', 'at'])
    tokens = [token for token in tokens if token not in junk_words]
    return tokens

spam_tokens = spam_df['v2'].apply(preprocess).sum()
ham_tokens = ham_df['v2'].apply(preprocess).sum()

n = 2 # Change n here to get different n-grams

spam_ngrams = ngrams(spam_tokens, n)
ham_ngrams = ngrams(ham_tokens, n)

spam_counts = Counter(map(lambda x: ' '.join(x), spam_ngrams))
ham_counts = Counter(map(lambda x: ' '.join(x), ham_ngrams))

top_spam = dict(spam_counts.most_common(20))
plt.bar(top_spam.keys(), top_spam.values())
plt.title(f'Top 20 {n}-grams in spam messages')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

top_ham = dict(ham_counts.most_common(20))
plt.bar(top_ham.keys(), top_ham.values())
plt.title(f'Top 20 {n}-grams in ham messages')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

