import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords')

df = pd.read_csv('spam.csv', encoding='latin-1')

spam_df = df[df['v1'] == 'spam']
ham_df = df[df['v1'] == 'ham']

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

spam_tokens = spam_df['v2'].apply(preprocess).sum()
ham_tokens = ham_df['v2'].apply(preprocess).sum()

spam_counts = Counter(spam_tokens)
ham_counts = Counter(ham_tokens)

top_spam = dict(spam_counts.most_common(20))
plt.bar(top_spam.keys(), top_spam.values())
plt.title('Top 20 words in spam messages')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

top_ham = dict(ham_counts.most_common(20))
plt.bar(top_ham.keys(), top_ham.values())
plt.title('Top 20 words in ham messages')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
