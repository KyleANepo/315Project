import pandas as pd
import numpy as np
import re

#Subclass for counting hashable objects, useful for the data handled
from collections import Counter
from sklearn.model_selection import train_test_split

# Import data (from CSV)
# Clean data (converting all text to lowercase, removing punctuation, etc.)
# Create bags of words (even visualizing it through a word cloud) or bags of n-grams
# Calculate frequencies through word vectors
# Store data in matrix, updating weights
# Calculate similarities
# Analysis

# Most of this comes from online
# Putting text in lowercase, removing spaces, etc...
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return ' '.join(text.split())

# Creating dataframe with panda
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

df['text'] = df['text'].apply(preprocess)
train_data, test_data, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Counter for spam and ham. Just getting the end result.
spam_count = Counter()
ham_count = Counter()
for i in range(len(train_data)):
    if train_labels.iloc[i] == 'spam':
        spam_count.update(train_data.iloc[i])
    else:
        ham_count.update(train_data.iloc[i])
        
spam_prior = len([label for label in train_labels if label == 'spam']) / len(train_labels)
ham_prior = len([label for label in train_labels if label == 'ham']) / len(train_labels)

# Got this from online
vocab = set(word for message in train_data for word in message)
spam_word_counts = {word: count + 1 for word, count in spam_count.items()}
ham_word_counts = {word: count + 1 for word, count in ham_count.items()}
spam_denominator = sum(spam_word_counts.values()) + len(vocab)
ham_denominator = sum(ham_word_counts.values()) + len(vocab)
spam_probs = {word: np.log((count + 1) / spam_denominator) for word, count in spam_word_counts.items()}
ham_probs = {word: np.log((count + 1) / ham_denominator) for word, count in ham_word_counts.items()}

# Sort all the messages, code below from online sources
def classify(message):
    words = preprocess(message)
    # Weights
    spam_score = np.log(spam_prior)
    ham_score = np.log(ham_prior)
    for word in words: # Updating weights
        if word in spam_probs:
            spam_score += spam_probs[word]
        else:
            spam_score += np.log(1 / spam_denominator)
        if word in ham_probs:
            ham_score += ham_probs[word]
        else:
            ham_score += np.log(1 / ham_denominator)
    if spam_score > ham_score:
        return 'spam'
    else:
        return 'ham'

# Print results!
predictions = [classify(message) for message in test_data]
accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == test_labels.iloc[i]) / len(predictions)
print('Accuracy:', accuracy)
