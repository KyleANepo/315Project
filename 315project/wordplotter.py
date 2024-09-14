import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# Create separate DataFrames for spam and ham messages
spam_df = df[df["v1"] == "spam"]
ham_df = df[df["v1"] == "ham"]

# Tokenize the messages using CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
spam_word_counts = vectorizer.fit_transform(spam_df["v2"])
ham_word_counts = vectorizer.fit_transform(ham_df["v2"])

# Get the total word counts for spam and ham
spam_word_counts = spam_word_counts.toarray().sum(axis=0)
ham_word_counts = ham_word_counts.toarray().sum(axis=0)

# Plot the word count distributions for spam and ham
plt.hist(spam_word_counts, bins=50, alpha=0.5, label="Spam")
plt.hist(ham_word_counts, bins=50, alpha=0.5, label="Ham")
plt.legend()
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()
