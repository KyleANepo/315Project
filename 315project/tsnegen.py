import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Convert the text messages into numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])

# Reduce the dimensionality of the data using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())

# Plot the t-SNE plot
plt.scatter(X_tsne[df['v1']=='ham', 0], X_tsne[df['v1']=='ham', 1], label='ham')
plt.scatter(X_tsne[df['v1']=='spam', 0], X_tsne[df['v1']=='spam', 1], label='spam')
plt.legend()
plt.show()
