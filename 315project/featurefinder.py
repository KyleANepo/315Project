import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1}) # map string labels to integer values

# Convert the text messages into numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])

# Select the 1000 most important features using the chi-squared test
selector = SelectKBest(chi2, k=1000)
X_new = selector.fit_transform(X, df['v1'])

# Get the boolean mask of the selected features
feature_mask = selector.get_support()

# Get the selected feature names
feature_names = [feature for bool, feature in zip(feature_mask, vectorizer.get_feature_names()) if bool]

# Print the selected feature names
print(feature_names)
