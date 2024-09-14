import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Filter out spam and ham messages
spam_messages = df.loc[df['v1'] == 'spam', 'v2'].str.cat(sep=' ')
ham_messages = df.loc[df['v1'] == 'ham', 'v2'].str.cat(sep=' ')

# Create wordclouds for spam and ham messages
spam_wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=50, contour_width=3, contour_color='steelblue')
spam_wordcloud.generate(spam_messages)
ham_wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=50, contour_width=3, contour_color='steelblue')
ham_wordcloud.generate(ham_messages)

# Plot the wordclouds
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
axs[0].imshow(spam_wordcloud)
axs[0].set_title('Spam messages')
axs[0].axis('off')
axs[1].imshow(ham_wordcloud)
axs[1].set_title('Ham messages')
axs[1].axis('off')
plt.savefig('wordcloud.png')
plt.show()
