# import common library
import pandas as pd
import numpy as np
from collections import Counter
import re

# for stemming and tokenization
import nltk
from gensim.corpora import Dictionary
# for one hot encoder imports
from sklearn.preprocessing import LabelEncoder
# imports visualization tools
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
%matplotlib inline
sns.set()



train_data = pd.read_csv('../input/train.csv')
​print(train_data.shape)
train_data.head(3)

# 1)Exploratory Data Analysis

# inspect for null values
train_data.isnull().sum()

# Inspect author variable and inspect label values
train_data.author.value_counts().index

fig, ax = plt.subplots(1,1,figsize=(8,6))

author_vc = train_data.author.value_counts()

ax.bar(range(3), author_vc)
ax.set_xticks(range(3))
ax.set_xticklabels(author_vc.index, fontsize=16)

for rect, c, value in zip(ax.patches, ['b', 'r', 'g'], author_vc.values):
    rect.set_color(c)
    height = rect.get_height()
    width = rect.get_width()
    x_loc = rect.get_x()
    ax.text(x_loc + width/2, 0.9*height, value, ha='center', va='center', fontsize=18, color='white')
# In conclusion: data is not skewed heavily for one author

# Inspect text variable
document_lengths = np.array(list(map(len, train_data.text.str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))
fig, ax = plt.subplots(figsize=(15,6))
​ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(document_lengths, bins=50, ax=ax);
# the data is in esence short scripts i.e.below 100 words


print(" {} documents has more than 100 words.".format(sum(document_lengths > 100)))
​shorter_documents = document_lengths[document_lengths <= 100]

sns.distplot(shorter_documents, bins=50, ax=ax);
fig, ax = plt.subplots(figsize=(15,6)
​
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(shorter_documents, bins=50, ax=ax);
