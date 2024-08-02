#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

url = ''
url = 'https://www.hindustantimes.com/india-news/pm-modi-in-gujarat-live-updates-sudarshan-setu-aiims-rajkot-february-25-2024-101708822751934.html'
response = requests.get(url)
text_data = response.text


# In[2]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

# Sample text
text = "Text mining is a fascinating field for natural language processing enthusiasts."

# Tokenization
tokens = word_tokenize(text)

# Removing stopwords and stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
cleaned_tokens = [stemmer.stem(word.lower()) for word in tokens if word.lower() not in stop_words]


# In[3]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create a word cloud
wordcloud = WordCloud(width=800, height=400).generate(' '.join(cleaned_tokens))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[4]:


from textblob import TextBlob

# Analyze sentiment
text = "I love this product! It's amazing."
analysis = TextBlob(text)
sentiment = analysis.sentiment.polarity

if sentiment > 0:
    print("Positive sentiment")
elif sentiment < 0:
    print("Negative sentiment")
else:
    print("Neutral sentiment")


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample text data (replace this with your actual data)
texts = [
    "This is a positive tweet.",
    "I feel great today.",
    "I am sad.",
    "I don't like this.",
    "This is awesome!"
]
labels = [1, 1, 0, 0, 1]  # 1 for positive, 0 for negative

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a model (using Naive Bayes as an example)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[6]:


#2nd data set 
#import install modules

get_ipython().system('pip install wordcount ')
get_ipython().system('pip install wordcloud')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[7]:


# Main
#impport denpendencues
import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
#from wordcloud import WordCloud

# For naive bayes
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[8]:


#load dataframe

# Reading the dataset with no columns titles and with latin encoding 
df_raw = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", header=None)

 # As the data has no column titles, we will add our own
df_raw.columns = ["sentiment", "time", "date", "query", "username", "tweet"]

# Show the first 10 rows of the dataframe.
df_raw.head()


# In[9]:


#drop unnecessary columns

# Ommiting every column except for the text and the label, as we won't need any of the other information
df = df_raw[['sentiment', 'tweet']]

# Replacing the label 4 with 1.
df['sentiment'] = df['sentiment'].replace(4,1)

df.head(10)


# In[10]:


#check balance


# Checking the data's output balance
# Label '4' denotes positive sentiment and '0' denotes negative sentiment
ax = df.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                               legend=False)
ax = ax.set_xticklabels(['Negative','Positive'], rotation=0)


# In[11]:


#trim dataset 

trim_df = False # If you set this to true -> trim dataframe to 1/80 for efficiency 

is_trimmed = False # This should always be initialized to false. Will be set to true if trimming occurs 

if trim_df:
    print("Trimming the dataset to 1/80")
    print("Nr rows before trim:", len(df))
    df_pos = df[df['sentiment'] == 1]
    df_neg = df[df['sentiment'] == 0] 
    df_pos = df_pos.iloc[:int(len(df_pos)/80)]
    df_neg = df_neg.iloc[:int(len(df_neg)/80)]
    df = pd.concat([df_pos, df_neg])
    trim_df = False # prevent running more than once  
    is_trimmed = True
    print("Nr rows after trim:", len(df))
else:
    print("No trimming done")
    
# Checking the data's output balance
# Label '4' denotes positive sentiment and '0' denotes negative sentiment
ax = df.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                               legend=False)
ax = ax.set_xticklabels(['Negative','Positive'], rotation=0)


# In[12]:


#cleaning and preproceesing data 

df.head(10)


# In[13]:


#preprocessing of data 

# Reading contractions.csv and storing it as a dict.
contractions = pd.read_csv('contraction.csv.zip', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

# Defining regex patterns.
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Defining regex for emojis
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"

def preprocess_apply(tweet):

    tweet = tweet.lower()

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern,'<url>',tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern,'<user>', tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace all emojis.
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)

    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    tweet = re.sub(alphaPattern, ' ', tweet)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet
df_raw = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", header=None)

 # As the data has no column titles, we will add our own
df_raw.columns = ["sentiment", "time", "date", "query", "username", "tweet"]

# Show the first 10 rows of the dataframe.
df_raw.head()


# In[14]:


#analayse the dataa 

df_raw.head(10)


# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example data
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Print metrics
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[ ]:




