#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[ ]:





# In[3]:


data = pd.read_csv("twitter_data.csv")
print(data.head())


# In[4]:


data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
print(data.head())


# In[5]:


data = data[["tweet", "labels"]]
print(data.head())


# In[6]:


import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


# In[7]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
print(data.head())


# In[8]:


x = np.array(data["tweet"])
y = np.array(data["labels"])


# In[9]:


cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[10]:


clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[11]:


def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)


# In[13]:


test_data="I will kill you "
df=cv.transform([test_data]).toarray()
print(clf.predict(df))


# In[14]:


test_data="you are awesome "
df=cv.transform([test_data]).toarray()
print(clf.predict(df))


# In[15]:


test_data="you are bad and i don't like you "
df=cv.transform([test_data]).toarray()
print(clf.predict(df))


# In[ ]:




