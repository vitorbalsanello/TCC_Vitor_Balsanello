#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[3]:


data_fake = pd.read_csv('C:\\Users\\admin\\Desktop\\TCC\\noticias\\verdadeira.csv')
data_true = pd.read_csv('C:\\Users\\admin\\Desktop\\TCC\\noticias\\falsa.csv')


# In[4]:


data_fake.head()


# In[5]:


data_true.head()


# In[6]:


data_fake["class"] = 0
data_true["class"] = 1


# In[7]:


data_fake.shape, data_true.shape


# In[8]:


data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis = 0, inplace = True)


# In[9]:


data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis = 0, inplace = True)


# In[10]:


data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1


# In[11]:


data_fake_manual_testing.head(10)


# In[12]:


data_true_manual_testing.head(10)


# In[13]:


dados_unidos = pd.concat([data_fake, data_true], axis = 0)


# In[14]:


dados_unidos.head(10)


# In[15]:


dados_unidos.columns


# In[16]:


data = dados_unidos.drop(['title','subject','date'], axis = 1)


# In[17]:


data.isnull().sum()


# In[18]:


data = data.sample(frac = 1)


# In[19]:


data.head(10)


# In[20]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)


# In[21]:


data.columns


# In[22]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[23]:


data['text'] = data['text'].apply(wordopt)


# In[24]:


x = data['text']
y = data['class']


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25) 


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)



# In[27]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[28]:


pred_lr = LR.predict(xv_test)


# In[29]:


LR.score(xv_test, y_test)


# In[30]:


print(classification_report(y_test, pred_lr))


# In[31]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[32]:


pred_dt = DT.predict(xv_test)


# In[33]:


DT.score(xv_test, y_test)


# In[49]:


print(classification_report(y_test, pred_dt))


# In[50]:


def tabela_de_saida(n):
    if n == 0:
        return "Noticia Falsa"
    elif n == 1:
        return "Não é notícia falsa"
    
def teste_manual(noticias):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    
    return print("\n\n LR Prediction: {}" .format(tabela_de_saida(pred_LR[0])))


# In[51]:


news = str(input())
teste_manual(news)


# In[52]:


news = str(input())
teste_manual(news)


# In[ ]:




