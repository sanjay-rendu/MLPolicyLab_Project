#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


engine = create_engine('postgres://aruns2@mlpolicylab.db.dssg.io:5432/bills3_database')


# In[5]:


sql = 'select doc from ml_policy_class.bill_texts limit1'
result_set = engine.execute(sql)
for rec in result_set:
    print(rec)
    break


# In[6]:


# #convert to dictionary
# all_data = [{column: value for column, value in rowproxy.items()} for rowproxy in result_set]
# headers = [i for i in all_data[0].keys()]
    
# csv_file= 'bill_text_csv'

# with open(csv_file, 'w') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=headers)
#     writer.writeheader()
#     for row in all_data:
#         writer.writerow(row)
# csvfile.close()
for rec in result_set:
    text = rec
    break
    
    


# In[7]:


text


# In[8]:


vectorizer = TfidfVectorizer()


# In[9]:


X = vectorizer.fit_transform(text)


# In[17]:


len(vectorizer.get_feature_names())


# In[31]:


vectorizer.transform(text).toarray()


# In[27]:


vector_text

