#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("bill_data.csv")


# In[4]:


df_new = df.drop(["introduced_date", "final_date", "present_date", "days_to_final", "days_from_introduction", "subjects"], axis = 1).drop_duplicates()


# In[6]:


bill_nopass = df_new[df_new["label"] == 0]
bill_pass = df_new[df_new["label"] == 1]


# In[7]:


print(len(df_new),len(bill_pass),len(bill_nopass),len(bill_pass)+len(bill_nopass))


# In[8]:


#bill_nopass.to_csv("NotpassedBills.csv")
#bill_pass.to_csv("passedBills.csv")


# In[9]:


import seaborn as sns


# In[10]:


sns.histplot(bill_pass, x = "is_bipartisan", discrete = True)
plt.title("Bipartisan support for passed bills")


# In[11]:


sns.histplot(bill_nopass, x = "is_bipartisan", discrete = True)
plt.title("Bipartisan support for NOt passed bills")


# In[12]:


sns.countplot(data = df_new, x = "is_bipartisan", hue = "label")
plt.title("Bipartisan support for passed and not passed bills")


# In[13]:


sns.countplot(data = df_new, x = "introduced_body", hue = "label")
plt.title("Introduced body for passed and not passed bills")


# In[14]:


sns.countplot(data = df_new, x = "session_id", hue = "label")
plt.title("Session wise passed and not passed bills")


# In[15]:


sns.countplot(data = df_new, x = "number_republicans", hue = "label")
plt.title("Session wise passed and not passed bills") 


# In[16]:


def label_race (row):
    if row['number_dems'] > row['number_republicans']:
        return 'Democrats'
    if row['number_dems'] < row['number_republicans']:
        return 'Republicans'
    else:
        return 'Others'
    
df_new['Major_support_from'] = df_new.apply (lambda row: label_race(row), axis=1)


# In[17]:


df_new


# In[52]:


sns.countplot(data = df_new, x = "Major_support_from", hue = "label")
plt.title("Party support and its impact on passed and not passed bills") 

