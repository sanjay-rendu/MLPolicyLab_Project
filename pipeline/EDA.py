#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("bill_data.csv")


# In[ ]:


df_new = df.drop(["introduced_date", "final_date", "present_date", "days_to_final", "days_from_introduction", "subjects"], axis = 1).drop_duplicates()


# In[ ]:


bill_nopass = df_new[df_new["label"] == 0]
bill_pass = df_new[df_new["label"] == 1]


# In[ ]:


print(len(df_new),len(bill_pass),len(bill_nopass),len(bill_pass)+len(bill_nopass))


# In[ ]:


#bill_nopass.to_csv("NotpassedBills.csv")
#bill_pass.to_csv("passedBills.csv")


# In[ ]:


import seaborn as sns


# In[ ]:


sns.histplot(bill_pass, x = "is_bipartisan", discrete = True)
plt.title("Bipartisan support for passed bills")


# In[ ]:


sns.histplot(bill_nopass, x = "is_bipartisan", discrete = True)
plt.title("Bipartisan support for NOt passed bills")


# In[ ]:


sns.countplot(data = df_new, x = "is_bipartisan", hue = "label")
plt.title("Bipartisan support for passed and not passed bills")


# In[ ]:


sns.countplot(data = df_new, x = "introduced_body", hue = "label")
plt.title("Introduced body for passed and not passed bills")


# In[ ]:


sns.countplot(data = df_new, x = "session_id", hue = "label")
plt.title("Session wise passed and not passed bills")


# In[ ]:


sns.countplot(data = df_new, x = "number_republicans", hue = "label")
plt.title("Session wise passed and not passed bills") 


# In[ ]:


def label_race (row):
    if row['number_dems'] > row['number_republicans']:
        return 'Democrats'
    if row['number_dems'] < row['number_republicans']:
        return 'Republicans'
    else:
        return 'Others'
    
df_new['Major_support_from'] = df_new.apply (lambda row: label_race(row), axis=1)


# In[ ]:


df_new


# In[52]:


sns.countplot(data = df_new, x = "Major_support_from", hue = "label")
plt.title("Party support and its impact on passed and not passed bills") 


# In[19]:


#from matplotlib.pyplot import show

sns.set(style="darkgrid")

total = float(len(df_new)) # one bill per row 

ax = sns.countplot(x="is_bipartisan", hue="label", data=df_new) # for Seaborn version 0.7 and more
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title("Bi_Partisan bill and its impact on passed and not passed bills") 
plt.show()


# In[20]:


sns.set(style="darkgrid")

total = float(len(df_new)) # one bill per row 

ax = sns.countplot(x="introduced_body", hue="label", data=df_new) # for Seaborn version 0.7 and more
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title("Introduced body for the bill and its impact on passed and not passed bills") 
plt.show()


# In[45]:



sns.set(style="darkgrid")

total = float(len(df_new)) # one bill per row 

ax = sns.countplot(x="Major_support_from", hue="label", data=df_new) # for Seaborn version 0.7 and more
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.3f}'.format(round(height/total,3)),
            ha="center") 
plt.title("Party support and its impact on passed and not passed bills") 
plt.show()


# In[46]:


sns.set(style="darkgrid")

total = float(len(df_new)) # one bill per row 

ax = sns.countplot(x="session_id", hue="label", data=df_new) # for Seaborn version 0.7 and more
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title("Session id wise passed and not passed bills") 
plt.legend(loc ="upper left")
plt.show()


# In[27]:


def with_hue(plot, feature, Number_of_categories, hue_categories):
    a = [p.get_height() for p in plot.patches]
    patch = [p for p in plot.patches]
    for i in range(Number_of_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
            percentage = '{:.1f}%'.format(100 * a[(j*Number_of_categories + i)]/total)
            x = patch[(j*Number_of_categories + i)].get_x() + patch[(j*Number_of_categories + i)].get_width() / 2 - 0.15
            y = patch[(j*Number_of_categories + i)].get_y() + patch[(j*Number_of_categories + i)].get_height() 
            ax.annotate(percentage, (x, y), size = 12)
    plt.show()

    
plt.figure(figsize = (7,5))
ax = sns.countplot(x="introduced_body", hue="label", data=df_new)
with_hue(ax, df_new.introduced_body,2,2)


# In[ ]:





# In[28]:


def with_hue(plot, feature, Number_of_categories, hue_categories):
    a = [p.get_height() for p in plot.patches]
    patch = [p for p in plot.patches]
    for i in range(Number_of_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
            percentage = '{:.1f}%'.format(100 * a[(j*Number_of_categories + i)]/total)
            x = patch[(j*Number_of_categories + i)].get_x() + patch[(j*Number_of_categories + i)].get_width() / 2 - 0.15
            y = patch[(j*Number_of_categories + i)].get_y() + patch[(j*Number_of_categories + i)].get_height() 
            ax.annotate(percentage, (x, y), size = 12)
    plt.show()

    
plt.figure(figsize = (7,5))
ax = sns.countplot(x="is_bipartisan", hue="label", data=df_new)
with_hue(ax, df_new.is_bipartisan,2,2)


# In[41]:


ax = sns.countplot(x="session_id", hue="label", data=df_new)
plt.title('session_id wise', fontsize=20)
total = float(len(df_new))
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
plt.yticks([])
plt.ylabel("Proportion")
plt.show()


# In[33]:


#df = sns.load_dataset("tips")
x, y, hue = "session_id", "proportion", "label"
hue_order = ["0", "1"]

(df_new[x]
 .groupby(df_new[hue])
 .value_counts(normalize=True)
 .rename(y)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))


# In[34]:


x, y, hue = "Major_support_from", "proportion", "label"
hue_order = ["0", "1"]

(df_new[x]
 .groupby(df_new[hue])
 .value_counts(normalize=True)
 .rename(y)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))

