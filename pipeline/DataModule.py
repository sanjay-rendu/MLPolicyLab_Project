#!/usr/bin/env python
# coding: utf-8

# In[7]:


import csv
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine


# In[8]:


engine = create_engine('postgres://aruns2@mlpolicylab.db.dssg.io:5432/bills3_database')


# In[5]:


#test command
sql = "SELECT status_id FROM catalogs.bill_status"

result_set = engine.execute(sql)
for rec in result_set:
    print(rec)


# In[4]:


# total number of bills
sql = "select count(distinct bill_id) from ml_policy_class.bill_progress"

result_set = engine.execute(sql)
for rec in result_set:
    total_bills = rec
    print(total_bills)
    
#total number of bills passed
sql = "select count(distinct bill_id) from ml_policy_class.bill_progress where bill_status =4"

result_set = engine.execute(sql)
for rec in result_set:
    total_passed_bills = rec
    print(total_passed_bills)


# In[5]:


#total number of bills in NY
sql = "select count(distinct bp.bill_id) from (select distinct bill_id from ml_policy_class.bill_progress) bp join ml_policy_class.bills b on b.bill_id = bp.bill_id join ml_policy_class.sessions s on s.session_id = b.session_id  where s.state_id = 32"

result_set = engine.execute(sql)
for rec in result_set:
    total_passed_bills = rec
    print(total_passed_bills)
    break

#total number of bills passed in NY
sql = "select count(distinct bp.bill_id) from (select distinct bill_id from ml_policy_class.bill_progress where bill_status =4) bp join ml_policy_class.bills b on b.bill_id = bp.bill_id join ml_policy_class.sessions s on s.session_id = b.session_id  where s.state_id = 32"
result_set = engine.execute(sql)
for rec in result_set:
    total_passed_bills = rec
    print(total_passed_bills)
    break


# In[18]:


#bills labels
sql = "select distinct m.bill_id, m.final_status from (select bill_id, (case when bill_status = 4 then 1 else 0 end) as final_status from ml_policy_class.bill_progress) m"
result_set = engine.execute(sql)
for rec in result_set:
    
    print(rec)
    break


# In[34]:


#bills details
sql = "select * from (select bp.bill_id,bp.final_status,s.session_id, s.state_id, s.special, s.year_start , s.year_end , b.bill_type , b.subjects, b.introduced_date, b.introduced_body, b.url from (select distinct m.bill_id as bill_id, m.final_status as final_status from (select bill_id, (case when bill_status = 4 then 1 else 0 end) as final_status from ml_policy_class.bill_progress) m) bp join ml_policy_class.bills b on b.bill_id = bp.bill_id join ml_policy_class.sessions s on s.session_id = b.session_id where s.state_id = 32) bill_details join ml_policy_class.bill_sponsors bs on bill_details.bill_id = bs.bill_id "
result_set = engine.execute(sql)
for rec in result_set:
    
    print(rec)
    break


# In[12]:


for rec in result_set:
    total_passed_bills = rec
    print(total_passed_bills[9])
    break


# In[35]:


all_data = [{column: value for column, value in rowproxy.items()} for rowproxy in result_set]
all_data[0]


# In[37]:


#headers
headers = [i for i in all_data[0].keys()]
headers
len(all_data)


# In[40]:


csv_file= 'output_csv'

with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for row in all_data:
        writer.writerow(row)
csvfile.close()


# In[2]:


def data_extractor(sql1, sql2):
    #import file
    import csv
    import matplotlib.pyplot as plt
    import pandas as pd
    from sqlalchemy import create_engine
    
    engine = create_engine('postgres://aruns2@mlpolicylab.db.dssg.io:5432/bills3_database')
    
    
    result_set = engine.execute(sql1)
    
    #convert to dictionary
    all_data = [{column: value for column, value in rowproxy.items()} for rowproxy in result_set]
    headers = [i for i in all_data[0].keys()]
    
    csv_file= 'output_csv'

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
    csvfile.close()
    
    result_set = engine.execute(sql2)
            #convert to dictionary
    all_data = [{column: value for column, value in rowproxy.items()} for rowproxy in result_set]
    headers = [i for i in all_data[0].keys()]
    
    csv_file= 'billprogress_csv'

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
    csvfile.close()

sql1 = "select * from (select bp.bill_id,bp.final_status,s.session_id, s.state_id, s.special, s.year_start , s.year_end , b.bill_type , b.subjects, b.introduced_date, b.introduced_body, b.url from (select distinct m.bill_id as bill_id, m.final_status as final_status from (select bill_id, (case when bill_status = 4 then 1 else 0 end) as final_status from ml_policy_class.bill_progress) m) bp join ml_policy_class.bills b on b.bill_id = bp.bill_id join ml_policy_class.sessions s on s.session_id = b.session_id where s.state_id = 32) bill_details join ml_policy_class.bill_sponsors bs on bill_details.bill_id = bs.bill_id "
#sql2 = "select * from ml_policy_class.bill_progress bp"
sql2 = 'select bill_id, session_id, introduced_date, final_date, present_date, (final_date - present_date) as "days_to_final", label from sketch.bill_processed order by present_date'

#data_extractor(sql)


# In[10]:


#getting bills progress
sql = "select * from ml_policy_class.bill_progress bp"

result_set = engine.execute(sql)
for rec in result_set:
    print(rec)
    break

        #convert to dictionary
all_data = [{column: value for column, value in rowproxy.items()} for rowproxy in result_set]
headers = [i for i in all_data[0].keys()]
    
csv_file= 'billprogress_csv'

with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for row in all_data:
         writer.writerow(row)
csvfile.close()

