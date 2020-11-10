# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:02:40 2020

@author: sharm
"""

"""
-- takes in .CSV the output of original SQL query (all present days)
-- outputs the filtered .CSV
"""
import csv
import numpy as np
import pandas as pd
import datetime

#reads bill_data csv - the output from the original sql query
dfnew = pd.read_csv('bill_data.csv')

# add the original start date for data
dfnew[["introduced_date", "final_date", "present_date"]] = dfnew[["introduced_date", "final_date", "present_date"]].apply(pd.to_datetime)
dfnew['original_date'] = pd.to_datetime("'2009-01-07'".replace("'",""))

#prepare df for filtering on every 7th day plus the final day for the bill or session
conditions =[(dfnew['present_date'] == dfnew['final_date'])]
choices = [8]

dfnew['day_from_week_start'] = np.select(conditions, choices, default = (dfnew['present_date'] - dfnew['original_date']).dt.days % 7 )

#filter
d = [0,8]
final_df = dfnew.loc[dfnew['day_from_week_start'].isin(d)]

#output
final_df.to_csv("modified_bill_data.csv")