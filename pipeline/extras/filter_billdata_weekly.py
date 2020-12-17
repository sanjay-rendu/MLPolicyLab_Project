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
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj
from daggit.core.base.factory import BaseOperator

class weekly_convert(BaseOperator):
    @property
    def inputs(self):
        return {"raw": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"df": Pandas_Dataframe(self.node.outputs[0])}
    def run(self):
        """
        converts dataframe to weekly labels
        """
        dfnew = self.inputs["raw"].read()

    #reads bill_data csv - the output from the original sql query
    #dfnew = pd.read_csv('bill_data.csv')
    
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
        final_df = final_df.drop(['original_date','day_from_week_start'],axis= 1)
        
        #output
        self.outputs["df"].write(df)
        #final_df.to_csv("modified_bill_data.csv")