import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator


class feature_eng(BaseOperator):
    
    @property
    def inputs(self):
        return {"raw": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"train": Pandas_Dataframe(self.node.outputs[0]),
                "test": Pandas_Dataframe(self.node.outputs[1]),
                "val": Pandas_Dataframe(self.node.outputs[2])}
    
    def run(self):
	"""Engineers features out of raw data. Saves and returns final dataframe.
	"""
        df = self.inputs["raw"].read()

	## convert party_id to string for OHE
	df['party_id'] = df['party_id'].apply(str)
	ohe1_df = pd.get_dummies(df[['bill_id', 'introduced_body', 'bill_type']]).drop_duplicates('bill_id')
	
	## convert to OHE for adding (# reps/senators + # party members)
	ohe_df = pd.get_dummies(df[['bill_id', 'role_name', 'party_id']])
	ohe_df = ohe_df.groupby(['bill_id']).sum()
	ohe_df = ohe_df.join(ohe1_df.set_index('bill_id'))

	## join with original data
	df = df[['bill_id', 'introduced_date', 'final_status']].drop_duplicates('bill_id')
	df = ohe_df.join(df.set_index('bill_id'), on='bill_id')

	df.to_csv("features.csv")        
	return df
