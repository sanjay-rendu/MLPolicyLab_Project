import pandas as pd
import numpy as np
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator


class feature_eng(BaseOperator):
    
    @property
    def inputs(self):
        return {"raw": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"df": Pandas_Dataframe(self.node.outputs[0])}
    
    def run(self, index, ohe_features, summed_features, other_features, out_path = "features.csv"):
	"""Engineers features out of raw data. Saves and returns final dataframe.
        Arguments:
            index: str
                Index features (bill_id)
            ohe_features: list[str]
                Features to be one hot encoded such as introduced_body and bill_type
            summed_features: list[str]
                Features to be summed such as role_name and party_id
            other_features: list[str]
                All other features of interest to be kept as is
            out_path: str
                Path where final features are saved
        Returns:
            pandas dataframe of features
	"""
        df = self.inputs["raw"].read()

	## convert party_id to string for OHE
        for feature in ohe_features:
            df[feature] = df[feature].apply(str)
	ohe1_df = pd.get_dummies(df[[index]+ohe_features]).drop_duplicates(index)
	
	## convert to OHE for adding (# reps/senators + # party members)
	ohe_df = pd.get_dummies(df[[index]+summed_features])
	ohe_df = ohe_df.groupby([index]).sum()
	ohe_df = ohe_df.join(ohe1_df.set_index(index))

	## join with original data
	df = df[[index]+other_features].drop_duplicates(index)
	df = ohe_df.join(df.set_index(index), on=index)

	df.to_csv(out_path)        
	return df
