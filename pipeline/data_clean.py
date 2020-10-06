import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.main.python.daggit.core.io.io import Pandas_Dataframe
from src.main.python.daggit.core.base.factor import BaseOperator


class feature_eng(BaseOperator):
    
    @property
    def inputs(self):
        return {"features": str(self.node.inputs[0]),
                "labels": str(self.node.inputs[1])}

    @property
    def outputs(self):
        return {"train": Pandas_Dataframe(self.node.outputs[0]),
                "test": Pandas_Dataframe(self.node.outputs[1]),
                "val": Pandas_Dataframe(self.node.outputs[2])}
    
    def run(self):
        df = self.inputs["input_path"].read()

        df['bill_session_date'] = df['session_end_date'] - df['bill_introduced_date']
        df['times_amended'] = df

        X = df.to_numpy()
        y = self.inputs["labels"].read()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_sze = .2, random_state=42)

        self.outputs["train"].write(train)
        self.outputs["test"].write(test)
        self.outputs["val"].write(val)
