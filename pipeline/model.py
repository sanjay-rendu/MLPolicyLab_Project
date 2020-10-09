import pandas as pd
import numpy as np
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator
import sklearn
import joblib


class logistic_regression_trainer(BaseOperator):

    @property
    def inputs(self):
        return {
            "train": Pandas_Dataframe(self.node.inputs[0]),
        }

    @property
    def outputs(self):
        return {
            "model_path": str(self.node.outputs[0])
        }

    def run(self, model_path, max_iter):
        df = self.inputs["train"].read()

        model_dir = self.arguments["model_dir"].read()

        X = df.as_matrix(columns=df.columns.remove('final_status'))
        y = df.as_matrix(columns='final_status')

        model = sklearn.linear_model.LogisticRegression(max_iter=max_iter)
        model.fit(X, y)

        joblib.dump(model, model_path)

        self.outputs["model_path"].write(model_path)


class validate(BaseOperator):

    @property
    def inputs(self):
        return {
            "val": Pandas_Dataframe(self.node.inputs[0]),
            "model_path": sr(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "accuracy": self.node.outputs[0]
        }

    def run(self, threshold):
        df = self.inputs["val"].read()
        model_path = self.inputs["model_path"].read()

        X = df.as_matrix(columns=df.columns.remove('final_status'))
        y = df.as_matrix(columns='final_status')

        model = joblib.load(model_path)
        y_pred = model.predict(X)
        y_pred = np.array(y_pred > threshold, dtype=np.float)

        acc = sklearn.metrics.accuracy_score(y_true, y_pred)

        self.outputs["accuracy"].write(acc)


class test(BaseOperator):

    @property
    def inputs(self):
        return {
            "test": Pandas_Dataframe(self.node.inputs[0]),
            "model_path": sr(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "accuracy": self.node.outputs[0]
        }

    def run(self, model_path, threshold):
        df = self.inputs["test"].read()
        model_path = self.inputs["model_path"].read()

        X = df.as_matrix(columns=df.columns.remove('final_status'))
        y = df.as_matrix(columns='final_status')

        model = joblib.load(model_path)
        y_pred = model.predict(X)
        y_pred = np.array(y_pred > threshold, dtype=np.float)

        acc = sklearn.metrics.accuracy_score(y_true, y_pred)

        self.outputs["accuracy"].write(acc)
