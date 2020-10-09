import pandas as pd
import numpy as np
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factor import BaseOperator


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

    def run(self):
        df = self.inputs["train"].read()

        model_dir = self.arguments["model_dir"].read()

        X = df.as_matrix(columns=df.columns.remove('status'))
        y = df.as_matrix(columns='status')

        model = sklearn.linear_model.LogisticRegression()
        model.fit(X, y)

        joblib.dump(model, model_dir + '/logistic_model.joblib')

        self.outputs["model_path"].write(model_dir + '/logistic_model.joblib')


class validate(BaseOperator):

    @property
    def inputs(self):
        return {
            "model_path": str(self.node.inputs[0]),
            "val": Pandas_Dataframe(self.node.inputs[1]),
        }

    @property
    def outputs(self):
        return {
            "accuracy": self.node.outputs[0]
        }

    def run(self):
        df = self.inputs["val"].read()
        model_path = self.inputs["model_path"].read()

        threshold = self.arguments["threshold"].read()

        X = df.as_matrix(columns=df.columns.remove('status'))
        y = df.as_matrix(columns='status')

        model = joblib.load(model_path)
        y_pred = model.predict(X)
        y_pred = np.array(y_pred > threshold, dtype=np.float)

        acc = sklearn.metrics.accuracy_score(y_true, y_pred))

        self.outputs["accuracy"].write(acc)


class test(BaseOperator):

    @property
    def inputs(self):
        return {
            "model_path": str(self.node.inputs[0]),
            "test": Pandas_Dataframe(self.node.inputs[1]),
        }

    @property
    def outputs(self):
        return {
            "accuracy": self.node.outputs[0]
        }

    def run(self):
        df = self.inputs["test"].read()
        model_path = self.inputs["model_path"].read()

        threshold = self.arguments["threshold"].read()

        X = df.as_matrix(columns=df.columns.remove('status'))
        y = df.as_matrix(columns='status')

        model = joblib.load(model_path)
        y_pred = model.predict(X)
        y_pred = np.array(y_pred > threshold, dtype=np.float)

        acc = sklearn.metrics.accuracy_score(y_true, y_pred))

        self.outputs["accuracy"].write(acc)
