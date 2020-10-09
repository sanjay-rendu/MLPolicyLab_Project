import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt
from daggit.core.base.factory import BaseOperator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class logistic_regression_trainer(BaseOperator):

    @property
    def inputs(self):
        return {
            "train": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "model": Pickle_Obj(self.node.outputs[0])
        }

    def run(self, target, max_iter):
        df = self.inputs["train"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        model = LogisticRegression(max_iter=max_iter)
        model.fit(X, y)

        self.outputs["model"].write(model)


class accuracy(BaseOperator):

    @property
    def inputs(self):
        return {
            "val": Pandas_Dataframe(self.node.inputs[0]),
            "model": Pickle_Obj(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "accuracy": File_Txt(self.node.outputs[0])
        }

    def run(self, target, threshold):
        df = self.inputs["val"].read()
        model = self.inputs["model"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        y_pred = model.predict(X)
        y_pred = np.array(y_pred > threshold, dtype=np.float)

        acc = accuracy_score(y, y_pred)

        self.outputs["accuracy"].write(str(acc))
