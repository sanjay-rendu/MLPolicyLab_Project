import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt
from daggit.core.base.factory import BaseOperator
import sklearn


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

    def run(self, max_iter):
        df = self.inputs["train"].read()

        X = df.as_matrix(columns=df.columns.remove('final_status'))
        y = df.as_matrix(columns='final_status')

        model = sklearn.linear_model.LogisticRegression(max_iter=max_iter)
        model.fit(X, y)

        self.outputs["model"].write(model)


class accuracy(BaseOperator):

    @property
    def inputs(self):
        return {
            "val": Pandas_Dataframe(self.node.inputs[0]),
            "model_path": Pickle_Obj(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "accuracy": File_Txt(self.node.outputs[0])
        }

    def run(self, threshold):
        df = self.inputs["val"].read()
        model = self.inputs["model_path"].read()

        X = df.as_matrix(columns=df.columns.remove('final_status'))
        y = df.as_matrix(columns='final_status')

        y_pred = model.predict(X)
        y_pred = np.array(y_pred > threshold, dtype=np.float)

        acc = sklearn.metrics.accuracy_score(y, y_pred)

        self.outputs["accuracy"].write(acc)
