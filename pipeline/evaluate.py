import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt
from daggit.core.base.factory import BaseOperator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class evaluate(BaseOperator):

    @property
    def inputs(self):
        return {
            "data": Pandas_Dataframe(self.node.inputs[0]),
            "model": Pickle_Obj(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "output": File_Txt(self.node.outputs[0]),
            "metrics": File_Txt(self.node.outputs[1])
        }

    def run(self, target, threshold):
        df = self.inputs["val"].read()
        model = self.inputs["model"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        y_prob = model.predict(X)
        y_pred = np.array(y_prob > threshold, dtype=np.float)

        acc = accuracy_score(y, y_pred)

        features = list(set(list(df.columns)) - {target} - {'bill_id'})
        df = df.drop(features)
        df['pred_probability'] = y_prob
        df = df.to_csv(sep=' ', index=False)

        self.outputs["output"].write(df)
        self.outputs["metrics"].write('Accuracy:' + str(acc))
