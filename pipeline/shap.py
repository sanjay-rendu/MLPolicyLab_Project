from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import Pandas_Dataframe
import shap

class shap(BaseOperator):
    @property
    def inputs(self):
        return {
            "model": Pickle_Obj(self.node.inputs[0]),
            "train": Pandas_Dataframe(self.node.inputs[1]),
            "val": Pandas_Dataframe(self.node.inputs[2])
        }

    @property
    def outputs(self):
        return {
            "model": Pickle_Obj(self.node.outputs[0])
        }

    def run(model_type = "misc", target="label"):
        model = self.inputs["model"].read()
        train = self.inputs["train"].read()
        val = self.inputs["val"].read()

        features = list(set(list(df.columns)) - {target})
        X_train = train.as_matrix(columns=features)
        X_val = val.as_matrix(columns=features)
        if model_type == "misc":
            explainer = shap.KernelExplainer(model.predict_proba, X_train, link="logit")
            shap_values = explainer.shap_values(X_val, nsamples=100)
