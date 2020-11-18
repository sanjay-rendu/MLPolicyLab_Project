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
            "shap_vals": File_Txt(self.node.outputs[0])
        }

    def run(model_type = "misc", target="label", graph_name="shap_bar"):
        model = self.inputs["model"].read()
        train = self.inputs["train"].read()
        val = self.inputs["val"].read()

        train = train.drop(target, axis=1)
        val = val.drop(target, axis=1)
        if "bill_id" in train.columns:
            train = train.drop("bill_id", axis=1)
            val = val.drop("bill_id", axis=1)

        if model_type == "misc":
            explainer = shap.KernelExplainer(model.predict_proba, train, link="logit")
            shap_values = explainer.shap_values(val, nsamples=100)
            shap.summary_plot(shap_values[1], val, plot_type="bar", show=False, feature_names=val.columns)
            plt.savefig("{}.png".format(graph_name))

        elif model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(train)
            shap.summary_plot(shap_values[1], train, plot_type="bar", feature_names = train.columns)
            plt.savefig("{}.png".format(graph_name))

        elif model_type == "deep":
            raise NotImplementedError

        for idx, val in enumerate(np.mean(np.abs(shap_values[1]), axis=0)):
            self.outputs['metrics'].write("{}: {}".format(train.columns[idx], val))
