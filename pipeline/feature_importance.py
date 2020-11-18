from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import Pandas_Dataframe, File_Txt
import shap, pickle, pandas as pd, numpy as np, matplotlib.pyplot as plt

class get_shap(BaseOperator):
    @property
    def inputs(self):
        return {
            # "model": Pickle_Obj(self.node.inputs[0]),
            # "train": Pandas_Dataframe(self.node.inputs[1]),
            # "val": Pandas_Dataframe(self.node.inputs[2])
        }

    @property
    def outputs(self):
        return {
            "shap_vals": File_Txt(self.node.outputs[0])
        }

    def run(self, model_type = "misc", target="label", graph_name="shap_bar"):
        with open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/metric_grid/top_models.csv", 'rb') as handle:
            model_list = pickle.load(handle)
        train = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/preprocess4/preprocessed_train.csv").drop("label", axis=1)
        val = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/preprocess4/preprocessed_test.csv").drop("label", axis=1)
        model = model_list[0]


        # model = self.inputs["model"].read()
        # train = self.inputs["train"].read()
        # val = self.inputs["val"].read()

        # train = train.drop(target, axis=1)
        # val = val.drop(target, axis=1)
        if "bill_id" in train.columns:
            train = train.drop("bill_id", axis=1)
            val = val.drop("bill_id", axis=1)

        # if model_type == "misc":
        #    print("hello")
        # shap.initjs()
        X_train = train.to_numpy()
        X_val = val.to_numpy()
        shap_train = shap.maskers.Independent(X_train)
        explainer = shap.LinearExplainer(model, shap_train)
        shap_val = X_val
        shap_values = explainer.shap_values(shap_val)
        shap.summary_plot(shap_values, shap_val, plot_type="bar", show=False, feature_names=val.columns)
        plt.tight_layout()
        plt.savefig("{}.png".format(graph_name))

        for idx, val in enumerate(np.mean(np.abs(shap_values), axis=0)):
            self.outputs['shap_vals'].write("{}: {}".format(train.columns[idx], val))
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(train)
            shap.summary_plot(shap_values[1], train, plot_type="bar", feature_names = train.columns)
            plt.savefig("{}.png".format(graph_name))

        elif model_type == "deep":
            raise NotImplementedError



