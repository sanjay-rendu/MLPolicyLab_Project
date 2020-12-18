from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import Pandas_Dataframe, File_Txt, Pickle_Obj
import shap, pickle, pandas as pd, numpy as np, matplotlib.pyplot as plt, math

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class get_shap(BaseOperator):
    @property
    def inputs(self):
        return {
            "top_models": Pickle_Obj(self.node.inputs[0]),
            "train": Pandas_Dataframe(self.node.inputs[1]),
            "val": Pandas_Dataframe(self.node.inputs[2])
        }

    @property
    def outputs(self):
        return {
            "cross_tabs": File_Txt(self.node.outputs[0])
        }

    def get_plot(self, model, train, val, i):
        if isinstance(model, DecisionTreeClassifier):
            vals = model.feature_importances_
            order = np.argsort(vals)
            vals = vals[order]
            y_pos = np.arange(len(train.columns))
            plt.barh(train.columns[order], vals, align='center')
            plt.yticks(ticks=y_pos, labels = train.columns[order]) 

        elif isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model, data= None)
            shap_values = explainer.shap_values(val)
            shap.summary_plot(shap_values[1],val, plot_type="bar", show=False)

        else:
            explainer = shap.KernelExplainer(model.predict_proba, train, link="logit")
            shap_values = explainer.shap_values(shap_val, nsamples = 100)
            shap.summary_plot(shap_values[1], val, plot_type="bar", show=False, feature_names=val.columns)

        plt.tight_layout()
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("Feature Importance for Random Forest {}".format(i))
        plt.savefig("plots/shap{}.png".format("rf"+str(i)), bbox_inches = 'tight')

    def get_feature_differences(self, model, val, max_vals, min_vals):
        y_prob = model.predict_proba(val)[:, 1]
        val["score"] = y_prob
        result = val.sort_values(by="score", ascending=False)
        topk = result.head(math.floor(.3*len(val)))

        top = topk.mean(axis=0).to_numpy()*(max_vals-min_vals)+min_vals
        bottom = result.tail(len(val) - math.floor(.3*len(val)))
        bottom_vals = bottom.mean(axis=0).to_numpy()*(max_vals-min_vals)+min_vals
        order = np.abs(topk.mean(axis=0).to_numpy()-bottom.mean(axis=0).to_numpy())

        cross_tabs = pd.DataFrame({"feature": val.columns, "top": top, "bottom": bottom_vals, "order": order})
        print(cross_tabs.sort_values(by="order", ascending=False))
    
    def run(self):
        """Feature importance module that outputs shap feature importance bar plot as well as cross-tabs values for 10 most different features.
        """
        
        top_models = self.inputs["top_models"].read()
        train = self.inputs["train"].read().drop("label", axis=1)
        val = self.inputs["val"].read().drop("label", axis=1)

        train = train.rename(columns = {"topic_0": "education", "topic_1": "health services", "topic_2": "county/city",
                                    "topic_3": "committees", "topic_4": "public service", "topic_5": "vehicles",
                                    "topic_6": "judiciary", "topic_7": "public funds", "topic_8": "other bills",
                                    "topic_9": "miscellaneous"})
        if "bill_id" in train.columns:
            val = val.drop("bill_id", axis=1)
        val = val.sample(frac = 0.001)
        i = 0
        max_vals = np.array([724, 92, 41, 1, 30, 0.964498, 0.929411, 0.995476, 0.989154, 0.969405, 0.978926, 0.998956, 0.890147, 0.953781, 0.972987, 1, 1, 1])
        min_vals = np.array([0, 0, 0, 0, 0, 6e-6, 3e-6, 5e-6, 2e-6, 9e-6, 6e-6, 14e-6, 10e-6, 15e-6, 12e-6, 0, 0, 0])
        for model in top_models:
            i += 1
            #self.get_plot(model, train, val, i)
            if "score" in val.columns:
                val = val.drop("score", axis = 1)
            print(model.get_params())
            self.get_feature_differences(model, val, max_vals, min_vals)

if __name__ == "__main__":
    with open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/dt_grid4/DecisionTreeClassifier13.pkl", 'rb') as handle:
        model = pickle.load(handle)['model']
    train = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/preprocess4/preprocessed_train.csv").drop("label", axis=1)
    val = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/preprocess4/preprocessed_test.csv").drop("label", axis=1)
    # model = model_list[0]
    train = train.rename(columns = {"topic_0": "education", "topic_1": "health services", "topic_2": "county/city",
                                    "topic_3": "committees", "topic_4": "public service", "topic_5": "vehicles",
                                    "topic_6": "judiciary", "topic_7": "public funds", "topic_8": "other bills",
                                    "topic_9": "miscellaneous"})
    vals = model.feature_importances_
    order = np.argsort(vals)
    vals = vals[order]
    print(vals)
    y_pos = np.arange(len(train.columns))
    plt.barh(train.columns[order], vals, align='center')
    plt.yticks(ticks=y_pos, labels = train.columns[order])
    #plt.set_yticklabels(train.columns)

    #plt.invert_yaxis()  # labels read top-to-bottom
    plt.tight_layout()
    plt.xlabel('Feature Importance')
    plt.ylabel("Feature")
    plt.title('Feature Importance of Decision Tree')
    
    plt.savefig("shap.png", bbox_inches='tight')
    if "bill_id" in train.columns:
        train = train.drop("bill_id", axis=1)
        val = val.drop("bill_id", axis=1)

    get_feature_differences(model, val)
