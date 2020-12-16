from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import Pandas_Dataframe, File_Txt
import shap, pickle, pandas as pd, numpy as np, matplotlib.pyplot as plt, math

class get_shap(BaseOperator):
    @property
    def inputs(self):
        return {
            # "model": Pickle_Obj(self.node.inputs[0]),
            "train": Pandas_Dataframe(self.node.inputs[0]),
            "val": Pandas_Dataframe(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "cross_tabs": File_Txt(self.node.outputs[0])
        }

    def run(self, model_dir="/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/metric_grid/top_models.csv", model_type = "Decision Tree", graph_name="shap_bar"):
        """Feature importance module that outputs shap feature importance bar plot as well as cross-tabs values for 10 most different features.
        Args:
            model_dir: directory of the model to evaluate
            model_type: general model type. All values beides "Decision Tree" and "Random Forest" are evaluated using the kernel explainer.
            graph_name: name of graph output
        """
       
        with open(model_dir, "rb") as handle:
            model = pickle.load(handle)
        train = self.inputs[1].read().drop("label", axis=1)
        val = self.inputs[1].read().drop("label", axis=1)

        train = train.rename(columns = {"topic_0": "education", "topic_1": "health services", "topic_2": "county/city",
                                    "topic_3": "committees", "topic_4": "public service", "topic_5": "vehicles",
                                    "topic_6": "judiciary", "topic_7": "public funds", "topic_8": "other bills",
                                    "topic_9": "miscellaneous"})
        if "bill_id" in train.columns:
            val = val.drop("bill_id", axis=1)
        val = val.sample(frac = 0.001)
        
        if model_type == "Decision Tree":
            vals = model.feature_importances_
            order = np.argsort(vals)
            vals = vals[order]
            y_pos = np.arange(len(train.columns))
            plt.barh(train.columns[order], vals, align='center')
            plt.yticks(ticks=y_pos, labels = train.columns[order]) 
        elif model_type == "Random Forest":
            explainer = shap.TreeExplainer(model, data= None)
            shap_values = explainer(val)
            shap.summary_plot(shap_values,val, plot_type="bar", show=False, feature_names=val.columns)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, train, link="logit")
            shap_values = explainer(shap_val, nsamples = 100)
            shap.summary_plot(shap_values, val, plot_type="bar", show=False, feature_names=val.columns)
        
        plt.tight_layout()
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance for {}".format(model_type))
        plt.savefig("{}.png".format(graph_name), bbox_inches = 'tight')

        get_feature_differences(model, val)


def get_feature_differences(model, val):
    y_prob = model.predict_proba(val)[:, 1]
    val["score"] = y_prob
    result = val.sort_values(by="score", ascending=False)
    topk = result.head(math.floor(.3*len(val)))
    print(topk.mean(axis=0))
    bottom = result.tail(len(val) - math.floor(.3*len(val)))
    print(bottom.mean(axis=0))
    print(topk.mean(axis=0)-bottom.mean(axis=0))

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
