import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator

class fairness_extract(BaseOperator):
    @property
    def inputs(self):
        return {"results": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"metrics": File_Txt(self.node.outputs[0])}

    def run(self, groups, metric="both"):
        """
        Calculates FDR or TPR for a group of interest. Groups is a dictionary with groups of interest and associated district codes.
        """
        df = self.inputs["results"].read()
        
        base = []
        for race in sorted(groups.keys(), reverse=True):
            if race == 'white':
                for model in model: 
                    new_group = df[df["district"].isin(groups[race])]
                    positive_pred = new_group[new_group["pred"] == 1]
                    fdr = np.sum(np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                    tpr = np.sum(1-np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                    precision = precision_score(df["label"], df["pred"])
                    base.append({"precision": precision, "fdr": fdr, "tpr": tpr, "model_name": model, "race": race})i
                base = pd.DataFrame(base)

            else:
                results = []
                for model in models:
                    new_group = df[df["district"].isin(groups[race])]
                    positive_pred = new_group[new_group["pred"] == 1]
                    fdr = np.sum(np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                    tpr = np.sum(1-np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                    precision = precision_score(df["label"], df["pred"])
                    base_model = base[base["model_name"] == model]
                    base_fdr = base_model["fdr"]
                    base_tpr = base_model["tpr"]
                    results.append({"precision": precision, "fdr": fdr/base_fdr, "tpr": tpr/base_tpr, "model_name": model, "race": race})
             
                results = pd.DataFrame(results)
                self.plot(results, race)


    def plot(self, data, group_name):
        plt.scatter("precision", "fdr", data = data)
        plt.xlabel("Precision at top 30%")
        plt.ylabel("False Discovery Rate Disparity")
        plt.title("FDR for {}".format(group_name))
        plt.save_fig("{}_fdr.csv".format(group_name))
        plt.close()

        plt.scatter("precision", "tpr", data=data)
        plt.xlabel("Precision at top 30%")
        plt.ylabel("True Positive Rate Disparity")
        plt.title("TPR for {}".format(group_name))
        plt.save_fig("{}_tpr.csv".format(group_name))
        plt.close()
