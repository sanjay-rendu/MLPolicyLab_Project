import pandas as pd, numpy as np, matplotlib.pyplot as plt, re, math, pickle
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
                    base.append({"precision": precision, "fdr": fdr, "tpr": tpr, "model_name": model, "race": race})
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
    
if __name__ == "__main__":
    """
    Calculates FDR or TPR for a group of interest. Groups is a dictionary with groups of interest and associated district codes.
    """

    def plot(data, group_name):
        plt.scatter("precision", "fdr", data = data)
        plt.xlabel("Precision at top 30%")
        plt.ylabel("False Discovery Rate Disparity")
        plt.title("FDR for {}".format(group_name))
        plt.savefig("{}_fdr.png".format(group_name))
        plt.close()

        plt.scatter("precision", "tpr", data=data)
        plt.xlabel("Precision at top 30%")
        plt.ylabel("True Positive Rate Disparity")
        plt.title("TPR for {}".format(group_name))
        plt.savefig("{}_tpr.png".format(group_name))
        plt.close()

    def parse_district(s):
        if s[0] == "S":
            body = "S"
        else:
            body = "H"
        part = re.sub(".*-0*", "", s)
        try:
            code = int(part)
        except:
            if part[0] == "D":
                code = int(part[-1])
            else:
                code = int(part[0])
        return pd.Series({"body_in": body, "code": code})

    # features4 = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/select4/fil_val.csv")
    # split4 = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/split4/val.csv")[["bill_id",
    #                                                                                                            "primary_sponsor_district"]]
    # val4 = features4.merge(split4, on = "bill_id", how = "left").dropna(subset=["primary_sponsor_district"])
    # val4 = val4.merge(val4["primary_sponsor_district"].apply(parse_district), left_index=True, right_index=True)

    features1 = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/select1/fil_val.csv")
    split1 = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/split1/val.csv")
    val1 = features1.merge(split1, on = "bill_id", how = "left").dropna(subset=["primary_sponsor_district"])
    val1 = val1.merge(val1["primary_sponsor_district"].apply(parse_district), left_index=True, right_index=True)
    base = []
    print("got data")
    rf1list = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/rf_grid1/RandomForestClassifier.pkl", "rb"))
#    svm1list = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/svm_grid1/SVC.pkl", "rb"))
    lr1list = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/lr_grid1/LogisticRegression.pkl", "rb"))
    dt1list = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/dt_grid1/DecisionTreeClassifier.pkl", "rb"))
    print("got models")
    models1 = rf1list + lr1list + dt1list
    house_races1 = {"hispanic": [6, 34, 35, 38, 39, 51, 53, 54, 68, 71, 72, 76, 77, 78, 79, 84, 85, 86],
                    "asian": [22, 24, 25, 26, 34, 35, 39, 47, 49, 51, 64],
                    "black": [18, 29, 31, 32, 33, 40, 42, 43, 55, 56, 57, 58, 70, 83, 133, 141],
                    "white": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 27, 28, 30, 36, 37, 41, 44, 45, 46, 48, 50, 52, 59, 60, 61, 62, 63, 65, 66, 67, 69, 73, 74, 75, 80, 81, 82, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150]
    }
    senate_races1 = {"hispanic": [13, 17, 28, 31, 32, 33],
                     "asian": [11, 13, 16, 22, 25],
                     "black": [10, 14, 18, 19, 20, 21, 30, 36],
                     "white": [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 23, 24, 26, 27, 29, 34, 35, 37, 38, 39, 40, 41, 42,
                               43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]}


# lr4list = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/lr_grid4/LogisticRegression.pkl", "rb"))
# dt4list = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/dt_grid4/DecisionTreeClassifier.pkl", "rb"))
# models4 = lr4list+dt4list
# house_races4 = {"hispanic": [6, 18, 71, 72, 115, 129, 137, 148],
#                 "asian": [9, 11, 18, 29, 32, 55, 56, 61, 69, 71, 74, 79, 100, 104, 108, 128, 137],
#                 "black": [6, 11, 18, 30, 39, 71, 102, 115, 117, 120, 128, 129],
#                 "white": [1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 73, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 116, 118, 121, 122, 123, 124, 125, 126, 127, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 150]
# }
# senate_races4 = {"hispanic": [4, 8, 30, 32, 45],
#                  "asian": [8, 14, 25, 30, 33],
#                  "black": [4, 8, 45, 48, 50, 53],
#                  "white": [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 31,
#                            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 49, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62]}

    #all_vals = [val1, val4]
    FEATURE_COLS = ['days_to_final_x',
                   'days_from_introduction_x', 'number_dems_x', 'number_republicans_x',
                   'is_bipartisan_x', 'day_from_month_start_x',
                      'topic_0', 'topic_1', 'topic_2','topic_3', 'topic_4', 'topic_5','topic_6', 'topic_7',
                      'topic_8', 'topic_9', 'A', 'S']
    def get_plots(df, models, house_races, senate_races):
        races = {}
        for race in house_races:
            races[race] = list(set(house_races[race] + senate_races[race]))
        intro = df["introduced_body_x"].tolist()
        introA = np.array([1 if x == 'A' else 0 for x in intro])
        introS = 1-introA
        df['A'] = introA
        df['S'] = introS
        df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
        for race in sorted(races.keys(), reverse=True):
            if race == 'white':
                base = pd.read_csv("white1.csv")
                #base = []
                #for model in models:
                #    X = df[FEATURE_COLS].to_numpy()
                #    preds = model['model'].predict_proba(X)
                #    df['prob'] = preds[:,1]
                #    df = df.sort_values("prob", ascending=False)
                #    df['pred'] = [1]*math.floor(len(df)*0.3) + [0]*(len(df) - math.floor(len(df)*0.3))
                #    new_group = df[df["code"].isin(races[race])]
    #                 X = new_group[FEATURE_COLS].to_numpy()
    #                 preds = model['model'].predict_proba(X)
    #                 new_group['prob'] = preds[:,1]
    #                 new_group = new_group.sort_values("prob", ascending=False)
    #                 new_group['pred'] = [1]*math.floor(len(new_group)*0.3) + [0]*(len(new_group) - math.floor(len(new_group)*0.3))

                #    positive_pred = new_group[new_group["pred"] == 1]
                #    fdr = np.sum(np.array(positive_pred["pred"]-positive_pred["label_x"]))/len(new_group)
                #    tpr = np.sum(1-np.array(positive_pred["pred"]-positive_pred["label_x"]))/len(new_group)
                #    precision = precision_score(df["label_x"], df["pred"])
                #    base.append({"precision": precision, "fdr": fdr, "tpr": tpr, "model_name": str(model['model']), "race": race})
                #    print("got base model {}".format(model))
                #base = pd.DataFrame(base)
                #base.to_csv("white1.csv") 
                print("got base")
                
            #elif race == 'hispanic':
            #    results = pd.read_csv("other_races.csv")
            #    plot(results, 'hispanic')
            #    print("got hispanic")
            else:
                results = []
                for model in models:
                    X = df[FEATURE_COLS].to_numpy()
                    preds = model['model'].predict_proba(X)
                    df['prob'] = preds[:,1]
                    df = df.sort_values("prob", ascending=False)
                    df['pred'] = [1]*math.floor(len(df)*0.3) + [0]*(len(df) - math.floor(len(df)*0.3))
                    new_group = df[df["code"].isin(races[race])]

                    positive_pred = new_group[new_group["pred"] == 1]
                    fdr = float(np.sum(np.abs(np.array(positive_pred["pred"]-positive_pred["label_x"])))/len(new_group))
                    tpr = float(np.sum(1-np.abs(np.array(positive_pred["pred"]-positive_pred["label_x"])))/len(new_group))
                    precision = precision_score(df["label_x"], df["pred"])
                    base_model = base[base["model_name"] == str(model['model'])]
                    base_fdr = base_model["fdr"].values[0]
                    base_tpr = base_model["tpr"].values[0]
                    results.append({"precision": precision, "fdr": fdr/base_fdr, "tpr": tpr/base_tpr, "model_name": str(model['model']), "race": race})


                results = pd.DataFrame(results)
                results.to_csv("{}1.csv".format(race))
                plot(results, str(race))
                print("got plot {}".format(str(race)))

    get_plots(val1, models1, house_races1, senate_races1)
