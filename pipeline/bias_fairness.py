import pandas as pd, numpy as np, matplotlib.pyplot as plt, re, math, pickle, os
from sklearn.metrics import precision_score
from daggit.core.io.io import Pandas_Dataframe, File_Txt, ReadDaggitTask_Folderpath
from daggit.core.base.factory import BaseOperator

class fairness_extract(BaseOperator):
    @property
    def inputs(self):
        return {"features": Pandas_Dataframe(self.node.inputs[0]),
                "split": Pandas_Dataframe(self.node.inputs[1]),
                "modelgrids": [os.path.dirname(ReadDaggitTask_Folderpath(x).read_loc())
                               for x in self.node.inputs][2:]}


    @property
    def outputs(self):
        return {"vals": Pandas_Dataframe(self.node.outputs[0])}

    
    def baseline(self, data):
        return data["number_dems"] - data["number_republicans"]

    def plot(self, data, group_name):
        plt.scatter("precision", "fdr", data = data)
        plt.axvline(x=0.022)
        plt.xlabel("Precision at top 30%")
        plt.ylabel("False Discovery Rate Disparity")
        plt.title("FDR for {}".format(group_name))
        plt.savefig("{}_fdr.png".format(group_name))
        plt.close()

        plt.scatter("precision", "tpr", data=data)
        plt.axvline(x=0.022)
        plt.xlabel("Precision at top 30%")
        plt.ylabel("True Positive Rate Disparity")
        plt.title("TPR for {}".format(group_name))
        plt.savefig("{}_tpr.png".format(group_name))
        plt.close()
    
    def parse_district(self, s):
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
    
    def get_preds(self, races, df, model, FEATURE_COLS):
        results = []
        for race in sorted(races.keys(), reverse=True):
            if race == 'white':
                
                X = df[FEATURE_COLS].to_numpy()
                preds = model['model'].predict_proba(X)
                df['prob'] = preds[:,1]
                df = df.sort_values("prob", ascending=False)
                df['pred'] = [1]*math.floor(len(df)*0.3) + [0]*(len(df) - math.floor(len(df)*0.3))
                new_group = df[df["code"].isin(races[race])]
                X = new_group[FEATURE_COLS].to_numpy()
                preds = model['model'].predict_proba(X)
                new_group['prob'] = preds[:,1]
                new_group = new_group.sort_values("prob", ascending=False)
                new_group['pred'] = [1]*math.floor(len(new_group)*0.3) + [0]*(len(new_group) - math.floor(len(new_group)*0.3))

                positive_pred = new_group[new_group["pred"] == 1]
                fdr = np.sum(np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                tpr = np.sum(1-np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                precision = precision_score(df["label"], df["pred"])
                base_model = {"precision": precision, "fdr": fdr, "tpr": tpr, "model_name": str(model['model']), "race": race}
            else:
                X = df[FEATURE_COLS].to_numpy()
                preds = model['model'].predict_proba(X)
                df['prob'] = preds[:,1]
                df = df.sort_values("prob", ascending=False)
                df['pred'] = [1]*math.floor(len(df)*0.3) + [0]*(len(df) - math.floor(len(df)*0.3))
                new_group = df[df["code"].isin(races[race])]

                positive_pred = new_group[new_group["pred"] == 1]
                fdr = float(np.sum(np.abs(np.array(positive_pred["pred"]-positive_pred["label"])))/len(new_group))
                tpr = float(np.sum(1-np.abs(np.array(positive_pred["pred"]-positive_pred["label"])))/len(new_group))
                precision = precision_score(df["label"], df["pred"])
                base_fdr = base_model["fdr"]
                base_tpr = base_model["tpr"]
                results.append({"precision": precision, "fdr": fdr/base_fdr, "tpr": tpr/base_tpr, "model_name": str(model['model']), "race": race})

        return base_model, results

    def get_baseline(self, races, df, FEATURE_COLS):
        results = []
        for race in sorted(races.keys(), reverse=True):
            if race == 'white':
                df['prob'] = self.baseline(df[FEATURE_COLS])
                df = df.sort_values("prob", ascending=False)
                df['pred'] = [1]*math.floor(len(df)*.3) + [0]*(len(df) - math.floor(len(df)*.3))
                new_group = df[df["code"].isin(races[race])]
                positive_pred = new_group[new_group["pred"] == 1]
                fdr = np.sum(np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                tpr = np.sum(1-np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                precision = precision_score(df["label"], df["pred"])
                base_model = {"precision": precision, "fdr": fdr, "tpr": tpr, "model_name": "baseline", "race": race}
            else:
                df['prob'] = self.baseline(df[FEATURE_COLS])
                df = df.sort_values("prob", ascending=False)
                df['pred'] = [1]*math.floor(len(df)*.3) + [0]*(len(df) - math.floor(len(df)*.3))
                new_group = df[df["code"].isin(races[race])]
                positive_pred = new_group[new_group["pred"] == 1]
                fdr = np.sum(np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                tpr = np.sum(1-np.array(positive_pred["pred"]-positive_pred["label"]))/len(new_group)
                precision = precision_score(df["label"], df["pred"]) 
                base_fdr = base_model["fdr"]
                base_tpr = base_model["tpr"]
                results.append({"precision": precision, "fdr": fdr/base_fdr, "tpr": tpr/base_tpr, "model_name": "baseline", "race": race})
        return results

    def get_plots(self, df, models, house_races, senate_races):
                
                results = pd.DataFrame(results)
                results.to_csv("{}4.csv".format(race))
                self.plot(results, str(race))
                print("got plot {}".format(str(race)))
    
    def run(self):
        """
        Calculates FDR or TPR for a group of interest. Groups is a dictionary with groups of interest and associated district codes.
        """
        features4 = self.inputs["features"].read()
        split4 = self.inputs["split"].read()[["bill_id", "primary_sponsor_district"]]
        val4 = features4.merge(split4, on = "bill_id", how = "left").dropna(subset=["primary_sponsor_district"])
        district_df = val4["primary_sponsor_district"].apply(self.parse_district)
        val4["body_in"] = district_df["body_in"]
        val4["code"] = district_df["code"]
        df = val4

        model_grids = self.inputs["modelgrids"]
        
        ## district codes derived from manual EDA
        house_races = {"hispanic": [6, 18, 71, 72, 115, 129, 137, 148],
                        "asian": [9, 11, 18, 29, 32, 55, 56, 61, 69, 71, 74, 79, 100, 104, 108, 128, 137],
                        "black": [6, 11, 18, 30, 39, 71, 102, 115, 117, 120, 128, 129],
                        "white": [1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 73, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 116, 118, 121, 122, 123, 124, 125, 126, 127, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 150]}
        senate_races = {"hispanic": [4, 8, 30, 32, 45],
                         "asian": [8, 14, 25, 30, 33],
                         "black": [4, 8, 45, 48, 50, 53],
                         "white": [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 31,
                                   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 49, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62]}
        results = []
        races = {}
        FEATURE_COLS = ['days_from_introduction', 'number_dems', 'number_republicans',
                       'is_bipartisan', 'day_from_month_start','topic_0', 'topic_1', 'topic_2','topic_3', 'topic_4', 'topic_5','topic_6', 'topic_7',
                        'topic_8', 'topic_9', 'introduced_body_A', 'introduced_body_S']
        for race in house_races:
            races[race] = list(set(house_races[race] + senate_races[race]))
        df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric)
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
        
        for grid in model_grids:
            for filename in os.listdir(grid):
                if filename.endswith(".pkl"):
                    print(grid + "/" + filename)
                    model = pickle.load(open(grid+"/"+filename, "rb"))
                    _, result = self.get_preds(races, df, model, FEATURE_COLS)
                    results += result

        ## get baseline
        results += self.get_baseline(races, df, FEATURE_COLS)
        results = pd.DataFrame(results)
        self.outputs["vals"].write(results)
        for race in house_races:
            df = results[results['race'] == race]
            self.plot(df, str(race))
            print("got plot {}".format(str(race)))

if __name__ == "__main__":
    """
    Calculates FDR or TPR for a group of interest. Groups is a dictionary with groups of interest and associated district codes.
    """



    black = pd.read_csv("black4.csv")
    plot(black, "black")
    asian = pd.read_csv("asian4.csv")
    plot(asian, "asian")
    hispanic = pd.read_csv("hispanic4.csv")
    plot(hispanic, "hispanic")

    print("done")
    features4 = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/preprocess4_bf/preprocessed_test.csv")
    print(features4["bill_id"])
    split4 = pd.read_csv("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/split4/val.csv")[["bill_id",
                                                                                                               "primary_sponsor_district"]]
    print(split4.dtypes)
    print(val4.columns)

    base = []
    print("got data")
    
    lr_dir = "/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/lr_grid4"
    lr_list = []
    for filename in os.listdir(lr_dir):
        if filename.endswith(".pkl"):
            try:
                lr_list.append(pickle.load(open(lr_dir+"/"+filename, "rb")))
            except:
                print(filename)
    dt_list = []
    dt_dir = "/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/dt_grid4"
    for filename in os.listdir(dt_dir):
        if filename.endswith(".pkl"):
            try:
                dt_list.append(pickle.load(open(dt_dir+"/"+filename, "rb")))
            except:
                print(filename)

    rf_dir = r'/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/rf_grid4'
    rf_list = []
    for filename in os.listdir(rf_dir):
        if filename.endswith(".pkl"):
            try:
                rf_list.append(pickle.load(open(rf_dir+"/"+filename, "rb")))
            except:
                print(filename)

    models4 = lr_list+dt_list+rf_list

