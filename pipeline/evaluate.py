import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt, ReadDaggitTask_Folderpath
from daggit.core.base.factory import BaseOperator
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns


class baseline(BaseOperator):

    @property
    def inputs(self):
        return {
            "data": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "baserate": Pandas_Dataframe(self.node.outputs[0]),
            "commonsense": Pandas_Dataframe(self.node.outputs[1])
        }

    def baserate(self, test):
        ''' Predicts that every bill will pass. Defines the baserate of bill passage.
        returns:
            score: list of scores
        '''
        score = [1] * len(test.index)
        return score

    def common_sense(self, train, colnames={'dem': 'number_dems', 'repub': 'number_republicans'}):
        ''' Score is # dem sponsors - # repub sponsors
        args:
            test: pandas.dataframe
            colnames: column names for # dem and #republican sponsors
                format: {'dem': **col_name**, 'repub': **col_name**}
                default: {'dem': 'num_dem_sponsors', 'repub': 'num_repub_sponsors'}
        '''
        score = (train[colnames['dem']] - train[colnames['repub']]).tolist()
        return score

    def run(self):
        df = self.inputs["data"].read()
        score = self.baserate(df)
        score1 = self.common_sense(df)

        baserate = pd.DataFrame(list(zip(list(df.label.values),score)), columns=['label', 'score'])
        common_sense = pd.DataFrame(list(zip(list(df.label.values), score1)), columns=['label', 'score'])


        self.outputs["baserate"].write(baserate)
        self.outputs["commonsense"].write(common_sense)

class load_model(BaseOperator):

    @property
    def inputs(self):
        return {
            "train": Pandas_Dataframe(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "model": Pickle_Obj(self.node.outputs[0])
        }

    def run(self, model_path):
        m = load(model_path)

        self.outputs["model"].write(m)

class predict_val(BaseOperator):

    @property
    def inputs(self):
        return {
            "data": Pandas_Dataframe(self.node.inputs[0]),
            "model": Pickle_Obj(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "prediction": Pandas_Dataframe(self.node.outputs[0])
        }

    def run(self, target):
        df = self.inputs["data"].read()
        model = self.inputs["model"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        y_prob = model.predict_proba(X)[:, 1]

        output = pd.DataFrame(list(zip(list(df[target].values),y_prob)), columns=['label', 'score'])

        self.outputs["prediction"].write(output)


class topk_metric(BaseOperator):

    @property
    def inputs(self):
        return {
            "predictions": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "metrics": File_Txt(self.node.outputs[0]),
            "topk_predictions": Pandas_Dataframe(self.node.outputs[1])
        }

    def topk(self, result, k=.3, colnames=None, metric='precision'):
        """ Returns the metric of the top k% of bills
        args:
            result: pandas.dataframe, csv with predicted labels, score, and true labels. Bill passed should be labeled with 1.
            k: float, decimal of top scores to check
                default: .3
            colnames: dict, used to specify column name for each feature of interest.
                format: {'label': **colname**, 'score': **colname**}
                default: {'label': 'label', 'score': 'score'}
            metric: str, either 'precision', 'recall', or 'both'
                default: 'precision'
        returns:
            precision or recall score. if both, then returns a tuple of (precision, recall)
        """
        if colnames is None:
            colnames = {'label': 'label', 'score': 'score'}
        result = result.sort_values(by=[colnames['score']], ascending=False)
        df_len = len(result.index)
        preds = [1] * math.floor(df_len * k)
        preds += [0] * (df_len - math.floor(df_len * k))
        labels = result[colnames['label']].tolist()

        result['preds'] = preds

        if metric == 'precision':
            return precision_score(labels, preds), result
        elif metric == 'recall':
            return recall_score(labels, preds), result
        else:
            return (precision_score(labels, preds), recall_score(labels, preds)), result

    def plot_prk(self, precisions, recalls, graph_loc):

        fig, ax = plt.subplots()

        assert len(precisions) == len(recalls)
        x = np.linspace(0, 1, len(precisions))
        ax.plot(x, precisions, color="red")
        ax.set_xlabel('Percent of Total Bills')
        ax.set_ylabel("Precision", color="red")
        ax.set_title('PR-k of model')
        ax.set_ylim(0, 1)

        ax2 = ax.twinx()
        ax2.plot(x, recalls,color="blue")
        ax2.set_ylabel("Recall", color="blue")
        ax2.set_ylim(0, 1)
        fig.savefig(graph_loc)

    def run(self, target, graph_loc):
        result = self.inputs["predictions"].read()
        precision = []
        recall = []
        for k in range(1, 101):
            temp, df_preds = self.topk(result, k=k / 100, metric='both')
            precision.append(temp[0])
            recall.append(temp[1])

        self.plot_prk(precision, recall, graph_loc)

        results, df_preds = self.topk(result, k=.3, metric='both')
        precision_30, recall_30 = results

        self.outputs['metrics'].write("Precision @ 30%: {} \nRecall @ 30%: {}".format(precision_30, recall_30))
        self.outputs['topk_predictions'].write(df_preds)


class topk_metric_grid(BaseOperator):

    @property
    def inputs(self):
        return {
            "data1": Pandas_Dataframe(self.node.inputs[0]),
            "data2": Pandas_Dataframe(self.node.inputs[1]),
            "data3": Pandas_Dataframe(self.node.inputs[2]),
            "data4": Pandas_Dataframe(self.node.inputs[3]),
            "models1": ReadDaggitTask_Folderpath(self.node.inputs[4]),
            "models2": ReadDaggitTask_Folderpath(self.node.inputs[5]),
            "models3": ReadDaggitTask_Folderpath(self.node.inputs[6]),
            "models4": ReadDaggitTask_Folderpath(self.node.inputs[7])
        }

    @property
    def outputs(self):
        return {
            "result": Pandas_Dataframe(self.node.outputs[0])
        }

    def baserate(self, test):
        ''' Predicts that every bill will pass. Defines the baserate of bill passage.
        returns:
            score: list of scores
        '''
        score = [1] * len(test.index)
        return score

    def common_sense(self, train, colnames={'dem': 'number_dems', 'repub': 'number_republicans'}):
        ''' Score is # dem sponsors - # repub sponsors
        args:
            test: pandas.dataframe
            colnames: column names for # dem and #republican sponsors
                format: {'dem': **col_name**, 'repub': **col_name**}
                default: {'dem': 'num_dem_sponsors', 'repub': 'num_repub_sponsors'}
        '''
        score = (train[colnames['dem']] - train[colnames['repub']]).tolist()
        return score

    def topk(self, result, k=.3, colnames=None, metric='precision'):
        """ Returns the metric of the top k% of bills
        args:
            result: pandas.dataframe, csv with predicted labels, score, and true labels. Bill passed should be labeled with 1.
            k: float, decimal of top scores to check
                default: .3
            colnames: dict, used to specify column name for each feature of interest.
                format: {'label': **colname**, 'score': **colname**}
                default: {'label': 'label', 'score': 'score'}
            metric: str, either 'precision', 'recall', or 'both'
                default: 'precision'
        returns:
            precision or recall score. if both, then returns a tuple of (precision, recall)
        """
        if colnames is None:
            colnames = {'label': 'label', 'score': 'score'}
        result = result.sort_values(by=[colnames['score']], ascending=False)
        df_len = len(result.index)
        preds = [1] * math.floor(df_len * k)
        preds += [0] * (df_len - math.floor(df_len * k))
        labels = result[colnames['label']].tolist()

        result['preds'] = preds

        if metric == 'precision':
            return precision_score(labels, preds), result
        elif metric == 'recall':
            return recall_score(labels, preds), result
        else:
            return (precision_score(labels, preds), recall_score(labels, preds)), result


    def run(self, target, save_loc):

        #idx_list = ['2011-07-01', '2013-07-01', '2015-07-01', '2017-07-01']
        idx_list = ['2011', '2013', '2015', '2017']
        result = pd.DataFrame(columns = ['split', 'model', 'config', 'precision'])

        for split in [1, 2, 3, 4]:
            df = self.inputs["data"+str(split)].read()
            model_dir = os.path.dirname(self.inputs["models"+str(split)].read_loc())

            directory = os.fsencode(model_dir)
            features = list(set(list(df.columns)) - {target})
            X = df.as_matrix(columns=features)

            for file in os.listdir(directory):
                filename = str(os.fsdecode(file))

                if filename.endswith(".pkl"):
                    with open(os.path.join(str(model_dir), filename), 'rb') as handle:
                        model_list = pickle.load(handle)

                y_prob = model_list['model'].predict_proba(X)[:, 1]

                res = pd.DataFrame(list(zip(list(df[target].values),y_prob)), columns=['label', 'score'])
                temp, df_preds = self.topk(res, k=0.3, metric='precision')

                result = result.append({'split': idx_list[split-1], 'model': filename[:-4], 'config': str(model_list['model']), 'precision': temp},
                ignore_index = True)
        
            score = self.baserate(df)
            score1 = self.common_sense(df)

            baserate = pd.DataFrame(list(zip(list(df.label.values), score)), columns=['label', 'score'])
            common_sense = pd.DataFrame(list(zip(list(df.label.values), score1)), columns=['label', 'score'])

            temp, df_preds = self.topk(baserate, k=0.3, metric='precision')
            result = result.append({'split': idx_list[split-1], 'model': 'baseline', 'config': '','precision': temp},
                ignore_index = True)

            temp, df_preds = self.topk(common_sense, k=0.3, metric='precision')
            result = result.append({'split': idx_list[split-1], 'model': 'commonsense', 'config': '', 'precision': temp},
                ignore_index = True)

        fig, ax = plt.subplots(1, figsize=(12, 5))
        sns.lineplot(x='split', y='precision', data=result,
                     hue='model', units=range(result.shape[0]), estimator=None,
                     ax=ax)
        ax.set_title('Precision@30% Over Time')
        plt.savefig(save_loc)

        self.outputs['result'].write(result)



class choose_best_two(BaseOperator):

    @property
    def inputs(self):
        return {
            "result": Pandas_Dataframe(self.node.inputs[0]),
        }

    @property
    def outputs(self):
        return {
            "model1": Pickle_Obj(self.node.outputs[0]),
            "model2": Pickle_Obj(self.node.outputs[1])
        }

    def run(self, split, save_path):
        result = self.inputs["result"].read()
        result = result.T

        result['model'] = list(np.arange(len(result.index)))
        result = result.sort_values(by=[result.columns[split-1]], ascending=False)

        model1 = load(save_path + 'model_split_{:d}_{:d}.joblib'.format(split, int(result['model'][0])))
        model2 = load(save_path + 'model_split_{:d}_{:d}.joblib'.format(split, int(result['model'][1])))

        self.outputs['model1'].write(model1)
        self.outputs['model2'].write(model2)

