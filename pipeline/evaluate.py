import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt, File_IO
from daggit.core.base.factory import BaseOperator
from sklearn.metrics import accuracy_score
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from joblib import dump, load

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
        df = self.inputs["data"].read()
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
            preds: list of predictions
            score: list of scores
        '''
        preds = [1] * len(test.index)
        score = [1] * len(test.index)
        return preds, score

    def common_sense(self, train, colnames={'dem': 'number_dems', 'repub': 'number_republicans'}):
        ''' Score is # dem sponsors - # repub sponsors
        args:
            test: pandas.dataframe
            colnames: column names for # dem and #republican sponsors
                format: {'dem': **col_name**, 'repub': **col_name**}
                default: {'dem': 'num_dem_sponsors', 'repub': 'num_repub_sponsors'}
        '''
        score = (train[colnames['dem']] - train[colnames['repub']]).tolist()
        preds = [x > 0 for x in score]
        return preds, score

    def run(self):
        df = self.inputs["data"].read()
        preds, score = self.baserate(df)
        preds1, score1 = self.common_sense(df)

        baserate = pd.DataFrame(list(zip(list(df.label.values),preds, score)), columns=['label', 'pred', 'score'])
        common_sense = pd.DataFrame(list(zip(list(df.label.values),preds1, score1)), columns=['label', 'pred', 'score'])


        self.outputs["baserate"].write(baserate)
        self.outputs["commonsense"].write(common_sense)


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

    def run(self, target, threshold):
        df = self.inputs["data"].read()
        model = self.inputs["model"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        y_prob = model.predict(X)
        y_pred = np.array(y_prob > threshold, dtype=np.float)

        output = pd.DataFrame(list(zip(list(df[target].values),y_pred,y_prob)), columns=['label', 'pred', 'score'])

        self.outputs["prediction"].write(output)


class predict_val_grid(BaseOperator):

    @property
    def inputs(self):
        return {
            "data": Pandas_Dataframe(self.node.inputs[0]),
            "model_files": File_Txt(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "prediction": Pandas_Dataframe(self.node.outputs[0])
        }

    def run(self, target, threshold):
        df = self.inputs["data"].read()
        model_files = self.inputs["model_files"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        for m in model_files:
            clf = load(m)
            y_prob = clf.predict(X)
            y_pred = np.array(y_prob > threshold, dtype=np.float)

        output = pd.DataFrame(list(zip(list(df[target].values),y_pred,y_prob)), columns=['label', 'pred', 'score'])

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

        ax2 = ax.twinx()
        ax2.plot(x, recalls,color="blue")
        ax2.set_ylabel("Recall", color="blue")
        fig.savefig(graph_loc)

    def run(self, target, threshold, graph_loc):
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
            "model_list": Pickle_Obj(self.node.inputs[4])
        }

    @property
    def outputs(self):
        return {
            "result": Pandas_Dataframe(self.node.outputs[0])
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


    def run(self, target, threshold, save_path):
        model_list = self.inputs["model_list"].read()

        #precisions = []
        #for clf in model_list:

        idx_list = ['2011-07-01', '2013-07-01', '2015-07-01', '2017-07-01']
        result = pd.DataFrame()

        for split in [1,2,3,4]:
            df = self.inputs["data"+str(split)].read()
            
            features = list(set(list(df.columns)) - {target})
            X = df.as_matrix(columns=features)

            precisions = []
            for i in range(len(model_list)):
                save_file = save_path + 'model_split_{:d}_{:d}.joblib'.format(split, i)
                clf = load(save_file)
        
                y_prob = clf.predict(X)
                y_pred = np.array(y_prob > threshold, dtype=np.float)

                res = pd.DataFrame(list(zip(list(df[target].values),y_pred,y_prob)), columns=['label', 'pred', 'score'])

                temp, df_preds = self.topk(res, k=0.3, metric='precision')

                precisions.append(temp)

            result[idx_list[split-1]] = precisions
        
        result = result.T

        result.plot(grid=True, legend=None)
        plt.savefig('model_grid.png')

        self.outputs['result'].write(result)



class plot_grid(BaseOperator):

    @property
    def inputs(self):
        return {
            "precision1": Pickle_Obj(self.node.inputs[0]),
            "precision2": Pickle_Obj(self.node.inputs[0]),
            "precision3": Pickle_Obj(self.node.inputs[0]),
            "precision4": Pickle_Obj(self.node.inputs[0]),
            "model_list": Pickle_Obj(self.node.inputs[1])
        }

    @property
    def outputs(self):
        return {
            "result": Pandas_Dataframe(self.node.outputs[0])
        }

    def run(self):
        precision1 = self.inputs["precision1"].read()
        precision2 = self.inputs["precision2"].read()
        precision3 = self.inputs["precision3"].read()
        precision4 = self.inputs["precision4"].read()
        model_list = self.inputs["model_list"].read()

        idx_list = ['2011-07-01', '2013-07-01', '2015-07-01', '2017-07-01']

        #result = pd.DataFrame(np.arange(len(model_list)), columns=['model'])
        result = pd.DataFrame()
        result[idx_list[0]] = precision1
        result[idx_list[1]] = precision2
        result[idx_list[2]] = precision3
        result[idx_list[3]] = precision4
        result = result.T

        result.plot(grid=True, legend=None)
        plt.savefig('model_grid.png')

        self.outputs['result'].write(result)
