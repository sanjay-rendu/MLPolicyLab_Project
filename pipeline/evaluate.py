import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt, File_IO
from daggit.core.base.factory import BaseOperator
from sklearn.metrics import accuracy_score
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

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


class topk_metric(BaseOperator):

    @property
    def inputs(self):
        return {
            "predictions": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "metrics": File_Txt(self.node.outputs[0])
        }

    def topk(self, result, k=.3, colnames=None, metric='precision'):
        """ Returns the metric of the top k% of bills
        args:
            result: pandas.dataframe, csv with predicted labels, score, and true labels. Bill passed should be labeled with 1.
            k: float, decimal of top scores to check
                default: .3
            colnames: dict, used to specify column name for each feature of interest.
                format: {'label': **colname**, 'pred': **colname**, 'score': **colname**}
                default: {'label': 'label', 'pred': 'pred', 'score': 'score'}
            metric: str, either 'precision', 'recall', or 'both'
                default: 'precision'
        returns:
            precision or recall score. if both, then returns a tuple of (precision, recall)
        """
        if colnames is None:
            colnames = {'label': 'label', 'pred': 'pred', 'score': 'score'}
        result = result.sort_values(by=[colnames['score']], ascending=False)
        df_len = len(result.index)
        top_k = result.head(math.floor(df_len * k))
        labels = top_k[colnames['label']].tolist()
        preds = top_k[colnames['pred']].tolist()

        if metric == 'precision':
            return precision_score(labels, preds)
        elif metric == 'recall':
            return recall_score(labels, preds)
        else:
            return (precision_score(labels, preds), recall_score(labels, preds))

    def plot_prk(self, precisions, recalls, model_name, graph_loc):

        # graph_loc is folder where the graphs are writtten

        assert len(precisions) == len(recalls)
        x = np.linspace(0, 1, len(precisions))
        plt.plot(x, precisions)
        plt.plot(x, recalls)
        plt.legend(['Precision', 'Recall'])
        plt.xlabel('Percent of Total Bills')
        plt.title('PR-k of model {}'.format(model_name))
        plt.savefig('{}/prk_graph_{}.png'.format(graph_loc, model_name))

    def run(self, target, threshold, graph_loc):
        result = self.inputs["predictions"].read()
        precision = []
        recall = []
        for k in range(10):
            temp = self.topk(result, k=k / 10, metric='both')
            precision.append(temp[0])
            recall.append(temp[1])

        self.plot_prk(precision, recall, 'test', graph_loc)


