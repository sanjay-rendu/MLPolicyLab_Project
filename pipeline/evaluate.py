import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt, ReadDaggitTask_Folderpath
from daggit.core.base.factory import BaseOperator
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import pickle
import os

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

        y = df[target].to_numpy()
        X = df.drop(target, axis=1).to_numpy()

        y_prob = model.predict_proba(X)[:, 1]

        output = pd.DataFrame(list(zip(list(y),y_prob)), columns=['label', 'score'])

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
            "topk_predictions": Pandas_Dataframe(self.node.outputs[1]),
            "result_df": Pandas_Dataframe(self.node.outputs[2])
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
        result = result.sort_values(by='score', ascending=False)
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

    def plot_prk(self, precisions, recalls, graph_name):

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
        fig.savefig(os.path.join(os.path.dirname(self.inputs["predictions"].data_location),"{}.png".format(graph_name)))

    def run(self, target, graph_name):
        result = self.inputs["predictions"].read()
        precision = []
        recall = []

        result_df = pd.DataFrame(columns=['model', 'k', 'precision', 'recall'])
        for k in range(1, 101):
            temp, df_preds = self.topk(result, k=k / 100, metric='both')
            precision.append(temp[0])
            recall.append(temp[1])
            result_df = result_df.append({'model_rank': 0, 'model': graph_name, 'k': k,
                                    'precision': temp[0], 'recall': temp[1]}, ignore_index=True)

        self.plot_prk(precision, recall, graph_name)

        results, df_preds = self.topk(result, k=.3, metric='both')
        precision_30, recall_30 = results



        self.outputs['metrics'].write("Precision @ 30%: {} \nRecall @ 30%: {}".format(precision_30, recall_30))
        self.outputs['topk_predictions'].write(df_preds)
        self.outputs['result_df'].write(result_df)


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
            "result": Pandas_Dataframe(self.node.outputs[0]),
            "top_models": Pickle_Obj(self.node.outputs[1])
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
        
        result = result.sort_values(by='score', ascending=False)
        df_len = len(result.index)
        preds = [1] * math.floor(df_len * k)
        preds += [0] * (df_len - math.floor(df_len * k))
        labels = result[colnames['label']].tolist()

        if metric == 'precision':
            return precision_score(labels, preds), result
        elif metric == 'recall':
            return recall_score(labels, preds), result
        else:
            return (precision_score(labels, preds), recall_score(labels, preds)), result


    def run(self, target, n_splits=4, top_n=2):

        idx_list = ['2011-07-01', '2013-07-01', '2015-07-01', '2017-07-01']
        #idx_list = ['2011', '2013', '2015', '2017']
        result = pd.DataFrame(columns = ['split', 'model', 'config', 'precision'])

        precisions = []
        models = []
        for split in range(1, n_splits+1):
            df = self.inputs["data"+str(split)].read()
            model_dir = os.path.dirname(self.inputs["models"+str(split)].read_loc())

            print(df.columns.tolist())
            directory = os.fsencode(model_dir)
            y = df[target].to_numpy()
            X = df.drop(target, axis=1).to_numpy()

            for file in sorted(os.listdir(directory)):
                filename = str(os.fsdecode(file))

                if filename.endswith(".pkl"):
                    with open(os.path.join(str(model_dir), filename), 'rb') as handle:
                        model_list = pickle.load(handle)

                    for clf in model_list:
                        y_prob = clf['model'].predict_proba(X)[:, 1]

                        res = pd.DataFrame(list(zip(list(y),y_prob)), columns=['label', 'score'])
                        temp, df_preds = self.topk(res, k=0.3, metric='precision')

                        result = result.append({'split': idx_list[split-1], 'model': filename[:-4],
                                                'config': str(clf['model']), 'precision': temp}, ignore_index = True)
                        if split == n_splits:
                            precisions.append(temp)
                            models.append(clf['model'])
        
            score = self.baserate(df)
            score1 = self.common_sense(df)

            baserate = pd.DataFrame(list(zip(list(y), score)), columns=['label', 'score'])
            common_sense = pd.DataFrame(list(zip(list(y), score1)), columns=['label', 'score'])

            temp, df_preds = self.topk(baserate, k=0.3, metric='precision')
            result = result.append({'split': idx_list[split-1], 'model': 'baseline', 'config': 'NA','precision': temp},
                ignore_index = True)

            temp, df_preds = self.topk(common_sense, k=0.3, metric='precision')
            result = result.append({'split': idx_list[split-1], 'model': 'commonsense', 'config': 'NA',
                                    'precision': temp}, ignore_index = True)

            top_models = [x[1] for x in sorted(zip(precisions, models), key=lambda x: x[0], reverse=True)][:top_n]

        self.outputs['result'].write(result)
        self.outputs['top_models'].write(top_models)




class plot_grid_results(BaseOperator):

    @property
    def inputs(self):
        return {
            "result": Pandas_Dataframe(self.node.inputs[0]),
        }

    @property
    def outputs(self):
        return {
            "result": Pandas_Dataframe(self.node.inputs[0])
        }

    def plot_best_only(result, prec, num_best):
        styles = ['b-'] * 3
        styles += ['k--'] + ['k:']

        prec_best = np.zeros((num_best+2, 4))
        idx = np.argsort(prec[:,num_best])[-num_best:]
        prec_best[:num_best, :] = prec[idx]

        prec_best[num_best, 0] = result[(result['split'] == '2011-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[num_best, 1] = result[(result['split'] == '2013-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[num_best, 2] = result[(result['split'] == '2015-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[num_best, 3] = result[(result['split'] == '2017-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[num_best+1, 0] = result[(result['split'] == '2011-07-01') & (result['model'] == 'commonsense')]['precision']
        prec_best[num_best+1, 1] = result[(result['split'] == '2013-07-01') & (result['model'] == 'commonsense')]['precision']
        prec_best[num_best+1, 2] = result[(result['split'] == '2015-07-01') & (result['model'] == 'commonsense')]['precision']
        prec_best[num_best+1, 3] = result[(result['split'] == '2017-07-01') & (result['model'] == 'commonsense')]['precision']

        return prec_best, styles_best

    def run(self, save_path, num_best=3):
        result = self.inputs["result"].read()

        idx_list = ['2011-07-01', '2013-07-01', '2015-07-01', '2017-07-01']

        styles = ['r-']*(len(result[result['model'] == 'DecisionTreeClassifier'])//4)
        styles += ['b-']*(len(result[result['model'] == 'LogisticRegression'])//4)
        styles += ['k--'] + ['k:']

        prec = np.zeros((len(result)//4, 4))
        prec[:,0] = result[result['split'] == '2011-07-01']['precision']
        prec[:,1] = result[result['split'] == '2013-07-01']['precision']
        prec[:,2] = result[result['split'] == '2015-07-01']['precision']
        prec[:,3] = result[result['split'] == '2017-07-01']['precision']
        """
        prec_best = np.zeros((5, 4))
        idx = np.argsort(y[:,3])[-3:]
        prec_best[:3, :] = y[idx]
        prec_best[3, 0] = result[(result['split'] == '2011-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[3, 1] = result[(result['split'] == '2013-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[3, 2] = result[(result['split'] == '2015-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[3, 3] = result[(result['split'] == '2017-07-01') & (result['model'] == 'baseline')]['precision']
        prec_best[4, 0] = result[(result['split'] == '2011-07-01') & (result['model'] == 'commonsense')]['precision']
        prec_best[4, 1] = result[(result['split'] == '2013-07-01') & (result['model'] == 'commonsense')]['precision']
        prec_best[4, 2] = result[(result['split'] == '2015-07-01') & (result['model'] == 'commonsense')]['precision']
        prec_best[4, 3] = result[(result['split'] == '2017-07-01') & (result['model'] == 'commonsense')]['precision']
        """

        fig, ax = plt.subplots(1, figsize=(12, 5))
        for i, style in enumerate(styles):
            ax.plot(idx_list, prec[i,:], style)
        ax.set_title('Precision@30% Over Time')
        plt.savefig('model_grid.png')

        prec_best, styles_best = self.plot_best_only(result, prec, num_best)

        fig, ax = plt.subplots(1, figsize=(12, 5))
        for i, style in enumerate(styles_best):
            ax.plot(idx_list, prec_best[i,:], style)
        ax.set_title('Precision@30% Over Time')
        plt.savefig('model_grid_best.png')

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


class top_prk(BaseOperator):

    @property
    def inputs(self):
        return {
            "data": Pandas_Dataframe(self.node.inputs[0]),
            "models": Pickle_Obj(self.node.inputs[1]),
            "baseline_prk": Pandas_Dataframe(self.node.inputs[2]),
            "commonsense_prk": Pandas_Dataframe(self.node.inputs[3])
        }

    @property
    def outputs(self):
        return {
            "prk_out": Pandas_Dataframe(self.node.outputs[0])
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

    def plot_prk(self, precisions, recalls, graph_name):

        fig, ax = plt.subplots()

        assert len(precisions) == len(recalls)
        x = np.linspace(0, 1, len(precisions))
        ax.plot(x, precisions, color="red")
        ax.set_xlabel('Percent of Total Bills')
        ax.set_ylabel("Precision", color="red")
        ax.set_title('PR-k of model')
        ax.set_ylim(0, 0.2)

        ax2 = ax.twinx()
        ax2.plot(x, recalls,color="blue")
        ax2.set_ylabel("Recall", color="blue")
        ax2.set_ylim(0, 1)
        fig.savefig(os.path.join(os.path.dirname(self.inputs["data"].data_location),"{}.png".format(graph_name)))

    def run(self, target):
        df = self.inputs["data"].read()
        models = self.inputs["models"].read()
        baseline = self.inputs["baseline_prk"].read()
        commonsense = self.inputs["commonsense_prk"].read()

        y = df[target].to_numpy()
        X = df.drop(target, axis=1).to_numpy()

        result = pd.DataFrame(columns=['model', 'k', 'precision', 'recall'])
        i = 1
        for model in models:
            y_prob = model.predict_proba(X)[:, 1]
            output = pd.DataFrame(list(zip(list(y), y_prob)), columns=['label', 'score'])
            precisions = []
            recalls = []
            for k in range(1, 101):
                temp, df_preds = self.topk(output, k=k / 100, metric='both')
                result = result.append({'model_rank': i, 'model': str(model), 'k': k,
                                        'precision': temp[0], 'recall':temp[1]}, ignore_index=True)
                precisions.append(temp[0])
                recalls.append(temp[1])

            #self.plot_prk(precisions, recalls, 'topk_model_'+str(i))
            i += 1

        result = pd.concat([result, baseline, commonsense],ignore_index=True)

        self.outputs["prk_out"].write(result)

class plot_best_prk(BaseOperator):

    @property
    def inputs(self):
        return {
            "prk_out": Pandas_Dataframe(self.node.inputs[0]),
        }

    @property
    def outputs(self):
        return {
            "result": Pandas_Dataframe(self.node.inputs[0])
        }

    def plot_prk(self, precisions, recalls, graph_name):

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
        fig.savefig(os.path.join(os.path.dirname(self.inputs["prk_out"].data_location),"{}.png".format(graph_name)))


    def run(self, save_path):
        prk = self.inputs["prk_out"].read()

        precision1 = prk[prk['model_rank'] == 1]['precision']
        precision2 = prk[prk['model_rank'] == 2]['precision']
        precision_baseline = prk[prk['model'] == 'baseline']['precision']
        precision_commonsense = prk[prk['model'] == 'commonsense']['precision']
        recall1 = prk[prk['model_rank'] == 1]['recall']
        recall2 = prk[prk['model_rank'] == 2]['recall']
        recall_baseline = prk[prk['model'] == 'baseline']['recall']
        recall_commonsense = prk[prk['model'] == 'commonsense']['recall']

        self.plot_prk(precision1, recall1, 'topk_model_1')
        self.plot_prk(precision2, recall2, 'topk_model_2')
        self.plot_prk(precision_baseline, recall_baseline, 'topk_model_baseline')
        self.plot_prk(precision_commonsense, recall_commonsense, 'topk_model_recall')

        self.outputs['result'].write(prk)
