import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

def topk(result, k=.3, colnames=None, metric='precision'):
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
    top_k = result.head(math.floor(df_len*k))
    labels = top_k[colnames['label']].tolist()
    preds = top_k[colnames['pred']].tolist()

    if metric == 'precision':
        return precision_score(labels, preds)
    elif metric == 'recall':
        return recall_score(labels, preds)
    else:
        return (precision_score(labels,preds), recall_score(labels,preds))

def plot_prk(precisions, recalls, model_name):
    assert len(precisions) == len(recalls)
    x = np.linspace(0, 1, len(precisions))
    plt.plot(x, precisions)
    plt.plot(x, recalls)
    plt.legend(['Precision', 'Recall'])
    plt.xlabel('Percent of Total Bills')
    plt.title('PR-k of model {}'.format(model_name))
    plt.savefig('../plots/prk_graph_{}.png'.format(model_name))

def test(result_file):
    result = pd.read_csv(result_file)
    precision = []
    recall = []
    for k in range(10):
        temp = topk(result, k=k/10, metric='both')
        precision.append(temp[0])
        recall.append(temp[1])

    plot_prk(precision, recall, 'test')

if __name__ == '__main__':
    test('../../data/test_results.csv')

