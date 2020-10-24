import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score

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

def test(result_file):
    print(pd.read_csv(result_file))

if __name__ == '__main__':
    test('../data/test_results.csv')

