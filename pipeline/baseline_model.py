import pandas as pd

def baserate(test):
    ''' Predicts that every bill will pass. Defines the baserate of bill passage.
    returns:
        preds: list of predictions
        score: list of scores
    '''
    preds = [1]*len(test.index)
    score = [1]*len(test.index)
    return preds, score

def common_sense(train, colnames={'dem': 'num_dem_sponsors', 'repub': 'num_repub_sponsors'}):
    ''' Score is # dem sponsors - # repub sponsors
    args:
        test: pandas.dataframe
        colnames: column names for # dem and #republican sponsors
            format: {'dem': **col_name**, 'repub': **col_name**} 
            default: {'dem': 'num_dem_sponsors', 'repub': 'num_repub_sponsors'}
    '''
    score = (train[colnames['dem']]-train[colnames['repub']]).tolist()
    preds = [x > 0 for x in score]
    return preds, score
    
