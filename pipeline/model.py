import numpy as np
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt
from daggit.core.base.factory import BaseOperator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF
import importlib
from joblib import dump, load

class logistic_regression_trainer(BaseOperator):

    @property
    def inputs(self):
        return {
            "train": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "model": Pickle_Obj(self.node.outputs[0])
        }

    def run(self, target, max_iter):
        df = self.inputs["train"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        model = LogisticRegression(max_iter=max_iter)
        model.fit(X, y)

        self.outputs["model"].write(model)

class rf_trainer(BaseOperator):

    @property
    def inputs(self):
        return {
            "train": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "model": Pickle_Obj(self.node.outputs[0])
        }

    def run(self, target, params):
        """ Set random forest parameters using params input as dictionary
        """
        df = self.inputs["train"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        model = RF(**params)
        model.fit(X, y)

        self.outputs["model"].write(model)

class model_grid(BaseOperator):
    """ 
    Saves trained models defined in the skeleton, returns saved file names
        inputs: 
            train: pandas.dataframe with train features and labels.
        args:
            target: str, column name containing labels.
            models: list of dicts, each dict defines a model class with the params
        returns:
            model_files: text file containing file names of all the saved models
        """
    @property
    def inputs(self):
        return {
            "train": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "model_list": Pickle_Obj(self.node.outputs[0])
        }

    def run(self, target, models, split, save_path):
        df = self.inputs["train"].read()

        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        i = 0
        save_files = []
        model_list = []
        for model in models:

            mod_name, func_name = model['model_name'].rsplit('.',1)
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            clf = func()

            #clf = model['model_name']()
            clf.set_params(**model['params'])
            clf.fit(X, y)

            save_file = save_path + 'model_split_{:d}_{:d}.joblib'.format(split, i)
            save_files.append(save_files)
            dump(clf, save_file)
            model_list.append(clf)
            i += 1

        self.outputs["model_list"].write(model_list)
