import os
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, ReadDaggitTask_Folderpath
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
            "train": Pandas_Dataframe(self.node.inputs[0]),
        }

    @property
    def outputs(self):
        return {
            "models": ReadDaggitTask_Folderpath(self.node.outputs[0])
        }

    def run(self, target, model_spec):
        df = self.inputs["train"].read()
        
        features = list(set(list(df.columns)) - {target})

        X = df.as_matrix(columns=features)
        y = df.as_matrix(columns=[target])

        mod_name, func_name = model_spec['model_name'].rsplit('.',1)
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        del model_spec['model_name']

        params_list = [dict(zip(model_spec, t)) for t in zip(*model_spec.values())]

        list_of_models = []
        for params in params_list:
            clf = func()
            clf.set_params(**params)
            clf.fit(X, y)
            params['model'] = clf
            list_of_models.append(params)

        model_dir = self.outputs["models"].read_loc()

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        save_file = model_dir + '{}.joblib'.format(func_name)
        dump(list_of_models, save_file)
