import os
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, ReadDaggitTask_Folderpath
from daggit.core.base.factory import BaseOperator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF
import importlib
import pickle
from distutils.dir_util import copy_tree
import itertools

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

        y = df[target].to_numpy()
        X = df.drop(target, axis=1).to_numpy()

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

        y = df[target].to_numpy()
        X = df.drop(target, axis=1).to_numpy()

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
        
        print(df.columns.tolist())

        y = df[target].to_numpy()
        X = df.drop(target, axis=1).to_numpy()

        print(X.shape)

        mod_name, func_name = model_spec['model_name'].rsplit('.',1)
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        del model_spec['model_name']

        params_list = [dict(zip(model_spec, t)) for t in list(itertools.product(*model_spec.values()))]

        #list_of_models = []
        model_dir = os.path.dirname(self.outputs["models"].read_loc())
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        for i, params in enumerate(params_list):
            params['class_weight'] = 'balanced'
            params['random_state'] = 123
            clf = func()
            clf.set_params(**params)
            clf.fit(X, y)
            params['model'] = clf

            save_file = os.path.join(model_dir, '{}{}.pkl'.format(func_name, i))
            with open(save_file, 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

class dummy_folder(BaseOperator):

    @property
    def inputs(self):
        return [os.path.dirname(ReadDaggitTask_Folderpath(x).read_loc()) for x in self.node.inputs]

    @property
    def outputs(self):
        return {
            "models_dir": os.path.dirname(ReadDaggitTask_Folderpath(self.node.outputs[0]).read_loc())
        }

    def run(self):
        dest = self.outputs["models_dir"]
        if not os.path.exists(dest):
            os.mkdir(dest)

        for src in self.inputs:
            copy_tree(src, dest)
