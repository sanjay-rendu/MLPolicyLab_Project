from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator
from datetime import datetime, timedelta
import pandas as pd

from sklearn.model_selection import train_test_split

class split(BaseOperator):

    @property
    def inputs(self):
        return {"all_data": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {
            "train_data": Pandas_Dataframe(self.node.outputs[0]),
            "val_data": Pandas_Dataframe(self.node.outputs[1]),
            "test_data": Pandas_Dataframe(self.node.outputs[2])
        }

    def run(self, val_size, test_size):
        """
        Splits data randomly into train, val, test
        :param val_size: fraction of val data
        :param test_size: fraction of test data
        :return: [train, val, test] data
        """
        all_data = self.inputs["all_data"].read()

        train_data, test_data = train_test_split(all_data, test_size=test_size)
        train_data, val_data = train_test_split(train_data, test_size=val_size)

        self.outputs["train_data"].write(train_data)
        self.outputs["val_data"].write(val_data)
        self.outputs["test_data"].write(test_data)

class timeSplit(BaseOperator):

    @property
    def inputs(self):
        return {"all_data": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {
            "train_data": Pandas_Dataframe(self.node.outputs[0]),
            "val_data": Pandas_Dataframe(self.node.outputs[1])
        }

    def run(self, prediction_date, validation_end_date ,label_window=180):
        """
        Splits data in training and validation splits based on the date of prediction
        :param prediction_date: date in %Y-%m-%d format
        :param validation_window: number of days in the future we intend to predict
        :return: [train, val] data
        """
        all_data = self.inputs["all_data"].read()
        all_data.present_date = pd.to_datetime(all_data.present_date, errors='coerce')

        present_date = datetime.strptime(prediction_date, "%Y-%m-%d")
        label_date = present_date + timedelta(days=label_window)
        validation_date = datetime.strptime(validation_end_date, "%Y-%m-%d")

        train_data = all_data[all_data.present_date < present_date]
        validation_data = all_data[(all_data.present_date >= label_date) &
                                   (all_data.present_date < validation_date)]
        self.outputs["train_data"].write(train_data)
        self.outputs["val_data"].write(validation_data)
