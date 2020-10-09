from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator

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
