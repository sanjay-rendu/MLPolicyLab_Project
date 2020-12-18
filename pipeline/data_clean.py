import pandas as pd
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator
import numpy as np

class feature_selector(BaseOperator):

    @property
    def inputs(self):
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "test": Pandas_Dataframe(self.node.inputs[1])}

    @property
    def outputs(self):
        return {"filtered_train": Pandas_Dataframe(self.node.outputs[0]),
                "filtered_test": Pandas_Dataframe(self.node.outputs[1])}

    def run(self, selected_features):
        train = self.inputs["train"].read()
        train = train[selected_features]

        test = self.inputs["test"].read()
        test = test[selected_features]

        print(train.columns.tolist())
        print(test.columns.tolist())

        self.outputs["filtered_train"].write(train)
        self.outputs["filtered_test"].write(test)

class monthly_row_selector(BaseOperator):

    @property
    def inputs(self):
        return {"raw_data": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"filtered_data": Pandas_Dataframe(self.node.outputs[0])}

    def run(self):
        dfnew = self.inputs["raw_data"].read()

        # add the original start date for data
        dfnew[["introduced_date", "final_date", "present_date"]] = dfnew[
            ["introduced_date", "final_date", "present_date"]].apply(pd.to_datetime, errors='coerce')
        dfnew['original_date'] = pd.to_datetime("'2009-01-07'".replace("'", ""), errors='coerce')

        # prepare df for filtering on every 7th day plus the final day for the bill or session
        conditions = [(dfnew['present_date'] == dfnew['final_date'])]
        choices = [30]

        dfnew['day_from_month_start'] = np.select(conditions, choices,
                                                 default=(dfnew['present_date'] - dfnew['original_date']).dt.days % 30)

        # filter
        d = [0, 30]
        final_df = dfnew.loc[dfnew['day_from_month_start'].isin(d)]

        print(final_df.columns.tolist())

        self.outputs["filtered_data"].write(final_df)


class merge_distric_data(BaseOperator):

    @property
    def inputs(self):
        return {"distric_csv_loc": self.node.inputs[0].external_ref,
                "raw_data": Pandas_Dataframe(self.node.inputs[1])}

    @property
    def outputs(self):
        return {"merged_data": Pandas_Dataframe(self.node.outputs[0])}

    def run(self):
        csv_loc = self.inputs["distric_csv_loc"]
        df = self.inputs["raw_data"].read()

        district = pd.read_csv(csv_loc)

        print(df.columns.tolist())
        print(df.dtypes.tolist())
        print(district.columns.tolist())
        print(district.dtypes.tolist())

        df = df.merge(district, on='primary_sponsor_district')
        print(df.columns.tolist())

        self.outputs["merged_data"].write(df)
