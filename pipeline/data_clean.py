import pandas as pd
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator
import numpy as np

class feature_eng(BaseOperator):

    @property
    def inputs(self):
        return {"raw": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"df": Pandas_Dataframe(self.node.outputs[0])}

    def run(self, index, ohe_features, summed_features, other_features, date):
        """
        Engineers features out of raw data. Saves and returns final dataframe.
            Arguments:
                index: str
                    Index features (bill_id)
                date: str
                    String argument for date of bill
                ohe_features: list[str]
                    Features to be one hot encoded such as introduced_body and bill_type
                summed_features: list[str]
                    Features to be summed such as role_name and party_id
                other_features: list[str]
                    All other features of interest to be kept as is
            Returns:
                pandas dataframe of features
        """
        df = self.inputs["raw"].read()
        ## df = df.fillna(0)
        ## convert party_id to string for OHE
        for feature in ohe_features:
            df[feature] = df[feature].apply(str)
        ohe1_df = pd.get_dummies(df[[index] + ohe_features]).drop_duplicates(index)

        ## convert to OHE for adding (# reps/senators + # party members)
        ohe_df = pd.get_dummies(df[[index] + summed_features])
        ohe_df = ohe_df.groupby([index]).sum()
        ohe_df = ohe_df.join(ohe1_df.set_index(index))

        ## join with original data
        df = df[[index] + [date]+ other_features].drop_duplicates(index)
        df = ohe_df.join(df.set_index(index), on=index)

        ## sort by date and convert to numeric
        df[date] = pd.to_datetime(df[date])
        df = df.sort_values(date)
        df['introduced_year'] = df[date].dt.year
        df['introduced_month'] = df[date].dt.month
        df['introduced_day'] = df[date].dt.day
        df = df.drop(date, axis=1)
        df.head(1) #check

        self.outputs["df"].write(df)

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
            ["introduced_date", "final_date", "present_date"]].apply(pd.to_datetime)
        dfnew['original_date'] = pd.to_datetime("'2009-01-07'".replace("'", ""))

        # prepare df for filtering on every 7th day plus the final day for the bill or session
        conditions = [(dfnew['present_date'] == dfnew['final_date'])]
        choices = [30]

        dfnew['day_from_week_start'] = np.select(conditions, choices,
                                                 default=(dfnew['present_date'] - dfnew['original_date']).dt.days % 30)

        # filter
        d = [0, 30]
        final_df = dfnew.loc[dfnew['day_from_week_start'].isin(d)]

        self.outputs["filtered_data"].write(final_df)