import pandas as pd
import numpy as np
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator


class get_labelled_data(BaseOperator):

    @property
    def inputs(self):
        return {
            "raw": Pandas_Dataframe(self.node.inputs[0])
            "progress": Pandas_Dataframe(self.node.inputs[0])
        }

    @property
    def outputs(self):
        return {
            "labelled": Pandas_Dataframe(self.node.outputs[0])
        }

    def run(self, n, out_path):
        """
            Computes labels for bills which are passed within a time frame and updates dataframe.
            Arguments:
                n: int
                    number of months for time frame
                out_path: str
                    Path where final features are saved
            Returns:
                updated pandas dataframe with labels
        """
        df = self.inputs["progress"].read()

        # convert dates from string to datetime object
        df['progress_date'] = pd.to_datetime(df['progress_date'], format='%Y-%m-%d')

        # get introduced and passed bills
        passed = df.loc[df['bill_status'] == 4]
        introduced = df.loc[(df['bill_id'].isin(passed['bill_id'])) & (df['bill_status'] == 1)]
        progress = pd.merge(introduced, passed, how='inner', on=['bill_id'])

        # set label for bills passed within 30*n days
        progress['days_between'] = (progress['progress_date_y'] - progress['progress_date_x']).dt.days
        progress = progress.loc[progress['days_between'] <= n*30]
        progress = progress.drop(columns=['progress_date_x',  'bill_status_x', 'progress_date_y',  'bill_status_y', 'days_between'])
        progress['final_status'] = 1

        # update final_status of bills
        df = self.inputs["raw"].read()
        df['final_status'] = 0
        df.update(progress)

        df.to_csv(out_path)

        return df