from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator

from airflow.hooks.postgres_hook import PostgresHook


class fetch_sql(BaseOperator):

    @property
    def inputs(self):
        return {"sql_loc": self.node.inputs[0].external_ref}

    @property
    def outputs(self):
        return {"dataframe": Pandas_Dataframe(self.node.outputs[0])}

    def run(self, conn_id='postgres_bills3'):
        """
        Fetches data from poostgres schema defined as airflow hook
        :param conn_id: schema to fetch from
        :return: dataframe containg results of sql query
        """
        sql_loc = self.inputs["sql_loc"]

        pg_hook = PostgresHook(postgres_conn_id=conn_id)
        data_out = pg_hook.get_pandas_df(open(sql_loc, "r").read())

        self.outputs["dataframe"].write(data_out)