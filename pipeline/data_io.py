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


class textSplit(BaseOperator):

    @property
    def inputs(self):
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "val": Pandas_Dataframe(self.node.inputs[1])}

    @property
    def outputs(self):
        return {"txt_train": Pandas_Dataframe(self.node.inputs[0]),
                "txt_val": Pandas_Dataframe(self.node.inputs[1])}

    def run(self, conn_id='postgres_bills3'):
        """
        Returns bill text for train and val datasets
        """
        train = self.inputs["train"].read()
        val = self.inputs["val"].read()

        train_ids = [str(x) for x in set(train["bill_id"].values)]
        val_ids = [str(x) for x in set(val["bill_id"].values)]
        del train
        del val

        train_sql_query = """
        select bill_id, doc, bill_description 
        from ml_policy_class.bill_texts
        where bill_id in ({})
        """.format(", ".join(train_ids))

        val_sql_query = """
        select bill_id, doc, bill_description 
        from ml_policy_class.bill_texts
        where bill_id in ({})
        """.format(", ".join(val_ids))

        pg_hook = PostgresHook(postgres_conn_id=conn_id)
        txt_train = pg_hook.get_pandas_df(train_sql_query)
        txt_val = pg_hook.get_pandas_df(val_sql_query)

        self.outputs["txt_train"].write(txt_train)
        self.outputs["txt_val"].write(txt_val)