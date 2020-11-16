import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj
from daggit.core.base.factory import BaseOperator
from sklearn.pipeline import Pipeline
from daggit.core.oplib.etl import DFFeatureUnion, ColumnExtractor
from daggit.core.oplib.etl import DFMissingStr, DFOneHot
from daggit.core.oplib.etl import DFMissingNum


class get_lda_model(BaseOperator):
    @property
    def inputs(self):
        return {"bill_texts": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"vectorizer": Pickle_Obj(self.node.outputs[0]),
                "lda_model": Pickle_Obj(self.node.outputs[1])}

    def run(self, col_names = {"bill_id": "bill_id", "doc": "doc"}, num_features = 1000, num_topics = 10):
        """
        Engineers features out of raw text data and bill_ids for train/test splits.
        Arguments:
                col_names: dict
                    Index names for bill_id and text in dataframe.
                    default: {"bill_ids": "bill_ids", "doc": "doc"}
                num_features: int, default 1000
                    Number of terms to include in the bag of words matrix
                num_topics: int, default 10
                    Number of topics we are modeling for
            Returns:
                train and test pandas dataframe of text features
        """
        bill_texts = self.inputs["bill_texts"].read()

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(bill_texts["doc"])
        print(len(tf_vectorizer.vocabulary_))
        del bill_texts

        lda = LDA(n_components=num_topics, max_iter=5, learning_method="online", learning_offset=50., random_state=0).fit(tf)

        self.outputs['vectorizer'].write(tf_vectorizer)
        self.outputs['lda_model'].write(lda)


class topic_model(BaseOperator):

    @property
    def inputs(self):
        return {"df": Pandas_Dataframe(self.node.inputs[0]),
                "bill_texts": Pandas_Dataframe(self.node.inputs[1]),
                "vectorizer": Pickle_Obj(self.node.inputs[2]),
                "lda_model": Pickle_Obj(self.node.inputs[3])}

    @property
    def outputs(self):
        return {"df_out": Pandas_Dataframe(self.node.outputs[0])}

    def run(self, col_names = {"bill_id": "bill_id", "doc": "doc"}, num_features = 1000, num_topics = 10):
        """
        Engineers features out of raw text data and bill_ids for train/test splits.
        Arguments:
                col_names: dict
                    Index names for bill_id and text in dataframe.
                    default: {"bill_ids": "bill_ids", "doc": "doc"}
                num_features: int, default 1000
                    Number of terms to include in the bag of words matrix
                num_topics: int, default 10
                    Number of topics we are modeling for
            Returns:
                train and test pandas dataframe of text features
        """
        df = self.inputs["df"].read()
        bill_texts = self.inputs["bill_texts"].read()
        vec = self.inputs["vectorizer"].read()
        lda = self.inputs["lda_model"].read()
        bill_df = bill_texts.drop("doc", axis = 1)

        tf = vec.transform(bill_texts["doc"])
        lda_vals = lda.transform(tf)
        del bill_texts
        del tf

        lda_cols = ["topic_{t}".format(t=t) for t in range(0, lda_vals.shape[1])]
        bill_df = bill_df.reindex(columns=bill_df.columns.tolist() + lda_cols)
        bill_df[lda_cols] = lda_vals

        df = df.merge(bill_df, on="bill_id", how = "left")
        self.outputs["df_out"].write(df)

class doc2vec(BaseOperator):
    
    @property
    def inputs(self):
        return {"bill_texts": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"doc2vec_model": Pickle_Obj(self.node.outputs[0])}

    def run(self, col_names = {"bill_id": "bill_id", "doc": "doc"}, num_features = 1000, num_topics = 10):

        bill_texts = self.inputs["bill_texts"].read()

        train_docs = bill_texts[col_names["doc"]].values.astype('U')
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_docs)]
        model = Doc2Vec(documents, vector_size=20, window=2, min_count=1, workers=4)

        self.outputs["doc2vec_model"].write(model)


class CustomPreprocess(BaseOperator):

    @property
    def inputs(self):
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "test": Pandas_Dataframe(self.node.inputs[1])}

    @property
    def outputs(self):
        return {"preprocessed_train": Pandas_Dataframe(self.node.outputs[0]),
                "preprocessed_test": Pandas_Dataframe(self.node.outputs[1])}

    def run(
            self,
            drop_missing_perc,
            target_variable,
            ignore_variables=None,
            categorical_impute=None,
            numeric_impute=None):
        train = self.inputs["train"].read()
        test = self.inputs["test"].read()

        if ignore_variables is not list:
            ignore_variables = [ignore_variables]

        data_availability = train.describe(
            include='all').loc['count'] / train.shape[0]
        selected_cols = data_availability[data_availability >
                                          drop_missing_perc].index
        selected_cols = set(selected_cols) - \
            (set(target_variable).union(set(*ignore_variables)))

        numeric_cols = list(
            set(train._get_numeric_data()).intersection(selected_cols))
        categorical_cols = list(selected_cols - set(numeric_cols))

        preprocess = Pipeline([("features",
                                DFFeatureUnion([("numeric",
                                                 Pipeline([("num_sel",
                                                            ColumnExtractor(numeric_cols)),
                                                           ("num_impute",
                                                            DFMissingNum(replace=numeric_impute))])),
                                                ("categorical",
                                                 Pipeline([("cat_sel",
                                                            ColumnExtractor(categorical_cols)),
                                                           ("str_impute",
                                                            DFMissingStr(replace=categorical_impute)),
                                                           ("one_hot",
                                                            DFOneHot())]))]))])

        processed_train = preprocess.fit_transform(train)
        processed_train[target_variable] = train[target_variable]
        processed_test = preprocess.transform(test)

        self.outputs["preprocessed_train"].write(processed_train)
        self.outputs["preprocessed_test"].write(processed_test)