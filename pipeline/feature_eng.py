import pandas as pd, numpy as np
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
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "val": Pandas_Dataframe(self.node.inputs[1]),
                "train_text": Pandas_Dataframe(self.node.inputs[2]),
                "val_text": Pandas_Dataframe(self.node.inputs[3])}

    @property
    def outputs(self):
        return {"doc2vec_model": Pickle_Obj(self.node.outputs[0]),
                "train": Pandas_Dataframe(self.node.outputs[1]),
                "val": Pandas_Dataframe(self.node.outputs[2])}

    def run(self, col_names = {"bill_id": "bill_id", "doc": "doc"}, vector_size = 10, window = 2, alpha=0.7):
        """ Module for running Doc2Vec feature extraction.
        Params:
            vector_size: int, default 10
                Specifies the size of the vector produced for each document
            window: int, default 2
                Maximum distance between the current and predicted word within a sentence
            alpha: float, default 0.7
                Learning rate
        """

        bill_texts = self.inputs["train_text"].read()

        train_docs = bill_texts[col_names["doc"]].values.astype('U')
        bill_texts = bill_texts.drop("doc", axis=1)
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_docs)]
        model = Doc2Vec(documents, vector_size=vector_size, alpha=alpha, window=window,
                        min_count=1, workers=4, dm=0, dbow_words=1, seed=17)

        self.outputs["doc2vec_model"].write(model)

        train_docs = [model.infer_vector(x) for x in train_docs]
        train_docs = np.concatenate(train_docs, axis=0)
        train_docs = pd.DataFrame(train_docs, columns = ["doc2vec{}".format(i) for i in range(vector_size)],
                                  index = bill_texts.index)
        bill_texts = pd.concat([bill_texts, train_docs], axis=1, sort=False)
        train = self.inputs["train"].read()
        train = train.merge(bill_texts, on="bill_id", how = "left")
        self.outputs["train"].write(train)
        del train
        del bill_texts
        del train_docs

        bill_texts = self.inputs["val_text"].read()
        val_docs = bill_texts["doc"].values.astype('U')
        bill_texts = bill_texts.drop("doc", axis=1)
        val_docs = [model.infer_vector(x) for x in val_docs]
        val_docs = np.concatenate(val_docs, axis=0)
        val_docs = pd.DataFrame(val_docs, columns = ["doc2vec{}".format(i) for i in range(vector_size)],
                                index = bill_texts.index)
        bill_texts = pd.concat([bill_texts, val_docs], axis=1, sort=False)
        val = self.inputs["val"].read()
        val = val.merge(bill_texts, on="bill_id", how="left")
        self.outputs["val"].write(val)
        # del train_docs
        # val_docs = self.inputs["val_text"].read()["doc"].values.astype("U")
        # val_features = [model.docvecs.infer_vector(x) for x in val_docs]
        # del val_docs
        # train = self.inputs["train"].read()
        # val = self.inputs["val"].read()




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
            ignore_variables = list(ignore_variables)

        data_availability = train.describe(
            include='all').loc['count'] / len(train)
        
        print(len(train))
        print(train.describe(include='all').loc['count'])
        print(data_availability)

        all_cols = list(data_availability[data_availability >
                                          drop_missing_perc].index)
        """
        selected_cols1 = set(all_cols) - \
            (set(target_variable).union(set(ignore_variables)))

        numeric_cols1 = list(
            set(train._get_numeric_data()).intersection(selected_cols1))
        categorical_cols1 = list(selected_cols1 - set(numeric_cols1))
        """

        all_cols.remove(target_variable)
        selected_cols = [col for col in all_cols if col not in ignore_variables]

        numeric_cols = [col for col in list(train._get_numeric_data()) if col in selected_cols]
        categorical_cols = [col for col in selected_cols if col not in numeric_cols]

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
        processed_test = preprocess.transform(test)

        processed_train = (processed_train-processed_train.min())/(processed_train.max()-processed_train.min())
        processed_test = (processed_test-processed_test.min())/(processed_test.max()-processed_test.min())

        processed_train[target_variable] = train[target_variable]
        processed_test[target_variable] = test[target_variable]

        print(processed_train.columns.tolist())
        print(processed_test.columns.tolist())

        self.outputs["preprocessed_train"].write(processed_train)
        self.outputs["preprocessed_test"].write(processed_test)

if __name__=="__main__":
    import pickle
    lda = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/lda_model4/lda_model.csv", "rb"))
    count_vectorizer = pickle.load(open("/data/groups/bills3/vrenduch/DAGGIT_HOME/daggit_storage/skeleton2/lda_model4/vectorizer.csv", "rb"))
    number_words = 5
    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print_topics(lda, count_vectorizer, number_words)
