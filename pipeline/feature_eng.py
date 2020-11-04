import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj
from daggit.core.base.factory import BaseOperator


class topic_model(BaseOperator):

    @property
    def inputs(self):
        return {"df": Pandas_Dataframe(self.node.inputs[0]),
                "train_ids": Pickle_Obj(self.node.inputs[1]),
                "test_ids": Pickle_Obj(self.node.inputs[2])}

    @property
    def outputs(self):
        return {"train": Pandas_Dataframe(self.node.outputs[0]),
                "val": Pandas_Dataframe(self.node.outputs[1])}

        def run(self, col_names = {"bill_ids": "bill_ids", "doc": "doc"}, num_features = 1000, num_topics = 10):
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
        train_ids = self.inputs["train_ids"].read()
        test_ids = self.inputs["test_ids"].read()
        
        train_mask = df[col_names["bill_ids"]].isin(train_ids)
        df_train = df[train_mask]
        df_test = df[~train_mask]
        train_docs = df_train[col_names["doc"]].tolist()
        test_docs = df_test[col_names["doc"]].tolist()

        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(train_docs)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(train_docs)
        tf_feature_names = tf_vectorizer.get_feature_names()

        nmf = NMF(n_components=num_topics, random_state = 1, alpha=.1, l1_ratio=.5, init="nndsvd").fit(tfidf)
        lda = LDA(n_components=num_topics, max_iter=5, learning_method="online", learning_offset=50., random_state=0).fit(tf)

        nmf_vals = nmf.transform(tfidf)
        lda_vals=lda.transform(tf)

        nmf_cols = ["nmf_{t}".format(t=t) for t in range(0,nmf_vals.shape[1])]
        lda_cols = ["topic_{t}".format(t=t) for t in range(0,lda_vals.shape[1])]
        df_train[nmf_cols] = nmf_vals
        df_train[lda_cols] = lda_vals

        tf = tf_vectorizer.transform(test_docs)
        tfidf = tfidf_vectorizer.transform(test_docs)
        df_test[nmf_cols] = nmf.transform(tfidf)
        df_test[lda_cols] = lda.transform(tf)
        self.outputs["train"].write(df)
        self.outputs["val"].write(test_df)

