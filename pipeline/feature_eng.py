import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj
from daggit.core.base.factory import BaseOperator


class get_lda_model(BaseOperator):
    @property
    def inputs(self):
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "bill_texts": Pandas_Dataframe(self.node.inputs[1])}

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
        train = self.inputs["train"].read()
        bill_texts = self.inputs["bill_texts"].read()
        train_ids = train[col_names["bill_id"]].unique()
        del train
        
        train_mask = bill_texts[col_names["bill_id"]].isin(train_ids)
        #train_docs = bill_texts[train_mask]["doc"].values.astype('U')
        #del bill_texts
        
        
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(bill_texts[train_mask]["doc"])
        print(len(tf_vectorizer.vocabulary))
        del bill_texts
        #del train_docs

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
        text_mask = bill_texts["bill_id"].isin(df["bill_id"].unique())
        bill_texts = bill_texts[text_mask]
        bill_df = bill_texts.drop("doc", axis = 1)

        tf = vec.transform(bill_texts["doc"])
        lda_vals = lda.transform(tf)
        del bill_texts
        del tf

        lda_cols = ["topic_{t}".format(t=t) for t in range(0, lda_vals.shape[1])]
        bill_df = bill_df.reindex(columns=bill_df.columns.tolist() + lda_cols)
        bill_df[lda_cols] = lda_vals

        df.merge(bill_df, on="bill_id", how = "left")
        self.outputs["df_out"].write(df)

class doc2vec(BaseOperator):
    
    @property
    def inputs(self):
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "val": Pandas_Dataframe(self.node.inputs[1]),
                "bill_texts": Pandas_Dataframe(self.node.inputs[2])}

    @property
    def outputs(self):
        return {"train": Pandas_Dataframe(self.node.outputs[0]),
                "val": Pandas_Dataframe(self.node.outputs[1])}

    def run(self, col_names = {"bill_id": "bill_id", "doc": "doc"}, num_features = 1000, num_topics = 10):
 
        train = self.inputs["train"].read()
        val = self.inputs["val"].read()
        bill_texts = self.inputs["bill_texts"].read()
        train_ids = train[col_names["bill_id"]].unique()
        
        train_mask = bill_texts[col_names["bill_id"]].isin(train_ids)
        df_train = bill_texts[train_mask]
        df_test = bill_texts[~train_mask]
        train_docs = df_train[col_names["doc"]].values.astype('U')
        test_docs = df_test[col_names["doc"]].values.astype('U')
        
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_docs)]
        model = Doc2Vec(documents, vector_size=20, window=2, min_count=1, workers=4)
        
