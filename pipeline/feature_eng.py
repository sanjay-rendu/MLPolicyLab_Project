import pandas as pd
from sklearn.feature_extraction.text import TdidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from daggit.core.io.io import Pandas_Dataframe
from daggit.core.base.factory import BaseOperator


class topic_model(BaseOperator):

    @property
    def inputs(self):
        return {"raw": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {"df": Pandas_Dataframe(self.node.outputs[0])}

    def run(self, text_features = "doc", num_features = 1000, num_topics = 20):
        """
        Engineers features out of raw data. Saves and returns final dataframe.
            Arguments:
                text_features: str, default "doc"
                    Index name for bill_text in dataframe.
                num_features: int, default 1000
                    Number of terms to include in the bag of words matrix
                num_topics: int, defautl 20
                    Number of topics we are modeling for
            Returns:
                pandas dataframe of features
        """
        df = self.inputs["raw"].read()
        documents = df[text_features].tolist()
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(documents)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()

        nmf = NMF(n_components=num_topics, random_state = 1, alpha=.1, l1_ratio=.5, init="nndsvd").fit(tfidf)
        lda = LDA(n_components=num_topics, max_iter=5, learning_method="online", learning_offset=50., random_state=0).fit(tf)

        nmf_vals = nmf.transform(tfidf)
        lda_vals=lda.transform(tf)

        nmf_cols = ["nmf_{t}".format(t=t) for t in range(0,nmf_vals.shape[1])]
        lda_cols = ["topic_{t}".format(t=t) for t in range(0,lda_vals.shape[1])]
        df[nmf_cols] = nmf_vals
        df[lda_cols] = lda_vals
        self.outputs["df"].write(df)

