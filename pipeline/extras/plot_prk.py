# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:05:55 2020

@author: sharm
"""
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file1 = "model_topmodels.csv"
pd_topk = pd.read_csv(file1)


log1 = pd_topk.loc[pd_topk["model_rank"] == 1].sort_values("k")
log1[["k","precision","recall"]]

log2 = pd_topk.loc[pd_topk["model_rank"] == 2].sort_values("k")
log2[["k","precision","recall"]]

base = pd_topk.loc[pd_topk["model"] == "baseline" ].sort_values("k")
base[["k","precision","recall"]]

commonsense = pd_topk.loc[pd_topk["model"] == "commonsense" ].sort_values("k")
commonsense[["k","precision","recall"]]

def plot_prk(precisions, recalls, graph_name):
        fig, ax = plt.subplots()
        assert len(precisions) == len(recalls)
        x = np.linspace(0, 1, len(precisions))
        ax.plot(x, precisions, color="red")
        ax.set_xlabel('Percent of Total Bills')
        ax.set_ylabel("Precision", color="red")
        ax.set_title('PR-k of model')
        ax.set_ylim(0, 1)
        ax2 = ax.twinx()
        ax2.plot(x, recalls,color="blue")
        ax2.set_ylabel("Recall", color="blue")
        ax2.set_ylim(0, 1)
        fig.savefig("{}".format(graph_name))
        #fig.savefig(os.path.join(os.path.dirname(self.inputs["predictions"].data_location),"{}.png".format(graph_name)))

plot_prk(log1["precision"], log1["recall"], "DecisionTree1(depth=3, min_sample_leaf =1, split =2).png")
plot_prk(log2["precision"], log2["recall"], "DecisionTree2(min_sample_split =4, depth =3).png")
plot_prk(base["precision"], base["recall"], "base_model.png")
plot_prk(commonsense["precision"], commonsense["recall"], "commonsense_model.png")