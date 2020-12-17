from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj

import re, torch
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(BaseOperator):

    @property
    def inputs(self):
        return {"train_text": Pandas_Dataframe(self.node.inputs[0]),
                "val_text": Pandas_Dataframe(self.node.inputs[1]),
                "train": Pandas_Dataframe(self.node.inputs[2]),
                "val": Pandas_Dataframe(self.node.inputs[3])}

    @property
    def outputs(self):
        return {"train": Pandas_Dataframe(self.node.outputs[0]),
                "val": Pandas_Dataframe(self.node.outputs[1]),
                "model": Pickle_Obj(self.node.outputs[2])}


    def run(self, params=None):
        train = self.inputs["train"].read()
        ## get the last row in a dataframe
        train = train.groupby('bill_id').tail(1)[["bill_id", "label"]]
        train_text = self.inputs["train_text"].read()
        train = train.merge(train_text, on = "bill_id", how="left")
        del train_text

        val = self.inputs["val"].read()
        val = val.groupby("bill_id").tail(1)[["bill_id", "label"]]
        val_text = self.inputs["val_text"].read()
        val = val.merge(val_text, on = "bill_id", how = "left")
        del val_text

        cnn = CNN_class(train, val)
        cnn.train()
        self.outputs["model"].write(cnn.cnn)

        train_features = cnn.get_features(cnn.train_iter)
        train = train.reindex(columns=train.columns.tolist() + ["cnn0", "cnn1"])
        train["cnn0"] = train_features[:,0]
        train["cnn1"] = train_features[:,1]
        del train_features

        val_features = cnn.get_features(cnn.val_iter)
        val = val.reindex(columns = val.columns.tolist() + ["cnn0", "cnn1"])
        val["cnn0"] = val_features[:,0]
        val["cnn1"] = val_features[:,1]
        del val_features

        self.outputs["train"].write(train)
        self.outputs["val"].write(val)


class MR(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def __init__(self, text_field, label_field, train = None, val = None, examples = None):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        text_field.preprocessing = data.Pipeline(self.clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            examples = []
            for idx, r in train.iterrows():
                examples.append(data.Example.fromlist([r['doc'], r['label']], fields))
            for idx, r in val.iterrows():
                examples.append(data.Example.fromlist([r['doc'], r['label']], fields))
        super(MR, self).__init__(examples, fields)

    @classmethod
    def splits(cls, text_field, label_field, train, val, val_index):
        examples = cls(text_field, label_field, train, val).examples

        return (cls(text_field, label_field, examples=examples[:val_index]),
                cls(text_field, label_field, examples=examples[val_index:]))

class CNN_Text(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes):
        super(CNN_Text, self).__init__()
        V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

class CNN_class():
    def __init__(self, train, val):
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)
        self.train_iter, self.val_iter = self.mr(text_field, label_field, train, val, len(train))
        embed_num = len(text_field.vocab)
        class_num = len(label_field.vocab) - 1
        kernel_sizes = [3, 4, 5]
        embed_dim = 128
        kernel_num = 100

        # model
        self.cnn = CNN_Text(embed_num, embed_dim, class_num, kernel_num, kernel_sizes)

    def mr(self, text_field, label_field, train, val, val_index):
        train_data, val_data = MR.splits(text_field, label_field, train, val, val_index)
        text_field.build_vocab(train_data, val_data)
        label_field.build_vocab(train_data, val_data)
        data_iter = data.Iterator.splits((train_data, val_data),
                                batch_sizes=(128, 128))
        return data_iter

    def get_features(self, data_iter):
        self.cnn.eval()
        all_logits = []
        for batch in data_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            feature, target = feature.cuda(), target.cuda()
            logit = self.cnn(feature).data.cpu().numpy()
            all_logits.append(logit)

        return np.concatenate(all_logits)

    def eval(self, data_iter):
        self.cnn.eval()
        corrects, avg_loss = 0, 0
        for batch in data_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            feature, target = feature.cuda(), target.cuda()
            logit = self.cnn(feature)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.item()
            correct = (torch.max(logit, 1)
                         [1].view(target.size()).data == target.data)
            corrects += correct.sum()

        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects/size
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size))
        return accuracy

    def save(self, model, save_dir, save_prefix, steps):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
        torch.save(model.state_dict(), save_path)

    def train(self):
        self.cnn.cuda()

        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=.001)

        steps = 0
        best_acc = 0
        last_step = 0
        self.cnn.train()
        for epoch in range(100):
            for batch in self.train_iter:
                feature, target = batch.text, batch.label
                feature.t_(), target.sub_(1)  # batch first, index align
                feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                logit = self.cnn(feature)
                loss = F.cross_entropy(logit, target)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % 1 == 0:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                    accuracy = 100.0 * corrects/batch.batch_size
                    print(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                 loss.item(),
                                                                                 accuracy.item(),
                                                                                 corrects.item(),
                                                                                 batch.batch_size))
        self.save(self.cnn, 'snapshot', 'final', steps)
