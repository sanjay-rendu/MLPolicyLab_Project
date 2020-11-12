import re
from torchtext import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def clean_str(string):
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

class bill_generator(data.Dataset):
    @static_method
    def sort_key(ex):
        return len(ex.text)
    
    def __init__(self, text_field, label_field, df):
         """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        
        examples = []
        for idx, r in df.iterrows():
            examples.append(data.Example.fromlist([r['doc'], r['label'], fields))
        super(MR, self).__init__(examples, fields, **kwargs)

    def get_data(cls, text_field, label_field, shuffle = True):
        examples = cls(text_field, label_field, df).examples
        if shuffle:
            random.shuffle(examples)

def mr(text_field, label_field):
    text_data = MR.get_data(text_field, label_field, df)
    text_field.build_vocab(text_data)
    label_field.build_vocab(text_data)
    data_iter = data.Iterator.splits((text_data))
    return data_iter

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

def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        correct = (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data)
        print(correct)
        corrects += correct.sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

def train(train_iter, model):
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(100):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
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
            if steps % 100 == 0:
                dev_acc = eval(train_iter, model)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    save(model, 'snapshot', 'best', steps)
                else:
                    if steps - last_step >= 1000:
                        print('early stop by {} steps.'.format(1000))
            elif steps % 500 == 0:
                save(model, 'snapshot', 'snapshot', steps)
    save(model, 'snapshot', 'final', steps)


if __name__ == "__main__":
    embed_num = len(text_field.vocab)
    class_num = len(label_field.vocab) - 1
    kernel_sizes = [3, 4, 5]
    embed_dim = 128
    kernel_num = 100

    # model
    cnn = CNN_Text(embed_num, embed_dim, class_num, kernel_num, kernel_sizes)
    train(train_iter, dev_iter, cnn)
