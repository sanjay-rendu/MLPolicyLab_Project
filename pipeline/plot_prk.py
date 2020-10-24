import matplotlib.pyplot as plt
import numpy as np

def plot_prk(precisions, recalls, model_name):
    assert len(precisions) == len(recalls)
    x = np.linspace(0, 1, len(precisions))
    plt.plot(x, precisions)
    plt.plot(x, recalls)
    plt.legend(['Precision', 'Recall'])
    plt.xlabel('Percent of Total Bills')
    plt.title('PR-k of model {}'.format(model_name))
    plt.savefig('prk_graph_{}.png'.format(model_name))
