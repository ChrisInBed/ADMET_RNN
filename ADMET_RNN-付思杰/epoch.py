# -*- coding: utf-8 -*-
"""
Run multiple epoches to evaluate of SolNet
Output file is in ./epoch.txt and the directory ./epoch
-----------------------------------------------------------------------------------------------------------
Author: fusijie@pku.edu.cn
Course: Machine Learning and its Applications in Chemistry, by Prof.Liu of Peking University
"""
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from model import SolNet, PlusSolNet, HIVNet, PlusHIVNet, nn
from train import ModelCV
from load_data import *

import matplotlib.pyplot as plt

model = ModelCV(SolNet, folds=5, inner_hidden_size=150, feature_size=50, hidden_size=100)
data = pd.read_csv('./datasets/ESOL.txt')
r = np.linspace(0.1, 0.9, 9)    # train test split ratio
train_r2 = []
test_r2 = []


def run_10_epoch(r=0.60, epoch=10, cycle=1):
    if not os.path.isdir('./epoch/' + '%d' % (epoch*cycle)):
        os.mkdir('./epoch/' + '%d' % (epoch*cycle))

    data_loader = SolData('./epoch/train.txt', batch_size=64)
    Optimizer = torch.optim.Adam
    optim_dict = {'lr': 1e-3}
    loss = nn.MSELoss()

    if cycle != 1:
        model.load_model('./epoch/%d/' % (epoch*(cycle-1)) + '%d' % (epoch*(cycle-1)))  # load from trained model

    model.train(data_loader, Optimizer, optim_dict, loss, early_stopping=False, tol=0.5)
    model.save_model('./epoch/%d/' % (epoch*cycle) + '%d' % (epoch*cycle))

    y_train_pre = model.predict(X_train).detach().numpy()
    y_test_pre = model.predict(X_test).detach().numpy()
    results_train = pd.DataFrame(np.column_stack((np.array(y_train), np.array(y_train_pre))),
                                 columns=['y_train', 'y_train_pre'])
    results_test = pd.DataFrame(np.column_stack((np.array(y_test), np.array(y_test_pre))),
                                columns=['y_test', 'y_test_pre'])
    results_train.to_csv('./epoch/' + '%d/' % (epoch*cycle) + 'results_train.txt', index=False, sep=',')
    results_test.to_csv('./epoch/' + '%d/' % (epoch*cycle) + 'results_test.txt', index=False, sep=',')

    train_r2.append(r2_score(y_train, y_train_pre))
    test_r2.append(r2_score(y_test, y_test_pre))


r = 0.40    # rate of data set for training
epoch = 10  # epoch per run
cycle_index = 0   # run time cycle count
keep_running = 1
print("Current epoch: %d.\n"
      "Run another 10 epoch?[y/n]\n"
      "y" % (cycle_index*epoch))

X_train, X_test, y_train, y_test = train_test_split(data['smiles'].values, data['solubility'].values,
                                                    test_size=1.0 - r)
train = pd.DataFrame(np.column_stack((X_train, y_train)), columns=['smiles', 'solubility'])
test = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['smiles', 'solubility'])
train.to_csv('./epoch/train.txt', index=False, sep=',')
test.to_csv('./epoch/test.txt', index=False, sep=',')

while keep_running:
    cycle_index = cycle_index + 1
    run_10_epoch(r=r, epoch=epoch, cycle=cycle_index)

    np.savetxt('./epoch/' + 'train_r2.txt', train_r2, fmt='%.6f')
    np.savetxt('./epoch/' + 'test_r2.txt', test_r2, fmt='%.6f')

    epoch_run = np.linspace(epoch, cycle_index*epoch, cycle_index)
    # train_r2 = np.loadtxt('./epoch/' + 'train_r2.txt')
    # test_r2 = np.loadtxt('./epoch/' + 'test_r2.txt')
    plt.ion()
    plt.plot(epoch_run, train_r2, 'b-', marker='^', label='train_r2')
    plt.plot(epoch_run, test_r2, 'r-', marker='*', label='test_r2')
    plt.axis([0, 10*(cycle_index+1), 0.4, 1.0])
    plt.ylabel('R2 score')
    plt.xlabel('epoch')
    plt.title('Training curve')
    plt.legend(loc='lower right')
    plt.savefig('./epoch/Training curve (%d epoch).png' % (epoch*cycle_index), dpi=600)
    plt.show()
    plt.pause(5)
    plt.close()

    if cycle_index < 5:
        keep_running = 'Y'
        print("Current epoch: %d.\n"
              "Run another 10 epoch?[y/n]y" % (cycle_index*epoch))
    else:
        keep_running = (input("Current epoch: %d.\n"
                              "Run another 10 epoch?[y/n]" % (cycle_index*epoch)).upper() == 'Y')
