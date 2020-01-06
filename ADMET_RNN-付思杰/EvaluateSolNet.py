# -*- coding: utf-8 -*-
"""
Learning score curve evaluation of SolNet
Output file is in ./EvaluateSolNet.txt and the directory ./EvaluateSolNet
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

new_dir = 'EvaluateSolNet'
if not os.path.isdir('./' + new_dir):
    os.mkdir('./' + new_dir)
new_dir = './' + new_dir + '/'


def run_sol(r, index=0, test_num=1):
    if not os.path.isdir(new_dir + 'r=%.2f' % r):
        os.mkdir(new_dir + 'r=%.2f' % r)
    if not os.path.isdir(new_dir + 'r=%.2f/' % r + '%d' % test_num):
        os.mkdir(new_dir + 'r=%.2f/' % r + '%d' % test_num)

    X_train, X_test, y_train, y_test = train_test_split(data['smiles'].values, data['solubility'].values,
                                                        test_size=1.0 - r)
    train = pd.DataFrame(np.column_stack((X_train, y_train)), columns=['smiles', 'solubility'])
    test = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['smiles', 'solubility'])
    train.to_csv(new_dir + 'r=%.2f/' % r + '%d/' % test_num + 'train.txt', index=False, sep=',')
    test.to_csv(new_dir + 'r=%.2f/' % r + '%d/' % test_num + 'test.txt', index=False, sep=',')
    train.to_csv(new_dir + 'train.txt', index=False, sep=',')
    test.to_csv(new_dir + 'test.txt', index=False, sep=',')
    data_loader = SolData(new_dir + 'train.txt', batch_size=64)
    Optimizer = torch.optim.Adam
    optim_dict = {'lr': 1e-3}
    loss = nn.MSELoss()
    # model.load_model('./temp_data/' + 'r=%.2f/' % r + '%d/' % test_num + 'SolNet')  # load from trained model file
    model.train(data_loader, Optimizer, optim_dict, loss, early_stopping=False, tol=0.5)

    model.save_model(new_dir + 'r=%.2f/' % r + '%d/' % test_num + 'SolNet')

    y_train_pre = model.predict(X_train).detach().numpy()
    y_test_pre = model.predict(X_test).detach().numpy()
    results_train = pd.DataFrame(np.column_stack((np.array(y_train), np.array(y_train_pre))),
                                 columns=['y_train', 'y_train_pre'])
    results_test = pd.DataFrame(np.column_stack((np.array(y_test), np.array(y_test_pre))),
                                columns=['y_test', 'y_test_pre'])
    results_train.to_csv(new_dir + 'r=%.2f/' % r + '%d/' % test_num + 'results_train.txt', index=False, sep=',')
    results_test.to_csv(new_dir + 'r=%.2f/' % r + '%d/' % test_num + 'results_test.txt', index=False, sep=',')
    train_r2[index].append(r2_score(y_train, y_train_pre))
    test_r2[index].append(r2_score(y_test, y_test_pre))


index = 0
for r in r:
    train_r2.append([])
    test_r2.append([])
    for test_num in range(1, 3):
        print('-----------------r=%.2f, test_num=%d-----------------------' % (r, test_num))
        run_sol(r, index, test_num)
    index += 1

np.savetxt(new_dir + 'train_r2.txt', train_r2, fmt='%.6f')
np.savetxt(new_dir + 'test_r2.txt', test_r2, fmt='%.6f')


def plot_learning_curve():
    train_r2score = np.loadtxt(new_dir + 'train_r2.txt')
    test_r2score = np.loadtxt(new_dir + 'test_r2.txt')
    plt.plot(r, train_r2score.mean(axis=1), 'b-', marker='^', label='train_r2')
    plt.plot(r, test_r2score.mean(axis=1), 'r-', marker='*', label='test_r2')
    plt.axis([0, 1.0, 0.5, 1.0])
    plt.ylabel('R2 score')
    plt.xlabel('ratio for training')
    plt.title('Learning score curve')
    plt.legend(loc='lower right')
    plt.savefig(new_dir + 'learning score curve.png', dpi=600)
    plt.show()


plot_learning_curve()
