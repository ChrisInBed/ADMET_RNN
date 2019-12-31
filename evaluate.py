# -*- coding: utf-8 -*-
"""
Example of the evaluation of final models
-----------------------------------------------------------------------------------------------------------
Author: Chris
Department: Machine Learning Course directed by Prof.Liu, Peking University
"""
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from model import SolNet, PlusSolNet, HIVNet, PlusHIVNet
from train import ModelCV

import matplotlib.pyplot as plt


def eval_solnet():
    model = ModelCV(SolNet, folds=5, inner_hidden_size=125, feature_size=40, hidden_size=100)  # Initialize a model
    model.load_model('models/SolNet/solnet_model')  # load from trained model file
    data_test = pd.read_csv('datasets/esol_test.txt')  # load test dataset
    smiles_test = data_test['smiles'].values
    target_test = data_test['solubility'].values
    target_pred = model.predict(smiles_test)

    # draw prediction-true value plot, calculate r^2 score.
    # R^2=1-\frac{\sum_{i=1}^{N}\left({\hat{y}}_i-y_i\right)^2}{\sum_{i=1}^{N}\left(y_i-\bar{y}\right)^2}
    fig, ax = plt.subplots()
    ax.scatter(target_test, target_pred, c='g', alpha=0.4)
    ax.plot(ax.get_xlim(), ax.get_ylim(), '-r')
    r2_result = r2_score(target_test, target_pred)
    ax.text(0.8 * ax.get_xlim()[0] + 0.2 * ax.get_xlim()[1], 0.2 * ax.get_ylim()[0] + 0.8 * ax.get_ylim()[1],
            fr'$\R^{{2}}\ =\ {r2_result:.3f}$')
    fig.show()
    # fig.savefig('graph/XXX.svg')

    rmse_score = mean_squared_error(target_test, target_pred) ** 0.5

    print(f'RMSE = {rmse_score:.3f}')
