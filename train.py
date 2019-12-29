# -*- coding: utf-8 -*-
"""
Model training and k-fold cross validation

train: basic train function of model, on a defined training set
ModelCV: Cross Validation wrapper of models in model.py
-----------------------------------------------------------------------------------------------------------
Author: Chris
Department: Machine Learning Course directed by Prof.Liu, Peking University
"""
import torch
from torch import nn
from torch.autograd import Variable

from model import SolNet, PlusSolNet, HIVNet, PlusHIVNet
from load_data import SolData, PlusSolData, HIVData, PlusHIVData


def make_variables(graphs):
    """
    wrap every serial value (atom and bonds) in Variable, for back_propagation in training process.
    :param graphs: Iterable of Graph
    """
    for graph in graphs:
        for node in graph.nodes:
            node.msg = Variable(node.msg)
            node.bonds = [Variable(bond) for bond in node.bonds]


def train(model, data_iterator, optimizer, loss, show_every=5, epochs=10):
    """
    basic train function of model, on a defined training set
    :param model: model.XXXNet
    :param data_iterator: load_data.XXXData object, or data_iterators acquired by load_data.XXXData.get_cv_sets()
    :param optimizer: optimizer in torch.optim
    :param loss: Function. loss function that accepts predicted values and targets, returns a scalar value
    :param show_every: int, print loss score after every show_every batches
    :param epochs: int
    """
    count = 0
    for epoch in range(epochs):
        for data, target in data_iterator:
            target = Variable(target)
            if model.plus:
                graphs, plus_features = data
                # for single_feature in plus_features.view(-1,):
                #     if torch.isnan(single_feature):
                #         print('hey, I am the problem')
                make_variables(graphs)
                plus_features = Variable(plus_features)
                data = (graphs, plus_features)
            else:
                make_variables(data)

            values_pred = model(data)
            loss_score = loss(values_pred, target)

            optimizer.zero_grad()
            loss_score.backward()
            optimizer.step()

            if count % show_every == 0:
                print(f'    count: {count}---loss: {loss_score.detach():.3f}')

            count += 1


class ModelCV(object):
    """
    Cross Validation wrapper of models in model.py
    :param
        Model: model in model.py
        folds: int, folds in k-fold cross validation
        plus: bool, whether the model is PlusXXXNet or not
        *args, **kwargs: arguments that the model takes

    :methods
        train(data_loader, Optimizer, optim_dict, loss, show_every=5, epochs=10, early_stopping=False, tol=0.5)
            :param data_loader: load_data.XXXData
            :param Optimizer: optimizer in torch.optim
            :param optim_dict: the kwargs dict that the optimizer takes
            :param loss: Function. loss function that accepts predicted values and targets, returns a scalar value
            :param show_every: int, print loss score after every show_every batches
            :param epochs: int
            :param early_stopping: bool, if True, the train process will stop when the loss score is less tha tol
            :param tol: float

        predict(smiles_es)
            :param smiles_es: iterable of smiles expressions
            :return: torch.Tensor, the average output of the models inside

    Example:
        model_cv = ModelCV(PlusSolNet, folds=5, inner_hidden_size=150, feature_size=50, hidden_size=100)
        data_loader = PlusSolData('datasets/plus_esol_train.txt', batch_size=64)
        Optimizer = torch.optim.Adam
        optim_dict = {'lr': 1e-3}
        loss = nn.MSELoss()
        model_cv.train(data_loader, Optimizer, optim_dict, loss, early_stopping=True, tol=0.5)
        model_cv.save_model('plus_solnet_model')
        model_cv.predict(["CCCO", "c1ccc(N)cc1C=O"])
    """

    def __init__(self, Model, folds=5, *args, **kwargs):
        self.models = [Model(*args, **kwargs) for _ in range(folds)]
        self.folds = folds

    def train(self, data_loader, Optimizer, optim_dict, loss, show_every=5, epochs=10, early_stopping=False, tol=0.5):
        optimizers = [Optimizer(params=model.parameters(), **optim_dict) for model in self.models]
        for epoch in range(epochs):
            scores = []
            for (data_iterator, data_val, target_val), model, optimizer in zip(data_loader.get_cv_sets(self.folds),
                                                                               self.models, optimizers):
                train(model, data_iterator, optimizer, loss, show_every, epochs=1)
                score_val = loss(model(data_val), target_val)
                scores.append(score_val)
            scores = torch.Tensor(scores).type(torch.float32)

            print(f'Validation Score over Epoch {epoch}: {scores.mean():.3f}')
            if early_stopping and scores.mean() < tol:
                break

        print('process ended')

    def save_model(self, fp):
        for index, model in enumerate(self.models):
            torch.save(model.state_dict(), f'{fp}_{index}.pkl')

    def load_model(self, fp):
        files = [f'{fp}_{index}.pkl' for index in range(self.folds)]
        for file, model in zip(files, self.models):
            model.load_state_dict(torch.load(file))

    def predict(self, smiles_es):
        council = [model.predict(smiles_es) for model in self.models]
        council = torch.cat([res.view(1, *res.shape) for res in council], dim=0)
        conclusion = council.mean(dim=0)
        return conclusion


def train_plus_solnet():
    model_cv = ModelCV(PlusSolNet, folds=5)
    data_loader = PlusSolData('datasets/plus_esol_train.txt', batch_size=64)
    Optimizer = torch.optim.Adam
    optim_dict = {'lr': 1e-3}
    loss = nn.MSELoss()
    model_cv.train(data_loader, Optimizer, optim_dict, loss, early_stopping=True, tol=0.5)
    model_cv.save_model('plus_solnet_model')


def train_plus_hivnet():
    model_cv = ModelCV(PlusHIVNet, folds=5)
    data_loader = PlusHIVData('datasets/plus_hiv_train.txt', batch_size=64)
    Optimizer = torch.optim.Adam
    optim_dict = {'lr': 1e-3}
    loss = nn.NLLLoss(weight=torch.Tensor([0.02, 0.98]))
    model_cv.train(data_loader, Optimizer, optim_dict, loss, early_stopping=True, tol=0.5)
    model_cv.save_model('plus_hivnet_model')


# def test():
#     model_cv = ModelCV(PlusHIVNet, folds=5)
#     print(model_cv.predict(['CCO', 'c1cc(O)ccc1C=O']))


if __name__ == '__main__':
    # train_plus_solnet()
    train_plus_hivnet()
