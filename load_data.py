import numpy as np, pandas as pd
import torch
from torch.autograd import Variable

from base import Graph

from sklearn.model_selection import KFold


class SolData(object):
    """
    Iterable.
    output in each iteration: (batch_graphs(np.ndarray, size=(batch_size, )),
                               batch_solubility(torch.Tensor, size=(batch_size, )))
    provide cross validation method: get_cv_sets(folds)
                                     return Iterator: output in each iteration: (data Iterator of one fold, validation data, validation target)
    """
    def __init__(self, filename, batch_size=64):
        self.batch_size = batch_size
        raw_data = pd.read_csv(filename)
        self.smiles = raw_data['smiles'].values
        self.solubility = raw_data['solubility'].values
        self.graphs = np.array([Graph(smiles) for smiles in self.smiles])
        self.size = self.graphs.shape[0]

    @classmethod
    def train_iterator(cls, data, target, batch_size):
        pointer = 0
        while True:
            if pointer >= data.shape[0]:
                break
            yield data[pointer: pointer + batch_size], torch.from_numpy(target[pointer: pointer + batch_size]).type(
                torch.float32)
            pointer += batch_size

    def get_cv_sets(self, folds=5):
        folder = KFold(n_splits=folds, shuffle=True)
        for train_index, test_index in folder.split(self.graphs):
            graph_train, graph_val = self.graphs[train_index], self.graphs[test_index]
            sol_train, sol_val = self.solubility[train_index], self.solubility[test_index]
            yield SolData.train_iterator(graph_train, sol_train, self.batch_size), graph_val, torch.from_numpy(
                sol_val).type(torch.float32)

    @classmethod
    def make_variables(cls, graphs):
        """
        wrap every serial value (atom and bonds) in Variable, for back_propagation in training process.
        :param graphs: Iterable of Graph
        """
        for graph in graphs:
            for node in graph.nodes:
                node.msg = Variable(node.msg)
                node.bonds = [Variable(bond) for bond in node.bonds]

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer >= self.size:
            raise StopIteration
        sample_index = np.random.randint(0, self.size, size=(self.batch_size,))
        batch_graphs = self.graphs[sample_index]
        batch_solubility = torch.from_numpy(self.solubility[sample_index]).type(torch.float32)
        self._pointer += self.batch_size
        return batch_graphs, batch_solubility


class PlusSolData(object):
    def __init__(self, filename, batch_size=64):
        self.batch_size = batch_size
        raw_data = pd.read_csv(filename)
        self.smiles = raw_data['smiles'].values
        self.solubility = raw_data['solubility'].values
        self.features = raw_data.iloc[:, 2:].values
        self.graphs = np.array([Graph(smiles) for smiles in self.smiles])
        self.size = self.graphs.shape[0]

    @classmethod
    def train_iterator(cls, data, feature, target, batch_size):
        pointer = 0
        while True:
            if pointer >= data.shape[0]:
                break
            yield (data[pointer: pointer + batch_size], torch.from_numpy(feature[pointer:pointer + batch_size]).type(
                torch.float32)), torch.from_numpy(target[pointer: pointer + batch_size]).type(torch.float32)
            pointer += batch_size

    def get_cv_sets(self, folds=5):
        folder = KFold(n_splits=folds, shuffle=True)
        for train_index, test_index in folder.split(self.graphs):
            graph_train, graph_val = self.graphs[train_index], self.graphs[test_index]
            sol_train, sol_val = self.solubility[train_index], self.solubility[test_index]
            feature_train, feature_val = self.features[train_index], self.features[test_index]
            yield PlusSolData.train_iterator(graph_train, feature_train, sol_train,
                                             self.batch_size), (
                      graph_val, torch.from_numpy(feature_val).type(torch.float32)), torch.from_numpy(sol_val).type(
                torch.float32)

    @classmethod
    def make_variables(cls, graphs):
        """
        wrap every serial value (atom and bonds) in Variable, for back_propagation in training process.
        :param graphs: Iterable of Graph
        """
        for graph in graphs:
            for node in graph.nodes:
                node.msg = Variable(node.msg)
                node.bonds = [Variable(bond) for bond in node.bonds]

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer >= self.size:
            raise StopIteration
        sample_index = np.random.randint(0, self.size, size=(self.batch_size,))
        batch_graphs = self.graphs[sample_index]
        batch_features = torch.from_numpy(self.features[sample_index]).type(torch.float32)
        batch_solubility = torch.from_numpy(self.solubility[sample_index]).type(torch.float32)
        self._pointer += self.batch_size
        return (batch_graphs, batch_features), batch_solubility


class HIVData(object):
    def __init__(self, filename, batch_size=64):
        self.batch_size = batch_size
        raw_data = pd.read_csv(filename)
        self.smiles = raw_data['smiles'].values
        self.activity = raw_data['activity'].values
        self.graphs = np.array([Graph(smiles) for smiles in self.smiles])
        self.size = self.graphs.shape[0]

    @classmethod
    def train_iterator(cls, data, target, batch_size):
        pointer = 0
        while True:
            if pointer >= data.shape[0]:
                break
            batch_data = data[pointer: pointer + batch_size]
            batch_targets = torch.from_numpy(target[pointer: pointer + batch_size]).type(torch.long)
            yield batch_data, batch_targets
            pointer += batch_size

    def get_cv_sets(self, folds=5):
        folder = KFold(n_splits=folds, shuffle=True)
        for train_index, test_index in folder.split(self.graphs):
            graph_train, graph_val = self.graphs[train_index], self.graphs[test_index]
            ac_train, ac_val = self.activity[train_index], self.activity[test_index]
            yield HIVData.train_iterator(graph_train, ac_train, self.batch_size), graph_val, torch.from_numpy(
                ac_val).type(torch.long)

    @classmethod
    def make_variables(cls, graphs):
        """
        wrap every serial value (atom and bonds) in Variable, for back_propagation in training process.
        :param graphs: Iterable of Graph
        """
        for graph in graphs:
            for node in graph.nodes:
                node.msg = Variable(node.msg)
                node.bonds = [Variable(bond) for bond in node.bonds]

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer >= self.size:
            raise StopIteration
        sample_index = np.random.randint(0, self.size, size=(self.batch_size,))
        batch_graphs = self.graphs[sample_index]
        batch_activity = torch.from_numpy(self.activity[sample_index]).type(torch.long)
        self._pointer += self.batch_size
        return batch_graphs, batch_activity


class PlusHIVData(object):
    def __init__(self, filename, batch_size=64):
        self.batch_size = batch_size
        raw_data = pd.read_csv(filename)
        self.smiles = raw_data['smiles'].values
        self.features = raw_data.iloc[:, 2:].values
        self.activity = raw_data['activity'].values
        self.graphs = np.array([Graph(smiles) for smiles in self.smiles])
        self.size = self.graphs.shape[0]

    @classmethod
    def train_iterator(cls, data, feature, target, batch_size):
        pointer = 0
        while True:
            if pointer >= data.shape[0]:
                break
            batch_data = data[pointer: pointer + batch_size]
            batch_features = torch.from_numpy(feature[pointer:pointer + batch_size]).type(torch.float32)
            batch_targets = torch.from_numpy(target[pointer: pointer + batch_size]).type(torch.long)
            yield (batch_data, batch_features), batch_targets
            pointer += batch_size

    def get_cv_sets(self, folds=5):
        folder = KFold(n_splits=folds, shuffle=True)
        for train_index, test_index in folder.split(self.graphs):
            graph_train, graph_val = self.graphs[train_index], self.graphs[test_index]
            ac_train, ac_val = self.activity[train_index], self.activity[test_index]
            feature_train, feature_val = self.features[train_index], self.features[test_index]
            yield PlusHIVData.train_iterator(graph_train, feature_train, ac_train,
                                             self.batch_size), (
                  graph_val, torch.from_numpy(feature_val).type(torch.float32)), torch.from_numpy(ac_val).type(
                torch.long)

    @classmethod
    def make_variables(cls, graphs):
        """
        wrap every serial value (atom and bonds) in Variable, for back_propagation in training process.
        :param graphs: Iterable of Graph
        """
        for graph in graphs:
            for node in graph.nodes:
                node.msg = Variable(node.msg)
                node.bonds = [Variable(bond) for bond in node.bonds]

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer >= self.size:
            raise StopIteration
        sample_index = np.random.randint(0, self.size, size=(self.batch_size,))
        batch_graphs = self.graphs[sample_index]
        batch_features = torch.from_numpy(self.features[sample_index]).type(torch.float32)
        batch_activity = torch.from_numpy(self.activity[sample_index]).type(torch.long)
        self._pointer += self.batch_size
        return (batch_graphs, batch_features), batch_activity


def test():
    soldata = HIVData('datasets/hiv_train.txt', 64)
    # iter(soldata)
    # print(next(soldata))
    soldata_cv = soldata.get_cv_sets(5)
    train_cv, test_data, test_target = next(soldata_cv)
    print(test_data.shape)
    print(test_target.shape)
    print(next(train_cv)[0].shape)


def test1():
    soldata = PlusSolData('datasets/plus_esol_train.txt', 64)
    # iter(soldata)
    # print(next(soldata))
    soldata_cv = soldata.get_cv_sets(5)
    train_cv, test_data, test_target = next(soldata_cv)
    print(test_data[1].shape)
    print(test_target.shape)
    print(next(train_cv)[0][1].shape)
