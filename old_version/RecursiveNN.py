# -*- coding: utf-8 -*-
"""
Example Recursive Neural Network
Equipped with a timer to analyse efficiency
"""
import torch
from torch import nn
from torch.autograd import Variable

from load_data import DataLoader, Graph

import time


def timer(func):
    def timed_func(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print(f'process ended after {end_time - start_time:.2f} seconds')
        return res

    return timed_func


class InnerRecurrentNet(nn.Module):
    def __init__(self, input_size=23, output_size=20):
        super(InnerRecurrentNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, output_size),
                                 nn.LeakyReLU(0.01))

    def forward(self, x):
        x = self.net(x)
        x = x.sum(dim=0)
        return x


class RecursiveNet(nn.Module):
    """
    input: Iterable of Graph, size=(N,)
    output: torch.Tensor, size=(N,)
    ----------------------------------------
    Example:
        model = RecursiveNet()
        graphs = [Graph('CC(=O)C'), Graph('CC(O)C')]
        RecursiveNet.make_variables(graphs)
        output = model(graphs)
        print(output)
        print(output.shape)
    """

    def __init__(self, atom_size=14, bond_size=3, inner_size=20, hidden_size=34, recursive_size=20,
                 output_hidden_size=25):
        super(RecursiveNet, self).__init__()
        self.inner_net = InnerRecurrentNet(input_size=recursive_size + bond_size, output_size=inner_size)
        self.net = nn.Sequential(nn.Linear(inner_size + atom_size, hidden_size),
                                 nn.LeakyReLU(0.01),
                                 nn.Linear(hidden_size, recursive_size),
                                 nn.LeakyReLU(0.01))
        self.net0 = nn.Sequential(nn.Linear(atom_size, hidden_size),
                                  nn.LeakyReLU(0.01),
                                  nn.Linear(hidden_size, recursive_size),
                                  nn.LeakyReLU(0.01))
        self.output_net = nn.Sequential(nn.Linear(recursive_size, output_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(output_hidden_size, 1))

    def forward(self, graphs):
        output = []
        for graph in graphs:
            output.append(self.single_forward(graph))
        return torch.cat(output, dim=0)

    def single_forward(self, graph):
        """
        predict only one graph at a time
        :param graph: Graph
        :return: torch.Tensor
        """
        output = torch.cat([self._recurse_node(tree.root) for tree in graph], dim=0).sum(dim=0).view(1, -1)
        output = self.output_net(output).view((-1,))
        return output

    def _recurse_node(self, node):
        """
        core recursive function
        """
        if not node.has_child():
            return self.net0(node.atom.view(1, -1))
        inner_node = torch.cat([self._recurse_node(child) for child in node.children], dim=0)
        bonds = torch.cat([bond.view(1, -1) for bond in node.tree_bonds], dim=0)
        inner_input = torch.cat([bonds, inner_node], dim=1)
        inner_vector = self.inner_net(inner_input).view(1, -1)
        inner_vector = torch.cat([node.atom.view(1, -1), inner_vector], dim=1)
        output = self.net(inner_vector.view(1, -1))
        return output

    @classmethod
    def make_variables(cls, graphs):
        """
        wrap every serial value (atom and bonds) in Variable, for back_propagation in training process.
        :param graphs: Iterable of Graph
        """
        for graph in graphs:
            for node in graph.nodes:
                node.atom = Variable(node.atom)
                node.bonds = [Variable(bond) for bond in node.bonds]

    def predict(self, smiles_es):
        graphs = [Graph(smiles) for smiles in smiles_es]
        return self(graphs)


loss = nn.MSELoss()


def get_optimizer(model):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    return optimizer


@timer
def train(model, data, loss, optimizer, show_every=5, epochs=10):
    count = 0
    for epoch in range(epochs):
        for graphs, values in data:
            RecursiveNet.make_variables(graphs)
            values = Variable(values)

            values_pred = model(graphs)
            loss_score = loss(values_pred, values)

            optimizer.zero_grad()
            loss_score.backward()
            optimizer.step()

            if count % show_every == 0:
                print(f'count: {count}---MSE loss: {loss_score.detach():.3f}')

            count += 1


def load_model(fn):
    model = RecursiveNet(atom_size=14, bond_size=3, inner_size=20, hidden_size=34, recursive_size=20,
                         output_hidden_size=25)
    model.load_state_dict(torch.load(fn))
    return model


def test_mse(model, test_graphs, test_values):
    values_pred = model(test_graphs)
    return loss(values_pred, test_values)


if __name__ == '__main__':
    data = DataLoader(r'Small Delaney Data Set.txt', batch_size=64, splitting=0.2)
    model = RecursiveNet(atom_size=14, bond_size=3, inner_size=20, hidden_size=34, recursive_size=20,
                         output_hidden_size=25)
    optimizer = get_optimizer(model)
    train(model, data, loss, optimizer, show_every=5, epochs=8)
    try:
        print(f'RMSE score on test set: {test_mse(model, *data.get_test_set()) ** 0.5:.3f}')
    except Exception as e:
        print(e)

    torch.save(model.state_dict(), 'RecursiveNN.pkl')
