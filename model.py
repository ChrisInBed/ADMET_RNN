import torch
from torch import nn

from base import Graph

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


class FeatureRNN(nn.Module):
    def __init__(self, atom_msg=125, bond_msg=12, hidden_size=125, output_size=40):
        super(FeatureRNN, self).__init__()
        self.bond_msg = bond_msg
        self.output_size = output_size
        self.net = nn.Sequential(nn.Linear(atom_msg + bond_msg + output_size, hidden_size),
                                 nn.LeakyReLU(0.01),
                                 nn.Linear(hidden_size, output_size),
                                 nn.LeakyReLU(0.01))

    def forward(self, graphs):
        outputs = []
        for graph in graphs:
            outputs.append(self.single_forward(graph))
        return torch.cat(outputs, dim=0)

    def single_forward(self, graph):
        """
        predict only one graph at a time
        :param graph: Graph
        :return: torch.Tensor
        """
        output = torch.cat([self._recurse_node(tree.root) for tree in graph.get_trees()], dim=0).sum(dim=0).view(1, -1)
        return output

    def _recurse_node(self, node):
        """
        core recursive function
        """
        if not node.has_child():
            return self.net(
                torch.cat([node.msg, torch.zeros(size=(self.bond_msg + self.output_size,))], dim=0).view(1, -1))
        upper_msgs = torch.cat([self._recurse_node(child) for child in node.children], dim=0)
        bonds = torch.cat([bond.view(1, -1) for bond in node.child_bonds], dim=0)
        msg = torch.mm(torch.ones(size=(bonds.shape[0], 1)), node.msg.view(1, -1))
        output = self.net(torch.cat([msg, bonds, upper_msgs], dim=1)).sum(dim=0).view(1, -1)
        return output


class SolNet(nn.Module):
    def __init__(self, atom_msg=125, bond_msg=12, inner_hidden_size=125, feature_size=40, hidden_size=100):
        super(SolNet, self).__init__()
        self.feature_net = FeatureRNN(atom_msg, bond_msg, inner_hidden_size, feature_size)
        self.net = nn.Sequential(nn.Linear(feature_size, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, 1))

    def forward(self, graphs):
        features = self.feature_net(graphs)
        return self.net(features).view(-1, )

    def predict(self, smiles_es):
        graphs = [Graph(smiles) for smiles in smiles_es]
        return self.forward(graphs)


class PlusSolNet(nn.Module):
    def __init__(self, atom_msg=125, bond_msg=12, inner_hidden_size=125, feature_size=40, hidden_size=100):
        super(PlusSolNet, self).__init__()
        self.feature_net = FeatureRNN(atom_msg, bond_msg, inner_hidden_size, feature_size)
        self.net = nn.Sequential(nn.Linear(feature_size + 200, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, 1))
        self.descriptors = [desc[0] for desc in Descriptors._descList]
        self.desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptors)

    def forward(self, graphs, plus_features):
        features = self.feature_net(graphs)
        features = torch.cat([features, plus_features], dim=1)
        return self.net(features).view(-1, )

    def predict(self, smiles_es):
        graphs = [Graph(smiles) for smiles in smiles_es]
        plus_features = [list(self.desc_calc.CalcDescriptors(Chem.MolFromSmiles(smiles))) for smiles in smiles_es]
        plus_features = torch.Tensor(plus_features).type(torch.float32)
        return self.forward(graphs, plus_features)


class HIVNet(nn.Module):
    def __init__(self, atom_msg=125, bond_msg=12, inner_hidden_size=125, feature_size=40, hidden_size=100):
        super(HIVNet, self).__init__()
        self.feature_net = FeatureRNN(atom_msg, bond_msg, inner_hidden_size, feature_size)
        self.net = nn.Sequential(nn.Linear(feature_size, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, 2),
                                 nn.LogSoftmax(1))

    def forward(self, graphs):
        features = self.feature_net(graphs)
        return self.net(features)

    def predict(self, smiles_es):
        graphs = [Graph(smiles) for smiles in smiles_es]
        return torch.exp(self.forward(graphs))


class PlusHIVNet(nn.Module):
    def __init__(self, atom_msg=125, bond_msg=12, inner_hidden_size=125, feature_size=40, hidden_size=100):
        super(PlusHIVNet, self).__init__()
        self.feature_net = FeatureRNN(atom_msg, bond_msg, inner_hidden_size, feature_size)
        self.net = nn.Sequential(nn.Linear(feature_size + 200, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, 2),
                                 nn.LogSoftmax(1))
        self.descriptors = [desc[0] for desc in Descriptors._descList]
        self.desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptors)

    def forward(self, graphs, plus_features):
        features = self.feature_net(graphs)
        features = torch.cat([features, plus_features], dim=1)
        return self.net(features)

    def predict(self, smiles_es):
        graphs = [Graph(smiles) for smiles in smiles_es]
        plus_features = [list(self.desc_calc.CalcDescriptors(Chem.MolFromSmiles(smiles))) for smiles in smiles_es]
        plus_features = torch.Tensor(plus_features).type(torch.float32)
        return torch.exp(self.forward(graphs, plus_features))


def test():
    net = FeatureRNN()
    from base import Graph
    graphs = [Graph('CCO'), Graph('c1cc(O)ccc1C=O')]
    print(net(graphs))


def test1():
    net = SolNet()
    from base import Graph
    smiles = ['CCO', 'c1cc(O)ccc1C=O']
    graphs = [Graph('CCO'), Graph('c1cc(O)ccc1C=O')]
    print(net(graphs))
    print(net.predict(smiles))


def test2():
    net = PlusSolNet()
    from base import Graph
    smiles = ['CCO', 'c1cc(O)ccc1C=O']
    graphs = [Graph('CCO'), Graph('c1cc(O)ccc1C=O')]
    # print(net(graphs))
    print(net.predict(smiles))


def test3():
    net = HIVNet()
    from base import Graph
    smiles = ['CCO', 'c1cc(O)ccc1C=O']
    graphs = [Graph('CCO'), Graph('c1cc(O)ccc1C=O')]
    # print(net(graphs))
    print(net.predict(smiles))


def test4():
    net = PlusHIVNet()
    from base import Graph
    smiles = ['CCO', 'c1cc(O)ccc1C=O']
    graphs = [Graph('CCO'), Graph('c1cc(O)ccc1C=O')]
    # print(net(graphs))
    print(net.predict(smiles))
