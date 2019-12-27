from rdkit import Chem
import torch


class Node(object):
    """
    Building block for both Graph and Tree, represent of atom.
    ----------------------------------------------------------------
    """
    num_atoms = 100  # types of elements
    num_bonds = 6  # num of bonds
    num_charges = 7
    num_Hs = 5

    def __init__(self, atom):
        self.symbol = atom.GetSymbol()
        self.msg = Node.get_msg(atom)
        self.neighbors = []
        self.bonds = []
        self.children = []
        self.child_bonds = []
        self.color = 'white'

    def add_neighbor(self, neighbor, bond, mol):
        self.neighbors.append(neighbor)
        self.bonds.append(Node.get_bond_msg(bond, mol))

    def add_child(self, child, bond_code):
        self.children.append(child)
        self.child_bonds.append(bond_code)

    def has_child(self):
        return bool(self.children)

    @classmethod
    def _one_hot_code(cls, size, index):
        code = torch.zeros(size=(size,))
        if 0 <= index < size:
            code[index] = 1
        return code

    @classmethod
    def get_msg(cls, atom):
        """atom_msg_size = 100 + 6 + 7 + 5 + 5 + 1 + 1 = 125"""
        one_hot_atom = cls._one_hot_code(cls.num_atoms, atom.GetAtomicNum() - 1)
        bonds = cls._one_hot_code(cls.num_bonds, len(atom.GetBonds()) - 1)
        charge = cls._one_hot_code(cls.num_charges, atom.GetFormalCharge() + 3)
        hydrogen = cls._one_hot_code(cls.num_Hs, atom.GetTotalNumHs())
        hybrid = cls._one_hot_code(5, atom.GetHybridization().real - 1)
        aroma = torch.Tensor([int(atom.GetIsAromatic()), ])
        atom_mass = torch.Tensor([atom.GetMass() / 100])
        return torch.cat([one_hot_atom, bonds, charge, hydrogen, hybrid, aroma, atom_mass], dim=0).type(torch.float32)

    @classmethod
    def _convert_bond_type(cls, bond):
        bond_type_dict = {1: 0, 2: 1, 3: 2, 12: 3}
        value = bond.GetBondType().real
        value = bond_type_dict[value] if value in bond_type_dict else 4
        return cls._one_hot_code(4, value)

    @classmethod
    def get_bond_msg(cls, bond, mol):
        """bond_msg_size = 4 + 1 + 1 + 6 = 12"""
        bond_type = cls._convert_bond_type(bond)
        conjugated = torch.Tensor([int(bond.GetIsConjugated())])
        in_ring = torch.Tensor([int(bond.GetIdx() in mol.GetRingInfo().BondRings())])
        stereo = cls._one_hot_code(6, bond.GetStereo().real)
        return torch.cat([bond_type, conjugated, in_ring, stereo], dim=0).type(torch.float32)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.__str__()


class Tree(object):
    """
    Connection of Nodes in Tree Structure. A simple wrapper of Nodes, by quoting the root of tree.
    :param
    root_node: Node, the root of tree

    For more specific utilization, please check the doc of Class Node.
    """

    def __init__(self, root):
        self.root = root

    def __str__(self):
        """
        :return: str. The recursive expression of the tree.
        """
        if not self.root.children:
            return self.root.symbol
        string = f'{self.root.symbol}-->({str(Tree(self.root.children[0]))}'
        for node, bond in zip(self.root.children[1:], self.root.child_bonds[1:]):
            string += f'+{str(Tree(node))}'
        string += ')'
        return string

    def __repr__(self):
        return self.__str__()


class Graph(object):
    """
    Connection of Nodes in Graph Structure
    :param
    smiles_expr: str, SMILES expression of a molecule
    """

    def __init__(self, smiles_expr):
        self.smiles = smiles_expr
        self.nodes = []
        self._build_graph()
        self.size = len(self.nodes)

    def _build_graph(self):
        mol = Chem.MolFromSmiles(self.smiles)
        for atom in mol.GetAtoms():
            self.nodes.append(Node(atom))
        for bond in mol.GetBonds():
            left = bond.GetBeginAtomIdx()
            right = bond.GetEndAtomIdx()
            self.nodes[left].add_neighbor(self.nodes[right], bond, mol)
            self.nodes[right].add_neighbor(self.nodes[left], bond, mol)

    def build_tree(self, root):
        self._color_init()
        root.color = 'black'
        self._build_tree_on_node(root)
        return Tree(root)

    def get_trees(self):
        for node in self.nodes:
            yield self.build_tree(node)

    def _color_init(self):
        for node in self.nodes:
            node.color = 'white'
            node.children = []
            node.child_bonds = []

    def _build_tree_on_node(self, node):
        for neighbor, bond in zip(node.neighbors, node.bonds):
            if neighbor.color == 'white':
                node.add_child(neighbor, bond_code=bond)
                neighbor.color = 'black'

        for child in node.children:
            self._build_tree_on_node(child)


def test():
    smiles = 'c1ccc(N)cc1C=O'
    graph = Graph(smiles)
    for tree in graph.get_trees():
        print(tree)
