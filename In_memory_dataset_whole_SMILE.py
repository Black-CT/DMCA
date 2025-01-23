from rdkit import Chem
from torch_geometric.data import (Data, InMemoryDataset,Dataset ,download_url,
                                  extract_gz)
import pickle
import torch
import os
import argparse

x_map = {
    'atomic_num':
        list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
        list(range(0, 11)),
    'formal_charge':
        list(range(-5, 7)),
    'num_hs':
        list(range(0, 9)),
    'num_radical_electrons':
        list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3'
        ,
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_file_names(self):
        return "our_sider.csv"

    def download(self):
        pass

    def process(self, dir):
        # load data
        print("load_data")
        print(os.getcwd())
        dataset_dir = dir + '/data.pkl'
        with open(dataset_dir, 'rb') as file:
           self.data_set = pickle.load(file)

        self.length = len(self.data_set)
        print(self.length)

        idx = 0
        datalist=[]
        for data in self.data_set:
            mol = Chem.MolFromSmiles(data[0])
            if mol is None:
                continue
            nodes_feature=[]
            for atom in mol.GetAtoms():
                node_feature = []
                node_feature.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                node_feature.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                node_feature.append(x_map['degree'].index(atom.GetTotalDegree()))
                node_feature.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                node_feature.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                node_feature.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                node_feature.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                node_feature.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                node_feature.append(x_map['is_in_ring'].index(atom.IsInRing()))
                nodes_feature.append(node_feature)
            x=torch.tensor(nodes_feature, dtype=torch.float).view(-1, 9)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # e = []
                # e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                # e.append(e_map['stereo'].index(str(bond.GetStereo())))
                # e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                edge_indices += [[i, j], [j, i]]
                # edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)

            y = torch.tensor(data[1], dtype=torch.float)

            smiles = data[0]
            data = Data(x=x, edge_index=edge_index, y=y, smiles=smiles)
            datalist.append(data)
            idx += 1
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])



if __name__ == '__main__':
    # test
    parser = argparse.ArgumentParser(description='process dataset')
    parser.add_argument('--dataset', type=str, default='Lipophilicity', help='Name of dataset')
    args = parser.parse_args()

    print("load dataset ", args.dataset)
    gnn_dataset = MyOwnDataset(root="drug_data/")
    gnn_dataset.process("drug_data/raw/{}".format(args.dataset))