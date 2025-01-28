import pandas as pd
import numpy as np
import torch
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# print(res_weight_table)



# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    # return c_size, features, edge_index

    return Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def target_to_graph( pro_seq, contact_map):
    target_edge_index = []


    # contact_file = os.path.join(contact_dir, str(target_key) + '.npy')
    # contact_map = np.load(contact_file)
    # target_size = len(contact_map)

    # contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = seq_feature (pro_seq)
    target_edge_index = np.array(target_edge_index)
    # return target_size, target_feature, target_edge_index
    return Data(x=torch.Tensor(target_feature), edge_index=torch.LongTensor(target_edge_index).transpose(1, 0))

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
               "H": 7, "I": 8, "K": 9, "L": 10, "M": 11,
               "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16,
               "T": 17, "V": 18, "W": 19,
               "Y": 20, "X": 21
               }
def label_sequence(line, MAX_SEQ_LEN=650):
    X = np.zeros(MAX_SEQ_LEN, dtype=int)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X


def label_smiles(line, max_smi_len=120):
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch]
    return X


# def build_dataset(dataset, smiles, seq, contact_dir,esm_dir, target_col):
#     processed = []
#     for idx in tqdm(range(len(dataset))):
#         # 分子图
#         Smiles = dataset.loc[idx, smiles]
#         Smiles_100 = label_smiles(Smiles, 100)
#         smiles_graph = smile_to_graph(Smiles)
#
#         # 蛋白质图
#         protein_seq = dataset.loc[idx, seq]
#         seq_1000 = label_sequence(protein_seq,650)
#         contact_map_file = os.path.join(contact_dir, f"{idx+1}.npy")
#         contact_map = np.load(contact_map_file)
#         protein_graph = target_to_graph(protein_seq, contact_map)
#
#         esm_file = os.path.join(esm_dir, f"{idx + 1}.npy")
#         esm = np.load(esm_file)
#         esm = torch.tensor(esm )
#         pooled_vector = torch.mean(esm, dim=0)
#
#
#         # 标签
#         label = torch.tensor(dataset.loc[idx, target_col], dtype=torch.float)
#
#         processed.append([Smiles_100,seq_1000,smiles_graph, protein_graph,pooled_vector, label,])
#     return processed, dataset

def build_dataset(dataset, smiles, seq,contact_dir, target_col):
    processed = []
    for idx in tqdm(range(len(dataset))):
        # 分子图
        Smiles = dataset.loc[idx, smiles]
        Smiles_100 = label_smiles(Smiles, 100)
        smiles_graph = smile_to_graph(Smiles)




        # 生成 Morgan 指纹
        mol = Chem.MolFromSmiles(Smiles)  # 将 SMILES 转换为分子对象
        ecfp_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        ecfp_array = np.zeros((1,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(ecfp_fp, ecfp_array)


        # 蛋白质图
        protein_seq = dataset.loc[idx, seq]
        seq_1000 = label_sequence(protein_seq, 500)


        contact_map_file = os.path.join(contact_dir, f"{idx+1}.npy")
        contact_map = np.load(contact_map_file)
        protein_graph = target_to_graph(protein_seq, contact_map)

        # 标签
        label = torch.tensor(dataset.loc[idx, target_col], dtype=torch.float)

        # 添加到 processed 列表
        processed.append([Smiles_100, seq_1000, smiles_graph, protein_graph, label])

    return processed, dataset



if __name__ == "__main__":
    try:

        # 测试集处理
        testing_file = '/data/stu1/saj_pycharm_project/KM/dataset/increased.csv'
        testing_contact_dir = '/data/stu1/saj_pycharm_project/KM/dataset/increased_contact_t12'
        # testing_esm = '/data/stu1/saj_pycharm_project/PTCA-PLA/Km/dataset/test480'
        output_testing = "dataset/processed/increased.pt"

        df_test = pd.read_csv(testing_file, sep=",")
        processed_data_test, dataset_test = build_dataset(df_test, "Smiles", "Sequence",testing_contact_dir,"log10_Km")
        torch.save(processed_data_test, output_testing)
        print(f"Testing data processed and saved to {output_testing}")

        # 训练集处理
        training_file = '/data/stu1/saj_pycharm_project/KM/dataset/decreased.csv'
        training_contact_dir = '/data/stu1/saj_pycharm_project/KM/dataset/decreased_contact_t12'
        # training_esm = '/data/stu1/saj_pycharm_project/PTCA-PLA/Km/dataset/train480'
        output_training = "dataset/processed/decreased.pt"

        df_train = pd.read_csv(training_file, sep=",")
        processed_data_train, dataset_train = build_dataset(df_train, "Smiles", "Sequence", training_contact_dir, "log10_Km")
        torch.save(processed_data_train, output_training)
        print(f"Training data processed and saved to {output_training}")

        # 验证集处理
        # validation_file = '/data/stu1/saj_pycharm_project/KM/dataset/wild-type-like.csv'
        # validation_contact_dir = '/data/stu1/saj_pycharm_project/KM/dataset/wild-type-like_contact_t12'
        # # validation_esm = '/data/stu1/saj_pycharm_project/PTCA-PLA/Km/dataset/valid480'
        # output_validation = "dataset/processed/wild-type-like.pt"
        #
        # df_val = pd.read_csv(validation_file, sep=",")
        # processed_data_val, dataset_val = build_dataset(df_val, "Smiles", "Sequence", validation_contact_dir,"log10_Km")
        # torch.save(processed_data_val, output_validation)
        # print(f"Validation data processed and saved to {output_validation}")


    except Exception as e:
        print(f"An error occurred during processing: {e}")
