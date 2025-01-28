import torch
import esm
import pandas as pd
import numpy as np
import os

output_path = 'wild-type_contact_t12'
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results



def Get_Residues_Adjacency(dis_matrix, contact_cutoff=0.5):
    size = len(dis_matrix)
    adj_matrix = [[0 for col in range(size)] for row in range(size)]

    for i in range(size):
        for j in range(size):
            if dis_matrix[i][j] > contact_cutoff:
                adj_matrix[i][j] = 1

    # 设置自连接：每个残基与自身的连接
    for i in range(size):
        adj_matrix[i][i] = 1  # 自连接，即对角线上的元素设置为1

    return adj_matrix


df = pd.read_csv('wild-type.csv')  # 假设您的文件名为 protein_sequences.csv

# 提取蛋白质名称和序列
protein_sequences = [(row['Sample_ID'], row['Sequence']) for _, row in df.iterrows()]

# Prepare a list to store adjacency matrices
adj_matrices = []

# 从第 5791 个样本开始处理
start_index = 0 # Python 中的索引从 0 开始，所以第 5791 个样本是索引 5790

# 处理每个蛋白质序列
for idx, (protein_name, sequence) in enumerate(protein_sequences[start_index:], start=start_index):
    # Convert sequence to tokens
    batch_labels, batch_strs, batch_tokens = batch_converter([(protein_name, sequence)])

    # Calculate sequence length
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Get the contact matrix from the model
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=True)

    # Get the contact matrix (dis_matrix)
    contact_matrix = results["contacts"][0].cpu().numpy()  # Convert to numpy array

    # Generate adjacency matrix
    adj_matrix = Get_Residues_Adjacency(contact_matrix, contact_cutoff=0.5)

    # 保存每个邻接矩阵为 .npy 文件
    npy_file_path = os.path.join(output_path, f'{protein_name}.npy')
    np.save(npy_file_path, adj_matrix)

    print(f'Successfully processed {protein_name} at index {idx+1}')

# Optionally, save all adjacency matrices together in one file
# np.save("all_adj_matrices.npy", np.array(adj_matrices))