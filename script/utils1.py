import torch
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
import math




def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min, lr_max, warmup=True):
    # warmup_epoch: Number of model warm-ups
    warmup_epoch = 10 if warmup else 0
    # Model warm-up phase
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    # Formal training phase of the model
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - max_epoch) / max_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            smile = data[0].to(device)
            sequence = data[1].to(device)
            smile_graph = data[2].to(device)
            pro_graph = data[3].to(device)

            output = model(smile,sequence,smile_graph,pro_graph)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data[4].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def collate(data_list):
    smiles = torch.tensor([data[0] for data in data_list])
    sequence = torch.tensor([data[1] for data in data_list])
    batchA = Batch.from_data_list([data[2] for data in data_list])
    # ecfp = torch.tensor([data[3] for data in data_list])
    batchB = Batch.from_data_list([data[3] for data in data_list])

    targets = torch.tensor([data[4] for data in data_list], dtype=torch.float)  # 标签批次
    return smiles, sequence, batchA,batchB, targets