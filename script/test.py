from datetime import datetime
from torch import  nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

from model2 import *
import os
import numpy as np
import torch.utils.data as Data
from utils1 import *
from emetrics import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SHOW_PROCESS_BAR = True
data_path = '../data/'

device = torch.device("cuda")














def calculate_metrics(Y, P):


    rmse = get_rmse(Y, P)
    r2 = get_r2(Y, P)


    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)




    print('r2:', r2)


    print('mse:', mse)
    print('rmse:', rmse)
    print('pearson', pearson)







if __name__ == '__main__':
    model = GNNNet()
    model = model.to(device)

    test_set = torch.load("/data/stu1/saj_pycharm_project/KM/dataset/processed/increased.pt")

    testloader = Data.DataLoader(test_set, batch_size=128, shuffle=True, collate_fn=collate)

    start = datetime.now()
    print('start at ', start)

    model.load_state_dict(torch.load('trained_model2.pt'))
    G, P = predicting(model, device, testloader)
    calculate_metrics(G, P)


    df = pd.DataFrame({
        'True Values': G.flatten(),  # 真实值
        'Predicted Values': P.flatten()  # 预测值
    })

    # 保存到 CSV 文件
    df.to_csv('predictions-increased.csv', index=False)
    print("The predictions and true values have been saved to 'predictions.csv'.")





