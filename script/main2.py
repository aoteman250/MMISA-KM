from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

n_epoch=100

model = GNNNet()
model = model.to(device)

train_set = torch.load("dataset/processed/training.pt")
# valid_set = torch.load("dataset/processed/validation.pt")
test_set = torch.load("dataset/processed/testing.pt")

trainloader = Data.DataLoader(train_set, batch_size=256, shuffle=True,collate_fn=collate)
# validloader = Data.DataLoader(valid_set, batch_size=128, shuffle=True,collate_fn=collate)
testloader = Data.DataLoader(test_set, batch_size=256, shuffle=True,collate_fn=collate)


Learning_rate = 1e-4
train_size = len(test_set )
optimizer = optim.AdamW(model.parameters(),lr=Learning_rate,weight_decay=1e-4)
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=Learning_rate, max_lr=Learning_rate * 10,
#                                         cycle_momentum=False,
#                                         step_size_up=train_size // 128)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5, eta_min=1e-3)
loss_function = nn.MSELoss()
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
#                                             epochs=n_epoch,
#                                           steps_per_epoch=len(trainloader))

start = datetime.now()
print('start at ', start)



best_epoch = -1
best_mse = 1000
for epoch in range(n_epoch):
    total_loss = 0.0
    tbar = tqdm(enumerate(trainloader), disable=not SHOW_PROCESS_BAR, total=len(trainloader))
    for idx, data in tbar:
        model.train()
        smile = data[0].to(device)
        sequence = data[1].to(device)
        smile_graph = data[2].to(device)
        pro_graph = data[3].to(device)

        optimizer.zero_grad()
        output = model(smile,sequence,smile_graph,pro_graph)
        loss = loss_function(output.view(-1), data[4].view(-1).to(device).float())
        loss.backward()
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=n_epoch, lr_min=1e-4,
                             lr_max=1e-3,
                             warmup=True)
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item()  # 累积当前批次的损失值
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() / len(data[4]):.3f}')


    # 打印当前 epoch 的总损失值
    avg_loss = total_loss / len(trainloader)
    print(f'Epoch {epoch}, Total Loss: {avg_loss:.3f}')
    print('-------------------------------------------------')
    print('predicting for valid data')
    G, P = predicting(model, device, testloader)
    val = get_mse(G, P)
    calculate_metrics(G, P)
    print('valid result:', val, best_mse)
    if val < best_mse:
        best_mse = val
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'trained_model2.pt')
        print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_mse, )
    else:
        print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, )




print('-------------------------------------------------')
print('-------------------------------------------------')
print('TEST')
model.load_state_dict(torch.load('trained_model2.pt'))
G, P = predicting(model, device, testloader)
calculate_metrics(G, P)

print('training finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
