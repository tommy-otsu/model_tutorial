import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1, 100)
        self.l2 = nn.Linear(100, 200)
        self.l3 = nn.Linear(200, 300)
        self.l4 = nn.Linear(300, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    
def get_dataset(n):
    x = 6*numpy.random.random(n) #0〜1のランダム値を取る
    y = numpy.sin(x) #ここを変えれば各種関数で計算できる
    x = x.reshape(n, 1) #(n, 1)の配列にする必要がある
    y = y.reshape(n, 1) #(n, 1)の配列にする必要がある
    x = torch.FloatTensor(x) # floatでないといけない
    y = torch.FloatTensor(y) # floatでないといけない
    return torch.utils.data.TensorDataset(x, y)

data_number = 10000 #準備するデータの数
batch_size  = 1000 # 1つのミニバッチのデータの数
data_loader = torch.utils.data.DataLoader(get_dataset(data_number), batch_size=batch_size)

model = Model()
criterion = nn.MSELoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_log = [] # 学習状況のプロット用
epoch = 100
for t in range(epoch):
    for xt, yt in data_loader: # 1ミニバッチずつ計算
        optimizer.zero_grad() 
        y_model = model(xt)
        loss = criterion(y_model, yt)
        loss.backward()
        optimizer.step()
    print(t, loss.item())
    loss_log.append(loss.item())
    
##学習したモデルと関数を比較する
x = numpy.linspace(0, 6, 100)
plt.plot(x, numpy.sin(x), label='sin(x)')
x_model = torch.FloatTensor(x.reshape(100,1))
y_model = model(x_model)
y = y_model.detach().numpy().reshape(1,100)[0] # プロット用に変換、detach()が必要
plt.plot(x, y, label='model')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.plot(loss_log)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.show()