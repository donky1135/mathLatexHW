import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

test_x = torch.Tensor( testset.data ) / 256.0 - 0.5
test_x = test_x.to(device)
test_y = torch.Tensor( testset.targets ).long()
test_y = test_y.to(device)
train_x = torch.Tensor( trainset.data ) / 256.0 - 0.5
train_x = train_x.to(device)
train_y = torch.Tensor( trainset.targets ).long()
train_y = train_y.to(device)

def get_batch(x, y, batch_size):
    n = x.shape[0]

    batch_indices = random.sample( [ i for i in range(n) ], k = batch_size )

    x_batch = x[ batch_indices ]
    y_batch = y[ batch_indices ]

    return x_batch, y_batch

def k_num(m,P):
    return 1+((P-10 - 795*m)//(m*(m+1)))

class autoEncoder(nn.Module):
    def __init__(self,k,m):
        super(autoEncoder, self).__init__()
        self.layer_inputF = torch.nn.Linear( in_features = 28*28*1, out_features = m, bias=True )
        self.linearsF = nn.ModuleList([nn.Linear(m, m) for i in range(k-1)])
        self.linearsG = nn.ModuleList([nn.Linear(m, m) for i in range(k-1)])
        self.layer_outputG = torch.nn.Linear(m,784, bias=True)
        self.normalize = nn.LayerNorm(m)
    def First(self, input_tensor):
        output = nn.Flatten()( input_tensor )
        output = self.layer_inputF(output)
        output = nn.ELU()(output)
        output = self.normalize(output)
        for l in self.linearsF:
            output = l(output)
            output = nn.ELU()(output)
            output = self.normalize(output)
        return output
    def G(self, compView):
        output = compView
        for l in self.linearsG:
            output = l(output)
            output = nn.ELU()(output)
            output = self.normalize(output)
        output = self.layer_outputG(output)
        return output
    def forward(self, input_tensor):
        comp = self.First(input_tensor)
        return self.G(comp)

P = 50000
batch_size = 1024
finalLossTest = []
finalLossTrain = []
trainingTime = []
loss_function = nn.MSELoss()
for k in range(10,20+1):
    model = autoEncoder(k_num(k,P),k)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001 )
    startTime = time.time()
    for epochs in range(25):
        total_loss = 0
        for batch in range( train_x.shape[0] // batch_size ):
            x_batch, y_batch = get_batch(train_x, train_y, batch_size)
            
            optimizer.zero_grad()
    
            loss = loss_function( model(x_batch), nn.Flatten()(x_batch) )
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        print( k,"input node(s), total Loss over Batches:",total_loss )
    endTime = time.time()
    trainingTime.append(endTime - startTime)
    finalLossTest.append(loss_function( model(test_x), nn.Flatten()(test_x) ))
    finalLossTrain.append(loss_function( model(train_x), nn.Flatten()(train_x) ))
    modelCollect.append(model)

print(finalLossTest)
print(finalLossTrain)
print(trainingTime)
