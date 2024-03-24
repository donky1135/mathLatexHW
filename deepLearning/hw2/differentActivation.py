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
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

test_x = torch.Tensor( testset.data, device=device ) / 256.0 - 0.5

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

def trainNetwork(model, device, train_x, train_y, text_x, test_y, loss_function, optimFunc, batch_size, learnRate):
    model = model.to(device)
    optimizer = optimFunc(model.parameters(), lr = learnRate )
    testLoss = loss_function(model(test_x),test_y).item()
    startTime = time.time()
    for epochs in range(30):
        total_loss = 0
        for batch in range( train_x.shape[0] // batch_size ):
            x_batch, y_batch = get_batch(train_x, train_y, batch_size)
            
            optimizer.zero_grad()
    
            logits = model( x_batch )
            loss = loss_function( logits, y_batch )
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        print("epochs: " + str(1+epochs) + " total Loss over Batches: " + str(total_loss) + " test loss: " + str(testLoss) )
        # if loss_function(model(test_x),test_y).item() > testLoss*1.1:
        #     break
        # else:
        testLoss = loss_function(model(test_x),test_y).item()
    endTime = time.time()
    return endTime-startTime, loss_function(model(train_x),train_y).item(), testLoss

class badVGGDecACTIVE(nn.Module):
    def __init__(self, activate):
        super(badVGGDecACTIVE, self).__init__()

        self.layer_input = nn.Sequential(nn.Linear( in_features = 28*28*1, out_features = 95, bias=True ), nn.LayerNorm(95), activate)
        self.layer_output = torch.nn.Linear( in_features = 82, out_features = 10, bias=True )
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(95, 90), nn.LayerNorm(90), activate),
                                      nn.Sequential(nn.Linear(90, 87), nn.LayerNorm(87), activate),
                                      nn.Sequential(nn.Linear(87, 82), nn.LayerNorm(82), activate)])
    def forward(self, input_tensor):
        output = nn.Flatten()( input_tensor )
        output = self.layer_input(output)
        for l in self.linears:
                output = l(output)
        output = self.layer_output(output)
        return output
loss_function = loss_function = torch.nn.CrossEntropyLoss()
telemVectReLU = []
telemVectSigmoid = []
telemVectTanh = []

a = int(sys.argv[2])
b = int(sys.argv[3])
c = int(sys.argv[1])
for i in range(a,b):
    modelReLU = badVGGDecACTIVE(nn.ReLU())
    modelSigmoid = badVGGDecACTIVE(nn.Sigmoid())
    modelTanh = badVGGDecACTIVE(nn.Tanh())

    if c == 1:
        telemVectReLU.append(trainNetwork(modelReLU, device, train_x, train_y, test_x, test_y, loss_function, optim.Adam, 2**i, 0.0004))
    elif c == 2:
        telemVectSigmoid.append(trainNetwork(modelSigmoid, device, train_x, train_y, test_x, test_y, loss_function, optim.Adam, 2**i, 0.0004))
    else:
        telemVectTanh.append(trainNetwork(modelTanh, device, train_x, train_y, test_x, test_y, loss_function, optim.Adam, 2**i, 0.0004))

print(telemVectReLU)
print(telemVectSigmoid)
print(telemVectTanh)