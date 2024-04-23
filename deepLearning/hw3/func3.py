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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)

class LinearSoftmaxRegression(nn.Module):
    def __init__(self):
        super(LinearSoftmaxRegression, self).__init__()

        self.layer_1 = torch.nn.Linear( in_features = 28*28*1, out_features = 10, bias=True )

    def forward(self, input_tensor):
        flattened = nn.Flatten()( input_tensor )

        logits = self.layer_1( flattened )

        return logits

class badVGGDec(nn.Module):
    def __init__(self):
        super(badVGGDec, self).__init__()

        self.layer_input = nn.Sequential(nn.Linear( in_features = 28*28*1, out_features = 95, bias=True ), nn.LayerNorm(95), nn.ELU())
        self.layer_output = torch.nn.Linear( in_features = 82, out_features = 10, bias=True )
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(95, 90), nn.LayerNorm(90), nn.ELU()),
                                      nn.Sequential(nn.Linear(90, 87), nn.LayerNorm(87), nn.ELU()),
                                      nn.Sequential(nn.Linear(87, 82), nn.LayerNorm(82), nn.ELU())])
    def forward(self, input_tensor):
        output = nn.Flatten()( input_tensor )
        output = self.layer_input(output)
        for l in self.linears:
                output = l(output)
        output = self.layer_output(output)
        return output

def get_batch(x, y, batch_size):
    n = x.shape[0]

    batch_indices = random.sample( [ i for i in range(n) ], k = batch_size )

    x_batch = x[ batch_indices ]
    y_batch = y[ batch_indices ]

    return x_batch, y_batch

def imshow(img):
    img = img / 256
    plt.imshow( img )
    plt.show()

def seqGen():
    ind = random.sample(range(10), 5)
    ind.sort()
    randmu = 2*torch.rand((5)) - 1
    T = random.randint(0, 99)
    haspoint = random.randint(0,1)
    seq = []
    for t in range(100):
        sample = torch.randn((10))
        if t >= T and haspoint == 1:
            j = 0
            for i in range(10):
                if i in ind:
                    sample[i] = sample[i] + randmu[j]
                    j = j+1
            seq.append(sample)
        else:
            seq.append(sample)
    return seq, ind, haspoint, T
def trainNetwork(model, device, train_x, train_y, test_x, test_y, loss_function, optimFunc, batch_size, learnRate):
    model = model.to(device)
    optimizer = optimFunc(model.parameters(), lr = learnRate )
    testLoss = loss_function(model(test_x),test_y).item()
    startTime = time.time()
    for epochs in range(25):
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
