import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import math
import time
import func3
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_X = torch.Tensor( trainset.data/255.0 - 0.5 )
train_X = train_X.permute( 0, 3, 1, 2 )


test_X = torch.Tensor( testset.data/255.0 - 0.5 )
test_X = test_X.permute( 0, 3, 1, 2 )
test_X = test_X.to(device)

train_Y = torch.Tensor( np.asarray( trainset.targets ) ).long()
test_Y = torch.Tensor( np.asarray( testset.targets ) ).long()
test_Y = test_Y.to(device)

def confusion_matrix( model, x, y ):
  identification_counts = np.zeros( shape = (10,10), dtype = np.int32 )

  logits = model( x )
  predicted_classes = torch.argmax( logits, dim = 1 )

  n = x.shape[0]

  for i in range(n):
    actual_class = int( y[i].item() )
    predicted_class = predicted_classes[i].item()
    identification_counts[actual_class, predicted_class] += 1

  return identification_counts

class badVGGprime(nn.Module):
    def __init__(self):
        super(badVGGprime, self).__init__()
        self.layer1In = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer1Main = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()) for _ in range(31)])
        
        self.layer2In = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.layer2Main = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()) for _ in range(15)])
        
        self.layer3In = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.layer3Main = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()) for _ in range(4)])
        
        self.layerOut = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linFinal = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),  # Adjusted input size based on max pooling
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 10))
        
    
        # self.layer1In = nn.Sequential(nn.Conv2d(in_channels=3, out_channels = 64, kernel_size = 3, stride = 1, bias=True, padding = "valid"),nn.BatchNorm2d(64),nn.ReLU())
        # self.layer1Main = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels = 64, kernel_size = 3, stride = 1, bias=True),nn.BatchNorm2d(64),nn.ReLU()) for i in range(31)])
        # self.layerOut = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.layer2In = nn.Sequential(nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, stride = 1, bias=True),nn.BatchNorm2d(128),nn.ReLU())
        # self.layer2Main = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels = 128, kernel_size = 3, stride = 1, bias=True),nn.BatchNorm2d(128),nn.ReLU()) for i in range(15)])
        # self.layer3In = nn.Sequential(nn.Conv2d(in_channels=128, out_channels = 256, kernel_size = 3, stride = 1, bias=True),nn.BatchNorm2d(256),nn.ReLU())
        # self.layer2Main = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels = 256, kernel_size = 3, stride = 1, bias=True),nn.BatchNorm2d(256),nn.ReLU()) for i in range(7)])
        # self.linFinal = nn.Sequential(nn.Linear(256*4, 1024), nn.LayerNorm(1024), nn.ELU(), nn.Linear(1024,1024),nn.LayerNorm(1024), nn.ELU(),nn.Linear(1024,1))
    # def forward(self,image):
    #     x = self.layer1In(image)
    #     for i in range(31):
    #         x = self.layer1Main[i]
    #     x = self.layerOut(x)
    #     x = self.layer2In(image)
    #     for i in range(15):
    #         x = self.layer2Main[i]
    #     x = self.layerOut(x)
    #     x = self.layer3In(image)
    #     for i in range(7):
    #         x = self.layer3Main[i]
    #     x = self.layerOut(x)
    #     return self.linFinal(x)
    def forward(self, image):
        x = self.layer1In(image)
        for layer in self.layer1Main:
            x = layer(x)
        x = self.layerOut(x)
        
        x = self.layer2In(x)  # Changed from `image` to `x`
        for layer in self.layer2Main:
            x = layer(x)
        x = self.layerOut(x)
        
        x = self.layer3In(x)  # Changed from `image` to `x`
        for layer in self.layer3Main:
            x = layer(x)
        x = self.layerOut(x)
        
        # Flatten before passing to the fully connected layers
        x = x.view(x.size(0), -1)
        
        return self.linFinal(x)
    
def get_batch(x, y, batch_size, device):
    n = x.shape[0]

    batch_indices = random.sample( [ i for i in range(n) ], k = batch_size )

    x_batch = x[ batch_indices ]
    y_batch = y[ batch_indices ]

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)    
    return x_batch, y_batch


batch_size = 256
learnRate = 0.01
model = badVGGprime().to(device)
optimizer = optim.Adam(model.parameters(), lr = learnRate )
loss_function = nn.CrossEntropyLoss()
test_x, test_y = get_batch(test_X, test_Y, 16, device)
print(test_x.shape)
print(test_y.shape)
print(test_y[0])
testLoss = loss_function(model(test_x),test_y).item()
startTime = time.time()
for epochs in range(25):
    total_loss = 0
    for batch in range( train_X.shape[0] // batch_size ):
        x_batch, y_batch = get_batch(train_X, train_Y, batch_size, device)
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
    # print(confusion_matrix(model, test_X, test_Y))
    test_x, test_y = get_batch(test_X, test_Y, 64, device)
    testLoss = loss_function(model(test_x),test_y).item()
    gc.collect()
endTime = time.time()
test_x, test_y = get_batch(test_X, test_Y, 1000, device)
print(confusion_matrix(model, test_X, test_Y))
print(endTime-startTime, loss_function(model(test_x), test_y).item(), testLoss)