import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch import nn
import random
import matplotlib.pyplot as plt
import torch.optim as optim # Importing built in optimizers, for instance for doing gradient descent


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

train_x = mnist_trainset.data / 256.0 # Scale values between 0 and 1
train_y = mnist_trainset.targets

test_x = mnist_testset.data / 256.0
test_y = mnist_testset.targets

print("Training Data Inputs:", train_x.shape )
print("Training Data Outputs:", train_y.shape )

# We have 60,000 images, each 28 x 28 pixels, where pixel values are between 0 and 1

example_image = train_x[0]

fig = plt.figure
plt.imshow(example_image, cmap='gray')
plt.show()

print("Label of Example Image:", train_y[0])

TRAIN_N = train_x.shape[0]
TEST_N = test_x.shape[0]

BATCH_SIZE = 64

def get_batch(x, y, batch_size):
    n = x.shape[0]
    
    batch_indices = random.sample( [ i for i in range(n) ], k = batch_size )

    x_batch = x[ batch_indices ]
    y_batch = y[ batch_indices ]
    
    return x_batch, y_batch

class SimpleMNISTSoftMaxModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTSoftMaxModel, self).__init__()
                
        self.linear_layer = torch.nn.Linear( in_features = 28*28, out_features = 10, bias=True )
        
    def forward(self, input_tensor):
        flattened = nn.Flatten()( input_tensor )
        
        output = self.linear_layer( flattened )
        
        logits = torch.nn.LogSoftmax( dim = 1 )( output )
        # Computes the softmax and then takes the log - built in, for additional
        # numerical stability
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax
        
        # If we wanted the final probabilities, we just need to take the result and to torch.exp
        # or np.exp and pass it these 'log-probabiities' or logits.
        return logits

def confusion_matrix( model, x, y ):
    identification_counts = np.zeros( shape = (10,10), dtype = np.int32 )
    
    logits = model( x )
    predicted_classes = torch.argmax( logits, dim = 1 )
    
    n = x.shape[0]
    
    for i in range(n):
        actual_class = y[i]
        predicted_class = predicted_classes[i].item()
        identification_counts[actual_class, predicted_class] += 1
    
    return identification_counts

model = SimpleMNISTSoftMaxModel()

optimizer = optim.SGD(model.parameters(), lr = 1.0 )

loss_function = torch.nn.CrossEntropyLoss()
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# Pre defined loss function
# we call it by evaluating loss_function( logits, target_labels )

print("Initial Confusion Matrix")
print( confusion_matrix( model, test_x, test_y ) )

for epochs in range(10):
    total_loss = 0
    for batch in range( TRAIN_N // BATCH_SIZE ):
        x_batch, y_batch = get_batch(train_x, train_y, BATCH_SIZE)
        
        optimizer.zero_grad()
        
        logits = model( x_batch )
        loss = loss_function( logits, y_batch )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print( "Total Loss over Batches:",total_loss )
    print("Current Confusion Matrix")
    print( confusion_matrix( model, test_x, test_y ) )
    print()

print( "Weight Matrix: ", model.linear_layer.weight )
print( "Bias Vectors:", model.linear_layer.bias )