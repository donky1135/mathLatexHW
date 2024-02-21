import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch import nn
import random
import matplotlib.pyplot as plt


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
        # Linear layers contain a weight matrix (28*28 columns, 10 rows here)
        # and a bias vector (because bias is true )
        
        # I'd like you to potentially show them
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Point out in particular how the weight and bias terms are initialized, U(-sqrt(k), sqrt(k)),
        # where k = 1 / in_features
        
        # Note that this means that the larger the input is, the smaller (though random) the initial
        # weights/bias term
        
    def forward(self, input_tensor):
        flattened = nn.Flatten()( input_tensor )
        
        output = self.linear_layer( flattened )
        # We can call the linear layer as a function, transforming the flattened (n x 784) data to (n x 10)

        probabilities = torch.nn.Softmax( dim = 1 )( output )
        
        return probabilities

def confusion_matrix( model, x, y ):
    identification_counts = np.zeros( shape = (10,10), dtype = np.int32 )
    
    probaiblities = model( x )
    predicted_classes = torch.argmax( probaiblities, dim = 1 )
    
    n = x.shape[0]
    
    for i in range(n):
        actual_class = y[i]
        predicted_class = predicted_classes[i].item() # We have to use item to get the value out of the tensor
        identification_counts[actual_class, predicted_class] += 1
    
    return identification_counts

model = SimpleMNISTSoftMaxModel()

print("Initial Confusion Matrix")
print( confusion_matrix( model, test_x, test_y ) )

alpha = 1.0

for epochs in range(10):
    total_loss = 0
    for batch in range( TRAIN_N // BATCH_SIZE ):
        x_batch, y_batch = get_batch(train_x, train_y, BATCH_SIZE)
        
        probabilities = model( x_batch )
        
        correct_class_probabilities = torch.gather( probabilities, dim = 1, index = torch.reshape( y_batch, (-1,1) ) ) 
        
        loss = torch.mean( -1 * torch.log( correct_class_probabilities + 0.00001 ) )
        
        loss.backward()
        
        with torch.no_grad():
            for weight in model.parameters():
                weight -= alpha * weight.grad
                weight.grad = None
        
        total_loss += loss.item()
    
    print( "Total Loss over Batches:",total_loss )
    print("Current Confusion Matrix")
    print( confusion_matrix( model, test_x, test_y ) )
    print()

print( "Weight Matrix: ", model.linear_layer.weight )
print( "Bias Vectors:", model.linear_layer.bias )