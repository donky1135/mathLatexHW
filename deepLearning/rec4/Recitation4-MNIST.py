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
        
        self.weight_matrix = torch.nn.Parameter( torch.randn(10, 28*28), requires_grad = True )
        self.bias_vector = torch.nn.Parameter( torch.randn(1,10), requires_grad = True )
        
    def forward(self, input_tensor):
        n = input_tensor.shape[0]
        
        flattened = nn.Flatten()( input_tensor )
        # This flattens each row of the tensor from a 28 x 28 matrix to a 28*28 or 784 dimensional vector
        
        output = torch.matmul( flattened, self.weight_matrix.t() ) + self.bias_vector
        # Taking the transpose of the weight matrix, we multiply each data point (row of flattened)
        # by the weight matrix, to get 10-dimensional output vectors

        # We then want to convert these to probabilities by taking softmax
        
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