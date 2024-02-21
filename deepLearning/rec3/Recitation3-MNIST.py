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

## NOTE: In google colab, after this is executed, if you check the files tab on the right <- (in my browswer),
## You should see the data sets stored locally

train_x = mnist_trainset.data / 256.0 # Note the pixel values are integers from 0 to 256, grayscale - we divide to scale them between 0 and 1
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

print("Label of Example Image:", train_y[0]) # Should be a five if your version agrees with mine


# You should get here that the 

TRAIN_N = train_x.shape[0]
TEST_N = test_x.shape[0]

BATCH_SIZE = 64

def get_batch(x, y, batch_size):
    n = x.shape[0] # get the number of data points in the data set
    
    batch_indices = random.sample( [ i for i in range(n) ], k = batch_size )
    # Generate a set of batch_size indices, to select those data points from the data set
    
    x_batch = x[ batch_indices ]
    # x_batch will be of size ( batch_size, 2 ) and be only the selected rows / data points from the data set
    y_batch = y[ batch_indices ]
    
    return x_batch, y_batch

class SimpleMNISTSoftMaxModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTSoftMaxModel, self).__init__()
        
        self.weights0 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias0 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights1 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias1 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights2 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias2 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights3 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias3 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights4 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias4 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights5 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias5 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights6 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias6 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights7 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias7 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights8 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias8 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        self.weights9 = torch.nn.Parameter( torch.randn(28 * 28) * 0.1, requires_grad = True )
        self.bias9 = torch.nn.Parameter( torch.randn(1) * 0.1, requires_grad = True )
        
        
    def forward(self, input_tensor):
        n = input_tensor.shape[0]
        
        flattened = nn.Flatten()( input_tensor )
        # This flattens each row of the tensor from a 28 x 28 matrix to a 28*28 or 784 dimensional vector
        
        linear_functions_by_class = torch.zeros( (n, 10) )
        linear_functions_by_class[:,0] = torch.sum( self.weights0 * flattened, axis = 1 ) + self.bias0
        linear_functions_by_class[:,1] = torch.sum( self.weights1 * flattened, axis = 1 ) + self.bias1
        linear_functions_by_class[:,2] = torch.sum( self.weights2 * flattened, axis = 1 ) + self.bias2
        linear_functions_by_class[:,3] = torch.sum( self.weights3 * flattened, axis = 1 ) + self.bias3
        linear_functions_by_class[:,4] = torch.sum( self.weights4 * flattened, axis = 1 ) + self.bias4
        linear_functions_by_class[:,5] = torch.sum( self.weights5 * flattened, axis = 1 ) + self.bias5
        linear_functions_by_class[:,6] = torch.sum( self.weights6 * flattened, axis = 1 ) + self.bias6
        linear_functions_by_class[:,7] = torch.sum( self.weights7 * flattened, axis = 1 ) + self.bias7
        linear_functions_by_class[:,8] = torch.sum( self.weights8 * flattened, axis = 1 ) + self.bias8
        linear_functions_by_class[:,9] = torch.sum( self.weights9 * flattened, axis = 1 ) + self.bias9
        
        # Note the use of the axis argument in sum - torch.sum( tensor ) by itself adds up all the values in the tensor to return a single scalar
        # But if we specify an axis, it only adds up things along that axis - so torch.sum( tensor, axis = 1 ) on an A x B tensor will return
        # An A tensor, where each entry is the sum of the B entries on that row
        
        
        # At the end of this, linear_functions by class will have n many rows
        # And each row will have 10 columns
        # Column i on row j corresponds to the weight vector for digit i dotted with the data point on row j (plus the bias term)
        
        # We then want to convert these ty probabilities by taking softmax
        
        probabilities = torch.nn.Softmax( dim = 1 )( linear_functions_by_class )
        
        # This will perform softmax over the 1-axis, rather than the entire tensor
        # So each row of 10 linear functions is converted to a row of 10 probabilities
        
        ## As a diagnostic, useful to uncomment these lines
        ## print( probabilities[0] )
        ## probabilities.shape
        
        return probabilities
        
        ## Note - a lot of this could be simplified if I stored the weights for each class in a matrix instead of individual vectors
        ## But I was trying to be as explicit as possible

def confusion_matrix( model, x, y ):
    identification_counts = np.zeros( shape = (10,10), dtype = np.int32 )
    # This is the first time I've used numpy arrays explicily in recitation code 
    
    # The confusion matrix is a way of assessing what a classifier is good or bad at
    # Row 4 of the confusion matrix for instance will list how many times
    # a 4-image was classified as a 0, a 1, a 2, a 3, etc
    
    # Ideally, the classification matrix will be diagonal (perfect classification)
    # But we can see how the matrix improves over training
    
    probaiblities = model( x )
    predicted_classes = torch.argmax( probaiblities, dim = 1 )
    # The use of argmax here, torch will return the index of the term with largest value, along dim 1 (i.e., for each row)
    
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
        # Returns a tensor of size (batch_size x 10) - for each data point in the batch, a vector of 10 probabilities for each class
        # To compute the loss, we need for each data point the probability that data point belongs to the 'true' class, which is stored in y_bach
        
        correct_class_probabilities = torch.gather( probabilities, dim = 1, index = torch.reshape( y_batch, (-1,1) ) ) 
        
        # Note that for data point i, i.e., image x_batch[i], the correct id for that digit is y_batch[i]
        # So the probability that i is assigned to that class is given by probabilities[i, y_batch[i]]
        
        # Gather is a clever slicing/index argument that for each row (indicated by the dim = 1 argument) of the probabilities tensor
        # will select the term given by the index stored in y_batch. Only tricky bit is having to reshape y_batch to get the dimensions right
        # But this command basically does
        # 
        # correct_class_probabilities = probabilities[:, y_batch] to select the desired probabilities using the labels in y_batch
        
        # This indexing trick took me some googling to figure out
        
        loss = torch.mean( -1 * torch.log( correct_class_probabilities + 0.00001 ) ) # softmax loss - we add up the -logs of the correct class probability predictions
        
        loss.backward()
        
        
        
        with torch.no_grad():
            for weight in model.parameters():
                weight -= alpha * weight.grad
                weight.grad = None # Always clear the gradient when you are done with it
        
        total_loss += loss.item()
    
    print( "Total Loss over Batches:",total_loss )
    print("Current Confusion Matrix")
    print( confusion_matrix( model, test_x, test_y ) )
    print()

# And we see that over time the confusion matrix is increasingly diagonal, meaning that the classifier is mostly getting the right idea

