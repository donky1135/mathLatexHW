import torch
import torch.nn as nn

class LinearLogisticModel(nn.Module):
    def __init__(self):
        super(LinearLogisticModel, self).__init__()
        
        # Input data will be of the form (x1, x2)
        # So for a linear model, we want to output sigmoid( a * x1 + b * x2 + c )
        
        self.a = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        self.b = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        self.c = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        
        # We intialize (trainable) parameters for each of these terms

    def forward(self, input_x):
        # the input tensor is of dimension (N,2) where N is the number of data points
        # the first column is going to be the x1 values for each data point
        # the second column is going to be the x2 values for each data point
        
        # the forward method is where we perform the forward computation
        
        ## Uncomment for first run through
        ##print("Input x shape:", input_x.shape )
        #####################################################
        output = self.a * input_x[:,0] + self.b * input_x[:,1] + self.c
        # Note the use of broadcasting here - c is a scalar term, but gets added to every term in the result
        
        ## Uncomment this for first run through
        ##print("Current output shape:", output.shape )
        #####################################################
        
        
        output = output.reshape( (-1,1) )
        # reshape is a funny command, you pass it the desired shape of a tensor and it will restructure the result to match
        # a 'size' of -1 basically says "shape the tensor here however it needs to be, to get the other dimension to match"
        # so a shape (100) tensor (essentially a 1-dimensional row of 100 values) reshaped by (-1,1) will turn into a (100,1) dimensional tensor
        
        ## Uncomment this for the first run through
        ##print("Final output shape:", output.shape )
        #####################################################
        ## output will be a tensor of size (N,1)
        
       
        
        output = torch.nn.Sigmoid()( output )
        # this applies the sigmoid function to each term of the tensor
        # the result will be a tensor of (N,1) many values, where each one is between 0 and 1
        
        ## Uncomment for the first run through
        ##print("Test Value:", output[-1])
        #####################################################
        
        return output

def generate_data(n):
    x = torch.randn( (n,2) )
    noise_values = torch.randn( n ) * 0.1 # we create a noise term for each data point, and scale it by 0.1 to get the right variance
    
    check_threshold = ( x[:,0] + x[:,1] + noise_values ) # we can compute the linear function of the data, as before, but a little bit more compact
    
    y = torch.zeros( (n,1) )
    for i in range(n):
        if check_threshold[i].item() > 0:
            y[i,0] = 1
        else:
            y[i,0] = 0
            
    # Note that the returned x vector here is only two columns
    # Because we deal with the constant/bias term in the module
    
    return x,y
    
def test_accuracy(x,y, model):
    n = x.shape[0] # the number of data points is the size of the 0-th dimension of the x tensor 
    
    y_prob = model(x)
    # note that all the parameters are wrapped up inside the model object
    # we just call it like a function, which triggers the 'forward' funcion call
    
    misclassifications = 0
    for i in range(n):
        if y_prob[i,0] >= 0.5 and y[i,0] == 0:# if probability is high but tag is 0, bad!
            misclassifications += 1
        elif y_prob[i,0] < 0.5 and y[i,0] == 1:# if probability is low but tag is 1, bad!
            misclassifications += 1
    
    return 1 - misclassifications / n # accuracy is 1 - (errors)

n_data = 100

##########################################################
x,y = generate_data( n_data )
print("Initial Data")
print( "X shape: ", x.shape )
print( "y shape: ", y.shape )

for i in range( n_data ):
    print("x:", x[i,:], "\ty:", y[i,0])
##########################################################

model = LinearLogisticModel()

print("Initializing Logistic Weights:")
for weight in model.named_parameters():
    print( weight )
print("Initial Model Accuracy ( prob >= 0.5, guess y = 1, prob < 0.5, guess y = 0):", test_accuracy(x,y, model))

alpha = 0.05

print("Entering Training Loop")

for i in range(1000):
    y_prob = model(x)
    loss_by_data_point = -1 * y * torch.log( y_prob ) - (1 - y) * torch.log( 1 - y_prob )
    loss = torch.mean( loss_by_data_point )
    
    loss.backward()
    # Compute derivatives
    
    with torch.no_grad():
        for weight in model.parameters():
            weight -= alpha * weight.grad
            weight.grad = None
    
    print("Current Loss:", loss.item())
    print("Current Accuracy:", test_accuracy(x,y, model))
    print()
    # What you should see that as the loss goes down, the accuracy goes up, and the weights more accurately
    # reflect that we want 0 + 1*x1 + 1*x2 > 0

print("Final Accuracy:", test_accuracy(x,y, model) ) # Final accuracy, hopefully close to 100%

print("Final Weights")
for weight in model.named_parameters():
    print( weight )

# Some things to note
# - The loop is done to 1000 iterations - this is arbitrary and could be changed. How?
# - The choice of stepsize? Arbitrary, could be changed.

# Last thing to point out, the final weights may not actually be close to [0,1,1]
# The weights could also be close to [0,2,2]
# or [0,10,10]
# Note that if the 'true' dividing line is w0 + w1 x1 + w2 x2 = 0, then any scalar multiple
# is going to give the exact same dividing line


