import torch
import torch.nn as nn

class QuadLogisticModel(nn.Module):
    def __init__(self):
        super(QuadLogisticModel, self).__init__()
        
        # Input data will be of the form (x1, x2)
        # So for a linear model, we want to output sigmoid( a * x1^2 + b * x1 * x2 + c * x2^2 + d * x1 + e * x2 + f )
        
        self.a = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        self.b = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        self.c = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        self.d = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        self.e = torch.nn.Parameter( torch.randn(1), requires_grad = True )
        self.f = torch.nn.Parameter( torch.randn(1), requires_grad = True )

    def forward(self, input_x):
        output = self.a * torch.square( input_x[:,0] ) # Note the broadcasting here, computing a vector of x1*x1 values
        output += self.b * input_x[:,0] * input_x[:,1] # Vector of x1*x2 values, scaled by b
        output += self.c * torch.square( input_x[:,1] ) # Vector of x2*x2 values, scaled by c
        output += self.d * input_x[:,0]
        output += self.e * input_x[:,1]
        output += self.f
        
        # output: a vector of a*x1*x1 + b*x1*x2 + c*x2*x2 + d*x1 + e*x2 + f

        output = output.reshape( (-1,1) ) # Reshaped to a column vector

        output = torch.nn.Sigmoid()( output ) # Apply the sigmoid to each term in the vector

        return output

def generate_data(n):
    x = torch.randn( (n,2) )
    noise_values = torch.randn( n ) * 0.1 # we create a noise term for each data point, and scale it by 0.1 to get the right variance
    
    check_threshold = torch.square( x[:,0] ) + torch.square( x[:,1] ) + noise_values
    
    y = torch.zeros( (n,1) )
    for i in range(n):
        if check_threshold[i].item() > 1: # if it is outside a circle of radius 1, class 1, else, class 0
            y[i,0] = 1
        else:
            y[i,0] = 0
    
    return x,y
    # Note, returned x tensor only has 2 columns, because we deal with the constant term and other features in the model
    
def test_accuracy(x,y, model):
    n = x.shape[0] # the number of data points is the size of the 0-th dimension of the x tensor 
    
    y_prob = model(x)
    
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

model = QuadLogisticModel()

print("Initializing Quadratic Logistic Weights:")
for weight in model.named_parameters():
    print( weight )
print("Initial Model Accuracy ( prob >= 0.5, guess y = 1, prob < 0.5, guess y = 0):", test_accuracy(x,y, model))

alpha = 0.05

print("Entering Training Loop")

for i in range(10):
    y_prob = model(x)
    loss_by_data_point = -1 * y * torch.log( y_prob + 0.0001 ) - (1 - y) * torch.log( 1 - y_prob + 0.0001 )
    loss = torch.mean( loss_by_data_point )
    
    loss.backward()

    with torch.no_grad():
        for weight in model.parameters():
            weight -= alpha * weight.grad
            weight.grad = None
    
    print("Current Loss:", loss.item())
    print("Current Accuracy:", test_accuracy(x,y, model))
    print()

print("Final Accuracy:", test_accuracy(x,y, model) ) # Final accuracy, hopefully close to 100%

print("Final Weights")
for weight in model.named_parameters():
    print( weight )