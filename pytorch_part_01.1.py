#6. Putting it all together
#import pytorch and mathplotlib
import torch
from torch import nn #nn contains all of pytorch's building block for neural network
import matplotlib.pyplot as plt
from pprint import pprint# Pretty Print
from pathlib import Path


#Check pytorch version
print("pytorch version: ",torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#Create weight and bias values
weight = 0.7
bias = 0.3
#Create range value
start = 0
end = 1
step = 0.02
#Create X and y (features and label)
X = torch.arange(start= start, end= end, step= step).unsqueeze(dim=1)# without unsqueeze, errors will happen later on (shapes within linear layers)
y = weight * X + bias
print(f"first 10 datas of X: {X[:10]}")
print(f"first 10 datas of y: {y[:10]}")

#Split data
train_split = int(0.8 * len(X))
X_train, y_train  = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(f"The length of X train, y train, X test, y test: {len(X_train)},{len(y_train)},{len(X_test)},{len(y_test)}")
#visualize
def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    #Plot training data, test data and compares predictions
    plt.figure(figsize=(10,7))
    #Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s = 4, label="Training data")
    #Plot test data in green
    plt.scatter(test_data, test_labels,c="g", s=4, label="Testing data")
    if predictions is not None:
        #Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data,predictions, c="r", s=4, label="Prediction")
    #Show the legend
    plt.legend(prop={"size": 15})
    plt.show()
plot_predictions()

#6.2 building a pytorch model
"""
instead of defining the weight and bias parameters 
of our model manually using nn.Parameter(), we'll use 
nn.Linear(in_features, out_features) to do it for us.
in_features: number of dimension of input data
out_features: number of dimensions of output data
"""
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        #Use nn.Linear() to create model parameter
        #in_features is the number of dimensions your input data has and out_features is the number of dimensions you'd like it to be output to.
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    #Define forward computation (input data x flows through nn.Linear())
    def forward(self, x:torch.Tensor) ->torch.Tensor:
        return self.linear_layer(x)
    
#Set munual seed (used for demonstration)
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(f"Model 1: {model_1}")
print(f"Model 1 state dict: {model_1.state_dict()}")

#Check model device
print(next(model_1.parameters()).device)
#Set model to GPU
model_1.to(device)
print(next(model_1.parameters()).device)

#6.3 Training
#Create loss function
loss_fn = nn.L1Loss()
#Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),# optimize newly created model's parameters
                            lr=0.01)

torch.manual_seed(42)
#Set the number of epochs = 1000
epochs = 1000
#Put data on available device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    ###Training
    model_1.train()# train mode is on by default after construction
    #1.Forward pass
    y_pred = model_1(X_train)
    #2 Calculate loss
    loss = loss_fn(y_pred, y_train)
    #3 Zero grad optimizer
    optimizer.zero_grad()
    #4 Loss backward
    loss.backward()
    #5 Step the optimizet
    optimizer.step()

    ###Testing
    model_1.eval()#Put model in evaluation mode for testing
    #1 Forward pass
    with torch.inference_mode():
        test_pred = model_1(X_test)
        #2 Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

#Find model's learned parameters
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weight and bias:")
print(f"Weight: {weight}, Bias: {bias}")

#6.4 Naking predictions
#Turn model's evaluation mode
model_1.eval()
#Make predictions on test
with torch.inference_mode():
    y_preds = model_1(X_test)
print(y_preds)   

# plot_predictions(predictions=y_preds) # -> won't work... data not on CPU
# Put data on the CPU and plot it
plot_predictions(predictions=y_preds.cpu())


#6.5 Saving and loading a model
#1. Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
#2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
#3.Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),# only saving the state_dict() only saves the models learned parameters)
           f=MODEL_SAVE_PATH)
#Load the model
#Instantiate the instance of LinearRegressionModelV2
loaded_model_1 = LinearRegressionModelV2()
#Load model state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
#Put model to target device
loaded_model_1.to(device=device)
print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device\n{next(loaded_model_1.parameters()).device}")
#Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
print(loaded_model_1_preds == y_preds)
