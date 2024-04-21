import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

from FlyObjectPlot import simulateFunction

# Define a simple Graph Neural Network model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)
    
def runSimulation(params):
    y_values = simulateFunction(params)
    return y_values

def generateSeed(bestSeed):
    seedCollector = [bestSeed]
    for _ in range(511):
        changeTime = np.random.randint(1,10)
        newSeed = np.copy(bestSeed)
        for _ in range(changeTime):
            newSeed[np.random.randint(10)] += np.random.rand()
        seedCollector.append(np.clip(newSeed,0,1))
    return np.concatenate((np.array(seedCollector),np.random.rand(512,10)),axis=0)

X_train = []
y_train = []

#Init
for _ in range(64):
    addX = np.random.rand(10)
    X_train.append(addX)
    y_train.append(runSimulation(addX))

print("Gen data completed")
X = torch.tensor(X_train,dtype=torch.float32)
Y = torch.tensor(y_train,dtype=torch.float32)

# You need to construct the edge_index based on your data's structure
# If you have a fully connected graph, you can construct it as follows:
num_nodes = 10
edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()

# Create a PyTorch Geometric Data object
data = Data(x=X, edge_index=edge_index)

# Define hyperparameters
input_dim = X.size(1)  # Input dimension
hidden_dim = 64  # Hidden dimension
output_dim = Y.size(1)  # Output dimension
lr = 0.01  # Learning rate
epochs = 100  # Number of training epochs

# Initialize model, optimizer, and loss function
model = GNNModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# After training, you can use the model for prediction
model.eval()
predicted_output = model(data)

# You can further evaluate the performance of your model
# For example, you can compute metrics like Mean Squared Error (MSE), etc.
