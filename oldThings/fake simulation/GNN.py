import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

# Define a simple Graph Neural Network with attention mechanism
class GNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNWithAttention, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Add self loops
        edge_index, _ = add_self_loops(edge_index)
        
        # Propagate input features through the graph
        h = F.relu(self.conv1(x, edge_index))
        
        # Compute attention weights
        att_weights = F.softmax(self.attention_layer(h), dim=0)
        
        # Apply attention to node representations
        h_att = torch.matmul(att_weights.transpose(1,0), h)
        
        # Final output
        output = self.output_layer(h_att)
        return output

# Main function to train the model
def train_model():
    # Hyperparameters
    input_dim = 5
    hidden_dim = 32
    output_dim = 5
    num_epochs = 100
    learning_rate = 0.001
    num_samples = 1000
    
    # Initialize model and optimizer
    model = GNNWithAttention(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate synthetic dataset and graph
    inputs = torch.randn(num_samples, input_dim)
    edge_index = torch.tensor([[i, i+1] for i in range(output_dim-1)], dtype=torch.long).t().contiguous()
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs_pred = model(inputs, edge_index)
        
        # Generate synthetic outputs
        outputs = torch.randn(num_samples, output_dim)
        
        # Compute loss
        loss = F.mse_loss(outputs_pred, outputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train the model
train_model()
