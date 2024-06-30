import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from AI_Model import UNet1d
import numpy as np
from sklearn.model_selection import train_test_split

from Libary.function import read_file

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# Example data
# returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Burning_2.txt")

# param = returnVal[0]
# strain = np.array(returnVal[1])/(-4)
# stress = np.array(returnVal[2])/(300)
# bodyOpen = np.array(returnVal[3])/(20)

# labels = np.concatenate((strain, stress, bodyOpen),axis=1)
# data = param

# data = data.reshape((len(param), 1, 11))
# labels = labels.reshape((len(param), 1, 153))

data = np.random.rand(100, 16)  # 100 samples, 16 length sequences
labels = np.random.rand(100, 16*3)

data = data.reshape((100, 1, 16))
labels = labels.reshape((100, 3, 16))

# Split the data
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_data = torch.tensor(val_data, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = CustomDataset(train_data, train_labels)
val_dataset = CustomDataset(val_data, val_labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Instantiate the model
model = UNet1d(in_channels=1, out_channels=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate training loss
    train_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            v_loss = criterion(val_outputs, val_labels)
            val_loss += v_loss.item()
    
    # Calculate validation loss
    val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Calculate accuracy (Mean Squared Error) for the training set
model.eval()
train_mse = 0.0
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        mse = criterion(outputs, labels)
        train_mse += mse.item()

train_mse = train_mse / len(train_loader)

# Calculate accuracy (Mean Squared Error) for the validation set
val_mse = 0.0
with torch.no_grad():
    for val_inputs, val_labels in val_loader:
        val_outputs = model(val_inputs)
        mse = criterion(val_outputs, val_labels)
        val_mse += mse.item()

val_mse = val_mse / len(val_loader)

print(f"Training MSE: {train_mse:.4f}, Validation MSE: {val_mse:.4f}")
torch.save(model.state_dict(), 'unet1d_model.pth')