import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot

from dataloader import train_dataloader, val_dataloader
from model import BRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 6
sequence_length = 335413
num_layers = 1
hidden_size = 125
num_classes = 4
learning_rate = 0.001
batch_size = 128
num_epochs = 1

device = 'mps'

model = BRNN(input_size, hidden_size, num_layers, num_classes, device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

print("training")

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    for batch_idx, (data, y) in enumerate(train_dataloader):
        optimizer.zero_grad()

        data = data.unsqueeze(dim=1)
        data = data.to(device = device)
        # one hot encoding
        y = one_hot(y, num_classes)
        y = y.to(device = device).float()

        y_pred = model(data)
        loss = criterion(y_pred, y)

        train_loss += loss
        train_acc += accuracy_fn(y_true=y.argmax(dim=1), y_pred=y_pred.argmax(dim=1))
        loss.backward()

        optimizer.step()
    
        
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    model.eval() # put model in eval mode
    # Turn on inference context manager
    val_loss = 0
    val_acc = 0
    with torch.inference_mode(): 
        for batch_idx, (data, y) in enumerate(val_dataloader):

            data = data.unsqueeze(dim=1)
            data = data.to(device = device)
            # one hot encoding
            y = one_hot(y, num_classes)
            y = y.to(device = device).float()
            # 1. Forward pass
            test_pred = model(data)
            
            # 2. Calculate loss and accuracy
            val_loss = criterion(test_pred, y)
            val_acc += accuracy_fn(y_true=y.argmax(dim=1), y_pred=y_pred.argmax(dim=1))
        
        # Adjust metrics and print out
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

    
    print(f"Epoch: {epoch} | Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}% | Val accuracy: {val_acc:.2f}%")
