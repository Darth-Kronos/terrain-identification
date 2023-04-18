import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np

from dataloader import train_dataloader, val_dataloader
from model import BRNN, OneDConvNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 6
sequence_length = 335413
num_layers = 2
hidden_size = 125
num_classes = 4
learning_rate = 0.01
batch_size = 128
num_epochs = 1

device='mps'

# model = BRNN(input_size, hidden_size, num_layers, num_classes, device).to(device)
model = OneDConvNet(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(X, y, model, optimizer, criterion):

    y_pred = model(X)
    predicted_classes = torch.argmax(y_pred.detach(), dim=1)

    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    corrects = torch.sum(y.data == predicted_classes)

    return loss.item(), corrects

def val_step(X, y, model, criterion):

    with torch.no_grad():
        
        y_pred = model(X)
        predicted_classes = torch.argmax(y_pred.detach(), dim=1)
        loss = criterion(y_pred, y)
        corrects = torch.sum(y.data == predicted_classes)

    return loss.item(), corrects, predicted_classes.detach().cpu().numpy()

best_val_loss = float("inf")
stats = {
  "acc_x": {
    "min": -39.13261,
    "max": 39.26
  },
  "acc_y": {
    "min": -38.92137,
    "max": 39.49
  },
  "acc_z": {
    "min": -31.50025,
    "max": 38.1886
  },
  "gyro_x": {
    "min": -11.62605,
    "max": 10.72668
  },
  "gyro_y": {
    "min": -12.19817,
    "max": 10.93212
  },
  "gyro_z": {
    "min": -6.345545,
    "max": 8.093803
  }
}
min = np.array([v["min"] for k, v in stats.items()])
max = np.array([v["max"] for k, v in stats.items()])

min = torch.from_numpy(min).float()
min = torch.unsqueeze((torch.unsqueeze(min, 0)), -1)
min = min.to(device)
max = torch.from_numpy(max).float().to(device)
max = torch.unsqueeze((torch.unsqueeze(max, 0)), -1)
max = max.to(device)

for epoch in range(num_epochs):
    # Train for "n" number of iterations
    running_loss = 0.
    running_acc = 0.
    for iteration, (X, y) in enumerate(train_dataloader):

        X = X.float().to(device)
        # Normalize
        X = (X - min) / (max - min)

        y = y.view(X.size(0)).to(device)

        loss, corrects = train_step(X, y, model, optimizer, criterion)

        # Running metrics
        running_loss = running_loss + loss * X.size(0)
        running_acc = running_acc + corrects

        

    train_loss = running_loss / len(train_dataloader)
    train_acc = running_acc / len(train_dataloader)

    # Validate
    running_val_loss = 0.
    running_val_acc = 0.
    for step, (X, y) in enumerate(val_dataloader):

        X = X.float().to(device)
        X = (X - min) / (max - min)

        y = y.view(X.size(0)).to(device)

        loss, corrects, predicted_classes = val_step(X, y, model, criterion)
        # Running metrics
        running_val_loss = running_val_loss + loss * X.size(0)
        running_val_acc = running_val_acc + corrects

    val_loss = running_val_loss / len(val_dataloader)
    val_acc = running_val_acc / len(val_dataloader)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        # Checkpoint model
        path = "checkpoint_model_run_1.pth"
        print(f"Saving model to {path}")
        torch.save(model.state_dict(), path)
        best_val_loss = val_loss

    print(f"Epoch: {epoch} | train_loss {train_loss} | train_acc: {train_acc} | val_loss: {val_loss} | val_acc: {val_acc}")


"""
for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    for batch_idx, (data, y) in enumerate(train_dataloader):
        optimizer.zero_grad()

        data = data.unsqueeze(dim=-1)
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
    # with torch.inference_mode(): 
    for batch_idx, (data, y) in enumerate(val_dataloader):

        data = data.unsqueeze(dim=-1)
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

torch.save(model.state_dict(), 'model_20.pth')
"""