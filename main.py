import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import train_dataloader
from model import BRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 6
sequence_length = 335413
num_layers = 2
hidden_size = 256
num_classes = 4
learning_rate = 0.001
batch_size = 128
num_epochs = 2

device = 'mps'

model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)
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
    for batch_idx, (data, y) in enumerate(train_dataloader):
        optimizer.zero_grad()

        data = data.unsqueeze(dim=1)
        data = data.to(device = device)
        # one hot encoding
        y = y.to(device = device)

        y_pred = model(data, device)
        loss = criterion(y_pred, y)

        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        loss.backward()

        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
