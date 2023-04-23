import torch
import torch.nn as nn
import torch.nn.functional as F

class BRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(BRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
        self.fc = nn.Sequential(
        #   nn.Dropout1d(),
        #   nn.Linear(hidden_size*2, hidden_size*2),
        nn.Linear(hidden_size*2, num_classes)
        )
  
    def forward(self, x):
        h0 = torch.randn(self.num_layers*2, 128, self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers*2, 128, self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
    
class OneDConvNet(nn.Module):
  def __init__(self, n_features, n_classes, base_filters=32, mul=120):
    super(OneDConvNet, self).__init__()

    self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=base_filters, kernel_size=3, stride=1, padding=1)
    self.norm1 = nn.LayerNorm(base_filters*mul)
    self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv1d(in_channels=base_filters, out_channels=base_filters*2, kernel_size=3, stride=1, padding=1)
    self.norm2 = nn.LayerNorm(base_filters*mul)
    self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
    self.conv3 = nn.Conv1d(in_channels=base_filters*2, out_channels=base_filters*4, kernel_size=3, stride=1, padding=1)
    self.norm3 = nn.LayerNorm(base_filters*mul)
    self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
    self.conv4 = nn.Conv1d(in_channels=base_filters*4, out_channels=base_filters*8, kernel_size=3, stride=1, padding=1)
    self.norm4 = nn.LayerNorm(base_filters*mul)
    self.fc1 = nn.Linear(base_filters*8, base_filters*16)
    self.dropout5 = nn.Dropout(0.4)
    self.fc2 = nn.Linear(base_filters*16, n_classes)
  
  def forward(self, x):
    x = self.conv1(x)
    N, C, T = x.shape
    x = x.view(N, C*T)
    x = self.norm1(x)
    x = x.view(N, C, T)
    x = self.pool1(F.relu(x))

    x = self.conv2(x)
    N, C, T = x.shape
    x = x.view(N, C*T)
    x = self.norm2(x)
    x = x.view(N, C, T)
    x = self.pool2(F.relu(x))

    x = self.conv3(x)
    N, C, T = x.shape
    x = x.view(N, C*T)
    x = self.norm3(x)
    x = x.view(N, C, T)
    x = self.pool3(F.relu(x))

    x = self.conv4(x)
    N, C, T = x.shape
    x = x.view(N, C*T)
    x = self.norm4(x)
    x = x.view(N, C, T)
    x = F.relu(x)

    N, C, T = x.size()
    x = x.mean(dim=-1) # Flatten
    x = self.dropout5(F.relu(self.fc1(x)))
    x = self.fc2(x)
    return x
