import torch
import torch.nn as nn

class BRNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
     super(BRNN, self).__init__()
     self.hidden_size = hidden_size
     self.num_layers = num_layers
     self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
     self.fc = nn.Linear(hidden_size*2, num_classes)
  
  def forward(self, x, device='cuda'):
    h0 = torch.randn(self.num_layers*2, 128, self.hidden_size).to(device)
    c0 = torch.randn(self.num_layers*2, 128, self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])

    return out