
import torch.nn as nn

class InventoryShortageNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(InventoryShortageNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.output(x))
        return x
