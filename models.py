
import torch
import torch.nn as nn


class FedANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(561, 200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FedConvNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)

        self.dropout = nn.Dropout(0.5)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(2740, 128)
        self.fc2 = nn.Linear(128, 6)

        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch, 561) -> (batch, 1, 561)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x
