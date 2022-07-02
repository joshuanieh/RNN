import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, dropout = 0.5, bidirectional = True, batch_first=True)
    def forward(self, x):
        #print(x.shape)
        x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
        #print(x.shape)
        x, y = self.lstm(x, None)
        #print(x.shape)
        x = x.squeeze()
        #print(x.shape)
        x = torch.reshape(x, (x.shape[0], 2, x.shape[1]//2))
        #print(x.shape)
        #print(torch.flip(x[:,1,:], dims=(0,)).shape)
        x = torch.cat([x[:,0,:], torch.flip(x[:,1,:], dims=(0,))], dim = 1)
        print(x.shape)
        #print(y)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            LSTM(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim*2),
            LSTM(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim*2),
            LSTM(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim*2),
            LSTM(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim*2),
            LSTM(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
