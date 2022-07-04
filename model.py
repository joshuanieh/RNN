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
        self.lstm = nn.LSTM(39, 39, num_layers=1, dropout = 0, bidirectional = True, batch_first=True)
    def forward(self, x):
        print(x.shape)
        # 512 * (39*19)
        x = torch.reshape(x, (x.shape[0], x.shape[1]//39, 39)) # 512 * 19 * 39
        print(x.shape)
        x, y = self.lstm(x, None) # 512 * 19 * 78
        print(x.shape)
        #x = x.squeeze()
        #print(x.shape)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 2, x.shape[2]//2)) # 512 * 19 * 2 * 39
        print(x.shape)
        #print(torch.flip(x[:,1,:], dims=(0,)).shape)
        x = torch.cat([x[:,:,0,:], torch.flip(x[:,:,1,:], dims=(1,))], dim = 2) # 512 * 19 * 78
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
        print(x.shape)
        #print(y)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            LSTM(input_dim, hidden_dim),
            LSTM(hidden_dim*2, hidden_dim),
            LSTM(hidden_dim*2, hidden_dim),
            LSTM(hidden_dim*2, hidden_dim),
            LSTM(hidden_dim*2, hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(input_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        print(x)
        x = self.fc(x)
        return x
