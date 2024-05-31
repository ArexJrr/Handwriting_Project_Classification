import torch
import torch.nn as nn

class Model:
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
            super(Model.RNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, is_validation=False):
            h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)
            out = self.dropout(out) if not is_validation else out
            out = self.fc(out[:, -1, :])
            return out

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, bidirectional=False):
            super(Model.LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
            #self.info()

        def forward(self, x, is_validation=False):
            num_directions = 2 if self.lstm.bidirectional else 1
            h0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out) if not is_validation else out
            out = self.fc(out[:, -1, :])
            return out
        
        #     # forse successivamente altra funz: print("========= MODEL INFO[i] =========")

    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, bidirectional=False):
            super(Model.GRU, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, is_validation=False):
            num_directions = 2 if self.gru.bidirectional else 1
            h0 = torch.zeros(self.gru.num_layers * num_directions, x.size(0), self.gru.hidden_size).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.dropout(out) if not is_validation else out
            out = self.fc(out[:, -1, :])
            return out
