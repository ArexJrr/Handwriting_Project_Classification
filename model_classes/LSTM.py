import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.45):
        super(LSTMClassifier, self).__init__()
        self.lstm_clc = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout) 
        self.fc = nn.Linear(hidden_size, num_classes)
        #flag che controlla se train o val 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Ricordo che deve essere (batch_size, sequence_length, input_size) da verificare
        h_lstm, _ = self.lstm_clc(x)
        h_lstm_last = h_lstm[:, -1, :]
        h_drop = self.dropout(h_lstm_last)
        out = self.fc(h_drop)
        return out
