import torch
import torch.nn as nn

"""
Created on Thu May 23 18:17 CET 2024

@author: andreapietro.arena@unikorestudent.it

Some description
"""

class Model:
    class RNN(nn.Module):
        """
        A standard recurrent neural network (RNN) for sequence processing.

        Parameters
        ----------
        input_size : int
            The size of the input vectors.
        hidden_size : int
            The size of the hidden layers of the RNN.
        output_size : int
            The size of the output vector.
        num_layers : int, optional
            The number of layers in the RNN. The default is 1.
        dropout : float, optional
            The dropout rate to be applied. The default is 0.

        Methods
        -------
        forward(x, is_validation=False).
            Performs the forward pass through the RNN network.
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
            super(Model.RNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, is_validation=False):
            """
            Performs the forward pass through the RNN network.

            Parameters
            ----------
            x : torch.Tensor
                Input data.
            is_validation : bool, optional
                Indicates whether the network is in validation mode (without dropout). The default is False.

            Returns
            -------
            torch.Tensor
                The output vector of the RNN network.
            """
            h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)
            out = self.dropout(out) if not is_validation else out
            out = self.fc(out[:, -1, :])
            return out

    class LSTM(nn.Module):
        """
        An LSTM neural network for sequence processing, with optional support for bidirectionality.

        Parameters
        ----------
        input_size : int
            The size of the input vectors.
        hidden_size : int
            The size of the hidden layers of the LSTM.
        output_size : int
            The size of the output vector.
        num_layers : int, optional
            The number of layers in the LSTM. The default is 1.
        dropout : float, optional
            The dropout rate to be applied. The default is 0.
        bidirectional : bool, optional
            Indicates whether the LSTM is bidirectional. The default is False.

        Methods
        -------
        forward(x, is_validation=False)
            Performs the forward pass through the LSTM network.
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, bidirectional=False):
            super(Model.LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
            #self.info()

        def forward(self, x, is_validation=False):
            """
            Performs the forward pass through the LSTM network.

            Parameters
            ----------
            x : torch.Tensor
                Input data.
            is_validation : bool, optional
                Indicates whether the network is in validation mode (without dropout). The default is False.

            Returns
            -------
            torch.Tensor
                The output vector of the LSTM network.
            """
            num_directions = 2 if self.lstm.bidirectional else 1
            h0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out) if not is_validation else out
            out = self.fc(out[:, -1, :])
            return out
        
        #     # forse successivamente altra funz: print("========= MODEL INFO[i] =========")

    class GRU(nn.Module):
        """
        A GRU neural network for sequence processing, with optional support for bidirectionality.

        Parameters
        ----------
        input_size : int
            The size of the input vectors.
        hidden_size : int
            The size of the hidden layers of the GRU.
        output_size : int
            The size of the output vector.
        num_layers : int, optional
            The number of layers in the GRU. The default is 1.
        dropout : float, optional
            The dropout rate to be applied. The default is 0.
        bidirectional : bool, optional
            Indicates whether the GRU is bidirectional. The default is False.

        Methods
        -------
        forward(x, is_validation=False).
            Performs the forward pass through the GRU network.
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, bidirectional=False):
            super(Model.GRU, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, is_validation=False):
            """
            Performs the forward pass through the GRU network.

            Parameters
            ----------
            x : torch.Tensor
                Input data.
            is_validation : bool, optional
                Indicates whether the network is in validation mode (without dropout). The default is False.

            Returns
            -------
            torch.Tensor
                The output vector of the GRU network.
            """
            num_directions = 2 if self.gru.bidirectional else 1
            h0 = torch.zeros(self.gru.num_layers * num_directions, x.size(0), self.gru.hidden_size).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.dropout(out) if not is_validation else out
            out = self.fc(out[:, -1, :])
            return out
