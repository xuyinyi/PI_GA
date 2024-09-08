import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, n_h=2, active='leaky_relu'):
        super(Model, self).__init__()

        if active == 'leaky_relu':
            self.active = nn.LeakyReLU()
        elif active == 'relu':
            self.active = nn.ReLU()
        elif active == 'tanh':
            self.active = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {active}")

        self.dropout = nn.Dropout(dropout)
        self.n_hidden = n_h

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.active,
            self.dropout))
        for _ in range(self.n_hidden):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.active,
                self.dropout))

        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, inputs):
        output = self.layers[0](inputs)
        for _ in range(self.n_hidden):
            output = self.layers[_ + 1](output)
        output = self.layers[-1](output)
        return output


class MultiFidelityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout, n_s=2, n_f=3,
                 active='leaky_relu'):
        super(MultiFidelityModel, self).__init__()

        if active == 'leaky_relu':
            self.active = nn.LeakyReLU()
        elif active == 'relu':
            self.active = nn.ReLU()
        elif active == 'tanh':
            self.active = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {active}")

        self.dropout = nn.Dropout(dropout)
        self.n_shared_hidden = n_s
        self.n_fidelity_hidden = n_f

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            self.active,
            self.dropout))

        self.layers.append(nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            self.active,
            self.dropout
        ))

        self.layers.append(nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim3),
            self.active,
            self.dropout
        ))
        for i in range(self.n_fidelity_hidden):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim2, hidden_dim2),
                self.active,
                self.dropout
            ))
        self.layers.append(nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            self.active,
            self.dropout
        ))

        for i in range(self.n_shared_hidden):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim3, hidden_dim3),
                self.active,
                self.dropout
            ))
        self.layers.append(nn.Linear(hidden_dim3, output_dim))

    def forward(self, inputs, fidelity):
        output1 = self.layers[0](inputs)
        if self.n_fidelity_hidden == 0 and self.n_shared_hidden == 0:
            output2 = self.layers[2](output1)
            output2 = self.layers[-1](output2)

            output3 = self.layers[1](output1)
            output3 = self.layers[3](output3)
            output3 = self.layers[-1](output3)
        elif self.n_fidelity_hidden != 0 and self.n_shared_hidden == 0:
            output2 = self.layers[2](output1)
            output2 = self.layers[-1](output2)

            output3 = self.layers[1](output1)
            for i in range(3, 3 + self.n_fidelity_hidden):
                output3 = self.layers[i](output3)
            output3 = self.layers[3 + self.n_fidelity_hidden](output3)
            output3 = self.layers[-1](output3)
        elif self.n_fidelity_hidden == 0 and self.n_shared_hidden != 0:
            output2 = self.layers[2](output1)
            for i in range(4, 4 + self.n_shared_hidden):
                output2 = self.layers[i](output2)
            output2 = self.layers[-1](output2)

            output3 = self.layers[1](output1)
            output3 = self.layers[3](output3)
            for i in range(4, 4 + self.n_shared_hidden):
                output3 = self.layers[i](output3)
            output3 = self.layers[-1](output3)
        else:
            output2 = self.layers[2](output1)
            for i in range(4 + self.n_fidelity_hidden, 4 + self.n_fidelity_hidden + self.n_shared_hidden):
                output2 = self.layers[i](output2)
            output2 = self.layers[-1](output2)

            output3 = self.layers[1](output1)
            for i in range(3, 3 + self.n_fidelity_hidden):
                output3 = self.layers[i](output3)
            output3 = self.layers[3 + self.n_fidelity_hidden](output3)
            for i in range(4 + self.n_fidelity_hidden, 4 + self.n_fidelity_hidden + self.n_shared_hidden):
                output3 = self.layers[i](output3)
            output3 = self.layers[-1](output3)

        output = torch.where(fidelity == 1, output2, output3)
        return output
