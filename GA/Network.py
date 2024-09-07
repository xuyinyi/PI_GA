import torch
import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim, d_shared_hidden_dim,
                 dropout, n_s, n_p, n_p_s, n_d, n_d_s, active='leaky_relu'):
        super(MultiTaskModel, self).__init__()

        if active == 'leaky_relu':
            self.active = nn.LeakyReLU()
        elif active == 'relu':
            self.active = nn.ReLU()
        elif active == 'tanh':
            self.active = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {active}")

        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.s_hidden_dim = s_hidden_dim
        self.p_hidden_dim = p_hidden_dim
        self.p_shared_hidden_dim = p_shared_hidden_dim
        self.d_hidden_dim = d_hidden_dim
        self.d_shared_hidden_dim = d_shared_hidden_dim
        self.n_shared_hidden = n_s
        self.n_permittivity_hidden = n_p
        self.n_permittivity_shared_hidden = n_p_s
        self.n_dielectricloss_hidden = n_d
        self.n_dielectricloss_shared_hidden = n_d_s

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Linear(self.input_dim, self.s_hidden_dim),
            self.active,
            self.dropout
        ))

        for _ in range(self.n_shared_hidden):
            self.layers.append(nn.Sequential(
                nn.Linear(self.s_hidden_dim, self.s_hidden_dim),
                self.active,
                self.dropout
            ))

        self.permittivity_layers = nn.ModuleList()
        self.permittivity_layers.append(nn.Sequential(
            nn.Linear(self.s_hidden_dim, self.p_hidden_dim),
            self.active,
            self.dropout
        ))
        for _ in range(self.n_permittivity_hidden):
            self.permittivity_layers.append(nn.Sequential(
                nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
                self.active,
                self.dropout
            ))

        self.permittivity_shared_layers = nn.ModuleList()
        self.permittivity_shared_layers.append(nn.Sequential(
            nn.Linear(self.s_hidden_dim, self.p_shared_hidden_dim),
            self.active,
            self.dropout
        ))
        self.permittivity_shared_layers.append(nn.Sequential(
            nn.Linear(self.p_hidden_dim, self.p_shared_hidden_dim),
            self.active,
            self.dropout
        ))
        for _ in range(self.n_permittivity_shared_hidden):
            self.permittivity_shared_layers.append(nn.Sequential(
                nn.Linear(self.p_shared_hidden_dim, self.p_shared_hidden_dim),
                self.active,
                self.dropout
            ))
        self.permittivity_shared_layers.append(nn.Linear(self.p_shared_hidden_dim, 1))

        self.dielectricloss_layers = nn.ModuleList()
        self.dielectricloss_layers.append(nn.Sequential(
            nn.Linear(self.s_hidden_dim, self.d_hidden_dim),
            self.active,
            self.dropout
        ))
        for _ in range(self.n_dielectricloss_hidden):
            self.dielectricloss_layers.append(nn.Sequential(
                nn.Linear(self.d_hidden_dim, self.d_hidden_dim),
                self.active,
                self.dropout
            ))

        self.dielectricloss_shared_layers = nn.ModuleList()
        self.dielectricloss_shared_layers.append(nn.Sequential(
            nn.Linear(self.s_hidden_dim, self.d_shared_hidden_dim),
            self.active,
            self.dropout
        ))
        self.dielectricloss_shared_layers.append(nn.Sequential(
            nn.Linear(self.d_hidden_dim, self.d_shared_hidden_dim),
            self.active,
            self.dropout
        ))
        for _ in range(self.n_dielectricloss_shared_hidden):
            self.dielectricloss_shared_layers.append(nn.Sequential(
                nn.Linear(self.d_shared_hidden_dim, self.d_shared_hidden_dim),
                self.active,
                self.dropout
            ))
        self.dielectricloss_shared_layers.append(nn.Linear(self.d_shared_hidden_dim, 1))

        self.vector_selector = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, fidelity, property_selector):
        # Get the vector selector mask
        mask = self.vector_selector(x)

        # Apply the mask to the input
        masked_x = x * mask

        # Shared layers
        share_output = self.layers[0](masked_x)
        for _ in range(self.n_shared_hidden):
            share_output = self.layers[_ + 1](share_output)

        output1 = self.permittivity_layers[0](share_output)
        for _ in range(self.n_permittivity_hidden):
            output1 = self.permittivity_layers[_ + 1](output1)
        output1 = self.permittivity_shared_layers[1](output1)
        for _ in range(self.n_permittivity_shared_hidden):
            output1 = self.permittivity_shared_layers[_ + 2](output1)
        output1 = self.permittivity_shared_layers[-1](output1)

        output2 = self.permittivity_shared_layers[0](share_output)
        for _ in range(self.n_permittivity_shared_hidden):
            output2 = self.permittivity_shared_layers[_ + 2](output2)
        output2 = self.permittivity_shared_layers[-1](output2)

        output3 = self.dielectricloss_layers[0](share_output)
        for _ in range(self.n_dielectricloss_hidden):
            output3 = self.dielectricloss_layers[_ + 1](output3)
        output3 = self.dielectricloss_shared_layers[1](output3)
        for _ in range(self.n_dielectricloss_shared_hidden):
            output3 = self.dielectricloss_shared_layers[_ + 2](output3)
        output3 = self.dielectricloss_shared_layers[-1](output3)

        output4 = self.dielectricloss_shared_layers[0](share_output)
        for _ in range(self.n_dielectricloss_shared_hidden):
            output4 = self.dielectricloss_shared_layers[_ + 2](output4)
        output4 = self.dielectricloss_shared_layers[-1](output4)

        output_p = torch.where(fidelity == 1, output2, output1)
        output_d = torch.where(fidelity == 1, output4, output3)

        outputs = torch.cat([output_p, output_d], 1)

        outputs = torch.mul(outputs, property_selector)
        return outputs


class TgModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, n_h=2, active='leaky_relu'):
        super(TgModel, self).__init__()

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
