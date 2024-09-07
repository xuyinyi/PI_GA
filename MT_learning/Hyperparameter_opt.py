import copy
import torch
import optuna
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim,
                 d_shared_hidden_dim, dropout, n_s, n_p, n_p_s, n_d, n_d_s, active='leaky_relu'):
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


def train(model, optimizer, x_train, y_train, nonzero_train, selector_train, criterion, device):
    model.train()

    x_data = torch.tensor(x_train[:, 1:], dtype=torch.float32, device=device)
    label = torch.tensor(x_train[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
    property_selector = torch.tensor(np.asarray(selector_train, dtype=float), dtype=torch.float32, device=device)
    y_data = torch.tensor(np.asarray(y_train, dtype=float), dtype=torch.float32, device=device)
    pred = model(x_data, label, property_selector)

    losses = [criterion(pred[:, i][nonzero_train[i]], y_data[:, i][nonzero_train[i]]) for i in range(2)]
    loss = ((losses[0] + losses[1]) / 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validate(model, x_val, y_val, nonzero_val, selector_val, criterion, device):
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_val[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x_val[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        property_selector = torch.tensor(np.asarray(selector_val, dtype=float), dtype=torch.float32, device=device)
        y_data = torch.tensor(np.asarray(y_val, dtype=float), dtype=torch.float32, device=device)
        pred = model(x_data, label, property_selector)
        losses = [criterion(pred[:, i][nonzero_val[i]], y_data[:, i][nonzero_val[i]]) for i in range(2)]
        loss = ((losses[0] + losses[1]) / 2)
    return loss


def test(model, x_test, y_test, nonzero_test, selector_test, criterion, device):
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_test[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x_test[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        property_selector = torch.tensor(np.asarray(selector_test, dtype=float), dtype=torch.float32, device=device)
        y_data = torch.tensor(np.asarray(y_test, dtype=float), dtype=torch.float32, device=device)
        pred = model(x_data, label, property_selector)
        losses = [criterion(pred[:, i][nonzero_test[i]], y_data[:, i][nonzero_test[i]]) for i in range(2)]
        loss = ((losses[0] + losses[1]) / 2)
    return loss


def objective(trial):
    input_dim = 107
    s_hidden_dim = trial.suggest_categorical('s_hidden_dim', [32, 64, 128, 256])
    p_hidden_dim = trial.suggest_categorical('p_hidden_dim', [32, 64, 128])
    p_shared_hidden_dim = trial.suggest_categorical('p_shared_hidden_dim', [32, 64, 128])
    d_hidden_dim = trial.suggest_categorical('d_hidden_dim', [32, 64, 128])
    d_shared_hidden_dim = trial.suggest_categorical('d_shared_hidden_dim', [32, 64, 128])
    n_s = trial.suggest_int('n_s', 0, 3)
    n_p = trial.suggest_int('n_p', 0, 3)
    n_p_s = trial.suggest_int('n_p_s', 0, 3)
    n_d = trial.suggest_int('n_d', 0, 3)
    n_d_s = trial.suggest_int('n_d_s', 0, 3)
    lr = trial.suggest_float('lr', 5e-4, 5e-3)
    dropout = trial.suggest_float('dropout', 0, 0.2)
    active = trial.suggest_categorical('active', ['relu', 'leaky_relu'])
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('data/PI_permittivity_dielectricLoss.csv').values

    x, y_1, y_2 = data[:, 3:], np.log10(np.array(data[:, 1].tolist())).reshape(-1, 1), np.log10(
        np.array(data[:, 2].tolist())).reshape(-1, 1)
    y = np.concatenate([y_1, y_2], axis=1)
    property_selector = np.ones_like(y)
    property_selector[y == 0] = 0

    x_Train, x_Test, y_Train, y_Test = train_test_split(x, np.concatenate([y, property_selector], axis=1),
                                                        test_size=0.1, random_state=2023)

    nonzero_test = [np.nonzero(y_Test[:, i]) for i in range(2)]
    selector_Train, selector_test = y_Train[:, 2:], y_Test[:, 2:]

    print("Model starts training！！！")

    kfold = KFold(n_splits=5, shuffle=True, random_state=None)
    TEST_LOSS = 0

    for train_idx, val_idx in kfold.split(x_Train):

        x_train, x_val = x_Train[train_idx], x_Train[val_idx]
        y_train, y_val = y_Train[train_idx], y_Train[val_idx]
        x_test, y_test = x_Test, copy.copy(y_Test)

        selector_train, selector_val = selector_Train[train_idx], selector_Train[val_idx]
        nonzero_train = [np.nonzero(y_train[:, i]) for i in range(2)]
        nonzero_val = [np.nonzero(y_val[:, i]) for i in range(2)]

        norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_train = norm_x.fit_transform(x_train)
        x_val = norm_x.transform(x_val)
        x_test = norm_x.transform(x_test)

        stdscaler_y_p = preprocessing.StandardScaler()
        p_value_train = stdscaler_y_p.fit_transform(y_train[:, 0].reshape(-1, 1)[nonzero_train[0]])
        y_train[:, 0].reshape(-1, 1)[nonzero_train[0]] = p_value_train
        p_value_val = stdscaler_y_p.transform(y_val[:, 0].reshape(-1, 1)[nonzero_val[0]])
        y_val[:, 0].reshape(-1, 1)[nonzero_val[0]] = p_value_val
        p_value_test = stdscaler_y_p.transform(y_test[:, 0].reshape(-1, 1)[nonzero_test[0]])
        y_test[:, 0].reshape(-1, 1)[nonzero_test[0]] = p_value_test

        stdscaler_y_d = preprocessing.StandardScaler()
        d_value_train = stdscaler_y_d.fit_transform(y_train[:, 1].reshape(-1, 1)[nonzero_train[1]])
        y_train[:, 1].reshape(-1, 1)[nonzero_train[1]] = d_value_train
        d_value_val = stdscaler_y_d.transform(y_val[:, 1].reshape(-1, 1)[nonzero_val[1]])
        y_val[:, 1].reshape(-1, 1)[nonzero_val[1]] = d_value_val
        d_value_test = stdscaler_y_d.transform(y_test[:, 1].reshape(-1, 1)[nonzero_test[1]])
        y_test[:, 1].reshape(-1, 1)[nonzero_test[1]] = d_value_test

        model = MultiTaskModel(input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim,
                               d_shared_hidden_dim, dropout, n_s, n_p, n_p_s, n_d, n_d_s, active)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(epochs):
            train_loss = train(model, optimizer, x_train, y_train, nonzero_train, selector_train, criterion, device)
            val_loss = validate(model, x_val, y_val, nonzero_val, selector_val, criterion, device)
            if epoch % 10 == 0 and epoch != 0:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                test_loss = test(model, x_test, y_test, nonzero_test, selector_test, criterion, device)
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping!")
                    break

        TEST_LOSS += test_loss
        print(f"Test Loss = {test_loss:.4f}")

    return TEST_LOSS / 5


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=300)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best test loss: {study.best_value:.4f}")

    with open('data/Hyperparameter_opt.log', 'w') as f:
        f.write(f"Best trial: {study.best_trial.params}" + '\n' + f"Best test loss: {study.best_value:.4f}")

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
