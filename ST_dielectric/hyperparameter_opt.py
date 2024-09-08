import optuna
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from Networks import *
from piplines import *


def objective(trial):
    input_dim = 110
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    n_h = trial.suggest_int('n_h', 0, 5)
    lr = trial.suggest_float('lr', 5e-4, 5e-3)
    dropout = trial.suggest_float('dropout', 0, 0.2)
    active = trial.suggest_categorical('active', ['relu', 'leaky_relu'])
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('output/permittivity/PI_permittivity_substructures_filter_0.001.csv').values

    x, y = data[:, 5:], np.log10(np.array(data[:, 3].tolist())).reshape(-1, 1)
    x_Train, x_test, y_Train, y_test = train_test_split(x, y, test_size=0.1, random_state=2023)

    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_Train = norm_x.fit_transform(x_Train)
    x_test = norm_x.transform(x_test)
    stdscaler_y = preprocessing.StandardScaler()
    y_Train = stdscaler_y.fit_transform(y_Train)
    y_test = stdscaler_y.transform(y_test)

    print("Model starts training.")

    kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
    generate = kfold.split(x_Train)
    TEST_LOSS = 0
    while True:
        try:
            train_idx, val_idx = next(generate)
            x_train, x_val = x_Train[train_idx], x_Train[val_idx]
            y_train, y_val = y_Train[train_idx], y_Train[val_idx]
            model = Model(input_dim, hidden_dim, dropout, n_h, active)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()

            best_val_loss = float('inf')
            early_stop_counter = 0
            for epoch in range(epochs):
                train_loss = train(model, optimizer, x_train, y_train, criterion, device)
                val_loss = validate(model, x_val, y_val, criterion, device)
                if epoch % 10 == 0 and epoch != 0:
                    print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

                # early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    test_loss = test(model, x_test, y_test, criterion, device)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping!")
                        break
            TEST_LOSS += test_loss
            print(f"Test Loss = {test_loss:.4f}")
        except StopIteration:
            break

    return TEST_LOSS / 5


def objective_m(trial):
    input_dim = 101
    hidden_dim1 = trial.suggest_categorical('hidden_dim1', [32, 64, 128, 256])
    hidden_dim2 = trial.suggest_categorical('hidden_dim2', [32, 64, 128, 256])
    hidden_dim3 = trial.suggest_categorical('hidden_dim3', [32, 64, 128, 256])
    output_dim = 1
    n_s = trial.suggest_int('n_s', 0, 3)
    n_f = trial.suggest_int('n_f', 0, 3)
    lr = trial.suggest_float('lr', 5e-4, 5e-3)
    dropout = trial.suggest_float('dropout', 0, 0.2)
    active = trial.suggest_categorical('active', ['relu', 'leaky_relu'])
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('output/permittivity/lasso/PI_permittivity_descriptors_fingerprint_lasso.csv').values

    x_h, y_h = data[:1301, 4:], np.log10(np.array(data[:1301, 3].tolist())).reshape(-1, 1)
    x_Train_h, x_test_h, y_Train_h, y_test_h = train_test_split(x_h, y_h, test_size=0.1, random_state=2023)

    x_l, y_l = data[1301:, 4:], np.log10(np.array(data[1301:, 3].tolist())).reshape(-1, 1)
    x_Train_l, x_test_l, y_Train_l, y_test_l = train_test_split(x_l, y_l, test_size=0.1, random_state=2023)

    x_Train = np.concatenate([x_Train_h, x_Train_l], axis=0)
    x_test = np.concatenate([x_test_h, x_test_l], axis=0)
    y_Train = np.concatenate([y_Train_h, y_Train_l], axis=0)
    y_test = np.concatenate([y_test_h, y_test_l], axis=0)

    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_Train = norm_x.fit_transform(x_Train)
    x_test = norm_x.transform(x_test)
    stdscaler_y = preprocessing.StandardScaler()
    y_Train = stdscaler_y.fit_transform(y_Train)
    y_test = stdscaler_y.transform(y_test)

    print("Model starts training.")

    kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
    generate = kfold.split(x_Train)
    TEST_LOSS = 0
    while True:
        try:
            train_idx, val_idx = next(generate)
            x_train, x_val = x_Train[train_idx], x_Train[val_idx]
            y_train, y_val = y_Train[train_idx], y_Train[val_idx]
            model = MultiFidelityModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout, n_s, n_f,
                                        active)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()

            best_val_loss = float('inf')
            early_stop_counter = 0
            for epoch in range(epochs):
                train_loss = train_m(model, optimizer, x_train, y_train, criterion, device)
                val_loss = validate_m(model, x_val, y_val, criterion, device)
                if epoch % 10 == 0 and epoch != 0:
                    print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

                # early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping!")
                        break
            test_loss = test_m(model, x_test, y_test, criterion, device)
            TEST_LOSS += test_loss
            print(f"Test Loss = {test_loss:.4f}")
        except StopIteration:
            break

    return TEST_LOSS / 5


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best test loss: {study.best_value:.4f}")

    with open('output/permittivity/xxx.log', 'w') as f:
        f.write(f"Best trial: {study.best_trial.params}" + '\n' + f"Best test loss: {study.best_value:.4f}")

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
