import math
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
from Hyperparameter_opt import MultiTaskModel


def train(model, optimizer, x_train, y_train, nonzero_train, selector_train, criterion, device):
    model.train()

    x_data = torch.tensor(x_train[:, 1:], dtype=torch.float32, device=device)
    label = torch.tensor(x_train[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
    property_selector = torch.tensor(np.asarray(selector_train, dtype=float), dtype=torch.float32, device=device)
    y_data = torch.tensor(np.asarray(y_train, dtype=float), dtype=torch.float32, device=device)
    pred = model(x_data, label, property_selector)

    losses = [criterion(pred[:, i][nonzero_train[i]], y_data[:, i][nonzero_train[i]]) for i in range(2)]
    r2 = [r2_score(y_train[:, i][nonzero_train[i]], pred[:, i].detach().numpy()[nonzero_train[i]]) for i in range(2)]

    loss = ((losses[0] + losses[1]) / 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return losses, r2, loss


def test(model, x_test, y_test, nonzero_test, selector_test, criterion, device):
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_test[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x_test[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        property_selector = torch.tensor(np.asarray(selector_test, dtype=float), dtype=torch.float32, device=device)
        y_data = torch.tensor(np.asarray(y_test, dtype=float), dtype=torch.float32, device=device)
        pred = model(x_data, label, property_selector)
        losses = [criterion(pred[:, i][nonzero_test[i]], y_data[:, i][nonzero_test[i]]) for i in range(2)]
        r2 = [r2_score(y_test[:, i][nonzero_test[i]], pred[:, i].detach().numpy()[nonzero_test[i]]) for i in range(2)]
        loss = ((losses[0] + losses[1]) / 2)
    return losses, r2


def train_alldata():
    input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim, d_shared_hidden_dim = [107, 256, 32, 64,
                                                                                                     32, 128]
    n_s, n_p, n_p_s, n_d, n_d_s, lr, dropout, active = [0, 2, 3, 0, 1, 1e-3, 0.05, 'leaky_relu']
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('data/PI_permittivity_dielectricLoss.csv').values

    x, y_1, y_2 = data[:, 3:], np.log10(np.array(data[:, 1].tolist())).reshape(-1, 1), np.log10(
        np.array(data[:, 2].tolist())).reshape(-1, 1)
    y = np.concatenate([y_1, y_2], axis=1)
    property_selector = np.ones_like(y)
    property_selector[y == 0] = 0
    nonzero = [np.nonzero(y[:, i]) for i in range(2)]

    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x = norm_x.fit_transform(x)
    with open('model/alldata/dielectric_scalerX.pkl', 'wb') as fw:
        pickle.dump(norm_x, fw)
    stdscaler_y_p = preprocessing.StandardScaler()
    p_value = stdscaler_y_p.fit_transform(y[:, 0].reshape(-1, 1)[nonzero[0]])
    with open('model/alldata/permittivity_scalerY.pkl', 'wb') as fw:
        pickle.dump(stdscaler_y_p, fw)
    y[:, 0].reshape(-1, 1)[nonzero[0]] = p_value
    stdscaler_y_d = preprocessing.StandardScaler()
    d_value = stdscaler_y_d.fit_transform(y[:, 1].reshape(-1, 1)[nonzero[1]])
    with open('model/alldata/dielectricLoss_scalerY.pkl', 'wb') as fw:
        pickle.dump(stdscaler_y_d, fw)
    y[:, 1].reshape(-1, 1)[nonzero[1]] = d_value

    model = MultiTaskModel(input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim,
                           d_shared_hidden_dim, dropout, n_s, n_p, n_p_s, n_d, n_d_s, active)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        model.train()
        x_data = torch.tensor(x[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        property_selector = torch.tensor(np.asarray(property_selector, dtype=float), dtype=torch.float32, device=device)
        y_data = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32, device=device)
        pred = model(x_data, label, property_selector)

        losses = [criterion(pred[:, i][nonzero[i]], y_data[:, i][nonzero[i]]) for i in range(2)]
        r2 = [r2_score(y[:, i][nonzero[i]], pred[:, i].detach().numpy()[nonzero[i]]) for i in range(2)]

        loss = ((losses[0] + losses[1]) / 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # early stopping
        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
            torch.save(model, 'model/alldata/dielectric.pth')
            print(f"epoch:{epoch}  R2_p = {r2[0]:.4f}, R2_d = {r2[1]:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter > patience:
                print("Early stopping!")
                break


def train_test():
    input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim, d_shared_hidden_dim = [107, 256, 32, 64,
                                                                                                     32, 128]
    n_s, n_p, n_p_s, n_d, n_d_s, lr, dropout, active = [0, 2, 3, 0, 1, 1e-3, 0.05, 'leaky_relu']
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
    nonzero_train = [np.nonzero(y_Train[:, i]) for i in range(2)]
    nonzero_test = [np.nonzero(y_Test[:, i]) for i in range(2)]
    selector_Train, selector_test = y_Train[:, 2:], y_Test[:, 2:]

    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_Train = norm_x.fit_transform(x_Train)
    x_Test = norm_x.transform(x_Test)

    stdscaler_y_p = preprocessing.StandardScaler()
    p_value_train = stdscaler_y_p.fit_transform(y_Train[:, 0].reshape(-1, 1)[nonzero_train[0]])
    y_Train[:, 0].reshape(-1, 1)[nonzero_train[0]] = p_value_train
    p_value_test = stdscaler_y_p.transform(y_Test[:, 0].reshape(-1, 1)[nonzero_test[0]])
    y_Test[:, 0].reshape(-1, 1)[nonzero_test[0]] = p_value_test

    stdscaler_y_d = preprocessing.StandardScaler()
    d_value_train = stdscaler_y_d.fit_transform(y_Train[:, 1].reshape(-1, 1)[nonzero_train[1]])
    y_Train[:, 1].reshape(-1, 1)[nonzero_train[1]] = d_value_train
    d_value_test = stdscaler_y_d.transform(y_Test[:, 1].reshape(-1, 1)[nonzero_test[1]])
    y_Test[:, 1].reshape(-1, 1)[nonzero_test[1]] = d_value_test

    model = MultiTaskModel(input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim,
                           d_shared_hidden_dim, dropout, n_s, n_p, n_p_s, n_d, n_d_s, active)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        train_loss, train_accuracy, loss = train(model, optimizer, x_Train, y_Train, nonzero_train,
                                                 selector_Train, criterion, device)

        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
            torch.save(model, 'model/split/dielectric.pth')
            print(f"epoch:{epoch + 1}/{epochs}  R2_p = {train_accuracy[0]:.4f}, R2_d = {train_accuracy[1]:.4f}")
            test_loss, test_accuracy = test(model, x_Test, y_Test, nonzero_test, selector_test, criterion, device)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping!")
                break

    print(f"Test Loss_p = {math.sqrt(test_loss[0]):.4f}, Test Loss_d = {math.sqrt(test_loss[1]):.4f}")
    print(f"Test R2_p = {test_accuracy[0]:.4f}, Test R2_d = {test_accuracy[1]:.4f}")


def main():
    input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim, d_shared_hidden_dim = [107, 256, 32, 64,
                                                                                                     32, 128]
    n_s, n_p, n_p_s, n_d, n_d_s, lr, dropout, active = [0, 2, 3, 0, 1, 1e-3, 0.05, 'leaky_relu']
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('data/PI_permittivity_dielectricLoss.csv').values

    x, y_1, y_2 = data[:, 3:], np.log10(np.array(data[:, 1].tolist())).reshape(-1, 1), np.log10(
        np.array(data[:, 2].tolist())).reshape(-1, 1)
    y = np.concatenate([y_1, y_2], axis=1)
    property_selector = np.ones_like(y)
    property_selector[y == 0] = 0

    print("Model starts training！！！")
    train_r2_p, train_r2_d, test_r2_p, test_r2_d, train_rmse_p, train_rmse_d, test_rmse_p, test_rmse_d, = [], [], [], [], [], [], [], []
    for _ in tqdm(range(20)):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
        generate = kfold.split(x)
        TRAIN_R2_p, TRAIN_R2_d, TEST_R2_p, TEST_R2_d, TRAIN_LOSS_p, TRAIN_LOSS_d, TEST_LOSS_p, TEST_LOSS_d = 0, 0, 0, 0, 0, 0, 0, 0
        while True:
            try:
                train_idx, test_idx = next(generate)
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                selector_train, selector_test = property_selector[train_idx], property_selector[test_idx]
                nonzero_train = [np.nonzero(y_train[:, i]) for i in range(2)]
                nonzero_test = [np.nonzero(y_test[:, i]) for i in range(2)]
                norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                x_train = norm_x.fit_transform(x_train)
                x_test = norm_x.transform(x_test)

                stdscaler_y_p = preprocessing.StandardScaler()
                p_value_train = stdscaler_y_p.fit_transform(y_train[:, 0].reshape(-1, 1)[nonzero_train[0]])
                y_train[:, 0].reshape(-1, 1)[nonzero_train[0]] = p_value_train
                p_value_test = stdscaler_y_p.transform(y_test[:, 0].reshape(-1, 1)[nonzero_test[0]])
                y_test[:, 0].reshape(-1, 1)[nonzero_test[0]] = p_value_test

                stdscaler_y_d = preprocessing.StandardScaler()
                d_value_train = stdscaler_y_d.fit_transform(y_train[:, 1].reshape(-1, 1)[nonzero_train[1]])
                y_train[:, 1].reshape(-1, 1)[nonzero_train[1]] = d_value_train
                d_value_test = stdscaler_y_d.transform(y_test[:, 1].reshape(-1, 1)[nonzero_test[1]])
                y_test[:, 1].reshape(-1, 1)[nonzero_test[1]] = d_value_test

                model = MultiTaskModel(input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim,
                                       d_shared_hidden_dim, dropout, n_s, n_p, n_p_s, n_d, n_d_s, active)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()

                best_train_loss = float('inf')
                early_stop_counter = 0

                for epoch in range(epochs):
                    train_loss, train_accuracy, loss = train(model, optimizer, x_train, y_train, nonzero_train,
                                                             selector_train, criterion, device)
                    if epoch % 10 == 0 and epoch != 0:
                        print(
                            f"Epoch {epoch + 1}/{epochs}: Train Loss1 = {train_loss[0]:.4f}, Train Loss2 = {train_loss[1]:.4f}" + '\n' +
                            f"Epoch {epoch + 1}/{epochs}: Train R21 = {train_accuracy[0]:.4f}, Train R22 = {train_accuracy[1]:.4f}")

                    # early stopping
                    if loss < best_train_loss:
                        best_train_loss = loss
                        early_stop_counter = 0
                        _train_loss, _train_accuracy = train_loss, train_accuracy
                        test_loss, test_accuracy = test(model, x_test, y_test, nonzero_test, selector_test, criterion,
                                                        device)
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            print("Early stopping!")
                            break

                TRAIN_R2_p += _train_accuracy[0]
                TRAIN_R2_d += _train_accuracy[1]
                TRAIN_LOSS_p += math.sqrt(_train_loss[0])
                TRAIN_LOSS_d += math.sqrt(_train_loss[1])
                TEST_R2_p += test_accuracy[0]
                TEST_R2_d += test_accuracy[1]
                TEST_LOSS_p += math.sqrt(test_loss[0])
                TEST_LOSS_d += math.sqrt(test_loss[1])
                print(f"Test Loss1 = {test_loss[0]:.4f}, Test Loss2 = {test_loss[1]:.4f}")
            except StopIteration:
                break
        train_r2_p.append(TRAIN_R2_p / 5), train_r2_d.append(TRAIN_R2_d / 5), test_r2_p.append(
            TEST_R2_p / 5), test_r2_d.append(TEST_R2_d / 5), train_rmse_p.append(TRAIN_LOSS_p / 5), train_rmse_d.append(
            TRAIN_LOSS_d / 5), test_rmse_p.append(TEST_LOSS_p / 5), test_rmse_d.append(TEST_LOSS_d / 5)
    with open('data/MT_result.txt', 'w') as f:
        f.write(
            f'train_r2_p:{np.mean(train_r2_p)}±{np.std(train_r2_p)}' + '\n' +
            f'train_r2_d:{np.mean(train_r2_d)}±{np.std(train_r2_d)}' + '\n' +
            f'test_r2_p:{np.mean(test_r2_p)}±{np.std(test_r2_p)}' + '\n' +
            f'test_r2_d:{np.mean(test_r2_d)}±{np.std(test_r2_d)}' + '\n' +
            f'train_rmse_p:{np.mean(train_rmse_p)}±{np.std(train_rmse_p)}' + '\n' +
            f'train_rmse_d:{np.mean(train_rmse_d)}±{np.std(train_rmse_d)}' + '\n' +
            f'test_rmse_p:{np.mean(test_rmse_p)}±{np.std(test_rmse_p)}' + '\n' +
            f'test_rmse_d:{np.mean(test_rmse_d)}±{np.std(test_rmse_d)}')


def n_kfold():
    input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim, d_shared_hidden_dim = [107, 256, 32, 64,
                                                                                                     32, 128]
    n_s, n_p, n_p_s, n_d, n_d_s, lr, dropout, active = [0, 2, 3, 0, 1, 1e-3, 0.05, 'leaky_relu']
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('data/PI_permittivity_dielectricLoss.csv').values

    x, y_1, y_2 = data[:, 3:], np.log10(np.array(data[:, 1].tolist())).reshape(-1, 1), np.log10(
        np.array(data[:, 2].tolist())).reshape(-1, 1)
    y = np.concatenate([y_1, y_2], axis=1)
    property_selector = np.ones_like(y)
    property_selector[y == 0] = 0

    print("Model starts training！！！")
    nfold_dict = {"n": [], "train_rmse_p": [], "train_rmse_p_std": [], "train_rmse_d": [], "train_rmse_d_std": [],
                  "test_rmse_p": [], "test_rmse_p_std": [], "test_rmse_d": [], "test_rmse_d_std": []}
    for i in tqdm(range(2, 11)):
        train_rmse_p, train_rmse_d, test_rmse_p, test_rmse_d = [], [], [], []
        for _ in range(20):
            kfold = KFold(n_splits=i, shuffle=True, random_state=2023)
            generate = kfold.split(x)
            TRAIN_LOSS_p, TRAIN_LOSS_d, TEST_LOSS_p, TEST_LOSS_d = 0, 0, 0, 0
            while True:
                try:
                    train_idx, test_idx = next(generate)
                    x_train, x_test = x[train_idx], x[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    selector_train, selector_test = property_selector[train_idx], property_selector[test_idx]
                    nonzero_train = [np.nonzero(y_train[:, i]) for i in range(2)]
                    nonzero_test = [np.nonzero(y_test[:, i]) for i in range(2)]
                    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                    x_train = norm_x.fit_transform(x_train)
                    x_test = norm_x.transform(x_test)

                    stdscaler_y_p = preprocessing.StandardScaler()
                    p_value_train = stdscaler_y_p.fit_transform(y_train[:, 0].reshape(-1, 1)[nonzero_train[0]])
                    y_train[:, 0].reshape(-1, 1)[nonzero_train[0]] = p_value_train
                    p_value_test = stdscaler_y_p.transform(y_test[:, 0].reshape(-1, 1)[nonzero_test[0]])
                    y_test[:, 0].reshape(-1, 1)[nonzero_test[0]] = p_value_test

                    stdscaler_y_d = preprocessing.StandardScaler()
                    d_value_train = stdscaler_y_d.fit_transform(y_train[:, 1].reshape(-1, 1)[nonzero_train[1]])
                    y_train[:, 1].reshape(-1, 1)[nonzero_train[1]] = d_value_train
                    d_value_test = stdscaler_y_d.transform(y_test[:, 1].reshape(-1, 1)[nonzero_test[1]])
                    y_test[:, 1].reshape(-1, 1)[nonzero_test[1]] = d_value_test

                    model = MultiTaskModel(input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim,
                                           d_shared_hidden_dim, dropout, n_s, n_p, n_p_s, n_d, n_d_s, active)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = torch.nn.MSELoss()

                    best_loss = float('inf')
                    early_stop_counter = 0
                    for epoch in range(epochs):
                        train_loss, train_accuracy, loss = train(model, optimizer, x_train, y_train, nonzero_train,
                                                                 selector_train, criterion, device)
                        if epoch % 10 == 0 and epoch != 0:
                            print(
                                f"Epoch {epoch + 1}/{epochs}: Train Loss1 = {train_loss[0]:.4f}, Train Loss2 = {train_loss[1]:.4f}")
                        if loss < best_loss:
                            best_loss = loss
                            early_stop_counter = 0
                            _train_loss, _train_accuracy = train_loss, train_accuracy
                            test_loss, test_accuracy = test(model, x_test, y_test, nonzero_test, selector_test,
                                                            criterion, device)
                        else:
                            early_stop_counter += 1
                            if early_stop_counter > patience:
                                print("Early stopping!")
                                break
                    TRAIN_LOSS_p += math.sqrt(_train_loss[0])
                    TRAIN_LOSS_d += math.sqrt(_train_loss[1])
                    TEST_LOSS_p += math.sqrt(test_loss[0])
                    TEST_LOSS_d += math.sqrt(test_loss[1])
                    print(f"Test Loss1 = {test_loss[0]:.4f}, Test Loss2 = {test_loss[1]:.4f}")
                except StopIteration:
                    break
            train_rmse_p.append(TRAIN_LOSS_p / i)
            train_rmse_d.append(TRAIN_LOSS_d / i)
            test_rmse_p.append(TEST_LOSS_p / i)
            test_rmse_d.append(TEST_LOSS_d / i)
        nfold_dict["n"].append(i)
        nfold_dict["train_rmse_p"].append(np.mean(train_rmse_p))
        nfold_dict["train_rmse_p_std"].append(np.std(train_rmse_p))
        nfold_dict["train_rmse_d"].append(np.mean(train_rmse_d))
        nfold_dict["train_rmse_d_std"].append(np.std(train_rmse_d))
        nfold_dict["test_rmse_p"].append(np.mean(test_rmse_p))
        nfold_dict["test_rmse_p_std"].append(np.std(test_rmse_p))
        nfold_dict["test_rmse_d"].append(np.mean(test_rmse_d))
        nfold_dict["test_rmse_d_std"].append(np.std(test_rmse_d))
    pd.DataFrame.from_dict(nfold_dict).to_csv('data/n_fold_result_MT.csv', index=False)


if __name__ == '__main__':
    # main()
    train_test()
