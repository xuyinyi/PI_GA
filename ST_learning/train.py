import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import KFold
from Networks import *
from piplines import *


def train_alldata():
    """
    Train all the data and save the model
    """
    input_dim, hidden_dim, dropout, n_h, lr, active = [110, 256, 0.15, 1, 7e-4, 'leaky_relu']
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('output/permittivity/PI_permittivity_substructures_filter_0.001.csv').values
    np.random.seed(2023)
    np.random.shuffle(data)

    x, y = data[:, 5:], np.log10(np.array(data[:, 3].tolist())).reshape(-1, 1)
    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x = norm_x.fit_transform(x)
    stdscaler_y = preprocessing.StandardScaler()
    y = stdscaler_y.fit_transform(y)

    model = Model(input_dim, hidden_dim, dropout, n_h, active)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        model.train()
        x_data = torch.tensor(x, dtype=torch.float32, device=device)
        y_data = torch.tensor(y, dtype=torch.float32, device=device)
        pred = model(x_data)
        loss = criterion(pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # early stopping
        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
            torch.save(model, 'model/all_data/permittivity_substructures.pth')
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {loss:.4f}")
            print(f"R2 = {r2_score(y, pred.detach().numpy()):.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping!")
                break


def main():
    input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim = [75, 128, 32, 128, 1]
    n_s, n_f, lr, dropout, active = [0, 3, 1e-3, 0.03, 'leaky_relu']
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('output/permittivity/rfe/PI_permittivity_descriptors_fingerprint_RFE.csv').values
    x, y = data[:, 4:], np.log10(np.array(data[:, 3].tolist())).reshape(-1, 1)
    train_r2, test_r2, train_rmse, test_rmse = [], [], [], []
    print("Model starts training！！！")
    for _ in tqdm(range(20)):

        kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
        generate = kfold.split(x)
        TRAIN_R2, TEST_R2, TRAIN_LOSS, TEST_LOSS = 0, 0, 0, 0
        while True:
            try:
                Train_idx, Test_idx = next(generate)
                x_Train, x_test = x[Train_idx], x[Test_idx]
                y_Train, y_test = y[Train_idx], y[Test_idx]
                norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                x_Train = norm_x.fit_transform(x_Train)
                x_test = norm_x.transform(x_test)
                stdscaler_y = preprocessing.StandardScaler()
                y_Train = stdscaler_y.fit_transform(y_Train)
                y_test = stdscaler_y.transform(y_test)

                model = MultiFidelityModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim,
                                           dropout, n_s, n_f, active)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()

                best_train_loss = float('inf')
                early_stop_counter = 0

                for epoch in range(epochs):
                    train_loss, train_accuracy = train_m(model, optimizer, x_Train, y_Train, criterion, device)
                    if epoch % 10 == 0 and epoch != 0:
                        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}")
                    # early stopping
                    if train_loss < best_train_loss:
                        best_train_loss = train_loss
                        early_stop_counter = 0
                        _train_loss, _train_accuracy = train_loss, train_accuracy
                        test_loss, test_accuracy = test_m(model, x_test, y_test, criterion, device)
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            print("Early stopping!")
                            break
                TRAIN_R2 += _train_accuracy
                TRAIN_LOSS += math.sqrt(_train_loss)
                TEST_R2 += test_accuracy
                TEST_LOSS += math.sqrt(test_loss)
            except StopIteration:
                break
        train_r2.append(TRAIN_R2 / 5), test_r2.append(TEST_R2 / 5), train_rmse.append(TRAIN_LOSS / 5), test_rmse.append(
            TEST_LOSS / 5)
    with open('output/permittivity/rfe/nn_fingerprint_result.txt', 'w') as f:
        f.write(
            f'train_r2:{np.mean(train_r2)}±{np.std(train_r2)}, test_r2:{np.mean(test_r2)}±{np.std(test_r2)}, train_rmse:{np.mean(train_rmse)}±{np.std(train_rmse)}, test_rmse:{np.mean(test_rmse)}±{np.std(test_rmse)}')


def n_kfold():
    input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim = [75, 128, 32, 128, 1]
    n_s, n_f, lr, dropout, active = [0, 3, 1e-3, 0.026, 'leaky_relu']
    epochs = int(1.5 / lr)
    device = torch.device('cpu')
    patience = int(0.1 / lr)

    data = pd.read_csv('output/permittivity/rfe/PI_permittivity_descriptors_fingerprint_RFE.csv').values
    x, y = data[:, 4:], np.log10(np.array(data[:, 3].tolist())).reshape(-1, 1)
    nfold_dict = {"n": [], "train_rmse": [], "train_rmse_std": [], "test_rmse": [], "test_rmse_std": []}
    print("Model starts training！！！")
    for i in tqdm(range(2, 11)):
        train_rmse, test_rmse = [], []
        for _ in tqdm(range(20)):

            kfold = KFold(n_splits=i, shuffle=True, random_state=2023)
            generate = kfold.split(x)
            TRAIN_LOSS, TEST_LOSS = 0, 0
            while True:
                try:
                    Train_idx, Test_idx = next(generate)
                    x_Train, x_test = x[Train_idx], x[Test_idx]
                    y_Train, y_test = y[Train_idx], y[Test_idx]
                    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                    x_Train = norm_x.fit_transform(x_Train)
                    x_test = norm_x.transform(x_test)
                    stdscaler_y = preprocessing.StandardScaler()
                    y_Train = stdscaler_y.fit_transform(y_Train)
                    y_test = stdscaler_y.transform(y_test)

                    model = MultiFidelityModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim,
                                               dropout, n_s, n_f, active)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = torch.nn.MSELoss()

                    best_loss = float('inf')
                    early_stop_counter = 0

                    for epoch in range(epochs):
                        train_loss, train_accuracy = train_m(model, optimizer, x_Train, y_Train, criterion, device)
                        if epoch % 10 == 0 and epoch != 0:
                            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}")
                        if train_loss < best_loss:
                            best_loss = train_loss
                            early_stop_counter = 0
                            test_loss, test_accuracy = test_m(model, x_test, y_test, criterion, device)
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= patience:
                                print("Early stopping!")
                                break
                    TRAIN_LOSS += math.sqrt(best_loss)
                    TEST_LOSS += math.sqrt(test_loss)
                except StopIteration:
                    break
            train_rmse.append(TRAIN_LOSS / i), test_rmse.append(TEST_LOSS / i)
        nfold_dict["n"].append(i)
        nfold_dict["train_rmse"].append(np.mean(train_rmse))
        nfold_dict["train_rmse_std"].append(np.std(train_rmse))
        nfold_dict["test_rmse"].append(np.mean(test_rmse))
        nfold_dict["test_rmse_std"].append(np.std(test_rmse))
    pd.DataFrame.from_dict(nfold_dict).to_csv('output/permittivity/n_fold_result_p.csv', index=False)


if __name__ == '__main__':
    main()
