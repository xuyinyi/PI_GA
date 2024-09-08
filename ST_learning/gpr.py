import math
import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def objective(trial):
    data = pd.read_csv('output/permittivity/ridge/xxx.csv').values
    x, y = data[:, 5:], np.log10(np.array(data[:, 3].tolist())).reshape(-1, 1)

    length_scale = trial.suggest_float('length_scale', 1e-1, 1e1)
    noise_lever = trial.suggest_float('noise_lever', 1e-6, 1e-1)
    alpha = trial.suggest_float('alpha', 1e-4, 1e0)
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_lever)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

    kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
    generate = kfold.split(x)
    TEST_LOSS = 0
    while True:
        try:
            train_idx, test_idx = next(generate)
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            x_train = norm_x.fit_transform(x_train)
            x_test = norm_x.transform(x_test)
            stdscaler_y = preprocessing.StandardScaler()
            y_train = stdscaler_y.fit_transform(y_train)
            y_test = stdscaler_y.transform(y_test)
            gp.fit(x_train, y_train)
            y_pred, sigma = gp.predict(x_test, return_std=True)
            mse = mean_squared_error(y_test, y_pred)
            TEST_LOSS += mse
        except StopIteration:
            break
    print(TEST_LOSS / 5)
    return TEST_LOSS / 5


def gpr():
    data = pd.read_csv('output/permittivity/ridge/xxx.csv').values
    x, y = data[:, 5:], np.log10(np.array(data[:, 3].tolist())).reshape(-1, 1)

    length_scale = 9.6
    noise_lever = 0.07
    alpha = 0.11
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_lever)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

    train_r2, test_r2, train_rmse, test_rmse = [], [], [], []
    for _ in tqdm(range(20)):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
        generate = kfold.split(x)
        TRAIN_R2, TEST_R2, TRAIN_LOSS, TEST_LOSS = 0, 0, 0, 0
        while True:
            try:
                train_idx, test_idx = next(generate)
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                x_train = norm_x.fit_transform(x_train)
                x_test = norm_x.transform(x_test)
                stdscaler_y = preprocessing.StandardScaler()
                y_train = stdscaler_y.fit_transform(y_train)
                y_test = stdscaler_y.transform(y_test)

                gp.fit(x_train, y_train)
                score = gp.score(x_train, y_train)
                TRAIN_R2 += score
                y_train_pred = gp.predict(x_train)
                y_pred, sigma = gp.predict(x_test, return_std=True)
                train_mse = mean_squared_error(y_train, y_train_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                TEST_R2 += r2
                TRAIN_LOSS += math.sqrt(train_mse)
                TEST_LOSS += math.sqrt(mse)
            except StopIteration:
                break
        train_r2.append(TRAIN_R2 / 5), test_r2.append(TEST_R2 / 5), train_rmse.append(TRAIN_LOSS / 5), test_rmse.append(
            TEST_LOSS / 5)
    with open('output/ridge/xxx.txt', 'w') as f:
        f.write(
            f'train_r2:{np.mean(train_r2)}±{np.std(train_r2)}, test_r2:{np.mean(test_r2)}±{np.std(test_r2)}, train_rmse:{np.mean(train_rmse)}±{np.std(train_rmse)}, test_rmse:{np.mean(test_rmse)}±{np.std(test_rmse)}')


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best test loss: {study.best_value:.4f}")

    with open('output/ridge/xxx.log', 'w') as f:
        f.write(f"Best trial: {study.best_trial.params}" + '\n' + f"Best test loss: {study.best_value:.4f}")

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
    # gpr()
