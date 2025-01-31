import os
import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
from Hyperparameter_opt import MultiTaskModel


class DataQualityCheck:
    def __init__(self, property=None):
        self.CurrentPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.PermittivityPath = os.path.join(self.CurrentPath, 'permittivity_v1.0/data/PI')
        self.DielectriclossPath = os.path.join(self.CurrentPath, 'dielectricLoss_v1.0/data/PI')
        self.property = property
        if self.property == "permittivity":
            self.ExperimentPath = os.path.join(self.PermittivityPath, f'PI_{self.property}_exp.csv')
            self.TheoryPath = os.path.join(self.PermittivityPath, f'PI_{self.property}_theory.csv')
            self.label = "ε'"
        elif self.property == "dielectricLoss":
            self.ExperimentPath = os.path.join(self.DielectriclossPath, f'PI_{self.property}_exp.csv')
            self.TheoryPath = os.path.join(self.DielectriclossPath, f'PI_{self.property}_theory.csv')
            self.label = 'ε"'

        self.outputdir = os.path.join(self.CurrentPath, 'MTlearning_v1.0/output/correlation_analysis')
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir, exist_ok=True)
        self.correlation_analysis()

    def correlation_analysis(self):
        self.ExperimentData = pd.read_csv(self.ExperimentPath).iloc[:, 3:-1].values.T
        self.TheoryData = pd.read_csv(self.TheoryPath).iloc[:, 3:-1].values.T
        experiment_correlation_matrix = np.around(np.corrcoef(self.ExperimentData), decimals=3)
        theory_correlation_matrix = np.around(np.corrcoef(self.TheoryData), decimals=3)

        self.plot_heatmap(experiment_correlation_matrix, 'experimental')
        self.plot_heatmap(theory_correlation_matrix, 'theoretical')

        experiment_output_path = os.path.join(self.outputdir, f'correlation_experiment_{self.property}.csv')
        theory_output_path = os.path.join(self.outputdir, f'correlation_theory_{self.property}.csv')
        columns = ["value", "temperature", "frequency"]
        pd.DataFrame(data=experiment_correlation_matrix, columns=columns).to_csv(experiment_output_path, index=False)
        pd.DataFrame(data=theory_correlation_matrix, columns=columns).to_csv(theory_output_path, index=False)

    def plot_heatmap(self, correlation_matrix, source):
        x_labels = [self.label, 'Temperature', 'Frequency']
        y_labels = [self.label, 'Temperature', 'Frequency']

        plt.figure(figsize=(10, 8))
        plt.rcParams['font.family'] = 'Arial'
        ax = sns.heatmap(correlation_matrix, vmin=-1, annot=True, fmt='.3f',
                         annot_kws={"size": 18, "weight": "bold"}, cmap='coolwarm',
                         cbar_kws={'fraction': 0.2, 'shrink': 0.6})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.linspace(-1, 1, 5))
        cbar.ax.tick_params(labelsize=18)

        ax.set_xticklabels(x_labels, fontsize=24, fontweight="bold")
        ax.set_yticklabels(y_labels, fontsize=24, fontweight="bold")
        plt.title(f"Correlation of {source} data on {self.label}",
                  fontdict={'fontsize': 28, 'fontweight': 'bold', 'fontname': 'Arial'}, pad=30)
        plt.savefig(os.path.join(self.outputdir, f'correlation_{source}_{self.property}.png'), dpi=600,
                    bbox_inches='tight')
        # plt.show()


class ablation_experiment:
    def __init__(self):
        self.CurrentPath = os.path.dirname(os.path.abspath(__file__))
        self.data_augmented_path = os.path.join(self.CurrentPath, 'data/PI_permittivity_dielectricLoss.csv')
        self.data_augmented = pd.read_csv(self.data_augmented_path)
        self.data_without_augmented_path = os.path.join(self.CurrentPath,
                                                        'data/PI_permittivity_dielectricLoss_without_augmented.csv')
        if not os.path.exists(self.data_without_augmented_path):
            self.prepare_data_without_augmented()
        self.data_without_augmented = pd.read_csv(self.data_without_augmented_path)

        self.frac_test_l = [round(0.05 * i, 2) for i in range(11, 20)]
        self.output_dir = os.path.join(self.CurrentPath, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.result_with_aug = {"p_RMSE_train": [], "p_RMSE_test": [], "p_R2_train": [], "p_R2_test": [],
                                "d_RMSE_train": [], "d_RMSE_test": [], "d_R2_train": [], "d_R2_test": [], "seed": []}

        self.result_without_aug = {"p_RMSE_train": [], "p_RMSE_test": [], "p_R2_train": [], "p_R2_test": [],
                                   "d_RMSE_train": [], "d_RMSE_test": [], "d_R2_train": [], "d_R2_test": [], "seed": []}
        self.run()
        # self.analysis()

    def prepare_data_without_augmented(self):
        data_without_augmented = self.data_augmented[self.data_augmented["label"] == 0]
        data_without_augmented.to_csv(self.data_without_augmented_path, index=False)

    def run(self):
        for frac_test in self.frac_test_l:
            self.reset_result()
            output_subdir = os.path.join(self.output_dir, f"test_{frac_test}")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir, exist_ok=True)
            for i in range(50):
                seed = np.random.randint(1, 500)
                train_dataset_with_aug, train_dataset_without_aug, test_dataset = self.data_partition(seed=seed,
                                                                                                      frac_test=frac_test)
                result_with_aug = self.training_model(train_dataset_with_aug, test_dataset)
                self.add_result(result_with_aug, seed, flag=True)
                result_without_aug = self.training_model(train_dataset_without_aug, test_dataset)
                self.add_result(result_without_aug, seed, flag=False)
            output_file_with_aug = os.path.join(output_subdir, 'with_augmented.csv')
            output_file_without_aug = os.path.join(output_subdir, 'without_augmented.csv')
            pd.DataFrame.from_dict(self.result_with_aug).to_csv(output_file_with_aug, index=False)
            pd.DataFrame.from_dict(self.result_without_aug).to_csv(output_file_without_aug, index=False)

    def reset_result(self):
        for key in self.result_with_aug.keys():
            self.result_with_aug[key] = []
        for key in self.result_without_aug.keys():
            self.result_without_aug[key] = []

    def data_partition(self, seed, frac_test):
        data_length_with_aug = self.data_augmented.shape[0]
        data_length_without_aug = self.data_without_augmented.shape[0]
        rand_state = np.random.RandomState(int(seed))
        indices_without_aug = [*range(data_length_without_aug)]
        rand_state.shuffle(indices_without_aug)
        indices_with_aug = indices_without_aug + [*range(data_length_without_aug, data_length_with_aug)]

        num_test = int(frac_test * data_length_without_aug)
        test_indices = indices_without_aug[:num_test]
        train_indices_with_aug = indices_with_aug[num_test:]
        train_indices_without_aug = indices_without_aug[num_test:]

        train_dataset_with_aug = self.data_augmented.iloc[train_indices_with_aug, :].values
        train_dataset_without_aug = self.data_without_augmented.iloc[train_indices_without_aug, :].values
        test_dataset = self.data_without_augmented.iloc[test_indices, :].values

        return train_dataset_with_aug, train_dataset_without_aug, test_dataset

    def training_model(self, train_dataset, test_dataset):
        x_Train, y_Train, nonzero_Train, selector_Train, x_Test, y_Test, nonzero_Test, selector_Test = self.data_preprocessing(
            train_dataset, test_dataset)
        input_dim = 107
        s_hidden_dim = 256
        p_hidden_dim = 32
        p_shared_hidden_dim = 64
        d_hidden_dim = 32
        d_shared_hidden_dim = 128
        n_s, n_p, n_p_s, n_d, n_d_s = [0, 2, 3, 0, 1]
        lr = 1e-3
        dropout = 0.05
        active = 'leaky_relu'
        epochs = int(1.5 / lr)
        device = torch.device('cpu')
        patience = int(0.1 / lr)

        model = MultiTaskModel(input_dim, s_hidden_dim, p_hidden_dim, p_shared_hidden_dim, d_hidden_dim,
                               d_shared_hidden_dim, dropout, n_s, n_p, n_p_s, n_d, n_d_s, active)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        best_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(epochs):
            train_loss, train_accuracy, loss = self.train(model, optimizer, x_Train, y_Train, nonzero_Train,
                                                          selector_Train, criterion, device)

            if loss < best_loss:
                best_loss = loss
                early_stop_counter = 0
                _train_loss, _train_accuracy = train_loss, train_accuracy
                # print(f"epoch:{epoch + 1}/{epochs}  R2_p = {train_accuracy[0]:.4f}, R2_d = {train_accuracy[1]:.4f}")
                test_loss, test_accuracy = self.test(model, x_Test, y_Test, nonzero_Test, selector_Test, criterion,
                                                     device)
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping!")
                    break

        return (math.sqrt(_train_loss[0]), math.sqrt(_train_loss[1]), math.sqrt(test_loss[0]), math.sqrt(test_loss[1]),
                train_accuracy[0], train_accuracy[1], test_accuracy[0], test_accuracy[1])

    def data_preprocessing(self, train_dataset, test_dataset):
        x_Train, y_1_Train, y_2_Train = train_dataset[:, 3:], np.log10(np.array(train_dataset[:, 1].tolist())).reshape(
            -1, 1), np.log10(np.array(train_dataset[:, 2].tolist())).reshape(-1, 1)

        x_Test, y_1_Test, y_2_Test = test_dataset[:, 3:], np.log10(np.array(test_dataset[:, 1].tolist())).reshape(
            -1, 1), np.log10(np.array(test_dataset[:, 2].tolist())).reshape(-1, 1)

        y_Train = np.concatenate([y_1_Train, y_2_Train], axis=1)
        selector_Train = np.ones_like(y_Train)
        selector_Train[y_Train == 0] = 0
        nonzero_Train = [np.nonzero(y_Train[:, i]) for i in range(2)]

        y_Test = np.concatenate([y_1_Test, y_2_Test], axis=1)
        selector_Test = np.ones_like(y_Test)
        selector_Test[y_Test == 0] = 0
        nonzero_Test = [np.nonzero(y_Test[:, i]) for i in range(2)]

        norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_Train = norm_x.fit_transform(x_Train)
        x_Test = norm_x.transform(x_Test)

        stdscaler_y_p = preprocessing.StandardScaler()
        p_value_train = stdscaler_y_p.fit_transform(y_Train[:, 0].reshape(-1, 1)[nonzero_Train[0]])
        y_Train[:, 0].reshape(-1, 1)[nonzero_Train[0]] = p_value_train
        p_value_test = stdscaler_y_p.transform(y_Test[:, 0].reshape(-1, 1)[nonzero_Test[0]])
        y_Test[:, 0].reshape(-1, 1)[nonzero_Test[0]] = p_value_test

        stdscaler_y_d = preprocessing.StandardScaler()
        d_value_train = stdscaler_y_d.fit_transform(y_Train[:, 1].reshape(-1, 1)[nonzero_Train[1]])
        y_Train[:, 1].reshape(-1, 1)[nonzero_Train[1]] = d_value_train
        d_value_test = stdscaler_y_d.transform(y_Test[:, 1].reshape(-1, 1)[nonzero_Test[1]])
        y_Test[:, 1].reshape(-1, 1)[nonzero_Test[1]] = d_value_test

        return x_Train, y_Train, nonzero_Train, selector_Train, x_Test, y_Test, nonzero_Test, selector_Test

    def train(self, model, optimizer, x_train, y_train, nonzero_train, selector_train, criterion, device):
        model.train()

        x_data = torch.tensor(x_train[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x_train[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        property_selector = torch.tensor(np.asarray(selector_train, dtype=float), dtype=torch.float32, device=device)
        y_data = torch.tensor(np.asarray(y_train, dtype=float), dtype=torch.float32, device=device)
        pred = model(x_data, label, property_selector)

        losses = [criterion(pred[:, i][nonzero_train[i]], y_data[:, i][nonzero_train[i]]) for i in range(2)]
        r2 = [r2_score(y_train[:, i][nonzero_train[i]], pred[:, i].detach().numpy()[nonzero_train[i]]) for i in
              range(2)]

        loss = ((losses[0] + losses[1]) / 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return losses, r2, loss

    def test(self, model, x_test, y_test, nonzero_test, selector_test, criterion, device):
        model.eval()
        with torch.no_grad():
            x_data = torch.tensor(x_test[:, 1:], dtype=torch.float32, device=device)
            label = torch.tensor(x_test[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
            property_selector = torch.tensor(np.asarray(selector_test, dtype=float), dtype=torch.float32, device=device)
            y_data = torch.tensor(np.asarray(y_test, dtype=float), dtype=torch.float32, device=device)
            pred = model(x_data, label, property_selector)
            losses = [criterion(pred[:, i][nonzero_test[i]], y_data[:, i][nonzero_test[i]]) for i in range(2)]
            r2 = [r2_score(y_test[:, i][nonzero_test[i]], pred[:, i].detach().numpy()[nonzero_test[i]]) for i in
                  range(2)]
            loss = ((losses[0] + losses[1]) / 2)
        return losses, r2

    def add_result(self, result, seed, flag):
        if flag:
            self.result_with_aug["p_RMSE_train"].append(result[0])
            self.result_with_aug["p_RMSE_test"].append(result[2])
            self.result_with_aug["p_R2_train"].append(result[4])
            self.result_with_aug["p_R2_test"].append(result[6])
            self.result_with_aug["d_RMSE_train"].append(result[1])
            self.result_with_aug["d_RMSE_test"].append(result[3])
            self.result_with_aug["d_R2_train"].append(result[5])
            self.result_with_aug["d_R2_test"].append(result[7])
            self.result_with_aug["seed"].append(seed)
        else:
            self.result_without_aug["p_RMSE_train"].append(result[0])
            self.result_without_aug["p_RMSE_test"].append(result[2])
            self.result_without_aug["p_R2_train"].append(result[4])
            self.result_without_aug["p_R2_test"].append(result[6])
            self.result_without_aug["d_RMSE_train"].append(result[1])
            self.result_without_aug["d_RMSE_test"].append(result[3])
            self.result_without_aug["d_R2_train"].append(result[5])
            self.result_without_aug["d_R2_test"].append(result[7])
            self.result_without_aug["seed"].append(seed)

    def analysis(self):
        result_with_aug = {"data_size": [], "p_RMSE_train": [], "p_RMSE_train_std": [], "p_RMSE_test": [],
                           "p_RMSE_test_std": [], "p_R2_train": [], "p_R2_train_std": [], "p_R2_test": [],
                           "p_R2_test_std": [], "d_RMSE_train": [], "d_RMSE_train_std": [], "d_RMSE_test": [],
                           "d_RMSE_test_std": [], "d_R2_train": [], "d_R2_train_std": [], "d_R2_test": [],
                           "d_R2_test_std": []}

        result_without_aug = {"data_size": [], "p_RMSE_train": [], "p_RMSE_train_std": [], "p_RMSE_test": [],
                              "p_RMSE_test_std": [], "p_R2_train": [], "p_R2_train_std": [], "p_R2_test": [],
                              "p_R2_test_std": [], "d_RMSE_train": [], "d_RMSE_train_std": [], "d_RMSE_test": [],
                              "d_RMSE_test_std": [], "d_R2_train": [], "d_R2_train_std": [], "d_R2_test": [],
                              "d_R2_test_std": []}
        for dir in os.listdir(self.output_dir):
            data_length_without_aug = self.data_without_augmented.shape[0]
            num_test = int(float(dir.split("_")[-1]) * data_length_without_aug)

            output_subdir = os.path.join(self.output_dir, dir)
            df_with_aug = pd.read_csv(os.path.join(output_subdir, 'with_augmented.csv'))
            top_30_indices = df_with_aug["p_RMSE_test"].sort_values().head(30).index
            top_30_with_aug = df_with_aug.iloc[top_30_indices, :].values
            result_with_aug["data_size"].append(num_test)
            for id, key in enumerate(result_with_aug.keys()):
                if id > 0:
                    if "std" not in key:
                        result_with_aug[key].append(np.mean(top_30_with_aug[:, id // 2]))
                    else:
                        result_with_aug[key].append(np.std(top_30_with_aug[:, (id // 2 - 1)]))

            df_without_aug = pd.read_csv(os.path.join(output_subdir, 'without_augmented.csv'))
            top_30_without_aug = df_without_aug.iloc[top_30_indices, :].values
            result_without_aug["data_size"].append(num_test)
            for id, key in enumerate(result_without_aug.keys()):
                if id > 0:
                    if "std" not in key:
                        result_without_aug[key].append(np.mean(top_30_without_aug[:, id // 2]))
                    else:
                        result_without_aug[key].append(np.std(top_30_without_aug[:, (id // 2 - 1)]))

        output_file_with_aug = os.path.join(self.output_dir, 'with_augmented.csv')
        output_file_without_aug = os.path.join(self.output_dir, 'without_augmented.csv')
        pd.DataFrame.from_dict(result_with_aug).to_csv(output_file_with_aug, index=False)
        pd.DataFrame.from_dict(result_without_aug).to_csv(output_file_without_aug, index=False)


if __name__ == "__main__":
    DataCheck = DataQualityCheck(property="dielectricLoss")
    # ablation_example = ablation_experiment()
