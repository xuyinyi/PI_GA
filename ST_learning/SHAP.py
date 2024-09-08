import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def SHAP_analyse(mode='', plot='dot'):
    """
    mode: '' or 'structure'
    plot: 'dot' or 'bar'
    """
    model_path = 'model/permittivity/all_data/xxx.pth'
    device = torch.device('cpu')
    pd_ = pd.read_csv('output/permittivity/rfe/PI_permittivity_descriptors_RFE.csv')
    data = pd_.values
    np.random.seed(2023)
    np.random.shuffle(data)

    x, y = data[:, 5:], np.log10(np.array(data[:, 3].tolist())).reshape(-1, 1)
    norm_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x = norm_x.fit_transform(x)
    x_data = torch.tensor(x[:, :], dtype=torch.float32, device=device)

    model = torch.load(model_path)
    explainer = shap.DeepExplainer(model, x_data)
    shap_values = explainer.shap_values(x_data)

    fig, ax = plt.subplots()
    if mode == 'structure':
        shap_values = shap_values[:, 2:]
        _x_data = torch.tensor(x[:, 2:], dtype=torch.float32, device=device)
        if plot == 'dot':
            shap.summary_plot(shap_values, _x_data, feature_names=list(pd_.columns)[7:], max_display=8, show=False,
                              cmap="viridis_r", color_bar=False, plot_type='dot', plot_size=(8, 8))
        elif plot == 'bar':
            shap.summary_plot(shap_values, _x_data, feature_names=list(pd_.columns)[7:], max_display=20, show=False,
                              plot_type='bar', plot_size=(8, 8))
    else:
        if plot == 'dot':
            shap.summary_plot(shap_values, x_data, feature_names=list(pd_.columns)[5:], max_display=10, show=False,
                              cmap="cool", color_bar=False, plot_type='dot', plot_size=(7.5, 8))
        elif plot == 'bar':
            shap.summary_plot(shap_values, x_data, feature_names=list(pd_.columns)[5:], max_display=15, show=False,
                              plot_type='bar', plot_size=(8, 8))

    # ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    for label in ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_weight('bold')
        label.set_fontname("Arial")
        label.set_rotation(30)
    for label in ax.get_xticklabels():
        label.set_fontsize(20)
        label.set_weight('bold')
        label.set_fontname("Arial")
    ax.tick_params(axis='x', which='major', width=4)
    ax.xaxis.get_label().set_fontsize(28)
    ax.xaxis.get_label().set_fontweight('bold')
    ax.xaxis.get_label().set_fontname("Arial")
    plt.gcf().axes[-1].tick_params(labelsize=24)
    ax.set_xlabel("Mean(|SHAP value|) (average impact on model output magnitude)", fontsize=28, fontweight='bold',
                  fontname="Arial")
    plt.savefig('output/figure/permittivity/xxx.png', dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    SHAP_analyse(mode='structure', plot='bar')
