import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def cole_cole(freq, eps_inf, delta_eps, tau, alpha):
    """
    Define the Cole-Cole equation
    """
    return eps_inf + delta_eps / (1 + (1j * 2 * np.pi * freq * tau) ** alpha)


def Havriliak_Negami(freq, eps_inf, delta_eps, tau, alpha, beta):
    """
    Define the Havriliak-Negami equation
    """
    return eps_inf + delta_eps / ((1 + (1j * 2 * np.pi * freq * tau) ** alpha) ** beta)


def fit_cole_cole(freq, eps_real, eps_imag):
    """
    The fitting function is defined to fit Cole-Cole equation and experimental data
    """

    def loss_func(*params):
        eps_fit = cole_cole(*params)
        return np.concatenate([eps_fit.real, -eps_fit.imag])

    initial_guess = (np.min(eps_real), np.max(eps_real) - np.min(eps_real), np.mean(1 / freq), 0.5)
    param_bounds = ([np.min(eps_real), 0, 0, 0], [np.inf, np.inf, np.inf, 1])
    params, _ = curve_fit(loss_func, freq, np.concatenate([eps_real, eps_imag]), p0=initial_guess, bounds=param_bounds)
    return params


def fit_Havriliak_Negami(freq, eps_real, eps_imag):
    """
    The fitting function is defined to fit the Havriliak-Negami equation with the experimental data
    """

    def loss_func(*params):
        eps_fit = Havriliak_Negami(*params)
        return np.concatenate([eps_fit.real, -eps_fit.imag])

    initial_guess = (np.min(eps_real), np.max(eps_real) - np.min(eps_real), np.mean(1 / freq), 0.5, 0.5)
    param_bounds = ([np.min(eps_real), 0, 0, 0, 0], [np.inf, np.inf, np.inf, 1, 1])
    params, _ = curve_fit(loss_func, freq, np.concatenate([eps_real, eps_imag]), p0=initial_guess, bounds=param_bounds)
    return params


def predict_cole_cole(freq, *params):
    """
    The fitting parameters are used to predict the dielectric constant and dielectric loss at high frequencies
    """
    return cole_cole(freq, *params)


def predict_Havriliak_Negami(freq, *params):
    """
    The fitting parameters are used to predict the dielectric constant and dielectric loss at high frequencies
    """
    return Havriliak_Negami(freq, *params)


def generate_highfreq_data_params():
    PI_d_p_path = '../raw_data/PI_p_d'
    freq_high = np.array([1e7 * (5 ** i) for i in range(8)])
    highfreq_real, highfreq_imag, param_list = [], [], []
    for pi in os.listdir(PI_d_p_path):
        data = pd.read_csv(PI_d_p_path + '/' + pi).values.tolist()
        tem = sorted(list(set([i[7] for i in data])))
        smile = data[0][0]
        for t in tem:
            tem_data = [j for j in data if j[7] == t]
            freq = np.array([i[-1] for i in tem_data])
            exp_real = np.array([i[3] for i in tem_data])
            exp_imag = np.array([i[6] for i in tem_data])
            popt = fit_Havriliak_Negami(freq=freq, eps_real=exp_real, eps_imag=exp_imag)
            eps = predict_Havriliak_Negami(freq_high, popt[0], popt[1], popt[2], popt[3], popt[4])
            freq_ = [i for i in freq_high]
            eps_real = [i for i in eps.real]
            eps_imag = [-i for i in eps.imag]
            if min(eps_imag) < 10e-5:
                continue
            else:
                param_list.append([pi, t, popt[0], popt[1], popt[2], popt[3], popt[4]])
                for n in range(len(freq_)):
                    highfreq_real.append([smile, eps_real[n], eps_real[n], eps_real[n], t, math.log10(freq_[n]), 1])
                    highfreq_imag.append([smile, eps_imag[n], eps_imag[n], eps_imag[n], t, math.log10(freq_[n]), 1])
    pd.DataFrame(np.array(param_list),
                 columns=["PI", "Temperature", "eps_inf", "delta_eps", "tau", "alpha", "beta"]).to_csv(
        'output/PI_theory_params.csv', index=False)
    pd.DataFrame(np.array(highfreq_real),
                 columns=["smile", "value_min", "value_max", "value_mean", "testTemperature", "testfrequency",
                          "labled"]).to_csv('output/PI_permittivity_theory.csv', index=False)
    pd.DataFrame(np.array(highfreq_imag),
                 columns=["smile", "value_min", "value_max", "value_mean", "testTemperature", "testfrequency",
                          "labled"]).to_csv('output/PI_dielectricLoss_theory.csv', index=False)


if __name__ == "__main__":
    freq = np.array([1, 10, 100, 1000, 10000, 1e5, 1e6])
    eps_real = np.array([3.43, 3.39, 3.34, 3.32, 3.3, 3.28, 3.27])
    eps_imag = np.array([0.0108, 0.0088, 0.0068, 0.0052, 0.0039, 0.0027, 0.0019])
    freq_ = np.array([1e7, 1e8, 1e9, 1e10, 1e11])

    popt = fit_cole_cole(freq=freq, eps_real=eps_real, eps_imag=eps_imag)
    _ = predict_cole_cole(freq, popt[0], popt[1], popt[2], popt[3])
    exp = np.concatenate([eps_real, eps_imag])
    pred = np.concatenate([_.real, -_.imag])

    _high = predict_cole_cole(freq_, popt[0], popt[1], popt[2], popt[3])
    print(_high.real, -_high.imag)

    popt = fit_Havriliak_Negami(freq=freq, eps_real=eps_real, eps_imag=eps_imag)
    _ = predict_Havriliak_Negami(freq, popt[0], popt[1], popt[2], popt[3], popt[4])
    exp = np.concatenate([eps_real, eps_imag])
    pred = np.concatenate([_.real, -_.imag])

    _high = predict_Havriliak_Negami(freq_, popt[0], popt[1], popt[2], popt[3], popt[4])
    print(_high.real, -_high.imag)

    # generate_highfreq_data_params()
