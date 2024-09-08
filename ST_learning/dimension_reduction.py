import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel, RFECV


def ridgeCV(data):
    """
    Ridge cross validation dimension reduction
    """
    smile_pd = data.iloc[:, :7]
    x, y = data.values[:, 7:], np.log10(np.array(data.values[:, 3].tolist())).reshape(-1, 1)
    a = np.logspace(-3, 2, num=50, endpoint=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=2023)
    reg = SelectFromModel(RidgeCV(alphas=a, cv=kf, fit_intercept=True, normalize=False)).fit(x, y)
    coefs = pd.DataFrame({'coefficient': np.abs(reg.estimator_.coef_.reshape(-1, ))}, index=data.columns[7:])
    coef = coefs[coefs['coefficient'] > reg.threshold_]
    print(
        f"best alpha is {reg.estimator_.alpha_}; score is {reg.estimator_.score(x, y)}; {coef.count(axis=0).values[0]} descriptors were picked.")
    pd.concat([smile_pd, data[coef.index]], axis=1).to_csv(
        'output/permittivity/ridge/PI_permittivity_descriptors_ridge.csv', index=False)


def lassoCV(data):
    """
    Lasso cross validation dimension reduction
    """
    smile_pd = data.iloc[:, :5]
    x, y = data.values[:, 5:], np.log10(np.array(data.values[:, 3].tolist())).reshape(-1, 1)
    a = np.logspace(-4, 1, num=50, endpoint=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=2023)
    reg = LassoCV(alphas=a, cv=kf, fit_intercept=True, normalize=False).fit(x, y)
    coefs = pd.DataFrame({'coefficient': reg.coef_}, index=data.columns[5:])
    coef = coefs[coefs['coefficient'] != 0]
    print(f"best alpha is {reg.alpha_}; score is {reg.score(x, y)}; {sum(reg.coef_ != 0)} descriptors were picked.")
    pd.concat([smile_pd, data[coef.index]], axis=1).to_csv(
        'output/permittivity/lasso/PI_permittivity_descriptors_lasso.csv', index=False)
    return reg.alpha_


def rfeCV(data, alpha):
    """
    RFE cross validation dimension reduction
    """
    smile_pd = data.iloc[:, :5]
    x, y = data.values[:, 5:], np.log10(np.array(data.values[:, 3].tolist())).reshape(-1, 1)
    reg = Lasso(alpha=alpha)  # alpha is the best alpha in lassoCV
    kf = KFold(n_splits=5, shuffle=True, random_state=2023)
    rfecv = RFECV(estimator=reg, step=1, cv=kf, n_jobs=-1, scoring='r2').fit(x, y)
    rank = pd.DataFrame({'ranking': rfecv.ranking_}, index=data.columns[5:])
    rank = rank[rank['ranking'] == 1]
    print(f"{rfecv.n_features_} descriptors were picked.")
    pd.concat([smile_pd, data[rank.index]], axis=1).to_csv(
        'output/permittivity/rfe/PI_permittivity_descriptors_RFE.csv', index=False)


if __name__ == "__main__":
    data = pd.read_csv(r'output/permittivity/xxx.csv')
    ridgeCV(data=data)
    alpha = lassoCV(data=data)
    rfeCV(data=data, alpha=alpha)
