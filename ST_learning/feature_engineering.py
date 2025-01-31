import mordred
import numpy as np
import pandas as pd
from tqdm import tqdm
from mordred import Calculator, descriptors
from rdkit.Chem import BRICS, rdchem, Descriptors
from rdkit.Chem import AllChem as Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn import preprocessing


def optimize_molecule(smiles):
    mol_opt = []
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        idx = [idx for idx, atomnum in enumerate([atom.GetAtomicNum() for atom in mol.GetAtoms()]) if atomnum == 0]
        [mol.GetAtomWithIdx(i).SetAtomicNum(1) for i in idx]
        mol_H = Chem.AddHs(mol)
        try:
            Chem.EmbedMolecule(mol_H, maxAttempts=5000)
            res = Chem.MMFFOptimizeMolecule(mol_H)
            if res == 0:
                mol_delH = Chem.RemoveHs(mol_H)
                mol_opt.append(mol_delH)
            elif res == 1:
                Chem.UFFOptimizeMolecule(mol_H)
                mol_delH = Chem.RemoveHs(mol_H)
                mol_opt.append(mol_delH)
            elif res == -1:
                mol_delH = Chem.RemoveHs(mol_H)
                mol_opt.append(mol_delH)
        except:
            mol_delH = Chem.RemoveHs(mol_H)
            mol_opt.append(mol_delH)

    print(len(smiles), len(mol_opt))
    return mol_opt


def cal_descriptor(smiles):
    mol_opt = optimize_molecule(smiles)

    calc = Calculator(descriptors, ignore_3D=True)
    result = calc.pandas(mol_opt)
    df_smiles = pd.DataFrame(np.array(smiles), columns=["smiles"])
    pd.concat([df_smiles, result], axis=1).to_csv('output/PI_descriptors.csv', index=False)


def cal_descriptor_rdkit(smiles):
    mol_opt = optimize_molecule(smiles)

    # 计算分子描述符
    descriptor_names = [x[0] for x in Descriptors._descList]
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = pd.DataFrame(
        [descriptor_calculator.CalcDescriptors(mol) for mol in mol_opt], columns=descriptor_names)
    df_smiles = pd.DataFrame(np.array(smiles), columns=["smiles"])
    pd.concat([df_smiles, descriptors], axis=1).to_csv('data/PI/PI_descriptors_rdkit.csv', index=False)


def wipe_off_data(data):
    smiles_pd = data.iloc[:, :7]
    column_list = [column for column in data][7:]

    data_np = data.values[:, 7:]
    m, n = data_np.shape
    print(m, n)

    filter_descriptors = {}
    for i in range(n):
        tag = 0
        sum = 0

        des_Name = column_list[i]
        des_value_column = data_np[:, i].tolist()
        if des_Name == 'GhoseFilter' or des_Name == 'Lipinski':
            c = []
            for j in des_value_column:
                if j == False:
                    c.append(0)
                else:
                    c.append(1)
            des_value_column = c

        for value in des_value_column:
            if isinstance(value, str):
                tag = 1
                break
            if np.isnan(value):
                tag = 1
                break
            sum += abs(value)
        if tag == 1:
            continue
        if sum == 0:
            continue

        filter_descriptors[des_Name] = des_value_column

    filter_descriptors_pd = pd.DataFrame(filter_descriptors)

    data_pd_out = pd.concat([smiles_pd, filter_descriptors_pd], axis=1)
    data_pd_out.to_csv('output/permittivity/PI_permittivity_descriptors_filter.csv', index=False)


def normalization_descriptors(df_descriptors_filter):
    column_list = [column for column in df_descriptors_filter][4:]
    smiles_pd = df_descriptors_filter.iloc[:, :4]
    data_des = df_descriptors_filter.values[:, 4:]
    n_des = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data_des = n_des.fit_transform(data_des)
    filter_descriptors = {}
    for c, column in enumerate(column_list):
        filter_descriptors[column] = data_des[:, c].tolist()
    df_descriptors_filter_normalized = pd.concat([smiles_pd, pd.DataFrame(filter_descriptors)], axis=1)
    return df_descriptors_filter_normalized


def descriptor_processing(var, df_descriptors_filter):
    """
    Remove columns with small variance and columns with large correlation coefficients
    """
    df_descriptors_filter_normalized = normalization_descriptors(df_descriptors_filter)
    smile_pd = df_descriptors_filter_normalized.iloc[:, :7]
    df_descriptors_filter_normalized = df_descriptors_filter_normalized.iloc[:, 7:]
    for column_name, rows in df_descriptors_filter_normalized.iteritems():
        for row in rows:
            if type(row) == mordred.error.Missing:
                df_descriptors_filter_normalized.drop([column_name], axis=1, inplace=True)
                break

    df_descriptors_filter_normalized.dropna(axis=1)
    df_descriptors_filter_normalized.describe()
    var_list_columns = df_descriptors_filter_normalized.var().index.tolist()
    for column in df_descriptors_filter_normalized.columns:
        if column not in var_list_columns:
            print(column)
            df_descriptors_filter_normalized.drop([column], axis=1, inplace=True)

    # Removes columns with variance var of 0
    for column in df_descriptors_filter_normalized.columns:
        if df_descriptors_filter_normalized[column].var() == 0:
            df_descriptors_filter_normalized.drop([column], axis=1, inplace=True)

    # Delete columns with variance var<var
    for column in df_descriptors_filter_normalized.columns:
        if df_descriptors_filter_normalized[column].var() < var:
            df_descriptors_filter_normalized.drop([column], axis=1, inplace=True)
    print(df_descriptors_filter_normalized.values.shape)

    # Remove descriptors that are strongly related to others (|r| > 0.9)
    for column in tqdm(df_descriptors_filter_normalized.columns):
        for r in df_descriptors_filter_normalized.corrwith(df_descriptors_filter_normalized[column]):
            if np.abs(r) > 0.9 and r != 1:
                try:
                    df_descriptors_filter_normalized.drop([column], axis=1, inplace=True)
                except:
                    pass

    print(df_descriptors_filter_normalized.values.shape)
    pd.concat([smile_pd, df_descriptors_filter_normalized], axis=1).to_csv(
        'output/permittivity/PI_permittivity_substructures_filter_%s.csv' % var, index=False)


def split_polymer(smiles):
    groups = []
    for smile in tqdm(smiles):
        group_BRICS = set()
        mol = Chem.MolFromSmiles(smile)
        mw = rdchem.RWMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                mw.ReplaceAtom(atom.GetIdx(), Chem.Atom(1))
        smile_new = Chem.MolToSmiles(mw)
        mol = Chem.MolFromSmiles(smile_new)
        pieces = BRICS.BRICSDecompose(mol)
        group_BRICS.update(pieces)
        for group in group_BRICS:
            group = group.replace('/', '').replace('\\', '')
            group = group.replace('([1*])', '').replace('[1*]', '')
            group = group.replace('([2*])', '').replace('[2*]', '')
            group = group.replace('([3*])', '').replace('[3*]', '')
            group = group.replace('([4*])', '').replace('[4*]', '')
            group = group.replace('([5*])', '').replace('[5*]', '')
            group = group.replace('([6*])', '').replace('[6*]', '')
            group = group.replace('([7*])', '').replace('[7*]', '')
            group = group.replace('([8*])', '').replace('[8*]', '')
            group = group.replace('([9*])', '').replace('[9*]', '')
            group = group.replace('([10*])', '').replace('[10*]', '')
            group = group.replace('([11*])', '').replace('[11*]', '')
            group = group.replace('([12*])', '').replace('[12*]', '')
            group = group.replace('([13*])', '').replace('[13*]', '')
            group = group.replace('([14*])', '').replace('[14*]', '')
            group = group.replace('([15*])', '').replace('[15*]', '')
            group = group.replace('([16*])', '').replace('[16*]', '')
            groups.append(group)
    groups = list(set(groups))
    return groups


def count_polymer_groups(smiles, data):
    groups = split_polymer(smiles)
    fingerprint = []
    for d in tqdm(data):
        group_ = []
        mol = Chem.MolFromSmiles(d[0])
        mw = rdchem.RWMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                mw.ReplaceAtom(atom.GetIdx(), Chem.Atom(1))
        smile_new = Chem.MolToSmiles(mw)
        mol = Chem.MolFromSmiles(smile_new)
        pieces = BRICS.BRICSDecompose(mol)
        for group in pieces:
            group = group.replace('/', '').replace('\\', '')
            group = group.replace('([1*])', '').replace('[1*]', '')
            group = group.replace('([2*])', '').replace('[2*]', '')
            group = group.replace('([3*])', '').replace('[3*]', '')
            group = group.replace('([4*])', '').replace('[4*]', '')
            group = group.replace('([5*])', '').replace('[5*]', '')
            group = group.replace('([6*])', '').replace('[6*]', '')
            group = group.replace('([7*])', '').replace('[7*]', '')
            group = group.replace('([8*])', '').replace('[8*]', '')
            group = group.replace('([9*])', '').replace('[9*]', '')
            group = group.replace('([10*])', '').replace('[10*]', '')
            group = group.replace('([11*])', '').replace('[11*]', '')
            group = group.replace('([12*])', '').replace('[12*]', '')
            group = group.replace('([13*])', '').replace('[13*]', '')
            group = group.replace('([14*])', '').replace('[14*]', '')
            group = group.replace('([15*])', '').replace('[15*]', '')
            group = group.replace('([16*])', '').replace('[16*]', '')
            group_.append(group)

        fingerprint_onehot = [0 for _ in range(len(groups))]
        for group in group_:
            for j in range(len(groups)):
                if group == groups[j]:
                    fingerprint_onehot[j] += 1
        fingerprint.append(fingerprint_onehot)

    pd.concat(
        [pd.read_csv('output/permittivity/PI_permittivity_descriptors_filter.csv'),
         pd.DataFrame(np.array(fingerprint), columns=groups)],
        axis=1).to_csv('output/PI_permittivity_descriptors_fingerprint.csv', index=False)


if __name__ == "__main__":
    # 1.Computation descriptor
    # smiles = pd.read_csv('../raw_data/PI_unique_smile.csv').values.reshape(-1, ).tolist()
    # cal_descriptor(smiles)
    # pd_data = pd.read_csv('output/PI_permittivity.csv')
    # data = pd_data.values.tolist()
    # pd_descriptor = pd.read_csv('output/PI_descriptors.csv')
    # descriptor = pd_descriptor.values.tolist()
    # column_list = [column for column in pd_descriptor][1:]
    # _d = []
    # for d in data:
    #     for des in descriptor:
    #         if des[0] == d[0]:
    #             _d.append(des[1:])
    #             break
    # d_DF = pd.DataFrame(np.array(_d).reshape(len(data), -1), columns=column_list)
    # pd.concat([pd_data, d_DF], axis=1).to_csv('output/permittivity/PI_permittivity_descriptors.csv', index=False)

    # 2.Cleaning descriptor
    # data = pd.read_csv('output/permittivity/PI_permittivity_descriptors.csv')
    # wipe_off_data(data)

    df_descriptors_filter = pd.read_csv('output/permittivity/PI_permittivity_substructures.csv')
    descriptor_processing(0.001, df_descriptors_filter)
