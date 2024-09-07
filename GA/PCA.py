import pickle
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import AllChem as Chem
from mordred import Calculator, descriptors
from sklearn.decomposition import PCA


def optimize_molecule(smiles):
    mol_opt = []
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        idx = [idx for idx, atomnum in enumerate([atom.GetAtomicNum() for atom in mol.GetAtoms()]) if atomnum == 0]
        [mol.GetAtomWithIdx(i).SetAtomicNum(1) for i in idx]
        mol_H = Chem.AddHs(mol)
        try:
            Chem.EmbedMolecule(mol_H, maxAttempts=1000)
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


def cal_descriptor(smiles, use_descriptor=False):
    smiles_list = smiles["SMILES"].values.tolist()
    mol_opt = optimize_molecule(smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)

    result = calc.pandas(mol_opt)
    if use_descriptor:
        use_des = pd.read_csv('descriptor_list.csv')["descriptor"].values.tolist()
        result = result[use_des]
    pd.concat([smiles, result], axis=1).to_csv(f'xxx/xxx/xxx.csv', index=False)
    del result


def pca(fit=False):
    if fit:
        with open('data/PCA_Distribution/transformer_pca.pkl', 'rb') as fw:
            pca_ = pickle.load(fw)
        fitness = pd.read_csv('xxx/xxx/xxx.csv')["Fitness"]
        data = pd.read_csv('xxx/xxx/xxx.csv').values[:, 8:]
        embedding = pca_.transform(data)
        print(embedding.shape)
        pd.concat([pd.DataFrame(data=embedding, columns=["pca1", "pca2"]), fitness], axis=1).to_csv(
            'xxx/xxx/xxx.csv', index=False)

    else:
        data = pd.read_csv('data/PCA_Distribution/PI_descriptors_train.csv').values[:, 1:]

        pca_ = PCA(n_components=2)
        embedding = pca_.fit_transform(data)
        with open('data/PCA_Distribution/transformer_pca.pkl', 'wb') as fw:
            pickle.dump(pca_, fw)

        pd.DataFrame(data=embedding, columns=["pca1", "pca2"]).to_csv(
            'data/PCA_Distribution/PI_pca_train.csv', index=False)


if __name__ == '__main__':
    pca(fit=True)
