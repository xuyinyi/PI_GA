import pandas as pd
from multiprocessing import freeze_support
from utils import GAdiCE

from Network import MultiTaskModel, TgModel


def various_cx():
    freeze_support()
    initialPopSize = 100
    selectedPopSize = 50
    mutateRate = 0.1
    numGeneration = 100
    geneA = pd.read_csv('data/geneA.csv', usecols=['SMILES']).values.reshape(-1).tolist()
    geneB = pd.read_csv('data/geneB.csv', usecols=['SMILES']).values.reshape(-1).tolist()

    crossRate_l = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for crossRate in crossRate_l:
        GAdiCE(initialPopSize, selectedPopSize, crossRate, mutateRate, numGeneration, geneA, geneB,
               output=f'output/xxx/')


def various_mut():
    freeze_support()
    initialPopSize = 100
    selectedPopSize = 50
    crossRate = 0.4
    numGeneration = 100
    geneA = pd.read_csv('data/geneA.csv', usecols=['SMILES']).values.reshape(-1).tolist()
    geneB = pd.read_csv('data/geneB.csv', usecols=['SMILES']).values.reshape(-1).tolist()

    mutateRate_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for mutateRate in mutateRate_l:
        GAdiCE(initialPopSize, selectedPopSize, crossRate, mutateRate, numGeneration, geneA, geneB,
               output=f'output/xxx/')


def main():
    freeze_support()
    initialPopSize = 100
    selectedPopSize = 50
    crossRate = 0.8
    mutateRate = 0.4
    numGeneration = 100
    geneA = pd.read_csv('data/geneA.csv', usecols=['SMILES']).values.reshape(-1).tolist()
    geneB = pd.read_csv('data/geneB.csv', usecols=['SMILES']).values.reshape(-1).tolist()

    GAdiCE(initialPopSize, selectedPopSize, crossRate, mutateRate, numGeneration, geneA, geneB,
           output=f'output/xxx/')


if __name__ == '__main__':
    main()
