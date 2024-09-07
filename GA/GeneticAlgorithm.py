import os
import pandas as pd
from multiprocessing import freeze_support
from utils import GAdiCE

from Network import MultiTaskModel, TgModel


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    freeze_support()

    initialPopSize = 5400
    selectedPopSize = 3000
    crossRate = 0.8
    mutateRate = 0.1
    numGeneration = 200
    geneA = pd.read_csv('data/geneA.csv', usecols=['SMILES']).values.reshape(-1).tolist()
    geneB = pd.read_csv('data/geneB.csv', usecols=['SMILES']).values.reshape(-1).tolist()

    myGA = GAdiCE(initialPopSize, selectedPopSize, crossRate, mutateRate, numGeneration, geneA, geneB)
