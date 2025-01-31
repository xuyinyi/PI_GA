import os
import gzip
import time
import math
import torch
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors
from mordred import Calculator, descriptors
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from Hyperparameter_opt import MultiTaskModel, TgModel


class GAdiCE:
    def __init__(self, initialPopSize, selectedPopSize, crossRate, mutateRate, numGeneration, geneA, geneB, output):
        self.CurrentPath = os.path.dirname(os.path.abspath(__file__))

        self.initialPopSize = initialPopSize
        self.selectedPopSize = selectedPopSize
        self.crossRate = crossRate
        self.mutateRate = mutateRate
        self.numGeneration = numGeneration

        self.template = 'A-B'
        self.genePool = {'A': geneA, 'B': geneB}
        self.geneType = list(self.genePool.keys())
        self.reactionTemplate = Chem.ReactionFromSmarts(
            '[#8:1]=[#6:2][#8][#6:3]=[#8:4].[#7:6]>>[#7:6]([#6:3]=[#8:4])[#6:2]=[#8:1]')

        self.importModels()
        self.refer_point = np.array([8, 0.5, 1, 10])
        self.RunID = datetime.now().strftime(f"cxpb{self.crossRate}_mutpb{self.mutateRate}_%Y-%m-%d_%H-%M-%S")
        self.TargetDirPath = os.path.join(self.CurrentPath, output)
        self.TargetDirPath = os.path.join(self.TargetDirPath, self.RunID)
        if not os.path.exists(self.TargetDirPath):
            os.makedirs(self.TargetDirPath, exist_ok=True)

        self.allData = pd.DataFrame(data=None,
                                    columns=['Generation', 'SMILES', 'geneA', 'geneB', 'P', 'D', 'T', 'S', 'rank',
                                             'distance'])
        self.HV = pd.DataFrame(data=None, columns=['Generation', 'HV'])
        self.timeLog = pd.DataFrame(data=None, columns=['Generation', 'RunTime'])

        self.parentPopulation = self.createInitialPopulation()
        self.childPopulation = []
        self.runGA()

    def importModels(self):
        ModelPath = os.path.join(self.CurrentPath, 'Model')
        DielectricmodelPath = os.path.join(ModelPath, 'dielectric.pth')
        DielectricdataPath = os.path.join(ModelPath, 'PI_permittivity_dielectricLoss.csv')
        DielectricScalerXPath = os.path.join(ModelPath, 'dielectric_scalerX.pkl')
        PermittivityScalerYPath, DielectricLossScalerYPath = os.path.join(ModelPath,
                                                                          'permittivity_scalerY.pkl'), os.path.join(
            ModelPath, 'dielectricLoss_scalerY.pkl')
        TgmodelPath = os.path.join(ModelPath, 'tg.pth')
        TgdataPath = os.path.join(ModelPath, 'tg_thermoplastic_descriptors_lasso_origin.csv')
        TgScalerXPath, TgScalerYPath = os.path.join(ModelPath, 'Tg_scalerX.pkl'), os.path.join(ModelPath,
                                                                                               'Tg_scalerY.pkl')
        FpScoresPath = os.path.join(ModelPath, 'fpscores.pkl.gz')

        self.DielectricModel = torch.load(DielectricmodelPath)
        with open(DielectricScalerXPath, 'rb') as fp:
            self.DielectricScalerX = pickle.load(fp)
        with open(PermittivityScalerYPath, 'rb') as fp1, open(DielectricLossScalerYPath, 'rb') as fp2:
            self.PermittivityScalerY, self.DielectricLossScalerY = pickle.load(fp1), pickle.load(fp2)
        self.TgModel = torch.load(TgmodelPath)
        with open(TgScalerXPath, 'rb') as fp1, open(TgScalerYPath, 'rb') as fp2:
            self.TgScalerX, self.TgScalerY = pickle.load(fp1), pickle.load(fp2)

        self.Dielectricdata = pd.read_csv(DielectricdataPath)
        self.Tgdata = pd.read_csv(TgdataPath)

        self.MordredCalculator = Calculator(descriptors)
        self.DielectricDescriptors = [column for column in self.Dielectricdata][6:49]
        self.DielectricFingerprints = [column for column in self.Dielectricdata][49:]
        self.TgDescriptors = [column for column in self.Tgdata][4:]

        FpScoresData = pickle.load(gzip.open(FpScoresPath))
        outDict = {}
        for i in FpScoresData:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        self.FpScores = outDict

    def createInitialPopulation(self):
        _initialPopulation = []

        for _ in range(self.initialPopSize):
            _currentChromosome = []
            for gT in self.template.split('-'):
                _currentGene = random.choice(self.genePool[gT])
                _currentChromosome.append(_currentGene)
            _initialPopulation.append(_currentChromosome)

        return _initialPopulation

    def runGA(self):
        for i in range(self.numGeneration + 1):
            startTime = time.time()
            print(
                'Analyze the {} generation individual information and start the {} generation optimization\n'.format(
                    i, i + 1))

            temPop = self.parentPopulation + self.childPopulation
            self.childPopulation, AllData, HV = self.runOpt(temPop, i)

            endTime = time.time()
            runTime = endTime - startTime
            print(
                'The optimization of the {} generation is completed, the duration is {} seconds, and the relevant files are saved'.format(
                    i + 1, runTime))

            m, _ = AllData.shape
            dataIndex = [i] * m
            dataSummary = np.hstack((np.array(dataIndex).reshape(-1, 1), AllData))
            data_temp = pd.DataFrame(dataSummary,
                                     columns=['Generation', 'SMILES', 'geneA', 'geneB', 'P', 'D', 'T', 'S', 'rank',
                                              'distance'])
            self.allData = self.allData.append(data_temp, ignore_index=True)

            HVIndex = [i]
            HVSummary = np.hstack((np.array(HVIndex).reshape(-1, 1), np.array(HV).reshape(-1, 1)))
            HV_temp = pd.DataFrame(data=HVSummary, columns=['Generation', 'HV'])
            self.HV = self.HV.append(HV_temp, ignore_index=True)

            log_temp = pd.DataFrame(data=np.array([i, runTime]).reshape(1, -1), columns=['Generation', 'RunTime'])
            self.timeLog = self.timeLog.append(log_temp, ignore_index=True)

            self.allData.to_csv(os.path.join(self.TargetDirPath, 'dataSummary.csv'), index=False)
            self.HV.to_csv(os.path.join(self.TargetDirPath, 'HV.csv'), index=False)
            self.timeLog.to_csv(os.path.join(self.TargetDirPath, 'log.csv'), index=False)

    def runOpt(self, calPopulation, numGeneration):
        # 1. Encode the genotype of the current sample as a phenotype (return 2D list)
        encodedPop = self.encoding(calPopulation)

        # 2. Verify the rationality of the current population
        valChromosome, valMolObject, inValChromosome = self.validateMolObjects(encodedPop)

        # 3. To evaluate the performance and fitness value of reasonable population
        pred_PDTS, Fits, ranks, distances, valChromosome = self.getFitness(valChromosome, valMolObject,
                                                                           inValChromosome)
        # Population consolidation and optimization
        if numGeneration > 0:
            pred_PDTS, Fits, ranks, distances, valChromosome = self.envSelect(pred_PDTS, Fits, ranks, distances,
                                                                              valChromosome)
        # Calculate hypervolume
        hv_value = self.cal_HV(Fits, ranks, valChromosome)

        # 4. Merge information
        allData = self.mergePopulation(pred_PDTS, ranks, distances)
        self.parentPopulation = valChromosome

        # 5. In the format of the tournament, selection is made (return the genotype and fitness values after selection)
        selectedPop_list, selectedFitness = self.select(valChromosome, Fits, ranks, distances)

        # 6. The current preferred sample is crossed by multiple points according to the cross rate
        childPopulation_list = self.crossover(selectedPop_list)

        # 7. The cross-over samples were mutated according to the mutation rate
        childPopulationNew_list = self.mutate(childPopulation_list)

        # 8. The current generation generates the individual after selecting cross-mutation, the previous generation information (SMILES, P, D, T, S, rank, dis), the current generation HV
        return childPopulationNew_list, allData, hv_value

    def encoding(self, _population):
        """
        :param _population: two-dimensional list, [[geneA, geneB],...]
        :return: two-dimensional list, [[[geneA, geneB], SMILES], ...]
        """

        Chromosome_SMILES = []

        for _pop in _population:
            _result = self.reaction(_pop)
            Chromosome_SMILES.append(_result)

        return Chromosome_SMILES

    def reaction(self, _chromosome):
        """
        According to the reaction rules, the chromosome is encoded into a phenotype, that is,
        chemical blocks are transformed into monomer SMILES by reaction
        :param _chromosome: a list of Mol object of gene [A,B]
        :return: chromosome, SMILES
        """

        def reaction_PI(smi_dianhydride, smi_diamine):
            """
            PI is formed by the reaction of dianhydride and diamine
            """
            reactionTemplate = Chem.ReactionFromSmarts(
                '[#8:1]=[#6:2][#8][#6:3]=[#8:4].[#6:5]-[#7:6]>>[#6:5]-[#7:6]([#6:3]=[#8:4])[#6:2]=[#8:1]')
            mol_dianhydride = Chem.MolFromSmiles(smi_dianhydride)
            mol_diamine = Chem.MolFromSmiles(smi_diamine)
            PI = reactionTemplate.RunReactants((mol_dianhydride, mol_diamine))
            product = []
            for pi in PI:
                pi_smi = Chem.MolToSmiles(pi[0])
                if Chem.MolFromSmiles(pi_smi):
                    product.append(Chem.MolToSmiles(pi[0]))
                else:
                    pass
                product = list(set(product))
                if len(product) > 1:
                    _SMILES = random.choice(product)
                elif len(product) == 1:
                    _SMILES = product[0]
                else:
                    _SMILES = None
            return _SMILES

        def replace_reaction(mol_PI):
            """
            Modify the generated PI molecule
            """
            reactionTemplate = Chem.ReactionFromSmarts(
                '[#8:1]=[#6:2][#8][#6:3]=[#8:4]>>[#6]-[#7]([#6:3]=[#8:4])[#6:2]=[#8:1]')
            PI = reactionTemplate.RunReactants((mol_PI,))
            product = []
            for pi in PI:
                pi_smi = Chem.MolToSmiles(pi[0])
                if Chem.MolFromSmiles(pi_smi):
                    product.append(Chem.MolToSmiles(pi[0]))
                else:
                    pass
                product = list(set(product))
                if len(product) > 1:
                    _SMILES = random.choice(product)
                else:
                    _SMILES = product[0]

            return _SMILES

        PI = reaction_PI(_chromosome[0], _chromosome[1])
        if PI:
            mol_PI = Chem.MolFromSmiles(PI)
            mols = []
            patt1 = Chem.MolFromSmarts('N')
            replace_1 = Chem.MolFromSmarts('C')
            mols.extend(Chem.ReplaceSubstructs(mol_PI, patt1, replace_1))
            PI = [Chem.MolToSmiles(mol) for mol in mols][0]

            mol_PI = Chem.MolFromSmiles(PI)
            PI = replace_reaction(mol_PI)
            _SMILES_std = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(PI)))
        else:
            _SMILES_std = None

        return _chromosome, _SMILES_std

    def validateMolObjects(self, _population):
        """
        :param _population: 2D list, [[chromosome, SMILES], ...].
        :return:
        """
        _validChromosome, _invalidChromosome, _validMolObject = [], [], []

        pool = mp.Pool()
        results = pool.map(self.structure_optimization, _population)
        pool.close()
        pool.join()
        for item in results:
            if item[1]:
                _validChromosome.append(item[0])
                _validMolObject.append(item[1])
            else:
                _invalidChromosome.append(item[0])

        return _validChromosome, _validMolObject, _invalidChromosome

    @staticmethod
    def structure_optimization(_pop):
        chromosome, SMILES = _pop[0], _pop[1]
        try:
            molObject = Chem.MolFromSmiles(SMILES)
            molObject = Chem.AddHs(molObject)
            Chem.EmbedMolecule(molObject, maxAttempts=1000)  # 2D->3D
            res = Chem.MMFFOptimizeMolecule(molObject)
            if res == 0:
                molObject = Chem.RemoveHs(molObject)
            elif res == 1:
                Chem.UFFOptimizeMolecule(molObject)
                molObject = Chem.RemoveHs(molObject)
            elif res == -1:
                molObject = Chem.RemoveHs(molObject)
            return [chromosome, SMILES], molObject
        except:
            return [chromosome, SMILES], None

    def getFitness(self, valChromosome, batchMol, invalChromosome):
        Permittivity_l, DielectricLoss_l, Tg_l, SAscore_l = [], [], [], []
        valid_indices = []
        for _id, mol in enumerate(batchMol):
            Permittivity, DielectricLoss, Tg = self.calculatePermittivityDielectricLossTg(mol)
            if np.isnan(Permittivity) or np.isnan(DielectricLoss) or np.isnan(Tg) or Permittivity < 0:
                invalChromosome.append(valChromosome[_id])
            else:
                Permittivity_l.append(Permittivity)
                DielectricLoss_l.append(DielectricLoss)
                Tg_l.append(Tg)
                valid_indices.append(_id)
                SAscore = self.calculateSAScore(mol)
                SAscore_l.append(SAscore)
        _valChromosome = [valChromosome[idx][0] for idx in valid_indices]
        smiles = np.array([valChromosome[idx][1] for idx in valid_indices]).reshape(-1, 1)
        geneA = np.array([valChromosome[idx][0][0] for idx in valid_indices]).reshape(-1, 1)
        geneB = np.array([valChromosome[idx][0][1] for idx in valid_indices]).reshape(-1, 1)

        Permittivity_array = np.array(Permittivity_l).reshape(-1, 1)
        DielectricLoss_array = np.array(DielectricLoss_l).reshape(-1, 1)
        if np.isnan(DielectricLoss_array).any():
            print('DielectricLossScore contains nan')
        Tg_array = np.array(Tg_l).reshape(-1, 1)
        SAscore_array = np.array(SAscore_l).reshape(-1, 1)
        PDTS_pred = np.hstack((smiles, geneA, geneB, Permittivity_array, DielectricLoss_array, Tg_array, SAscore_array))

        Tg_normed = np.array([self.stepFunc(Tg) for Tg in Tg_l]).reshape(-1, 1)
        Fits = np.hstack((Permittivity_array, DielectricLoss_array, Tg_normed, SAscore_array))
        ranks = self.nonDominationSort(valChromosome, Fits)
        distances = self.crowdingDistanceSort(valChromosome, Fits, ranks)

        return PDTS_pred, Fits, ranks, distances, _valChromosome

    def calculateSAScore(self, _mol):
        # fragment score
        fp = rdMolDescriptors.GetMorganFingerprint(_mol, 2)  # 2 is the *radius* of the circular fingerprint
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += self.FpScores.get(sfp, -4) * v
        score1 /= nf

        # features score
        def numBridgeheadsAndSpiro(mol, ri=None):
            nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            return nBridgehead, nSpiro

        nAtoms = _mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(_mol, includeUnassigned=True))
        ri = _mol.GetRingInfo()
        nBridgeheads, nSpiro = numBridgeheadsAndSpiro(_mol, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    def calculatePermittivityDielectricLossTg(self, _mol):
        des_df = self.MordredCalculator.pandas([_mol])
        groups = []
        value = [0, 200, 10]
        for des in self.DielectricDescriptors:
            value.append(des_df[[des]].values[0][0])
        for f in self.DielectricFingerprints:
            sub_mol = Chem.MolFromSmiles(f)
            matches = _mol.GetSubstructMatches(sub_mol)
            value.append(len(matches))

        all_value_dielectric = np.asarray([value])

        value = []
        for des in self.TgDescriptors:
            if des in des_df:
                value.append(des_df[[des]].values[0][0])
            else:
                if des not in groups:
                    value.append(0)
                else:
                    value.append(groups.count(des))
        all_value_Tg = []
        all_value_Tg.append(value)
        all_value_Tg = np.asarray(all_value_Tg)

        device = torch.device('cpu')

        x = self.DielectricScalerX.transform(all_value_dielectric)
        x_data = torch.tensor(x[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        property_selector = torch.tensor(np.ones((1, 2)), dtype=torch.float32, device=device)
        self.DielectricModel.eval()
        pred = self.DielectricModel(x_data, label, property_selector).cpu().detach().numpy()
        y_pred_p = self.PermittivityScalerY.inverse_transform(pred[:, 0].reshape(-1, 1))
        base_test = np.ones_like(y_pred_p) * 10
        y_pred_p = np.power(base_test, y_pred_p)
        y_pred_p = y_pred_p.reshape(1, ).tolist()
        score_Permittivity = y_pred_p[0]

        y_pred_d = self.DielectricLossScalerY.inverse_transform(pred[:, 1].reshape(-1, 1))
        base_test = np.ones_like(y_pred_d) * 10
        y_pred_d = np.power(base_test, y_pred_d)
        y_pred_d = y_pred_d.reshape(1, ).tolist()
        score_DielectricLoss = y_pred_d[0]

        self.TgModel.eval()
        x = torch.tensor(self.TgScalerX.transform(all_value_Tg), dtype=torch.float32, device=device)
        y_pred = self.TgScalerY.inverse_transform(self.TgModel(x).cpu().detach().numpy())
        Tg = y_pred.tolist()[0][0]

        return score_Permittivity, score_DielectricLoss, Tg

    @staticmethod
    def stepFunc(Tg):
        if Tg > 500:
            score = 0
        elif Tg <= 40:
            score = 1.0
        else:
            score = (500 - Tg) / 460
        return score

    @staticmethod
    def nonDominationSort(pops, fits):
        """
        Non-dominated sorting algorithm
        pops: population, [[[geneA, geneB], SMILES],...]
        fits: fitness, [[fit1, fit2,...],...]
        return: ranks, [1, 2, 3, 4, 5,...]
        """
        num_Pop = len(pops)
        num_Object = fits.shape[1]
        ranks = np.zeros(num_Pop, dtype=np.int32)
        num_Pop_dominated = np.zeros(num_Pop)  # The amount each individual is dominated by other individuals
        set_Pop_dominated = []
        for i in range(num_Pop):
            i_set = []
            for j in range(num_Pop):
                if i == j:
                    continue
                isDom1 = fits[i] <= fits[j]
                isDom2 = fits[i] < fits[j]
                if sum(isDom1) == num_Object and sum(isDom2) >= 1:
                    i_set.append(j)
                if sum(~isDom2) == num_Object and sum(~isDom1) >= 1:
                    num_Pop_dominated[i] += 1
            set_Pop_dominated.append(i_set)
        rank = 0
        indices = np.arange(num_Pop)
        while sum(num_Pop_dominated == 0) != 0:
            rIdices = indices[num_Pop_dominated == 0]
            ranks[rIdices] = rank
            for rIdx in rIdices:
                i_set = set_Pop_dominated[rIdx]
                num_Pop_dominated[i_set] -= 1
            num_Pop_dominated[rIdices] = -1
            rank += 1
        return ranks

    @staticmethod
    def crowdingDistanceSort(pops, fits, ranks):
        """
        Congestion sorting algorithm
        pops: population, [[[geneA, geneB], SMILES],...]
        fits: fitness, [[fit1, fit2,...],...]
        ranks: ranks, [1, 2, 3, 4, 5,...]
        return: dis, [float, float,...]
        """
        num_Pop = len(pops)
        num_Object = fits.shape[1]
        dis = np.zeros(num_Pop)
        num_Rank = ranks.max()
        indices = np.arange(num_Pop)
        for rank in range(num_Rank + 1):
            rIdices = indices[ranks == rank]  # Index of the current rank of population
            rFits = fits[ranks == rank]  # Fitness of the current rank of population
            rSortIdices = np.argsort(rFits, axis=0)  # Index for vertical sorting
            rSortFits = np.sort(rFits, axis=0)
            fMax = np.max(rFits, axis=0)
            fMin = np.min(rFits, axis=0)
            n = len(rIdices)
            for i in range(num_Object):
                orIdices = rIdices[rSortIdices[:, i]]
                j = 1
                while n > 2 and j < n - 1:
                    if fMax[i] != fMin[i]:
                        dis[orIdices[j]] += (rSortFits[j + 1, i] - rSortFits[j - 1, i]) / \
                                            (fMax[i] - fMin[i])
                    else:
                        dis[orIdices[j]] = np.inf
                    j += 1
                dis[orIdices[0]] = np.inf
                dis[orIdices[n - 1]] = np.inf
        return dis

    @staticmethod
    def mergePopulation(validProp, ranks, distances):
        _allData = np.hstack((validProp, ranks.reshape(-1, 1), distances.reshape(-1, 1)))

        return _allData

    def select(self, pops, fits, ranks, distances):
        nPop = len(pops)
        nF = fits.shape[1]
        newPops = []
        newFits = np.zeros((self.selectedPopSize, nF))

        indices = np.arange(nPop).tolist()
        i = 0
        while i < self.selectedPopSize:
            idx1, idx2 = random.sample(indices, 2)
            idx = self.compare(idx1, idx2, ranks, distances)
            newPops.append(pops[idx])
            newFits[i] = fits[idx]
            i += 1

        return newPops, newFits

    @staticmethod
    def compare(idx1, idx2, ranks, distances):
        if ranks[idx1] < ranks[idx2]:
            idx = idx1
        elif ranks[idx1] > ranks[idx2]:
            idx = idx2
        else:
            if distances[idx1] <= distances[idx2]:
                idx = idx2
            else:
                idx = idx1
        return idx

    def crossover(self, _population):
        lenPop, lenGene = len(_population), len(self.geneType)

        _childPopulation = []
        pop_ = np.array(_population)
        pop_copy = pop_.copy()

        for parent in pop_:
            if np.random.rand() < self.crossRate:
                index_ = np.random.randint(0, lenPop, size=1)  # 从现有种群中任选一个个体(基因型,i.e., A, B)进行交叉
                cPs_1 = np.random.randint(0, 2, size=lenGene).astype(np.bool)  # 随机产生多个交叉点(及选取换哪个基因)
                cPs_2 = ~cPs_1
                child_1, child_2 = parent.copy(), parent.copy()
                child_1[cPs_1], child_2[cPs_2] = pop_copy[index_, cPs_1], pop_copy[index_, cPs_2]
                _childPopulation.append(child_1.tolist())
                _childPopulation.append(child_2.tolist())
            else:
                _childPopulation.append(parent.tolist())

        return _childPopulation

    def mutate(self, _population):
        def mutatePop(chromosome_):
            for i in range(len(chromosome_)):
                if np.random.rand() < self.mutateRate:
                    geneType = self.template.split('-')[i]
                    while True:
                        newGene = random.choice(self.genePool[geneType])
                        if newGene == chromosome_[i]:
                            continue
                        chromosome_[i] = newGene
                        break
            return chromosome_

        childPopulation = []
        for pop_ in _population:
            res_ = mutatePop(pop_)
            childPopulation.append(res_)

        return childPopulation

    def envSelect(self, pred_PDTS, Fits, ranks, distances, Pops):
        """
        Population consolidation and optimization
        Return:
            newPops, newFits
        """
        nF = Fits.shape[1]
        smiles = pred_PDTS[:, 0].reshape(-1).tolist()
        geneA = pred_PDTS[:, 1].reshape(-1).tolist()
        geneB = pred_PDTS[:, 2].reshape(-1).tolist()
        new_smiles, new_geneA, new_geneB = [], [], []
        new_pred_PDTS = np.zeros((self.initialPopSize, nF))
        newFits = np.zeros((self.initialPopSize, nF))
        newRanks = np.zeros((self.initialPopSize,))
        newDistances = np.zeros((self.initialPopSize,))
        newPops = []

        indices = np.arange(len(Pops))
        r = 0
        i = 0
        rIndices = indices[ranks == r]
        while i + len(rIndices) <= self.initialPopSize:
            new_pred_PDTS[i:i + len(rIndices)] = pred_PDTS[:, 3:][rIndices]
            newFits[i:i + len(rIndices)] = Fits[rIndices]
            newRanks[i:i + len(rIndices)] = ranks[rIndices]
            newDistances[i:i + len(rIndices)] = distances[rIndices]
            for _ in rIndices:
                new_smiles.append(smiles[_])
                new_geneA.append(geneA[_])
                new_geneB.append(geneB[_])
                newPops.append(Pops[_])

            r += 1
            i += len(rIndices)
            rIndices = indices[ranks == r]

        if i < self.initialPopSize:
            rDistances = distances[rIndices]
            rSortedIdx = np.argsort(rDistances)[::-1]
            surIndices = rIndices[rSortedIdx[:(self.initialPopSize - i)]]
            new_pred_PDTS[i:] = pred_PDTS[:, 3:][surIndices]
            newFits[i:] = Fits[surIndices]
            newRanks[i:] = ranks[surIndices]
            newDistances[i:] = distances[surIndices]
            for _ in surIndices:
                new_smiles.append(smiles[_])
                new_geneA.append(geneA[_])
                new_geneB.append(geneB[_])
                newPops.append(Pops[_])
        new_pred_PDTS = np.hstack((np.array(new_smiles).reshape(-1, 1), np.array(new_geneA).reshape(-1, 1),
                                   np.array(new_geneB).reshape(-1, 1), new_pred_PDTS))

        return new_pred_PDTS, newFits, newRanks, newDistances, newPops

    def cal_HV(self, Fits, ranks, Pops):
        ind = HV(ref_point=self.refer_point)
        indices = np.arange(len(Pops))
        rIndices = indices[ranks == 0]
        non_dominated_set = Fits[rIndices]
        hv_value = ind(non_dominated_set)
        return hv_value


if __name__ == '__main__':
    pass
