import os
import math
import gzip
import time
import copy
import torch
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdchem, BRICS
from rdkit.Chem import rdMolDescriptors
from mordred import Calculator, descriptors

from Network import MultiTaskModel, TgModel


class GAdiCE:
    def __init__(self, initialPopSize, selectedPopSize, crossRate, mutateRate, numGeneration, geneA, geneB):
        self.CurrentPath = os.path.dirname(os.path.abspath(__file__))

        self.initialPopSize = initialPopSize
        self.selectedPopSize = selectedPopSize
        self.crossRate = crossRate
        self.mutateRate = mutateRate
        self.numGeneration = numGeneration

        # Converts gene from SMILES to Mol object
        self.geneAs = self.convertGene(geneA)
        self.geneBs = self.convertGene(geneB)

        # Declaration of dianhydride and diamine monomer structure template related settings, reaction rules
        self.template = 'A-B'
        self.genePool = {'A': self.geneAs, 'B': self.geneBs}
        self.geneType = list(self.genePool.keys())
        self.reactionTemplate = Chem.ReactionFromSmarts(
            '[#8:1]=[#6:2][#8][#6:3]=[#8:4].[#7:6]>>[#7:6]([#6:3]=[#8:4])[#6:2]=[#8:1]')

        self.DielectricModel = None
        self.TgModel = None

        self.MordredCalculator = None
        self.FpScores = None
        self.importModels()

        self.RunID = datetime.now().strftime("InverseByGA_%Y-%m-%d_%H-%M-%S")
        self.TargetDirPath = os.path.join(self.CurrentPath, self.RunID)
        if not os.path.exists(self.TargetDirPath):
            os.mkdir(self.TargetDirPath)

        self.allData = pd.DataFrame(data=None, columns=['Generation', 'SMILES', 'S', 'P', 'D', 'T', 'Fitness'])
        self.allFitness = pd.DataFrame(data=None, columns=['Generation', 'Fitness'])
        self.selectedFitness = pd.DataFrame(data=None, columns=['Generation', 'Fitness'])
        self.timeLog = pd.DataFrame(data=None, columns=['Generation', 'RunTime'])

        self.currentPopulation = self.createInitialPopulation()

        self.runGA()

    @staticmethod
    def convertGene(genesSMILES):
        assert isinstance(genesSMILES, list), 'The type of gene variable is not list'

        genesMol = [Chem.MolFromSmiles(gene) for gene in genesSMILES]

        for _i in range(len(genesMol)):
            if genesMol[_i] is None:
                assert genesSMILES[_i] == '*H', 'The gene object is None, please check {}'.format(genesSMILES[_i])

        return genesMol

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
                'Analyze the {} generation individual information and start the {} generation optimization\n'.format(i,
                                                                                                                     i + 1))

            tempPop = self.currentPopulation
            self.currentPopulation, AllFit, SelectedFit, AllData = self.runOpt(tempPop)

            endTime = time.time()
            runTime = endTime - startTime
            print(
                'The optimization of the {} generation is completed, the duration is {} seconds, and the relevant files are saved'.format(
                    i + 1, runTime))

            m, _ = AllData.shape
            dataIndex = [i] * m
            dataSummary = np.hstack((np.array(dataIndex).reshape(-1, 1), AllData))
            data_temp = pd.DataFrame(dataSummary, columns=['Generation', 'SMILES', 'S', 'P', 'D', 'T', 'Fitness'])
            self.allData = self.allData.append(data_temp, ignore_index=True)

            AllFitnessIndex = [i] * len(AllFit)
            AllFitnessSummary = np.hstack((np.array(AllFitnessIndex).reshape(-1, 1), np.array(AllFit).reshape(-1, 1)))
            all_temp = pd.DataFrame(data=AllFitnessSummary, columns=['Generation', 'Fitness'])
            self.allFitness = self.allFitness.append(all_temp, ignore_index=True)

            SelectedFitnessIndex = [i] * len(SelectedFit)
            SelectedFitnessSummary = np.hstack(
                (np.array(SelectedFitnessIndex).reshape(-1, 1), np.array(SelectedFit).reshape(-1, 1)))
            selected_temp = pd.DataFrame(data=SelectedFitnessSummary, columns=['Generation', 'Fitness'])
            self.selectedFitness = self.selectedFitness.append(selected_temp, ignore_index=True)

            log_temp = pd.DataFrame(data=np.array([i, runTime]).reshape(1, -1), columns=['Generation', 'RunTime'])
            self.timeLog = self.timeLog.append(log_temp, ignore_index=True)

            self.allData.to_csv(os.path.join(self.TargetDirPath, 'dataSummary.csv'), index=False)
            self.allFitness.to_csv(os.path.join(self.TargetDirPath, 'allFitness.csv'), index=False)
            self.selectedFitness.to_csv(os.path.join(self.TargetDirPath, 'selectedFitness.csv'), index=False)
            self.timeLog.to_csv(os.path.join(self.TargetDirPath, 'log.csv'), index=False)

    def runOpt(self, _currentPopulation):
        # 1. Encode the genotype of the current sample as a phenotype (return 2D list)
        encodedPop = self.encoding(_currentPopulation)

        # 2. Verify the rationality of the current population
        valChromosome, valMolObject, inValChromosome = self.validateMolObjects(encodedPop)

        # 3. To evaluate the performance and fitness value of reasonable population
        predSPDTF, predFitness, valChromosome, inValChromosome = self.getFitness(valChromosome, valMolObject,
                                                                                 inValChromosome)

        # 4. Merge information
        rawPop_list, rawFitness, allData = self.mergePopulation(inValChromosome, valChromosome, predSPDTF, predFitness)

        # 5. In roulette form, the selection is made (returns the genotype and fitness values after selection)
        selectedPop_list, selectedFitness = self.select(rawPop_list, rawFitness)

        # 6. The current preferred sample is crossed by multiple points according to the cross rate
        childPopulation_list = self.crossover(selectedPop_list)

        # 7. The cross-over samples were mutated according to the mutation rate
        childPopulationNew_list = self.mutate(childPopulation_list)

        return childPopulationNew_list, rawFitness, selectedFitness, allData

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
        Chemical Blocks are transformed into monomer SMILES by reaction
        :param _chromosome: a list of Mol object of gene [A,B]
        :return: chromosome, SMILES
        """

        def reaction_PI(mol_dianhydride, mol_diamine):
            """The dianhydride reacts with diamine to form PI"""
            reactionTemplate = Chem.ReactionFromSmarts(
                '[#8:1]=[#6:2][#8][#6:3]=[#8:4].[#6:5]-[#7:6]>>[#6:5]-[#7:6]([#6:3]=[#8:4])[#6:2]=[#8:1]')
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
            """Modify the generated PI molecule"""
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

        PI = reaction_PI(_chromosome[0], _chromosome[1])  # PI的Smiles
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

        _validChromosome = []
        _invalidChromosome = []

        _tempChromosome, _tempMolObject = [], []

        pool = mp.Pool()
        results = pool.map(self.validateMaxAtomNum, _population)
        pool.close()
        pool.join()
        for item in results:
            if item[1]:
                _tempChromosome.append(item[0])
                _tempMolObject.append(item[1])
            elif not item[1]:
                _invalidChromosome.append(item[0])

        _validMolObject = []
        # des_df = self.MordredCalculator.pandas(_tempMolObject)
        # desRawValues_np = des_df[self.targetdescriptor].values
        # _m, _ = desRawValues_np.shape
        # for _i in range(_m):
        #     if np.isnan(desRawValues_np[_i, :].astype('float')).any():
        #         _invalidChromosome.append(_tempChromosome[_i])
        #     else:
        #         _validChromosome.append(_tempChromosome[_i])
        #         _validMolObject.append(_tempMolObject[_i])
        _validMolObject = _tempMolObject
        _validChromosome = _tempChromosome

        return _validChromosome, _validMolObject, _invalidChromosome

    @staticmethod
    def validateMaxAtomNum(_pop):
        chromosome, SMILES = _pop[0], _pop[1]
        try:
            molObject = Chem.MolFromSmiles(SMILES)
            molObject = Chem.AddHs(molObject)
            Chem.EmbedMolecule(molObject, maxAttempts=1000)
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
        scorePermittivity, scoreDielectricLoss, Tg_Pred = [], [], []
        valid_indices = []
        for _id, mol in enumerate(batchMol):
            score_Permittivity, score_DielectricLoss, Tg = self.calculatePermittivityDielectricLossTgScore(mol)
            if np.isnan(score_Permittivity) or np.isnan(score_DielectricLoss) or np.isnan(Tg) or score_Permittivity < 0:
                invalChromosome.append(valChromosome[_id])
            else:
                scorePermittivity.append(score_Permittivity)
                scoreDielectricLoss.append(score_DielectricLoss)
                Tg_Pred.append(Tg)
                valid_indices.append(_id)
        valChromosome = [valChromosome[idx] for idx in valid_indices]

        PermittivityScore = np.array(scorePermittivity).reshape(-1, 1)
        DielectricLossScore = np.array(scoreDielectricLoss).reshape(-1, 1)
        if np.isnan(DielectricLossScore).any():
            print('DielectricLossScore contains nan')
        Tg_Pred = np.array(Tg_Pred).reshape(-1, 1)

        scoreSA = []
        for idx in valid_indices:
            mol = batchMol[idx]
            score = self.calculateSAScore(mol)
            scoreSA.append(score)
        SaScore = np.array(scoreSA).reshape(-1, 1)

        SPDT_Pred = np.hstack((SaScore, PermittivityScore, DielectricLossScore, Tg_Pred))

        obj_copy = np.hstack((PermittivityScore, DielectricLossScore))

        obj_min, obj_max = obj_copy.min(axis=0), obj_copy.max(axis=0)
        Obj_normed = (obj_copy - obj_min) / (obj_max - obj_min + 1e-10)

        TgScore = np.array([self.stepFunc(Tg) for Tg in Tg_Pred.reshape(-1)]).reshape(-1, 1)
        SATmScore = np.hstack((SaScore, TgScore))
        ST_min, ST_max = SATmScore.min(axis=0), SATmScore.max(axis=0)
        SATmScore_normed = (SATmScore - ST_min) / (ST_max - ST_min + 1e-10)

        ScoreAll = np.hstack((SATmScore_normed, Obj_normed))
        OverallScore = np.average(ScoreAll, axis=1)
        OverallScore_normed = (OverallScore - OverallScore.min()) / (OverallScore.max() - OverallScore.min() + 1e-10)
        Fitness_normed = 1. - OverallScore_normed + 1e-5
        Fitness_normed = Fitness_normed.reshape(-1, 1)

        SPDTF_pred = np.hstack((SPDT_Pred, Fitness_normed))

        return SPDTF_pred, Fitness_normed, valChromosome, invalChromosome

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

    def calculatePermittivityDielectricLossTgScore(self, _mol):
        des_df = self.MordredCalculator.pandas([_mol])
        # groups = self.getGC(_mol)
        groups = []
        value = [0, 0, 0]
        for des in self.DielectricDescriptors:
            value.append(des_df[[des]].values[0][0])
        for f in self.DielectricFingerprints:
            sub_mol = Chem.MolFromSmiles(f)
            matches = _mol.GetSubstructMatches(sub_mol)
            value.append(len(matches))
        all_value_dielectric = []
        for i in range(8):
            v = copy.copy(value)
            v[1] = i * 25
            v[2] = 10
            all_value_dielectric.append(v[:])
        all_value_dielectric = np.asarray(all_value_dielectric)

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
        property_selector = torch.tensor(np.ones((8, 2)), dtype=torch.float32, device=device)
        pred = self.DielectricModel(x_data, label, property_selector).cpu().detach().numpy()
        y_pred_p = self.PermittivityScalerY.inverse_transform(pred[:, 0].reshape(-1, 1))
        base_test = np.ones_like(y_pred_p) * 10
        y_pred_p = np.power(base_test, y_pred_p)
        y_pred_p = y_pred_p.reshape(8, ).tolist()
        score_Permittivity = np.std(y_pred_p) * np.mean(y_pred_p)

        y_pred_d = self.DielectricLossScalerY.inverse_transform(pred[:, 1].reshape(-1, 1))
        base_test = np.ones_like(y_pred_d) * 10
        y_pred_d = np.power(base_test, y_pred_d)
        y_pred_d = y_pred_d.reshape(8, ).tolist()
        score_DielectricLoss = np.std(y_pred_d) * np.mean(y_pred_d)

        x = torch.tensor(self.TgScalerX.transform(all_value_Tg), dtype=torch.float32, device=device)
        y_pred = self.TgScalerY.inverse_transform(self.TgModel(x).cpu().detach().numpy())
        Tg = y_pred.tolist()[0][0]

        return score_Permittivity, score_DielectricLoss, Tg

    @staticmethod
    def getGC(mol):
        group_BRICS = set()
        groups = []
        mw = rdchem.RWMol(mol)
        index = 0

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                mw.ReplaceAtom(index, Chem.Atom(1))
            index = index + 1
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

        return groups

    @staticmethod
    def stepFunc(Tg):
        if Tg > 400:
            TF = 1.0
        elif 300 < Tg <= 400:
            TF = 3.0
        elif 200 < Tg <= 300:
            TF = 5.0
        elif 100 < Tg <= 200:
            TF = 7.0
        else:
            TF = 9.0
        return TF

    @staticmethod
    def mergePopulation(invalidPop, validPop, validProp, validFit):
        ivP, vP = np.array(invalidPop, dtype=object), np.array(validPop, dtype=object)
        miv = len(ivP)
        mv = len(vP)
        assert mv > 0, 'Error'
        if miv > 0:
            inValFit = np.zeros(miv)
            _allPop = vP[:, 0].tolist() + ivP[:, 0].tolist()
            _allFit = np.append(validFit, inValFit)
            _allFit = _allFit.reshape(-1)
        else:
            _allPop = vP[:, 0].tolist()
            _allFit = validFit
            _allFit = _allFit.reshape(-1)

        _allData = np.hstack((vP[:, 1].reshape(-1, 1), validProp))

        return _allPop, _allFit, _allData

    def select(self, _population, _fitness):
        _population = np.array(_population)
        idx = np.random.choice(np.arange(len(_population)), size=self.selectedPopSize, replace=True,
                               p=_fitness / _fitness.sum())

        return _population[idx].tolist(), _fitness[idx]

    def crossover(self, _population):
        lenPop, lenGene = len(_population), len(self.geneType)

        _childPopulation = []
        pop_ = np.array(_population)
        pop_copy = pop_.copy()

        for parent in pop_:
            if np.random.rand() < self.crossRate:
                index_ = np.random.randint(0, lenPop, size=1)
                cPs_1 = np.random.randint(0, 2, size=lenGene).astype(np.bool)
                cPs_2 = ~cPs_1
                child_1, child_2 = parent.copy(), parent.copy()
                child_1[cPs_1], child_2[cPs_2] = pop_copy[index_, cPs_1], pop_copy[index_, cPs_2]
                _childPopulation.append(child_1)
                _childPopulation.append(child_2)
            else:
                _childPopulation.append(parent)

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
