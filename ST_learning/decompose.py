import pickle
import collections
import numpy as np
import pandas as pd
from pathlib import Path
import rdkit.Chem as Chem
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from gspan.chemutils import (
    get_mol,
    get_smiles,
)
from gspan.features import get_identifier
from gspan.gspan import gSpan
from gspan.config import parser
from gspan.main import main
from gspan.utils import (
    create_gspan_dataset_nx,
    gspan_to_mol,
    preprocess_mols,
)
from rdkit.Chem import Draw


@dataclass
class DefaultConfig:
    data_path: Path
    support: int = 1
    lower: int = 2
    upper: int = 2
    directed: bool = False
    is_save: bool = True
    output_csv: bool = False
    method: str = "raw"


def check_substructures(path, max_nums=100, **kwargs):
    df = pd.read_csv(path, index_col=0)
    sub_mols = [Chem.MolFromSmiles(s) for s in df["0"]]
    Draw.MolsToGridImage(sub_mols[0:max_nums], **kwargs).show()


def get_matrix(report_df, wl_kernel, y):
    # report_df = pd.read_csv(self._config.path.with_name(f"{name}_info.csv"))[
    #     "support_ids"
    # ]
    # report_df = gspan._report_df['support_ids']
    ncols = report_df.shape[0]
    nums = report_df.to_numpy()
    mat = np.zeros((len(y), ncols))
    for i in range(ncols):
        cnt = Counter(nums[i].split(","))
        for key, val in cnt.items():
            mat[int(key), i] = val

    mat = np.array(mat)  # pd.DataFrame(mat)
    X = np.c_[wl_kernel.X[0].X.toarray(), mat]
    return X


class GraphMining:
    def __init__(self, config: DefaultConfig):
        self.config = config
        self.save_name = None

    def _run_gspan(self):
        args_str = (
            f"-s {self.config.support} -d {self.config.directed} -l {self.config.lower} -u {self.config.upper} "
            f"-p False -w False {self.config.data_path}"
        )
        FLAG, _ = parser.parse_known_args(args=args_str.split())
        result = main(FLAG)
        return result

    def decompose(self, graphs):
        if any(graphs):
            create_gspan_dataset_nx(nx_graphs=graphs)
        gspan_object = self._run_gspan()
        return gspan_object


class MolsMining(GraphMining):
    def __init__(self, config: DefaultConfig):
        super(MolsMining, self).__init__(config=config)
        self.config = config
        self.save_name = None

    def decompose(self, mols: Optional[List[Chem.Mol]] = None) -> gSpan:
        if any(mols):
            preprocess_mols(
                mols, fname=self.config.data_path, method=self.config.method
            )

        gspan_object = self._run_gspan()
        if self.config.is_save:
            self.save_csv(gspan_object=gspan_object, mols=mols)

        return gspan_object

    def save_csv(
            self,
            gspan_object: gSpan,
            mols: Optional[List[Chem.Mol]] = None,
            suffix: str = ".pickle",
    ):
        cnf = self.config
        save_name = (
            f"{cnf.data_path.name.split('.')[0]}_s{cnf.support}l{cnf.lower}u{cnf.upper}"
        )

        # fpath = cnf.path.with_name(save_name).with_suffix(suffix)
        self.save_name = save_name

        # Save as CSV
        smiles = [get_smiles(m) for m in mols]
        sub_mols = self.gspan_to_mols(gspan_object, smiles_list=smiles)
        sub_smiles = [get_smiles(m) for m in sub_mols]
        pd.DataFrame(sub_smiles).to_csv(
            cnf.data_path.with_name(save_name).with_suffix(".csv")
        )
        gspan_object._report_df["support_ids"].to_csv(
            cnf.data_path.with_name(f"{save_name}_info.csv")
        )

    def gspan_to_mols(self, gspan: gSpan, smiles_list: Optional[List[str]] = None):
        return gspan_to_mol(gspan, self.config.method, smiles_list=smiles_list)

    @staticmethod
    def save(fpath: Path, obj: gSpan):
        with fpath.open("wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(fpath):
        with fpath.open("rb") as f:
            obj = pickle.load(f)
        return obj


if __name__ == "__main__":
    test_smiles = pd.read_csv(r'xxx\xxx\xxx.csv').values[:, 0]
    test_mols = [get_mol(s) for s in test_smiles]

    cnf = DefaultConfig(data_path=Path("outputs/xxx.data"), method="jt")
    runner = MolsMining(config=cnf)
    # gspan_obj = runner._run_gspan()
    gspan_obj = runner.decompose(test_mols)
    runner.save(Path("outputs/xxx.pickle"), gspan_obj)
    # test = runner.load(Path("outputs/test/gspan_jt.pickle"))
    sub_mols = gspan_to_mol(gspan_obj, method=cnf.method, smiles_list=test_smiles)
    sub_smiles = [get_smiles(m) for m in sub_mols]

    dat, info = get_identifier(test_mols[0], radius=2)
    dat = pd.DataFrame(dat).T.to_numpy().flatten()
    dat = [d for d in dat if d is not None]
    cd = collections.Counter(dat)
    ds = np.unique(dat)
