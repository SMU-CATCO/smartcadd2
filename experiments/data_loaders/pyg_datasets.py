import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter

import pandas as pd

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor(
    [
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        1.0,
        1.0,
        KCALMOL2EV,
        1.0,
        1.0,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        KCALMOL2EV,
        KCALMOL2EV,
    ]
)


class QM40(InMemoryDataset):
    r"""The QM40 dataset consisting of molecules with 16 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | Internal_E(0K)                   | Internal energy at 0K                                                             | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | HOMO                             | Energy of HOMO                                                                    | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | LUMO                             | Energy of LUMO                                                                    | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | HL_gap                           | Energy difference of (HOMO - LUMO)                                                | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | Polarizability                   | Isotropic polarizability                                                          | :math:`a_0^3`                               |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | spatial_extent                   | Electronic spatial extent                                                         | :math:`a_0^2`                               |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | dipol_moment                     | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | ZPE                              | Zero point energy                                                                 | :math:`\textrm{Kcal/mol}`                   |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | rot1                             | Rotational constant 1                                                             | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | rot2                             | Rotational constant 2                                                             | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | rot3                             | Rotational constant 3                                                             | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | Inter_E(298)                     | Internal energy at 298.15K                                                        | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | Enthalpy                         | Enthalpy at 298.15K                                                              | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | Free_E                           | Free energy at 298.15K                                                            | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | CV                               | Heat capacity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | Entropy                          | Entropy at 298.15K                                                                | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    """

    raw_url = None
    processed_url = None

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa

            return ["QM40_main.csv", "QM40_xyz.csv", "QM40_bond.csv"]
        except ImportError:
            return ["qm40.pt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    # def download(self) -> None:
    #     try:
    #         import rdkit  # noqa

    #         file_path = download_url(self.raw_url, self.raw_dir)
    #         extract_zip(file_path, self.raw_dir)
    #         os.unlink(file_path)

    #     except ImportError:
    #         path = download_url(self.processed_url, self.raw_dir)
    #         extract_zip(path, self.raw_dir)
    #         os.unlink(path)
    def download(self) -> None:
        pass

    # def load(path):

    def process(self) -> None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType

            RDLogger.DisableLog("rdApp.*")  # type: ignore
            WITH_RDKIT = True

        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT:
            print(
                (
                    "Using a pre-processed version of the dataset. Please "
                    "install 'rdkit' to alternatively process the raw data."
                ),
                file=sys.stderr,
            )

            data_list = fs.torch_load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return

        types = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
        type_idx_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        print("Loading raw data...")
        main_df = pd.read_csv(self.raw_paths[0])
        xyz_df = pd.read_csv(self.raw_paths[1])
        bond_df = pd.read_csv(self.raw_paths[2])

        # Pre-process xyz_df and bond_df
        xyz_df_grouped = xyz_df.groupby("Zinc_id")
        bond_df_grouped = bond_df.groupby("Zinc_id")

        data_list = []
        print("Processing raw data...")
        for mol_idx, row in tqdm(main_df.iterrows(), total=len(main_df)):
            ID = row["Zinc_id"]
            SMILES = row["smile"]
            y = torch.tensor(
                row.iloc[2:].values.astype(np.float32), dtype=torch.float
            )

            mol_xyz = xyz_df_grouped.get_group(ID).reset_index(drop=True)
            mol_bonds = bond_df_grouped.get_group(ID).reset_index(drop=True)

            mol = Chem.MolFromSmiles(SMILES)
            mol = Chem.AddHs(mol)
            N = mol.GetNumAtoms()

            # Vectorize atom property extraction
            atoms = mol.GetAtoms()
            atomic_numbers = np.array([atom.GetAtomicNum() for atom in atoms])
            type_idx = np.array([type_idx_map[num] for num in atomic_numbers])
            aromatic = np.array([int(atom.GetIsAromatic()) for atom in atoms])
            hybridizations = np.array(
                [atom.GetHybridization() for atom in atoms]
            )
            sp = (hybridizations == HybridizationType.SP).astype(int)
            sp2 = (hybridizations == HybridizationType.SP2).astype(int)
            sp3 = (hybridizations == HybridizationType.SP3).astype(int)

            # Set charges and positions
            conf = Chem.Conformer(N)
            for i, row in mol_xyz.iterrows():
                atoms[i].SetFormalCharge(int(round(row["charge"])))
                conf.SetAtomPosition(
                    i, (row["final_x"], row["final_y"], row["final_z"])
                )

            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            z = torch.tensor(atomic_numbers, dtype=torch.long)

            # Process bonds
            bond_data = [
                (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bonds[bond.GetBondType()],
                )
                for bond in mol.GetBonds()
            ]
            rows, cols, edge_types = zip(*bond_data)
            rows, cols = rows + cols, cols + rows  # Add reverse edges
            edge_types = edge_types + edge_types

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))
            edge_attr2 = torch.tensor(
                mol_bonds["lmod"].tolist(), dtype=torch.float
            )

            # Sort edges
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            # Count hydrogens
            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            # Create node features
            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    np.array([aromatic, sp, sp2, sp3, num_hs]),
                    dtype=torch.float,
                )
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=SMILES,
                edge_attr=edge_attr,
                edge_attr2=edge_attr2,
                y=y * conversion.view(1, -1),
                name=ID,
                idx=mol_idx,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
