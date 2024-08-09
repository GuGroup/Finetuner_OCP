from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import read
import numpy as np
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
import argparse

'''
Welcome!
Convert ase.Atoms -> torch.geometric.data.data.Data -> Save on LMDB 
or      ase.Atoms -> Save on aseDB
'''
class converter:
    def __init__(self, name, path):
        self.name = str(name)
        self.path = str(path)
        # AtomsToGraphs determines how to convert ase.Atoms -> torch.geometric.data.data.Data
        self.a2g = AtomsToGraphs(
                    max_neigh=50,
                    radius=6,
                    r_energy=True,
                    r_forces=True,
                    r_distances=False,
                    r_fixed=True
                )
        os.makedirs("data", exist_ok=True)
        
    # Tagging each atom into Fixed: 0, Slab: 1, Adsorbate: 2
    def _set_and_get_tags(self, atoms, ads=None): 
        if ads==None: 
            # If you have different adsorbate atoms, add its atomic symbol here
            ads = ['H', 'C', 'O', 'He', 'N'] 
        atoms.set_tags(np.ones(len(atoms)))
        if atoms.constraints:
            for idx in atoms.constraints[0].index:
                atoms[idx].tag=0
        for atom in atoms:
            if atom.symbol in ads:
                atom.tag=2
        
        return atoms.get_tags()

    def _make_lmdb_files(self): 
        self.lmdb_train = lmdb.open(
            f"data/{self.name}_train.lmdb",
            map_size=1099511627776 * 2, # About 60GB LMDB file is okay with this map size. But if you want to make more more large LMDB, size should be larger
            subdir=False,
            meminit=False,
            map_async=True,
        )

        self.lmdb_test = lmdb.open(
            f"data/{self.name}_test.lmdb",
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )

        self.asedb_test = connect(
            f"data/{self.name}_ase_test.db"
        )

        self.lmdb_val = lmdb.open(
            f"data/{self.name}_val.lmdb",
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )

    def _make_asedb_files(self):
        self.asedb_train = connect(
            f"data/{self.name}_ase_train.db"
        )

        self.asedb_test = connect(
            f"data/{self.name}_ase_test.db"
        )

        self.asedb_val = connect(
            f"data/{self.name}_ase_val.db"
        )

    def asedb_convert(self):
        self._make_asedb_files()
        traj = read(self.path, index=':')

        for _, atoms in tqdm(enumerate(traj), total=len(traj)):
            i = np.random.randint(100)

            if i<=80:
                atoms.set_tags(self._set_and_get_tags(atoms)) #it looks funny..
                calc = atoms.calc
                energy = calc.get_potential_energy()
                forces = (calc.get_forces()).tolist()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                self.asedb_train.write(atoms)

            if i<=90 and i>=81:
                atoms.set_tags(self._set_and_get_tags(atoms))
                calc = atoms.calc
                energy = calc.get_potential_energy()
                forces = (calc.get_forces()).tolist()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                self.asedb_test.write(atoms)

            if i>=91:
                atoms.set_tags(self._set_and_get_tags(atoms))
                calc = atoms.calc
                energy = calc.get_potential_energy()
                forces = (calc.get_forces()).tolist()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                self.asedb_val.write(atoms) 


    # sid: Security id, fid: File id
    def lmdb_convert(self):
        self._make_lmdb_files()
        traj = read(self.path, index=':')

        fid_train = 0
        fid_test = 0
        fid_val = 0

        for idx, atoms in tqdm(enumerate(traj), total=len(traj)):
            i = np.random.randint(100) # Manual split method..

            # data means torch.geometric.data.data
            # ase.Atoms converted into torch.geometric.data.data and get id and tag
            # data is stored to each lmdbfile with txn (transaction)
            if i<=80:
                data = self.a2g.convert(atoms)
                data.sid = torch.LongTensor([0])
                data.tags = torch.LongTensor(self._set_and_get_tags(atoms))
                data.fid = torch.LongTensor([fid_train])
                txn = self.lmdb_train.begin(write=True)
                txn.put(f"{fid_train}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()
                fid_train += 1

            if i<=90 and i>=81:
                data = self.a2g.convert(atoms)
                data.sid = torch.LongTensor([0])
                data.tags = torch.LongTensor(self._set_and_get_tags(atoms))
                data.fid = torch.LongTensor([fid_test])
                txn = self.lmdb_test.begin(write=True)
                txn.put(f"{fid_test}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()
                fid_test += 1
            
                # You might need this..
                atoms.set_tags(self._set_and_get_tags(atoms))
                calc = atoms.calc
                energy = calc.get_potential_energy()
                forces = (calc.get_forces()).tolist()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                self.asedb_test.write(atoms)

            if i>=91:
                data = self.a2g.convert(atoms)
                data.sid = torch.LongTensor([0])
                data.tags = torch.LongTensor(self._set_and_get_tags(atoms))
                data.fid = torch.LongTensor([fid_val])
                txn = self.lmdb_val.begin(write=True)
                txn.put(f"{fid_val}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()
                fid_val += 1
        
        # Total length of dataset required in LMDB
        txn = self.lmdb_train.begin(write=True)
        txn.put(f"length".encode("ascii"), pickle.dumps(fid_train, protocol=-1))
        txn.commit()

        txn = self.lmdb_test.begin(write=True)
        txn.put(f"length".encode("ascii"), pickle.dumps(fid_test, protocol=-1))
        txn.commit()
    
        txn = self.lmdb_val.begin(write=True)
        txn.put(f"length".encode("ascii"), pickle.dumps(fid_val, protocol=-1))
        txn.commit()

        self.lmdb_train.sync()
        self.lmdb_test.sync()
        self.lmdb_val.sync()
        self.lmdb_train.close()
        self.lmdb_test.close()
        self.lmdb_val.close()

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", dest="name", action="store")
parser.add_argument("--asedb", dest="mode", action="store_true")
parser.add_argument("-p", "--path", dest="path", action="store")
args = parser.parse_args()

con = converter(args.name, args.path)
if args.mode:
    con.asedb_convert()
else:
    con.lmdb_convert()

