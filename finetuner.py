from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset
from fairchem.core.common.tutorial_utils import generate_yml_config
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.models.model_registry import available_pretrained_models
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
import time
import subprocess
from abc import ABC, abstractmethod
from typing import List, Union
import pathlib
'''

Total assembly of 
Prepare dataset - Make configuration file - Fine-tuning

Code structure is updated, different with previous files
If you want to modify this one, I recommend do it after reading other files

ykt's parameter pick:
Dataset: lmdb
batch_size: 16
eval_every: 10

'''

class TrajLoader:
    def load_file(
        self,
        path: str
    ) -> List[ase.Atoms]:
        traj = read(self.path, index=':')
        return traj


class FileMaker(ABC):
    @abstractmethod
    def make_file(
        self,
        name: str
    ) -> List[Environment]:
        pass


class LmdbMaker(FileMaker):
    def make_file(
        self,
        name: str
    ) -> List[Environment]:
        lmdb_train = lmdb.open(
            f"data/{name}_train.lmdb",
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        lmdb_test = lmdb.open(
            f"data/{name}_test.lmdb",
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        lmdb_val = lmdb.open(
            f"data/{name}_val.lmdb",
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        return [lmdb_train, lmdb_test, lmdb_val]


class AsedbMaker(FileMaker):
    def make_file(
        self, 
        name: str
    ) -> List[Environment]:
        asedb_train = connect(
            f"data/{name}_train.db"
        )
        asedb_test = connect(
            f"data/{name}_test.db"
        )
        asedb_val = connect(
            f"data/{name}_val.db"
        )
        return [asedb_train, asedb_test, asedb_val]


class Tagger:
    def __init__(self, ads: List[str] = None):
        if ads is None:
            ads = ['H', 'C', 'O', 'He', 'N']
        self.ads = ads

    def _set_and_get_tags(self, atoms: ase.Atoms) -> atoms.tags:
        atoms.set_tags(np.ones(len(atoms))
        if atoms.constraints:
            for idx in atoms.constraints[0].index:
                atoms[idx].tag = 0
        for atom in atoms:
            if atom.symbol in self.ads:
                atom.tag = 2
        return atoms.get_tags()


class Converter(ABC):
    def __init__(self, tagger: Tagger = None) -> None:
        self.tagger = tagger if tagger is not None else Tagger()
        self.a2g = AtomsToGraphs(
                    max_neigh=50,
                    radius=6,
                    r_energy=True,
                    r_forces=True,
                    r_distances=False,
                    r_fixed=True
        )        

    @abstractmethod
    def convert(self, atoms: ase.Atoms) -> Union[torch_geometric.data.data.Data, ase.Atoms]: 
        pass


class LmdbConverter(Converter):
    def convert(self, atoms: ase.Atoms) -> torch_geometric.data.data.Data:
        data = self.a2g.convert(atoms)
        data.sid = torch.LongTensor([0])
        data.tags = torch.LongTensor(self.tagger._set_and_get_tags(atoms))
        return data


class AsedbConverter(Converter):
    def convert(self, atoms: ase.Atoms) -> ase.Atoms:
        atoms.set_tags(self.tagger._set_and_get_tags(atoms))
        calc = atoms.calc
        energy = calc.get_potential_energy()
        forces = (calc.get_forces()).tolist()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        return atoms


class DataSaver(ABC):
    @abstractmethod
    def save_data(self,
                  path: Union[Environment, ase.db.sqlite.SQLite3Database],
                  data: Union[torch_geometric.data.data.Data, ase.atoms],
                  fid: int
    ) -> None:
    pass

 
class LmdbSaver(DataSaver):
    def save_data(self,
                  path: Environmnet,
                  data: torch_geometric.data.data.Data,
                  fid: int
    ) -> None:
    txn = path.begin(write=True)
    txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()


class AsedbSaver(DataSaver):
    def save_data(self,
                  path: ase.db.sqlite.SQLite3Database,
                  data: ase.atoms,
                  fid: int
    ) -> None:
    path.write(data)


class Splitter:
    def __init__(
        self,
        converter: Converter,
        data_saver: DataSaver
    ) -> None:
        self.converter = converter
        self.data_saver = data_saver
    
    def split(
        self,
        environ_list: List[Environment],
        traj: List[ase.Atoms]
    ) -> None:

        fid
        for idx, atoms in tqdm(enumerate(traj), total = len(traj)):
            i = np.random.randint(100)
            data = self.converter.convert(atoms)
            
            if i<=80:
                self.data_saver.save_data(path=environ_list[0], data=data, fid=fid[0])
                fid[0] += 1

            if i<=90 and i>=81:
                self.data_saver.save_data(path=environ_list[1], data=data, fid=fid[1])
                fid[1] += 1

            if i>=91:
                self.data_saver.save_data(path=environ_list[2], data=data, fid=fid[2])
                fid[2] +=1

class YmlMaker:
    def show_and_get_model_name(self) -> None:
        s2ef = []
        for name in available_pretrained_models:
            if "S2EF" in name:
                s2ef.append(name)
        print("**Available S2EF models:")
        print("(IS2RE not supported)")
        print(*s2ef, sep='\n')
        print("**Type a model you want to Fine-tune")
        while True:
            self.model_name = input()
            if self.model_name in names:
                break
            else:
                print(f"Typed {self.model_name} is not in available models")
        self.checkpoint = model_name_to_local_file(self.model_name, local_cache='./')
        return self.checkpoint

    def make_yml(self, 
                 db: str, 
                 name: str
        ) -> pathlib.PosixPath:
        if db=='ase':
            yml = generate_yml_config(self.checkpoint, 'config.yml',
                            delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes',
                                    'dataset', 'test_dataset', 'val_dataset'],
                            update={'gpus': 1,
                                    'task.dataset': 'ase_db',
                                    'optim.eval_every': 10,
                                    'optim.max_epochs': 1,
                                    'optim.batch_size': 16,
                                    'logger': 'tensorboard',
                                    'dataset.train.src': f'./data/{name}_train.db',
                                    'dataset.train.format': 'ase_db',
                                    'dataset.train.a2g_args.r_energy': True,
                                    'dataset.train.a2g_args.r_forces': True,
                                    'dataset.val.src': f'./data/{name}_val.db',
                                    'dataset.val.format': 'ase_db',
                                    'dataset.val.a2g_args.r_energy': True,
                                    'dataset.val.a2g_args.r_forces': True,
                                })


        else:
            yml = generate_yml_config(self.checkpoint, 'config.yml',
                            delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes',
                                    'dataset', 'test_dataset', 'val_dataset'],
                            update={'gpus': 1,
                                    'task.dataset': 'lmdb',
                                    'optim.eval_every': 10,
                                    'optim.max_epochs': 1,
                                    'optim.batch_size': 16,
                                    'logger': 'tensorboard',
                                    'dataset.train.src': f'./data/{name}_train.lmdb',
                                    'dataset.train.format': 'lmdb',
                                    'dataset.train.a2g_args.r_energy': True,
                                    'dataset.train.a2g_args.r_forces': True,
                                    'dataset.val.src': f'./data/{name}_val.lmdb',
                                    'dataset.val.format': 'lmdb',
                                    'dataset.val.a2g_args.r_energy': True,
                                    'dataset.val.a2g_args.r_forces': True,

        return yml


class ConsoleTyper:
    def run_command(self, 
                    checkpoint,
                    yml: pathlib.PosixPath, 
                    name: str
        ) -> None:
        t0 = time.time()
        command = [
                'python',
                'main.py',
                '--mode', 'train',
                '--config-yml', yml,
                '--checkpoint', checkpoint,
                '--run-dir', 'fine-tuning',
                '--identifier', f'{name}',
                '--amp'
        ]
        with open('train.txt', 'w') as outfile:
            subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT)
        print(f'Elapsed time = {time.time() - t0:1.1f} seconds')
        with open('train.txt', 'r') as file:
            for line in file:
                if "checkpoint_dir:" in line:
                    cpline = line.strip()
                    break
        cpdir = cpline[0].split(':')[-1].strip()
        print(cpdir)


class Processor:
    def __init__(
        self,
        traj_loader: TrajLoader,
        file_maker: FileMaker,
        splitter: Splitter
        yml_maker: YmlMaker,
        console_typer: ConsoleTyper
    ) -> None:
        self.traj_loader = traj_loader
        self.file_maker = file_maker
        self.splitter = splitter
        self.yml_maker = yml_maker
        self.console_typer = console_typer

    def process(
        self,
        db: str,
        path: str,
        name: str,
    ) -> None:
        traj = self.traj_loader.load_file(path)
        environ_list = self.file_maker.make_file(name)
        self.splitter.split(traj=traj, environ_list=environ_list)
        checkpoint = yml_maker.show_and_get_model_name()
        yml = yml_maker.make_yml(db=db, name=name)
        console_typer.run_command(checkpoint=checkpoint, yml=yml, name=name)


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", dest="name", action="store")
parser.add_argument("-d", "--db", dest="db", action="store")
parser.add_argument("-p", "--path", dest="path", action="store")
args = parser.parse_args()

converter = Converter()
data_saver = DataSaver()
splitter = Splitter(
           converter=converter,
           data_saver=data_saver
           )

processor = Processor(
    traj_loader=TrajLoader(),
    file_maker=FileMaker(),
    splitter=splitter,
    yml_maker=YmlMaker(),
    console_typer=ConsoleTyper()
)

processor.process(db="lmdb", path="", name="ytk")










