from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from tqdm import tqdm
from ase.db import connect
import json
import matplotlib.pyplot as plt
import os

# Due to timestamping on folder name (2024-08-...), You need to change path of the checkpoint with your situation
checkpoint = './fine-tuning/checkpoints/2024-08-09-10-59-28-hi/best_checkpoint.pt'
newcalc = OCPCalculator(checkpoint_path=checkpoint, cpu=True)

# You need to change name of the ase_test.db to your situation
db = connect('./data/hi_ase_test.db')
atoms_data = []
for row in db.select():
    atoms_data.append([row.toatoms(), row.get('energy', None)])

dfts = []
total_dfts = []
ocps = []
total_ocps = []


for data in tqdm(atoms_data, total=len(atoms_data)):
    dft = []
    ocp = []
    atoms = data[0]
    atoms.calc=newcalc
    pe = atoms.get_potential_energy()/len(atoms)
    de = data[1]/len(atoms)
    dfts.append(de)
    ocps.append(pe)
    total_dfts.append(de*len(atoms))
    total_ocps.append(pe*len(atoms))

plt.scatter(dfts, ocps, alpha=0.5)
plt.xlabel('DFT predicted energy (eV/atom)')
plt.ylabel('OCP predicted energy (eV/atom)')

# dfts.json or ocps_finetuned.json: eV/atom (energy per atom)
# total_dfts.json or total_ocps_finetuned.json: eV (total energy of slab & adsorbate)
os.makedirs('energys', exist_ok=True)
json.dump(ocps, open('./energys/ocps_finetuned.json','w'))
json.dump(dfts, open('./energys/dfts.json','w'))
json.dump(total_ocps, open('./energys/total_ocps_finetuned.json','w'))
json.dump(total_dfts, open('./energys/total_dfts.json', 'w'))
plt.savefig('./energys/comparison.png')






