# Manual for fine-tuning on Open Catalyst Project (OCP)
![Cent OS](https://img.shields.io/badge/cent%20os-002260?style=for-the-badge&logo=centos&logoColor=F0F0F0)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
---

#### Welcome.

#### You might be able to fine-tune OCP models with your dataset after reading this manual.

# Good luck.

---

## A tiny description before you start

Entire fine-tuning process can be devided into four parts.

* Set a conda environment
* (Fast fine-tuning)
* Prepare a dataset
* Make a configuration file
* Fine-tune pretrained model

(There is utilization part on bottom)

Easier than running a DFT calculation, so you don't need to worry.

I shall be with you in every step.

---

All code is executed on our server.

*-Linux CentOS release 7.9.2009*

Mac OS X compatible with ase.db

Window not checked


---
## Set a conda environment

The environment setup for Meta lab differs slightly from the OCP recommended setup.

I highly recommend creating a new environment using the **'env.gpu.yml'** file.

Follow me.
  

#### 1. Login to server

  ssh, MobaXTerm, or anything you use

  
#### 2. Download files through this github code

Type this on linux console

```shell

git clone https://github.com/GuGroup/Finetuner_OCP

```

  
#### 3. Navigate to the **Finetuner_OCP** folder

```shell

cd Finetuner_OCP

```

  
#### 4. Create a new environment with the following code

```shell

conda env create -f env.gpu.yml

```

(This process takes a few minutes.)

>  [!TIP]
> You can change the name of the environment via **'vi env.gpu.yml'**
  
  
#### 5. Activate git-chem

```shell

conda activate git-chem

```

  
#### 6. Download fairchem-core

```shell

pip install fairchem-core==1.0.0

```

Now you are ready for fine-tuning!

---
## (Fast fine-tuning)

**Only if you are busy, tired, sick, or sad...**

You can fine-tune with ytk's parameter pick.

Copy your ase file (OUTCAR, *.extxyz or *.traj) to Finetuner_OCP folder

```shell

cp OUTCAR /home/[your name]/Finetuner_OCP

```

Navigate to the **'Finetuner_OCP'** folder

```shell

cd ~/Finetuner_OCP

ls
```
```shell
OUTCAR README.md config.yml dbmaker.py dftocp.py env.gpu.yml finetuner.py main.py ymlmaker.py

```

Type this line on linux console

```shell

python finetuner.py --name ytk --path ./OUTCAR

```
>[!NOTE]
>If name of your ase file is not OUTCAR, so like ytk.extxyz or gu.traj, change line like this
>
>python finetuner.py --name ytk --path **./ytk.extxyz**

Then you can get fine-tuned Gemnet-OC S2EF model directly

It is a compressed version of upcoming 3 processes

For usage of fine-tuned model, scroll down to final part of this manual


---
## Prepare a dataset

Convert ase file (OUTCAR, *.extxyz, or *.traj) into LMDB or aseDB

I recommend using LMDB, faster data loading and lower memory pressure

>[!TIP]
>Assembling every OUTCAR into one *.extxyz is recommended
>
>Information about extxyz: https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#extxyz

#### 1. Get a path of ase file

Navigate to the folder that stores the ase file

```shell
pwd

```
Or just put the ase file into Finetuner_OCP folder

```shell
cp OUTCAR /home/[your name]/Finetuner_OCP

```

  
#### 2. run dbmaker.py

Navigate back to your Finetuner_OCP folder

```shell

cd home/[your_name]/Finetuner_OCP

```

Run this line in linux console

```shell
python dbmaker.py --name [name_that_you_want] --path [path_that_you_got]

```

For example, name: Ag111 and path: /home/ytk/Finetuner_OCP/OUTCAR

```shell
python dbmaker.py --name Ag111 --path /home/ytk/Finetuner_OCP/OUTCAR

```

> [!TIP]
> Name has no effect on processing. you can name it whatever you want!

Navigate to the 'data' folder

```shell
cd data

ls
```

You can see train, test, and val data splitted with [0.8, 0.1, 0.1] ratio

```shell
Ag111_train.lmdb Ag111_train.lmdb-lock 

Ag111_test.lmdb Ag111_test.lmdb-lock Ag111_ase_test.db 

Ag111_val.lmdb Ag111_val.lmdb-lock

```
> [!NOTE]
> .lmdb-lock file is calm and kind. Don't care about it.

You can check it with this code in python

```python
from fairchem.core.datasets import LmdbDataset
train = LmdbDataset({'src':'Ag111_train.lmdb'})
print(train[0])
```

You might see like this

```shell
Data(pos=[20, 3], cell=[1, 3, 3], atomic_numbers=[20], natoms=20, tags=[20], edge_index=[2, 815], cell_offsets=[815, 3], energy=-110.66750456, forces=[20, 3], fixed=[20], sid=[1], fid=[1])

```

<br/>
  
Or if you want it to be with asedb, just put --asedb to the end of the line

```shell
python dbmaker.py --name Ag111 --path /home/ytk/Finetuner_OCP/OUTCAR --asedb
```

Navigate to the 'data' folder

```shell
cd data

ls
```

You can see train, test, and val data splitted with [0.8, 0.1, 0.1] ratio

```shell
Ag111_ase_train.db Ag111_ase_test.db Ag111_ase_val.db 
```

You can check it with this line in linux console

```shell
ase db Ag111_ase_train.db
```

```shell
id|age|user|formula|calculator|  energy|natoms| fmax|pbc| volume|charge|    mass
 1|15m|ytk |Pt16H3N|unknown   |-110.668|    20|0.488|TTT|816.864| 0.000|3138.375
 2|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.506|TTT|816.864| 0.000|3138.375
 3|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.471|TTT|816.864| 0.000|3138.375
 4|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.505|TTT|816.864| 0.000|3138.375
 5|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.471|TTT|816.864| 0.000|3138.375
 6|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.466|TTT|816.864| 0.000|3138.375
 7|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.512|TTT|816.864| 0.000|3138.375
 8|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.470|TTT|816.864| 0.000|3138.375
 9|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.508|TTT|816.864| 0.000|3138.375
10|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.470|TTT|816.864| 0.000|3138.375
11|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.466|TTT|816.864| 0.000|3138.375
12|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.512|TTT|816.864| 0.000|3138.375
13|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.508|TTT|816.864| 0.000|3138.375
14|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.469|TTT|816.864| 0.000|3138.375
15|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.507|TTT|816.864| 0.000|3138.375
16|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.466|TTT|816.864| 0.000|3138.375
17|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.470|TTT|816.864| 0.000|3138.375
18|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.508|TTT|816.864| 0.000|3138.375
19|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.471|TTT|816.864| 0.000|3138.375
20|15m|ytk |Pt16H3N|unknown   |-110.666|    20|0.507|TTT|816.864| 0.000|3138.375
Rows: 56 (showing first 20)
```

---
## Make a configuration file

you need **'config.yml'** for fine-tuning

What is the role of **'config.yml'**?

**'config.yml'** passes information about the location of dataset, model specification, and hyperparameter to OCP model. 

If you are familiar with DFT, you can treat **'config.yml'** as INCAR

With **'ymlmaker.py'**, you can make the **'config.yml'** file

you can find in the Finetuner_OCP folder

```shell
ls
```
```shell
OUTCAR data dbmaker.py dftocp.py env.gpu.yml finetuner.py main.py ymlmaker.py
```


This is a brief description about each parameter in **'ymlmaker.py'**

Recommend check each line in this code for your fine-tuning

```shell
vi ymlmaker.py
```

But if you are busy, just check 'dataset.(train/test/val).src'

```python

from fairchem.core.common.tutorial_utils import generate_yml_config
from fairchem.core.models.model_registry import model_name_to_local_file

checkpoint = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='./')

yml = generate_yml_config(checkpoint, 'config.yml',
                delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes',
                        'optim.loss_force',
                        'optim.load_balancing',
                        'dataset', 'test_dataset', 'val_dataset'], # There are unused parameter in Gemnet-OC, so delete
                update={'gpus': 1, # number of GPU in fine-tuning
                        'task.dataset': 'lmdb' # if you train with asedb, write 'asedb'
                        'optim.eval_every': 10, # In each 10 steps, reschedule learning rate by the loss of validation data, This affect to process time strongly, so set deliberately
                        'optim.max_epochs': 1,
                        'optim.batch_size': 4, # A6000 has about 20-30 max batch_size handle power. 16 batch size recommended for larger datset
                        'logger': 'tensorboard',
                        # Train data
                        'dataset.train.src': './data/Ag111_train.lmdb', # location of data should be in here.
                        'dataset.train.format': 'lmdb', # if you train with asedb, write 'asedb'
                        'dataset.train.a2g_args.r_energy': True,
                        'dataset.train.a2g_args.r_forces': True,
                        # Test data - prediction only so no regression
                        'dataset.test.src': './data/Ag111_test.lmdb', # same procedure with test and val
                        'dataset.test.format': 'lmdb',
                        'dataset.test.a2g_args.r_energy': False,
                        'dataset.test.a2g_args.r_forces': False,
                        # val data
                        'dataset.val.src': './data/Ag111_val.lmdb',
                        'dataset.val.format': 'lmdb',
                        'dataset.val.a2g_args.r_energy': True,
                        'dataset.val.a2g_args.r_forces': True,
                        })

```

>[!WARNING]
>You have to change 'dataset.(train/test/val).src': **'./data/Ag111_(train/test/val).lmdb'** into the name you written in 'Prepare a dataset' part.

After you run this code, you can find **'config.yml'** in Finetuner_OCP folder.

```shell

python ymlmaker.py
ls

```

```shell
OUTCAR config.yml data dbmaker.py dftocp.py env.gpu.yml gnoc_oc22_oc20_all_s2ef.pt finetuner.py main.py ymlmaker.py
```

Now we are very close to fine-tuning.


## Fine-tune pretrained model

Type this line on your linux console

```shell

python main.py --mode train --config-yml config.yml --run-dir fine-tuning --checkpoint ./gnoc_oc22_oc20_all_s2ef.pt --amp > train.txt 2>&1

```
Fine-tuning is on process.

(If you want to change parameter or pretrained model, just change **'config.yml'** file and --checkpoint )

Now we can see how fine-tuning is going with **'train.txt'**

type this in linux console

```shell
tail -f -n 4 train.txt
```

And watch!

```shell
2024-08-08 13:41:34 (INFO): energy_forces_within_threshold: 0.0000, energy_mae: 24.6883, forcesx_mae: 0.0952, forcesy_mae: 0.1123, forcesz_mae: 0.4739, forces_mae: 0.2271, forces_cosine_similarity: -0.1308, forces_magnitude_error: 0.3816, loss: 25.2326, epoch: 0.5882
```

```shell
2024-08-08 13:41:35 (INFO): Writing results to ./results/2024-08-08-13-41-36/ocp_predictions.npz
2024-08-08 13:41:39 (INFO): Total time taken: 16.31741714477539
```

After you see 'Total time taken' its all over.

Thanks.

---
## Utilization

How to play with fine-tuned model?

Open ase.Atoms in python and put your fine-tuned model as calculator!

```python
from farichem.core.common.relaxation.ase_utils import OCPCalculator
from ase.io import read

# Due to timestamping on folder name (2024-08-...), You need to change path of the checkpoint with your situation
checkpoint = './fine-tuning/checkpoints/2024-08-09-10-59-28-ytk/best_checkpoint.pt' 

# If you run this on a GPU node, cpu=False
newcalc = OCPCalculator(checkpoint_path=checkpoint, cpu=True)
traj = read('./OUTCAR', index=':')

atoms = traj[0]
atoms.calc = newcalc
print(atoms.get_potential_energy())
```
```shell
-118.8056945800
```

I uploaded **'dftocp.py'** in Finetuner_OCP folder.

You can get .json file of potential energy in dft and ocp with it.

I recommend playing with it more..


---

# This is the end...





