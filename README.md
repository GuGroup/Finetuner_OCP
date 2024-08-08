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
* Prepare a dataset
* Make a configuration file
* Fine-tune pretrained model 

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

How to make **'config.yml'**

What is the role of **'config.yml'**?

**'config.yml'** passes information about the location of dataset, model specification, and hyperparameter to OCP model. 

If you are familiar with DFT, you can treat **'config.yml'** as INCAR

With **'ymlmaker.py'**, you can make the **'config.yml'** file

you can find in the Finetuner_OCP folder

```shell
ls
vi ymlmaker.py
```

This is a brief description about each parameter in **'ymlmaker.py'**

I think it is better to check this code for your fine-tuning

But if you are busy, just check 'dataset.(train/test/val).src'

```python

from fairchem.core.common.tutorial_utils import generate_yml_config
# I cannot directly download pre-trained model in server, because of an error with HTTP
# So pre-trained model file is saved in the Finetuner_OCP folder

checkpoint = './gnoc_oc22_oc20_all_s2ef.pt' # Gemnet-OC_S2EF_OC20+OC22

yml = generate_yml_config(checkpoint, 'config.yml',
                delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes',
                        'optim.loss_force',
                        'optim.load_balancing',
                        'dataset', 'test_dataset', 'val_dataset'], # There are unused parameter in Gemnet-OC, so delete
                update={'gpus': 1, # number of GPU in fine-tuning
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





