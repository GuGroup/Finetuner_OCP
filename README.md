# Manual for fine-tuning on Open Catalyst Project (OCP)
![Cent OS](https://img.shields.io/badge/cent%20os-002260?style=for-the-badge&logo=centos&logoColor=F0F0F0)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
---
**Based on fair-chem and its tutorial website (before Open Catalyst Project)**

https://fair-chem.github.io/index.html

---

#### Welcome.

#### You might be able to fine-tune OCP models with your dataset after reading this manual.

# Good luck.

---

## A tiny description before you start

Entire fine-tuning process can be devided into two parts.

* Set a conda environment
* Fine-tuning

(There is an utilization part on bottom)

Much easier than running a DFT calculation, so you don't need to worry.

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
## Fine-tuning

**One Step Fine-tuning**

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
OUTCAR README.md dftocp.py env.gpu.yml finetuner.py main.py

```

Type this line on linux console

```shell

python finetuner.py --name [name_that_you_want] --path ./OUTCAR --db lmdb

```
>[!NOTE]
>If name of your ase file is not OUTCAR, so like ytk.extxyz or gu.traj, change line like this
>
>python finetuner.py --name ytk --path **./ytk.extxyz** --db lmdb


First, This will be displayed on your console

```shell
Now making DB files in data folder...
100%|███████████████████████████████████████████████████████████████████████████████████| 73/73 [00:02<00:00, 30.13it/s]
```

It converts your ase file into lmdb (or asedb) and split into train, test, and val as 80:10:10 ratio, in **data** folder

Then you can see these lines

```shell
**Available S2EF models:
(IS2RE not supported)
CGCNN-S2EF-OC20-200k
CGCNN-S2EF-OC20-2M
CGCNN-S2EF-OC20-20M
CGCNN-S2EF-OC20-All
DimeNet-S2EF-OC20-200k
DimeNet-S2EF-OC20-2M
SchNet-S2EF-OC20-200k
SchNet-S2EF-OC20-2M
SchNet-S2EF-OC20-20M
SchNet-S2EF-OC20-All
DimeNet++-S2EF-OC20-200k
DimeNet++-S2EF-OC20-2M
DimeNet++-S2EF-OC20-20M
DimeNet++-S2EF-OC20-All
SpinConv-S2EF-OC20-2M
SpinConv-S2EF-OC20-All
GemNet-dT-S2EF-OC20-2M
GemNet-dT-S2EF-OC20-All
PaiNN-S2EF-OC20-All
GemNet-OC-S2EF-OC20-2M
GemNet-OC-S2EF-OC20-All
GemNet-OC-S2EF-OC20-All+MD
GemNet-OC-Large-S2EF-OC20-All+MD
SCN-S2EF-OC20-2M
SCN-t4-b2-S2EF-OC20-2M
SCN-S2EF-OC20-All+MD
eSCN-L4-M2-Lay12-S2EF-OC20-2M
eSCN-L6-M2-Lay12-S2EF-OC20-2M
eSCN-L6-M2-Lay12-S2EF-OC20-All+MD
eSCN-L6-M3-Lay20-S2EF-OC20-All+MD
EquiformerV2-83M-S2EF-OC20-2M
EquiformerV2-31M-S2EF-OC20-All+MD
EquiformerV2-153M-S2EF-OC20-All+MD
SchNet-S2EF-force-only-OC20-All
DimeNet++-Large-S2EF-force-only-OC20-All
DimeNet++-S2EF-force-only-OC20-20M+Rattled
DimeNet++-S2EF-force-only-OC20-20M+MD
GemNet-dT-S2EFS-OC22
GemNet-OC-S2EFS-OC22
GemNet-OC-S2EFS-OC20+OC22
GemNet-OC-S2EFS-nsn-OC20+OC22
GemNet-OC-S2EFS-OC20->OC22
EquiformerV2-lE4-lF100-S2EFS-OC22
SchNet-S2EF-ODAC
DimeNet++-S2EF-ODAC
PaiNN-S2EF-ODAC
GemNet-OC-S2EF-ODAC
eSCN-S2EF-ODAC
EquiformerV2-S2EF-ODAC
EquiformerV2-Large-S2EF-ODAC
**Type a model you want to Fine-tune
```

These are available pre-trained model names

In example of using GemNet-OC-S2EFS-OC20+OC22, 

```shell

GemNet-OC-S2EFS-OC20+OC22
```

Pre-trained model is downloaded in your folder, as [pre-trained_model_name].pt

You might see some warnings.

```shell
WARNING:root:Detected old config, converting to new format. Consider updating to avoid potential incompatibilities.
```

```shell
WARNING:root:Unrecognized arguments: ['symmetric_edge_symmetrization']
```

```shell
WARNING:root:Using `weight_decay` from `optim` instead of `optim.optimizer_params`.Please update your config to use `optim.optimizer_params.weight_decay`.`optim.weight_decay` will soon be deprecated.
```

```shell
WARNING:root:No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run
```

Finally, fine-tuning is in progress

```shell
**Fine-tuning is in progress, check "train.txt"
```

In the same folder, **train.txt** is created

You can see how fine-tuning is going with this line.

```shell
tail -f -n 4 train.txt
```


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





