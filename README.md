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
* Preparing a dataset
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

> [!TIP]
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

Convert ase (OUTCAR or *.extxyz or *.traj) to aseDB or LMDB

I recommend using LMDB, faster data loading and lower memory pressure









