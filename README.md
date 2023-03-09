# latent-gan
This repository contains code for modified version of latent-gan https://github.com/Dierme/latent-gan

## Install

Python 3.7+ is required. We will assume you have `conda` installed on your system.
First, use `conda` to create a virtual environment.

```bash
conda env create --file environment.yml
```

Activates created environment and run below line.
```bash
bash install-dependencies.sh
pip install ipykernel
python -m ipykernel install --user --name latent_gan_env --display-name latent_gan
```

## Train

```bash
Arguments:
-sf input_smiles.smi
-st /.. /../latent-gan/saving_directory 
--n-epochs number_of_epochs 
--sample-n number_of_outputs
```
