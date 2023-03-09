#!/bin/bash
# A script to install everything needed to run MOSES in a new environment
conda create -n latent_gan_env python=3.6
eval "$(conda shell.bash hook)"
conda activate latent_gan_env
git clone https://github.com/pcko1/Deep-Drug-Coder.git --branch moses
git clone https://github.com/EBjerrum/molvecgen.git
conda env update --file environment.yml
mv Deep-Drug-Coder/ddc_pub/ .
mv molvecgen/molvecgen tmp/
rm -r -f molvecgen/
mv tmp/ molvecgen/
