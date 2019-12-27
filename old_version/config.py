# -*- coding: utf-8 -*-
import os

env_loc = input('please, input the path of Anaconda environment')
with os.popen('pip list') as reader:
    packages = reader.read()
if "rdkit" in packages:
    print('rdkit packages already exist')
    exit()
with os.popen(f'conda activate {env_loc}\nconda install -c rdkit rdkit') as reader:
    print(reader.read())
