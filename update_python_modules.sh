#!/bin/bash

cd "/home/data/git/ars-est-in-v_rb0"
. venv/bin/activate

xhost local:

which python
python --version

which python3
python3 --version


#exit
echo "---------------------------------"
# python3 -m pip install --upgrade pip setuptools wheel ; python3 -m pip freeze | cut -d'=' -f1 | xargs -n1 python3 -m pip install -U ; python3 -m pip freeze > requirements.txt
echo "---------------------------------"
python3 -m pip install --upgrade pip setuptools wheel
echo "---------------------------------"
python3 -m pip freeze | cut -d'=' -f1 | xargs -n1 python3 -m pip install -U
echo "---------------------------------"
python3 -m pip freeze > requirements.txt




# howto setup venv and install python modules:
# /usr/bin/python3 -m venv venv
# . venv/bin/activate
# python3 -m pip install --upgrade pip setuptools wheel
# python3 -m pip install -r requirements.txt
# python3 -m pip install --upgrade pip setuptools wheel ; python3 -m pip freeze | cut -d'=' -f1 | xargs -n1 python3 -m pip install -U ; python3 -m pip freeze > requirements.txt


