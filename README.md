# visuWordArt
## Purpose
- Collecting some ideas for an art project written in Python
- Make some notes of the step-by-step process


# Diary
## Day-0001 - 2022-12-25
### General installation of Python
- Install latest Python version: 
  - Create folder /opt/python
  - Download version `3.11.1` from [python.org/downloads](https://www.python.org/downloads/)
  - `wget https://www.python.org/ftp/python/3.11.1/Python-3.11.1.tgz`
  - Extract in folder `/opt/python` : `tar xzf Python-3.11.1.tgz`
  - Change into `/opt/python/Python-3.11.1` and execute `./configure --prefix=/opt/python --enable-optimizations`
  - Execute `make -s -j4`
  - Execute `make test`
  - Execute `make install`
  - Update `$HOME/.bashrc`  
    `export PATH=/opt/python/Python-3.11.1:/opt/python/bin:$PATH`  
    `export PYTHONPATH=opt/python/Python-3.11.1`
- Upgrade `pip`: `/opt/python/bin/pip3 install --upgrade pip`  
    Check pip-version: `python3 -m pip --version`
    
### General project setup
- Setup a virtual environment for the project
  - Go to your home directory and create folder `cd $HOME ; mkdir visuWordArt`
  - Change into project folder `cd visuWordArt`
  - Setup the virtual environment `/opt/python/bin/python3 -m venv venv`
- Switch on virtual environment: `. venv/bin/activate`
- Upgrade/update modules `pip`, `setuptools`, `wheels`:  
  `python3 -m pip install --upgrade pip setuptools wheel`
  
### Specific project setup
Maybe check later if some of these packages are really needed...
- Install SW packages on operating system (don't know if they are really needed):  
  `sudo apt-get install python-dev python-pip ffmpeg libffi-dev`  
  `sudo apt-get install libxml2-dev libxslt-dev`  
  `sudo apt-get install libcairo2`
- Install python modules:  
  `python3 -m pip install gizeh svgutils svgwrite`  
  `python3 -m pip install pycairo` (does not install properly)
  
### Install Visual Studio Code 
- Download VS Code: [code.visualstudio.com/download](https://code.visualstudio.com/download)
- Extract `.tar.gz`-file into folder `/opt/VSCode`
- Install plugins: **TODO** Python, Python Indent, Python Snippets
- Setup: **TODO**
