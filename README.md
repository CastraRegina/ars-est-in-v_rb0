# ars-est-in-verbo
## Purpose
- Scientific research journey into the art of words by means of coding
- Making notes of the step-by-step process
- Collecting ideas


## Rules / Guidelines / Best Practices
- Code Style  
  - Code style according to [PEP-0008 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)  
  - See also [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- DocStrings  
  - DocStrings according to [PEP 257 – Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)  
  - DocStrings in style of [pandas docstring guide](https://pandas.pydata.org/docs/development/contributing_docstring.html)
- Typing  
  - Use type annotations to define static types and use checking tool(s)  
  - Typing according to [PEP 484 – Type Hints](https://www.python.org/dev/peps/pep-0484/)
  - See [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)  
  - Use [http://mypy-lang.org/](http://mypy-lang.org/) for checking
- Automated Testing  
  - See [The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/tests/)  
  - Documentation of [unittest](https://docs.python.org/3/library/unittest.html)  
  - Documentation of [doctest](https://docs.python.org/3/library/doctest.html)
- Logging  
  - See [Logging-basic-tutorial](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial) 
  - Maybe use `from loguru import logger`
- Config
  - Prefer YAML over JSON for standard config files
  - Separate public and private data, e.g. credentials ...  
    See [How to hide sensitive credentials using Python](https://www.geeksforgeeks.org/how-to-hide-sensitive-credentials-using-python/)  
  - `.env`-file together with `python-dotenv`
- Folder Structure  
  - See [Sampleproject by the "Python Packaging Authority"](https://github.com/pypa/sampleproject)   
  - folders: `bin`, `data`, `src`, `tests`, ... `build`, `dist`, ...   
- Parsing arguments from command line  
  - Maybe use [argparser](https://docs.python.org/3/library/argparse.html)
  - See [PEP 389 – argparse - New Command Line Parsing](https://www.python.org/dev/peps/pep-0389/)
- Versioning
  - Use `MAJOR.MINOR.PATCH`, see [semver.org](https://semver.org)
  - See [Git-Basics-Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging)


## Git examples
- Repository --> local
  - Clone the repository (i.e. creates the repository-folder in your local folder):  
    `git clone https://<your-link-to-the-repository>`  
  - Fetch file(s):  
    `git fetch`
  - Pull file(s) = fetch and merge file(s):  
    `git pull`
  - Checkout branch:  
    `git checkout <branch>`
  - Checkout certain commitID:  
    `git checkout <commitID>`
- Local --> repository
  - Add file(s):  
    `git add <files>`
  - Commit file(s):  
    `git commit -m "...Commit message..."`
  - Push file(s) to (remote) repository:  
    `git push`
  - Create branch:  
    `git branch <branch>`
  - Push a newly created branch (including its changes):  
    `git push --set-upstream origin <branch>`
  - Commit file(s) using the last commit (= do a correction of the last commit):  
    `git commit --amend -m "...Commit message..."`
  - Tagging:  
    `git tag -a "v1.3.0-beta" -m "version v1.3.0-beta"`  
    `git push origin --tags`
- Further examples:
  - `git status` List which files are staged, unstaged, and untracked
  - `git log`    Display the entire commit history using the default format
  - `git diff`   Show unstaged changes between your index and working directory
  - `gitk`       a git GUI


---

# Step-by-step process
## General installation of Python
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
    
## General project setup
- Setup a virtual environment for the project
  - Go to your home directory and create folder `cd $HOME ; mkdir ars-est-in-verbo`
  - Change into project folder `cd ars-est-in-verbo`
  - Setup the virtual environment `/opt/python/bin/python3 -m venv venv`
- Switch on virtual environment: `. venv/bin/activate`
- Upgrade/update modules `pip`, `setuptools`, `wheels`:  
  `python3 -m pip install --upgrade pip setuptools wheel`
  
## Specific project setup
Maybe check later if some of these packages are really needed...
- Install SW packages on operating system (don't know if they are really needed):  
  `sudo apt-get install python-dev python-pip ffmpeg libffi-dev`  
  `sudo apt-get install libxml2-dev libxslt-dev`  
  `sudo apt-get install libcairo2`
- Install python modules:  
  `python3 -m pip install gizeh svgutils svgwrite`  
  `python3 -m pip install pycairo` (does not install properly)
- Remark: Later use `requirements.txt` to install needed PyPI packages:  
  `python3 -m pip install -r requirements.txt`  
  `python3 -m pip freeze > requirements.txt`
  
## Install Visual Studio Code 
- Download VS Code: [code.visualstudio.com/download](https://code.visualstudio.com/download)
- Extract `.tar.gz`-file into folder `/opt/VSCode`
- Start VS Code: `/opt/VSCode/code`
- Install extensions:
  - Python extension for Visual Studio Code (ms-python.python)
  - Python indent (KevinRose.vsc-python-indent)
  - autoDocstring - Python Docstring Generator (njpwerner.autodocstring)
  - Pylance (ms-python.vscode-pylance) (seems to be already installed by ms-python.python)
  - GitLens - Git supercharged (eamodio.gitlens)
  - Markdown Preview Mermaid Support (bierner.markdown-mermaid) for diagrams and flowcharts
- Extensions to check later:
  - Code Runner (formulahendry.code-runner)
  - Python Extension Pack (donjayamanne.python-extension-pack)
  - Tabnine AI Autocomplete (TabNine.tabnine-vscode)
  - GitHub Copilot (GitHub.copilot) for autocompletion
  - python snippets (frhtylcn.pythonsnippets)
  - AREPL for python (almenon.arepl)
  - Vim (vscodevim.vim)
- Setup / modify settings:
  - Python Analysis Type Checking Mode
  - Editor Format On Save
  - Python Formatting Provider
  - Python Linting Pylint Enabled

## Git setup
- Clone github project `ars-est-in-verbo`:  
  `cd $HOME`  
  `git clone https://github.com/CastraRegina/ars-est-in-verbo`


---

# Next steps / check in future / ToDos / Reminders
- How to handle examples / spikes / testing / unit-test 
- Pipeline architecture
- Check ArjanCodes YT Channel videos

### Ideas
- Palindrome --> see [Wikipedia](https://en.wikipedia.org/wiki/Palindrome)  
  Example: "SATOR AREPO TENET OPERA ROTAS"
- Ambigram --> see [Wikipedia](https://en.wikipedia.org/wiki/Ambigram)
- Pictures of...  
    Big Ben, Silhouette of town, towers of town,
    Black/white of portraits, Banksy,
    Einstein, Mona Lisa, The Creation of Adam,
    Nude front/back, Face(s)
    Dragonfly, Dear, Boar, Donkey half view,
    lucky pig, shamrock/four-leaf clover, heart, Yin and Yang, globe/world-maps

### Fonts
- Variable Fonts
  - Grandstander --> see [fonts.google.com](https://fonts.google.com/specimen/Grandstander)
  - Recursive --> see [fonts.google.com](https://fonts.google.com/specimen/Recursive)
  - Roboto Flex --> see [fonts.google.com](https://fonts.google.com/specimen/Roboto+Flex)

