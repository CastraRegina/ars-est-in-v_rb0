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
  - DocStrings according to [PEP-0257 – Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)  
  - DocStrings in style of [pandas docstring guide](https://pandas.pydata.org/docs/development/contributing_docstring.html)
- Typing  
  - Use type annotations to define static types and use checking tool(s)  
  - Typing according to [PEP-0484 – Type Hints](https://www.python.org/dev/peps/pep-0484/)
  - See [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)  
  - Use [http://mypy-lang.org/](http://mypy-lang.org/) for checking
- Linting
  - See [Python Code Quality: Tools & Best Practices](https://realpython.com/python-code-quality/)
  - User pylint and Pylama (= contains others)
- Automated Testing  
  - See [The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/tests/)
  - See [4 Techniques for Testing Python Command-Line (CLI) Apps](https://realpython.com/python-cli-testing/)
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
  - folders `mkdir -p {bin,data,docs,examples,src,src/examples,tests}`:   
```
        bin
        [configs]
        data
        [dist]
        docs
        [examples]
        [logs]
        src
        src/examples
        tests
        [tools]
        (venv)
```
- Naming Conventions of file and folders
  - Naming according to
    [PEP-0008 - Style Guide for Python Code - Prescriptive: Naming Conventions](https://peps.python.org/pep-0008/#prescriptive-naming-conventions)
    - module = `.py`-file
    - package = folder with `__init__.py`-file
- Parsing arguments from command line  
  - Maybe use [argparser](https://docs.python.org/3/library/argparse.html)
  - See [PEP-0389 – argparse - New Command Line Parsing](https://www.python.org/dev/peps/pep-0389/)
- Versioning
  - Use `MAJOR.MINOR.PATCH`, see [semver.org](https://semver.org)
  - See [Git-Basics-Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- Good git commit message
  - Questions
    - Why have I made these changes? (Not what has been made)
    - What effect have my changes made?
    - Why was the change needed?
    - What are the changes in reference to?
  - Remarks
    - First commit line is the most important
    - Use present tense
    - Be specific
- Scripts should be...
  - [idempotent](https://en.wikipedia.org/wiki/Idempotence):
    Regardless of how many times the script is again executed with the same input, the output must always remain the same
    

## Git
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
  - `git log --all --graph --decorate --oneline`   Nice history view
- Git without server
  - See [Bare Repositories in Git](https://www.geeksforgeeks.org/bare-repositories-in-git/)
  - See [Git workflow without a server](https://stackoverflow.com/questions/5947064/git-workflow-without-a-server) 
  - Server-like repository:  
    `mkdir foo.git ; cd foo.git`  
    `git init --bare`
  - Local repository:  
    `mkdir foo ; cd foo`  
    `git init`  
    ... add your files ...  
    `git add .`  
    `git commit -m "Initial commit"`  
    `git remote add origin /path/to/server/like/repository/foo.git`  
    `git push origin main`
  - Every further local repository like standard clone:  
    `git clone /path/to/server/like/repository/foo.git`
    
    
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
  
## Install Visual Studio Code or VSCodium
- Install [Visual Studio Code](https://code.visualstudio.com/) or [VSCodium](https://vscodium.com)
  - Download: [code.visualstudio.com/download](https://code.visualstudio.com/download)
    or [github.com/VSCodium](https://github.com/VSCodium/vscodium/releases)
  - For VS Code...
    - Extract `.tar.gz`-file into folder `/opt/VSCode`
    - Start VS Code: `/opt/VSCode/code`
  - VSCodium is available in [Snap Store](https://snapcraft.io/) as [Codium](https://snapcraft.io/codium) ...
    - Install: `snap install codium --classic`
- Install extensions:
  - Python extension for Visual Studio Code (ms-python.python)
  - Python indent (KevinRose.vsc-python-indent)
  - autoDocstring - Python Docstring Generator (njpwerner.autodocstring)
  - Pylance (ms-python.vscode-pylance) (seems to be already installed by ms-python.python)
  - Pylint (ms-python.pylint)
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
- Setting for python `src`-folder
  - See [Setting Python source folders in Visual Studio Code](https://binx.io/2020/03/05/setting-python-source-folders-vscode/)
  - Modify `settings.json`
  ```
        {
          "terminal.integrated.env.osx": {
            "PYTHONPATH": "${workspaceFolder}/src",
          },
          "terminal.integrated.env.linux": {
            "PYTHONPATH": "${workspaceFolder}/src",
          },
          "terminal.integrated.env.windows": {
            "PYTHONPATH": "${workspaceFolder}/src",
          },
          "python.envFile": "${workspaceFolder}/.env"
        }
  ```
  - Modify `.env` : `PYTHONPATH=./src`
  - or:  
  ```
        {
          "terminal.integrated.env.osx": {
            "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/src",
          },
          "terminal.integrated.env.linux": {
           "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/src",
          },
          "terminal.integrated.env.windows": {
            "PYTHONPATH": "${env:PYTHONPATH};${workspaceFolder}/src",
          }
        }
  ```
  - ... and: `PYTHONPATH=${PYTHONPATH}:./src`
- Helpful Shortcuts
  - `Ctrl+Shift+P` to open the Command Palette

## Folder and Git setup
- Git ssh setup
  - Read [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
  - Generate ssh-key on your local machine:  
    `cd $HOME/.ssh`  
    `ssh-keygen -o -t ed25519 -C "git@github.com" -f id_ed25519_github`  
    `ssh-add id_ed2551_github`  
    `ssh-add -L` to check if the key was added
  - Copy public-key `cat id_rsa_github` to github->Settings->SSH and GPG keys->SSH keys->New SSH key...  
  - Test ssh connection:  
    `ssh -T git@github.com` should show: `Hi ...! You've successfully authenticated...`
- Clone github project `ars-est-in-verbo`:  
  `cd $HOME`  
  ~~`git clone https://github.com/CastraRegina/ars-est-in-verbo`~~  
  `git clone git@github.com:CastraRegina/ars-est-in-verbo.git`
- Create folders:  
  `cd ars-est-in-verbo`  
  `mkdir -p {bin,data,docs,examples,src,src/examples,tests}`
- Set username for this repository:  
  `git config user.name "Regina Castra"`  
  `git config user.name` to check the settings
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
    Nude front/back, Face(s),
    Dragonfly, Butterfly, Dear, Boar, Donkey half view,
    lucky pig, unicorn, dragon,
    shamrock/four-leaf clover, heart, globe/world-maps,
    Yin and Yang, zodiac signs

### Fonts
- Variable Fonts
  - Grandstander --> see [fonts.google.com](https://fonts.google.com/specimen/Grandstander)
  - Recursive --> see [fonts.google.com](https://fonts.google.com/specimen/Recursive)
  - Roboto Flex --> see [fonts.google.com](https://fonts.google.com/specimen/Roboto+Flex)

