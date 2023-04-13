# ars-est-in-verbo

## Content
[Rules / Guidelines / Best Practices](#rules--guidelines--best-practices)  
[Git](#git)  
[Scripts](#scripts)  
[Fonts](#fonts)  
[Ideas](#ideas)  
[Know How](#know-how)


## Purpose
- Scientific research journey into the art of words by means of coding
- Making notes of the step-by-step process
- Collecting ideas


## Rules / Guidelines / Best Practices
- General
  - See Pranav Kapur's [Structuring Python Code - Best practices from over 10 blogs](https://medium.com/analytics-vidhya/structuring-python-code-best-practices-from-over-10-blogs-2e33cbb83c49)
- Code Style  
  - Code style according to [PEP-0008 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)  
  - See also [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- DocStrings  
  - DocStrings according to [PEP-0257 - Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)  
  - DocStrings in style of [pandas docstring guide](https://pandas.pydata.org/docs/development/contributing_docstring.html)
- Typing  
  - Use type annotations to define static types and use checking tool(s)  
  - Typing according to [PEP-0484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
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
- Use [Context manager](https://book.pythontips.com/en/latest/context_managers.html)
  for handling file access:
  ```
    with open('filename.txt', 'w') as file:
        file.write('Hello World!')
  ```
- Folder Structure
  - See [Python Packages - Package structure](https://py-pkgs.org/04-package-structure.html#package-structure)
  - See Jason C. McDonald's [Dead Simple Python: Project Structure and Imports](https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6)
  - See [Sampleproject by the "Python Packaging Authority"](https://github.com/pypa/sampleproject)   
  - folders `mkdir -p {bin,data,docs,examples,src,src/examples,tests}`:   
    ```
        bin
        [configs]
        data
        [data/input]
        [data/output]
        [dist]
        docs
        [examples]
        [fonts]
        [fonts/zips]
        [logs]
        src
        src/examples
        [src/subpackage]
        tests
        [tools]
        (venv)
    ```
- Naming Conventions of file and folders
  - Naming according to
    [PEP-0008 - Style Guide for Python Code - Prescriptive: Naming Conventions](https://peps.python.org/pep-0008/#prescriptive-naming-conventions)
    - **module = `.py`-file**:  
      files should have short, all-lowercase names, underscores `_` are allowed
    - **package = folder** with `__init__.py`-file:  
      folders should have short, all-lowercase names, use of underscores `_` is discouraged
    - Linters (install using `pip`): `pycodestyle`, `flake8`
    - Autoformatters (install using `pip`): `black` -> usage: `black --line-length=79 code.py`
- Parsing arguments from command line  
  - Maybe use [argparser](https://docs.python.org/3/library/argparse.html)
  - See [PEP-0389 - argparse - New Command Line Parsing](https://www.python.org/dev/peps/pep-0389/)
- Versioning
  - Use `MAJOR.MINOR.PATCH`, see [semver.org](https://semver.org)
  - See [Git-Basics-Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging) and
    illustrated example: [medium.com: Major.Minor.Patch](https://medium.com/fiverr-engineering/major-minor-patch-a5298e2e1798)
- Branches
  - *Main branch* for latest stable (productive) version
  - Use *feature branches* for developing and testing new features
- Good git commit message
  - Questions
    - Why have I made these changes? (Not what has been made)
    - What effects do my changes have?
    - What is the reason for these changes?
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
  - Replace local changes (in case of modified files)
    - Either replace only the changes in your working tree with the last content in HEAD.  
      Changes already added to the index (stage), as well as new files, will be kept:  
        `git checkout -- <filename>`
    - or drop all your local changes and commits, fetch the latest history from the server and point your local main branch at it:  
        `git fetch origin`
        `git reset --hard origin/main`
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
- Create an executable (for Windows)
  - Create folders `build` and `dist`
  - `pyinstaller --distpath dist --workpath build --onefile --clean <aScript.py>`
- Further examples
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
- Git submodules and subtrees
  - See [Managing Git Projects: Git Subtree vs. Submodule](https://gitprotect.io/blog/managing-git-projects-git-subtree-vs-submodule/)
  - From [Managing Git projects with submodules and subtrees](https://opensource.com/article/20/5/git-submodules-subtrees):  
    If there is a *sub*-repository you want to push code back to, use Git *submodule* as it is easier to push.  
    If you want to import code you are unlikely to push (e.g. third-party code), use Git *subtree* as it is easier to pull.
  - Practical advice, see [How do I manage the file structure of GIT sub modules/trees?](https://stackoverflow.com/questions/68950221/how-do-i-manage-the-file-structure-of-git-sub-modules-trees)
    
## Scripts
- Convert utf-8 file (`README.md`) to ascii:  
  `cat README.md | iconv -f utf-8 -t ascii//TRANSLIT > output.txt`
- Remove empty / blank lines from text file `file.txt`:  
  `sed -i '/^\s*$/d' file.txt`
- Remove `\r` at end of each line:  
  `sed -i 's/\r$//g' file.txt`
- Replace each newline by a space:  
  `sed -i ':a;N;$!ba;s/\n/ /g' file.txt`
- Replace all multiple spaces and tabs by one single space:  
  `sed -i 's/\s\+/ /g' file.txt`  
  or replace all kind of "spaces" by one single space:  
  `sed -i 's/[ \t\r\n\v\f]\+/ /g' file.txt`
---

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
- Activate virtual environment: `. venv/bin/activate`
- Upgrade/update modules `pip`, `setuptools`, `wheels`:  
  `python3 -m pip install --upgrade pip setuptools wheel`
- Upgrade/update all already installed modules:  
  `python3 -m pip freeze | cut -d'=' -f1 | xargs -n1 python3 -m pip install -U`  
  after that update also the `requirements.txt` file by executing  
  `python3 -m pip freeze > requirements.txt`
## Specific project setup
Maybe check later if some of these packages are really needed...
- Install SW packages on operating system (don't know if they are really needed):  
  `sudo apt-get install python-dev python-pip ffmpeg libffi-dev`  
  `sudo apt-get install libxml2-dev libxslt-dev`  
  `sudo apt-get install libcairo2`  
  `sudo apt-get install libgeos++-dev libgeos-dev libgeos-doc`
- Check which SW packages are already installed, e.g.  
  `sudo apt list --installed | grep -i geos`
- Install python modules:  
  `python3 -m pip install pycodestyle flake8 black autopep8`  
  `python3 -m pip install lxml unicode scipy numpy`  
  `python3 -m pip install gizeh svgutils svgwrite svgpathtools svgelements cairosvg`  
  `python3 -m pip install fonttools[ufo,lxml,woff,unicode,type1]`  
  `python3 -m pip install fonttools[interpolatable,symfont,pathops,plot]`  
  `python3 -m pip install shapely svg.path svgpath2mpl matplotlib pytest`  
  `python3 -m pip install pillow opencv-python pypng`  
  `python3 -m pip install pangocffi cairocffi pangocairocffi freetype-py`  
  `python3 -m pip install pycairo` (does not install properly as libcairo2 is too old on my machine)  
  [OpenAI's ChatGPT](https://chat.openai.com) summarizes the function of the Python libraries as following:
  - gizeh
      is a Python library for creating vector graphics using the Cairo library.
      It is designed for creating simple shapes and complex shapes with a minimum of code.
  - svgutils
      is a library for working with SVG files,
      it allows you to easily combine multiple SVG files into a single document,
      manipulate individual SVG elements, and extract information from SVG files.
  - [svgwrite](https://svgwrite.readthedocs.io/en/latest/)
      is a library for creating new SVG files, 
      it provides a simple and easy to use interface for creating and manipulating SVG elements.
  - svgpathtools
      is a library for manipulating and analyzing SVG path elements,
      it provides tools for parsing, transforming, and simplifying SVG paths.
  - svgelements
      is a library for working with individual SVG elements,
      it provides a simple and easy to use interface for creating, manipulating, and analyzing SVG elements.
  - cairosvg
      is a library for converting SVG files to other formats such as PNG and PDF using the Cairo library.
  - cairocffi
      is a CFFI-based Python binding to the Cairo graphics library.
  - pangocffi
      is a Python library for creating and manipulating text layout using the Pango library
  - pangocairocffi
      is a library for creating and manipulating text layout using the Pango and Cairo libraries.
  - freetype-py
      is a Python wrapper for the FreeType library, which is used for rendering text.
  - fonttools
      is a library for manipulating font files, it provides tools for parsing, editing
      and converting fonts between different formats.
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
  - XML (redhat.vscode-xml)
  - Code Spell Checker (streetsidesoftware.code-spell-checker)
  - Todo Tree (Gruntfuggly.todo-tree)
  - Flake8 (ms-python.flake8)
- Extensions to check later:
  - Code Runner (formulahendry.code-runner)
  - Python Extension Pack (donjayamanne.python-extension-pack)
  - Tabnine AI Autocomplete (TabNine.tabnine-vscode)
  - GitHub Copilot (GitHub.copilot) for autocompletion
  - python snippets (frhtylcn.pythonsnippets)
  - AREPL for python (almenon.arepl)
  - Vim (vscodevim.vim)
- Setup / modify settings:
  - Python Analysis Type Checking Mode: on
  - Editor Format On Save: on
  - Editor Default Formatter: Python (ms-python.python)
  - Python Formatting Provider: autopep8
  - Python Linting Enabled: check-on
  - Python Linting Flake8 Enabled: check-on
  - Edit `$HOME/.config/Code/User/settings.json`:  
    `"editor.rulers": [79]`
  - Python Select Interpreter: `./venv/bin/python`
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
- Helpful Keyboard Shortcuts (`keybindings.json`)
  - `Ctrl+Shift+P` to open the Command Palette
  - `Crtl+Shift+7` Fold All Block Comments
  - `Crtl+x`       Remove whole line (if nothing is selected)
  - `Crtl+RETURN`  Python: Run Python File in Terminal (assigned by using `Ctrl+Shift+P`)

## Folder and Git setup
- Git ssh setup
  - Read [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
  - Generate ssh-key on your local machine:  
    `cd $HOME/.ssh`  
    `ssh-keygen -o -t ed25519 -C "git@github.com" -f id_ed25519_github`  
    `eval "$(ssh-agent)"`  (unsure if this command is really needed)  
    `ssh-add id_ed2551_github`  
    `ssh-add -L` to check if the key was added
  - Copy public-key `cat id_ed2551_github.pub` and add it to github  
    *Settings->SSH and GPG keys->SSH keys->New SSH key*...  
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


## Fonts
### Installation
- Install variable Fonts:  
  Find a font, check its license (SIL Open Font License (OFL) recommended), download it,  
  double-click on the `.ttf`-file and click *install*  in the opening *GNOME Font Viewer* window.
- Alternative installation method (quick and easy):  
  Create folder `$HOME/.fonts` and copy the `.ttf`-files into the folder.
### Font selection
- Resources / Overview
  - [fonts.google.com](https://fonts.google.com/)
  - [v-fonts.com](https://v-fonts.com/), e.g. a [funny "variable" cat font](https://v-fonts.com/fonts/zycon)
  - [axis-praxis.org](https://www.axis-praxis.org), see also the tool
      [samsa](https://www.axis-praxis.org/samsa)
  - [Building variable fonts with Feature Variations](https://github.com/irenevl/variable-fonts-with-feature-variations)
- Check the overview and axes of the 
  [Google variable fonts](https://fonts.google.com/variablefonts#font-families).
- Grandstander ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL))
    --> see [fonts.google.com](https://fonts.google.com/specimen/Grandstander)  
    Only axis *weight* (wght) and *italic*.  
    Range of standard Grandstander *weight*: 100-[400]-900.  
    The width of a single character does not seem to change when the *weight* changes.
    ```
    Available axes of Grandstander Thin:
      - wght:   100.0 to   900.0, default:   100.0 
    ```
- Noto Sans Mono ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL)) 
    --> see [fonts.google.com](https://fonts.google.com/noto/specimen/Noto+Sans+Mono)  
    Only two axes *weight* (wght) and *width* (wdth):  
    The width of a single character does not seem to change when the *weight* changes.  
    As it is a *mono* font all characters have the same *width*.
    ```
    Available axes of Noto Sans Mono:
      - wght:   100.0 to   900.0, default:   400.0 
      - wdth:    62.5 to   100.0, default:   100.0 
    ```
- Recursive ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL))
    --> see [fonts.google.com](https://fonts.google.com/specimen/Recursive)  
    Range of standard Recursive *weight*: 300-[400]-1000.  
    The height of some capital letters is "jumping" when *weight* is changed.  
    ```
    Available axes of Recursive Sans Linear Light:
      - MONO:     0.0 to     1.0, default:     0.0 
      - CASL:     0.0 to     1.0, default:     0.0 
      - wght:   300.0 to  1000.0, default:   300.0 
      - slnt:   -15.0 to     0.0, default:     0.0 
      - CRSV:     0.0 to     1.0, default:     0.5 
    ```
    My selection of main axes for my purpose:
    - `wght` - Weight: 300 - [400] - 1000
    - `slnt` - Slant: 0
    - `CASL` - Casual: 1
    - `MONO` - Monospace: 0
    - `CRSV` - Cursive: 1
- Roboto Flex ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL))
    --> see [fonts.google.com](https://fonts.google.com/specimen/Roboto+Flex)
    ```
    Available axes of Roboto Flex:
      - wght:   100.0 to  1000.0, default:   400.0 
      - wdth:    25.0 to   151.0, default:   100.0 
      - opsz:     8.0 to   144.0, default:    14.0 
      - GRAD:  -200.0 to   150.0, default:     0.0 
      - slnt:   -10.0 to     0.0, default:     0.0 
      - XTRA:   323.0 to   603.0, default:   468.0 
      - XOPQ:    27.0 to   175.0, default:    96.0 
      - YOPQ:    25.0 to   135.0, default:    79.0 
      - YTLC:   416.0 to   570.0, default:   514.0 
      - YTUC:   528.0 to   760.0, default:   712.0 
      - YTAS:   649.0 to   854.0, default:   750.0 
      - YTDE:  -305.0 to   -98.0, default:  -203.0 
      - YTFI:   560.0 to   788.0, default:   738.0 
    ```
    My selection of main axes for my purpose:  
    - `opsz` - (Optical Size: default)
    - `wght` - Weight 100-400-1000
    - `wdth` - Width 25-100-151
    - `YTAS` - (Ascender Height) (so that it looks good for f,i,j,ä,ö,ü)
    - `YTDE` - Descender Depth (so that it looks good for g,j,p,q,y)
    - `GRAD` - Grade (so that it looks good)
    - `YTLC` - Lowercase Height (so that it looks good for i,j,ä,ö,ü,a,c,e,g,s)
    - `YOPQ` - (Thin Stroke) (so that it looks good for a,e,f,g,s)
- Roboto Mono ([Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0))
    --> see [fonts.google.com](https://fonts.google.com/specimen/Roboto+Mono)  
    Only axis *weight* (wght) and *italic*.  
    The width of a single character does not seem to change when the *weight* changes, 
    but the dots of e.g. 'ä' and 'i' are "jumping" when *weight* is changed.  
    As it is a *mono* font all characters have the same *width*.
    ```
    Available axes of Roboto Mono:
      - wght:   100.0 to   700.0, default:   400.0 
    ```

### Links to information
- Check [freetype.org](https://freetype.org/) for information about fonts,
  especially the [FreeType Glyph Conventions](https://freetype.org/freetype2/docs/glyphs/index.html).  
  See also [freetype-py](https://github.com/rougier/freetype-py) and 
  [w3.org:SVG/text](https://www.w3.org/TR/SVG/text.html).
- Also check [wikipedia:Typeface](https://en.wikipedia.org/wiki/Typeface) and a lot more linked sites.
- For knowledge a good starting point is [fonts.google.com](https://fonts.google.com/knowledge), 
  especially check [Using type](https://fonts.google.com/knowledge/using_type)
- Also see [Awesome Typography](https://github.com/Jolg42/awesome-typography), a curated list about digital typography
- For text (i.e. glyph) layout engine check [pango.org](https://pango.gnome.org/)
- For font manipulation see [fonttools](https://github.com/fonttools/fonttools)
  and the [fonttools-documentation](https://fonttools.readthedocs.io/en/latest/)

## Artist name
Create an artist name, check for it on [namecheckr.com](https://www.namecheckr.com/) and register/secure it at several online services:  
- Googlemail (pwd+2factor)
- Youtube (login using google account)
- Pinterest (login using google account)
- TikTok (login using google account)
- Etsy (login using google account)  
- reddit (login using google account)
- patreon (login using google account)
- Twitter (login using google account)
- Instagram (pwd+2factor)
- Xing (pwd+2factor)  
- LinkedIn (pwd+2factor) 
- GitHub (pwd+2factor+2ndSMS)
-    redbubble (pwd w/o 2factor authentication)
- ... further ...
-    artmajeur.com
-    inprnt.com
-   society6
-   Kunstkopie.de
-   Europosters.de
-   Juniqe.de
-   Meisterdrucke.com
-   arsmundi.de
-   ARTGalerie.de, artlandgmbh.de
-   sell.wayfair.com
- Amazon
- Ebay
- Dribbble
- Behance
-  flickr
-  tumblr
- ello
-  deviantArt
- theartling
- saatchiart
- singulart
- artgallery.co.uk
-   Shutterstock
-   iStock
-   Adobe Stock
-   Getty Images
-   alamy.de
-   YellowKorner
- ... and further more like...
- a payment service
- a file sharing service
- ...

## Ideas
- Palindrome --> see [Wikipedia](https://en.wikipedia.org/wiki/Palindrome)  
  Example of an ambigram and palindrome:  
  "SATOR AREPO TENET OPERA ROTAS",
  see [Wikipedia - Sator Square](https://en.wikipedia.org/wiki/Sator_Square)
  (N and S reversed)  
  See also [Wikipedia: List of English palindromic phrases](https://en.wikipedia.org/wiki/List_of_English_palindromic_phrases)

- Word square --> see [Wikipedia](https://en.wikipedia.org/wiki/Word_square)

- Ambigram --> see [Wikipedia](https://en.wikipedia.org/wiki/Ambigram)  
  Example painting: *ME -> WE*, see [Wikipedia](https://en.wikipedia.org/wiki/Ambigram#/media/File:Me_we_co.jpg)

- Pictures (black and white, silhouettes, ...) of...
  - Architecture and monuments
    - Big Ben, Silhouette of town, towers of town, castle, palace,
    - Mount Everest, Mont Blanc, Matterhorn, Uluru - Ayers Rock,
  - Flora
    - fruits: strawberry, banana, apple, pear, cherry, kiwi, pineapple, paprika/pepper, 
    - flower, Nelumbo - lotus effect, daisy chain, blossom, tomato, broccoli, cabbage, pickle, pumpkin, 
    - fern, gingko leaf, maple leaf, chestnut leaf, cannabis leaf, oak leaf, peppermint, acorn, tree, grain ear, sunflower, artichoke,
  - Fauna  
    - dragonfly (half), butterfly (half), bat, ladybird, bee, bumblebee, spider, turtle,
    - rabbit, hedgehog, raccoon, sloth, koala, squirrel, mouse, fawn / bambi, penguin, 
    - horse (in various gaits like walk, trot, canter, gallop, pace and also pesade), 
    - dear (partly), boar, donkey (partly), cow, zebra, gnu, dog, dalmatian dog, greyhound, wolf+moon,
    - tiger, cat, lion, cheetah, cougar, panther, polar bear, fox, elephant, mammoth, rhinoceros, giraffe, camel, moose, panda, kangaroo,
    - owl, eagle, pigeon, sparrow, pelican, flamingo, cockatoo, swallow, black bird, 
      robin, crow, peacock, ostrich, hummingbird, kiwi-bird, 
    - cock, weathercock, meerkat/suricate, 
    - crocodile, whale, snail shell, jakobs shell, ammonite, mussel, scallop/jacobs shell, octopus, seagull, fish, thunnus, seahorse,
      [nautilus shell](https://upload.wikimedia.org/wikipedia/commons/0/08/NautilusCutawayLogarithmicSpiral.jpg),
    - salamander, lizard, iguana, camelion, 
    - egg, egg-shape, dinosaur, frog, 
    - cat head with flashing eyes, cat looking up at a balloon from below, 2 swans showing a heart,
    - cat's paw, dog's paw, wolf's paw,
    - goldfish in spherical glass, animals that look into the picture from the side,
    - animal paw prints and animal tracks,
  - Technology
    - car, train/locomotive, tractor, airplane, lunar module, lunar roving vehicle, sailing ship, paddle steamer, scooter,
    - zeppelin, hot-air balloon,
    - musical instruments: violin, guitar, hunting horn, trumpet, harp, 
    - light bulb, candle, reading lamp, clock, clock face, 
    - gun, pistol, revolver, colt, rifle,
  - Hollow Ware
    - vase, bottle, cup (of coffee), teapot, glass, wine glass, beer mug,
  - Art
    - Banksy, Vermeer, Rembrandt, Vincent van Gogh, Pablo Picasso, 
    - M. C. Escher (Penrose triangle, Penrose stairs), Triskelion, three hares, op-art,
    - Optical illusion (e.g. moving points),
    - The Creation of Adam,
    - Mona Lisa,
    - Praying Hands (Duerer),
    - Golden ration - The Vitruvian Man (Leonardo da Vinci),
    - Manneken Pis,
    - The Great Wave off Kanagawa (Katsushika Hokusai), 
    - Andy Warhol - "Shot Marilyns" paintings (1964),
  - Fantasy
    - unicorn, dragon, Chinese dragon, angel, monk, mermaid, medusa, devil,
      ghost, skeleton, magician, wizard, sorcerer, witch, leprechaun, troll, dwarf, fairy,
      dracula, nosferatu, Grim Reaper - Death in cowl, nun,
      Rudolph the Red-Nosed Reindeer, Easter Bunny, garden gnome,
    - Greek mythology like Icarus, Daedalus, Achilles, Sisyphus, Heracles, Prometheus, ...
  - Portraits
    - Einstein: I am convinced that He (God) does not play dice,
    - Goethe, Heinrich Heine, Leonardo da Vinci, Darwin, Kelvin, Tesla, 
    - Newton, Albert Schweitzer, Samuel Hahnemann, Richard Feynman,
    - Friedrich Nietzsche, Machiavelli, Niels Bohr, Zeno of Elea, Argonauts,
    - Karl Valentin, Charlie Chaplin, Yoda, Nostradamus, Baron Munchausen, roman emperors,
    - Mozart, Ludwig van Beethoven, Bach, Chopin, Joseph Haydn, Antonio Vivaldi, 
    - Johannes Kepler, Galileo Galilei, Nicolaus Copernicus, 
    - Odysseus, Christopher Columbus, Amerigo Vespucci, Ferdinand Magellan, James Cook, Marco Polo,
    - black/white of portraits / face(s): eye, mouth, lips, ears, butt, breast,
    - black/white portraits of animal faces like tiger, lion, 
  - Signs / themes
    - Luck: lucky pig, shamrock/four-leaf clover, Hans-in-luck = boy-with-goose, rich pig - poor pig,
    - Love: heart, cupid, kiss mouth, 
    - Sports: baseball, football, soccer-ball, tennis ball, basketball, badminton, shuttlecock,
    - Yoga - positons,
    - Peace-health-freedom: peace dove + human rights, tree-of-life,
    - Mathematics: pi, infinity, delta, gradient, phi, sum, product, sine, cosine, tangent, 
      pythagorean theorem / triangles, circle+pi,
    - Penrose tiling, einstein problem,
    - Rorschach test,
    - False Friends: freedom, this is not without,
    - Professional Category: rod of asclepius (medical) + Hippocratic Oath, Justitia, toque blanche, ...
    - Food: pizza, doner, burger, hotdog, curry sausage, french fries, gummy bears, abstract fish sign, 
    - Drink: coffee, tea, water, whisky, scotch, beer, 
    - chess, checkerboard, chessmen, teddy bear, feather, viking helmet, 
    - globe/world-maps, european countries, states of Germany, islands like Sylt,
    - yin and yang, buddha, peace sign, zodiac signs, zodiac wheel, maltese cross,
    - collection of icons, money-signs (dollar, euro, Yen, bitcoin and other crypto-coins, ... money-reflection),
    - clef / music sign, QR-code, dice, atom bomb mushroom cloud,
    - traffic signs: STOP, Autobahn, blue bicycle, traffic light man, video surveillance,
    - gender signs, cute couple as parking sign / parking garage, home of a single, ...
    - cloud, snowflake, crescent moon, sunglasses, footprint,
    - coins (maple leaf, krugerrand, old roman coins, chinese coins with holes, bitcoins, ...),
    - nude front/back, chauvinistic slogans, woman legs with seamed stockings in (red) pumps, red pumps,
    - brezel, Bavarian veal sausage, ass with ears, cigarette with smoke, no smoking,
    - knife, fork, spoon, garden gnome, hat, cap, baseball bat, condoms save...,
    - anchor, ship's propeller, aircraft's propeller, yellow submarine,
    - Bavarian laptop and leather pants and smartphone, bavarian lion, 
    - pirate flag, country flags in general, ratisbona's crossed keys,
    - drop into water with round waves, surfer wave,
    - "folder"-drawing (=Nude drawing), toilet paper, pepper mill, fire hydrant,
    - silhouettes of various flying insects in a glass case,
    - silhouette of sexy girl bending down for a colorful flower,
    - silhouette of sexy girl bending down to a dwarf, "Bend over fairy, wish is wish", 
    - silhouette of a butler offering something on a tray,
    - masks: venetian masks, scream ghostface, Guy Fawkes mask, 
    - a bunch of child's balloons, three wise monkeys (variations: see, hear, speak good), 
    - stylized cartoon eyes, Heart with a bite, fruit with a bite,
    - two puzzle parts fitting together like a couple,
    - statue, greek torso, diamonds and the cut of gems, 
    - Eye of Providence - All-Seeing Eye of God, the good God knows everything the neighborhood even more, 
    - Don Quixote, Tutankhamun, Cleopatra, Pl.yb.y bunny,

- Themes and texts like...  
  - luck and happiness: smile, be happy, don't worry be happy, home, sweet home, 
      welcome, good vibes only, do more what makes you happy,
      smiling can make you happy - just try it, 
      enjoy every moment, happy mind, live more - worry less,
  - YES, NO, maybe, 
  - automatized generation using popular first names,
  - big brother is watching you, adjust your crown and keep going, follow the money, 
  - don't marry be happy, body mind soul, emotional damage,
  - latin citations: errare humanum est, in dubio pro reo iudicandum est, 
    panem et circenses, ora et labora, cave canem, nomen est omen,
    veni vidi vici, alea jacta est, pecunia non olet, cui bono, 
    carpe diem et noctem, odi et amo,
  - Jack Nicholson as Edward Cole - The Bucket List (2007):  
    "Three things to remember when you get older:
     never pass up a bathroom, never waste a hard-on, and never trust a fart."
  - It takes nine months to have a baby, no matter how many men are working on it
  - It is not rocket science, too good to be true, it is a no-brainer, easygoing
  - Don't judge a book by its cover
  - Just because I am not talking doesn't mean I'm in a bad mood - sometimes I just like being quiet.
  - Pure energy, Heureka, be inspired, high life (cannabis leaf),
  - give us today our daily bread, those who die sooner are dead longer,
  - if looks could kill, 
  - crumpy cat - crumpy by nature
  - Do not get too excited, it's just a card with Easter greetings
  - Don't play with my heart, juggling with my feelings

- Further ideas and variations
  - Shape
    - outer contour in any shape - not only rectangle,
      e.g. heart, hole like broken, parts showing out of the hole ...
  - Text
    - slightly skewed text
    - descriptive text (e.g. bridge, tower, window, church, river, ...)
    - black/white as filled/stroke-only text
    - black/white using only two fonts of different thickness
    - use of different text heights on/over the page
  - Dots
    - gray-scale as single dots in different sizes (diameters)
    - gray-scale as tiny dots of constant size (diameter), but with different density 
  - Lines
    - gray-scale as lines with different thickness
    - gray-scale as lines with different density
  - Op art, optical art

## Copyright topics
- Reuse of Wikipedia texts:
  - See [Wikipedia: Reusing Wikipedia content](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content), especially:
    - [Wikipedia: Example notice](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content#Example_notice)
- Reuse of Wikipedia / Wikimedia images:
  - See [Wikimedia: Reusing content outside Wikimedia](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia),
    especially:
    - [Wikipedia: Ten things you may not know about images on Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Ten_things_you_may_not_know_about_images_on_Wikipedia)
    - [Wikipedia: File copyright tags](https://en.wikipedia.org/wiki/Wikipedia:File_copyright_tags)
- Reuse assistance tools
  - [Attribution Generator - lizenzhinweisgenerator.de](https://www.lizenzhinweisgenerator.de/?lang=en)
- Further Wikipedia / Wikimedia infos:
  - https://en.wikipedia.org/wiki/Public_domain
  - https://commons.wikimedia.org/wiki/Commons:Reuse_of_PD-Art_photographs
  - https://en.wikipedia.org/wiki/List_of_countries%27_copyright_lengths


## Know-How
### Images / Paintings - Sizes and Resolution
- General  
  300 DPI, sRGB-colors, .png or .jpg  
  Example: 20x20cm : 
  1.5cm frame + 2cm margin (white space) --> picture ca. 15x15cm  
  ... on these 14 to 16cm are approximately 50 rows of text  --> ca. 3mm text-height.  
  Further examples: 60x60cm --> 47x47cm and 40x40cm --> 32x32cm.
- P.n.erest  
  ratio 2:3, 1000x1500px, 600x1260px, 600x600px, 600x900px
- P.i.tful  
  - Poster(inch): 20.32x25.40cm - 60.96x91.44cm - deviation up to 0.51 cm  
    **8x10**, **10x10**, 12x12, **11x14**, **12x16**, 12x18, 14x14, 16x16, **16x20**, 18x18, 18x24, **24x36** inch  
  - Poster(cm): 21x30cm - 70x100cm - deviation up to 0.51 cm  
    **21x30**, **30x40**, **50x70**, **61x91**, 70x100 cm
- R.db.bble
  - see webpage "Dimensions & Format" --> "Wall Art"
- I.EA
  - picture frame: 10x15, 13x18, 20x25, 21x30, 23x23, 30x40, 32x32, 40x50, 50x70, 61x91cm
  - see also other sizes, passepartout-sizes and more ...

### SVG
- SVG-Coordinates
  - (0,0) at top left corner
  - x-direction: positive from left to right ("width")
  - y-direction: positive from top to bottom ("height")
- To be compatible with inkscape, 
  - use the units `[mm]` only once for `width` and `height`.
  - All other dimension are given in user units, which are independent of the physical units of length.  
  - Use `viewBox` to define the drawing canvas width in such a way that
    `RECT_WIDTH` corresponds to size `1` in user units (my own definition).  
  - Therefore all other sizes need to be scaled by `VB_RATIO = 1 / RECT_WIDTH`.
  - Example of a `140x100mm` rectangle in the middle of a DIN A4 page (`210x297mm`),
    drawn with `strokewidth = 0.1mm`.  
    Python code:
    ```python
    import svgwrite

    OUTPUT_FILE = "data/output/example/svg/din_a4_page_rectangle.svg"

    CANVAS_UNIT   = "mm"  # Units for CANVAS dimensions
    CANVAS_WIDTH  = 210   # DIN A4 page width in mm
    CANVAS_HEIGHT = 297   # DIN A4 page height in mm

    RECT_WIDTH  = 140  # rectangle width in mm
    RECT_HEIGHT = 100  # rectangle height in mm

    VB_RATIO = 1 / RECT_WIDTH  # multiply each dimension with this ratio

    # Center the rectangle horizontally and vertically on the page
    vb_w =  VB_RATIO * CANVAS_WIDTH
    vb_h =  VB_RATIO * CANVAS_HEIGHT
    vb_x = -VB_RATIO * (CANVAS_WIDTH - RECT_WIDTH) / 2
    vb_y = -VB_RATIO * (CANVAS_HEIGHT - RECT_HEIGHT) / 2

    # Set up the SVG canvas
    dwg = svgwrite.Drawing(OUTPUT_FILE,
                          size=(f"{CANVAS_WIDTH}mm", f"{CANVAS_HEIGHT}mm"),
                          viewBox=(f"{vb_x} {vb_y} {vb_w} {vb_h}")
                          )

    # Draw the rectangle
    dwg.add(
        dwg.rect(
            insert=(0, 0),
            size=(VB_RATIO*RECT_WIDTH, VB_RATIO*RECT_HEIGHT),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1*VB_RATIO,
            fill="none",
        )
    )

    # Save the SVG file
    dwg.save()
    ```  
    SVG-output:
    ```svg
    <?xml version="1.0" encoding="utf-8"?>
    <svg baseProfile="full" version="1.1"
        width="210mm" height="297mm"
        viewBox="-0.25 -0.7035714285714285 1.5 2.1214285714285714"
        xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events"
        xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs />
        <rect width="1.0" height="0.7142857142857143" x="0" y="0"
              stroke="black" stroke-width="0.0007142857142857143"
              fill="none"/>
    </svg>
    ``` 
- SVG font 
  - Attributes
    - [`baseline-shift`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/baseline-shift):
              Specifies the distance from the dominant baseline of the parent text content element
              to the dominant baseline of this text content element.
              A shifted object might be a sub- or superscript.
    - [`dominant-baseline`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/dominant-baseline):
              Specifies the baseline used to align text.
    - [`font-family`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/font-family):
              Specifies the font family.
    - [`font-size`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/font-size):
              Specifies the size of the font, measured from 
              [baseline](https://en.wikipedia.org/wiki/Baseline_(typography))
              to baseline in y-direction
    - [`font-size-adjust`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/font-size-adjust):
              Specifies the aspect value of the font.
              It helps preserve the font's x-height when the font-size is scaled.
    - [`font-stretch`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/font-stretch):
              Specifies the horizontal scaling of the font.
    - [`font-style`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/font-style):
              Specifies the style of the font (normal, italic, or oblique).
    - [`font-variant`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/font-variant):
              Specifies the variant of the font (normal or small-caps).
    - [`font-weight`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/font-weight):
              Specifies the weight of the font (normal or bold).
    - [`letter-spacing`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/letter-spacing):
              Specifies the space between characters.
    - [`text-anchor`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/text-anchor):
              Specifies the position relative to a given point.
    - [`text-decoration`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/text-decoration):
              Specifies the decoration applied (underline, overline, line-through, or blink).
    - [`word-spacing`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/word-spacing):
              Specifies the space between words.
    - [`glyph-orientation-horizontal`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/glyph-orientation-horizontal) - deprecated:
              Specifies the orientation of the glyphs used to render.
    - [`glyph-orientation-vertical`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/glyph-orientation-vertical) - deprecated:
              Specifies the orientation of the glyphs used to render.
    - [`kerning`](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/kerning) - deprecated:
              Specifies the amount of space between two characters.
    - [`font-variation-settings`](https://developer.mozilla.org/en-US/docs/Web/CSS/font-variation-settings) - CSS:
              Specifies the values for the axes.
  - Font axis & abbreviations  
      | Axis & Abbreviation | CSS attribute              | font-variation-settings syntax        |
      |---------------------|----------------------------|---------------------------------------|
      | Italic (ital)       | font-style: italic;        | font-variation-settings: 'ital' 1;    |
      | Optical size (opsz) | font-optical-sizing auto;  | font-variation-settings: 'opsz' 16;   |
      | Slant (slnt)        | font-style: oblique 14deg; | font-variation-settings: 'slnt' 14;   |
      | Weight (wght)       | font-weight: 375;          | font-variation-settings: 'wght' 375;  |
      | Width (wdth)        | font-stretch: 115%;        | font-variation-settings: 'wdth' 115;  |
  - How to embed the font directly into a SVG file by converting the ttf file using base64...   
    see [stackoverflow.com -> how-to-specify-font-family-in-svg](https://stackoverflow.com/questions/30024943/how-to-specify-font-family-in-svg)
- SVG path
  - Path direction (explaination given by [OpenAI's ChatGPT](https://chat.openai.com)):  
    A closed path that is drawn in a counterclockwise direction represents a filled path,
    while a closed path that is drawn in a clockwise direction represents a subtractive path.  
    **Note:** Here "clockwise" and "counterclockwise" direction are applied to a coordinate system with bottom-left origin, while SVG originally has a top-left origin.  
    Example using `svgpathtools` to check the direction by calculating the area:  
    ```
    from svgpathtools import parse_path

    # Define a closed counterclockwise path: right, up, left, down (close) 
    path1 = parse_path('M 10 10 L 20 10 L 20 20 L 10 20 Z')

    # Define a closed clockwise path: up, right, down, left (close)
    path2 = parse_path('M 30 30 L 30 40 L 40 40 L 40 30 Z')

    # Check the direction of the paths
    print(path1.area())  # positive, counterclockwise
    print(path2.area())  # negative, clockwise
    ```
- [Shapely](https://shapely.readthedocs.io/en/stable/manual.html)
  is a Python package for working with vector geometries
  - For the defintions / conventions used by `Shapely` and `JTS` see  
  [JTS Discussion stored at archive.org](https://web.archive.org/web/20160719195511/http://www.vividsolutions.com/jts/discussion.htm),  
  [Simple Feature Access - Part 1: Common Architecture](https://www.ogc.org/standard/sfa/)  
  and also the [Introduction to Spatial Data Programming with Python](https://geobgu.xyz/py/shapely.html).  
  See also the [Well-known text representation of geometry](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry).  
  **Note:** "clockwise" and "counterclockwise" direction are applied to a coordinate system with bottom-left origin. 
  - The
  [Dimensionally Extended Nine-Intersection Model (DE-9IM)](https://giswiki.hsr.ch/images/3/3d/9dem_springer.pdf)
    describes the topological predicates and their corresponding meanings.
  - According to 
  [JTS Technical Specs.pdf](https://raw.githubusercontent.com/locationtech/jts/master/doc/JTS%20Technical%20Specs.pdf)... 
    - a `LinearRing` represents an ordered sequence of point tuples `(x, y[, z])`.  
      The sequence may be **explicitly closed** by passing identical values in the first and last indices.  
      Otherwise, the sequence will be **implicitly closed** by copying the first tuple to the last index.  
      A `LinearRing` may not cross itself, and may not touch itself at a single point.
    - a `Polygon` is implemented as a single `LinearRing` for the outer shell and an array of `LinearRings` for the holes.  
      The **outer shell** (exterior ring) is oriented **clockwise** (CW) and the **holes** (interior rings) are oriented **counterclockwise** (CCW),
      see [`shapely.geometry.polygon.orient()`](https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.polygon.orient).
    - a `MultiPolygon` is a collection of `Polygons`.

### Fonts
- Font-Coordinates
  - (0,0) at left side (of *EM box*) on baseline (y-direction)
  - x-direction: positive from left to right ("width")
  - y-direction: positive from bottom to top ("height")
- Unit conversion: `16px = 12pt = 1Pica` (Pixel->Point->Pica)  
  According to
  [wikipedia.org](https://en.wikipedia.org/wiki/Point_(typography))
  `1point = 1pt` is defined as  
  ```
  typographic units  -->  1/12 picas
  imperial/US units  -->  1/72 in
  metric (SI) units  -->  0.3528 mm
  ```
- [Classic Typographic Scale](https://retinart.net/typography/typographicscale/): 
  6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 21, 24, 36, 48, 60, 72
- Weight: `font-weight`, `wght` in CSS determines e.g.: 
  Normal (Regular), Medium, Bold, ...
- According to the
  [OpenType specification](https://docs.microsoft.com/en-us/typography/opentype/spec/name#name-ids)
  following `nameIDs` are defined for example for
  `NotoSansMono-VariableFont_wdth,wght.ttf`:
  ```
     1 : font family name         --- Noto Sans Mono
     2 : font subfamily name      --- Regular
     4 : font full name           --- Noto Sans Mono Regular
    13 : font license description --- ... SIL Open Font License ... 
  ```
- According to 
  [w3.org](https://www.w3.org/TR/SVG11/text.html#FontsTablesBaselines)
  the geometry of a font is based on a coordinate system defined by the *EM box* or *em square*.  
  The box of `1EM` high and `1EM` wide specifies the *design space*.
  Each glyph is sized relative to that *EM box*.  
  The SVG `font-size` defines the `1em` value of a typeface set, e.g.:  
  `font-size = 12px --> 1em = 12px --> 1en=6px`  
  The geometric coordinates are given in certain units,
  i.e. `EM` is devided into a number of units per `EM`.  
  According to 
  [w3.org](http://www.w3.org/TR/2008/REC-CSS2-20080411/fonts.html#emsq)
  common `unitsPerEm (head)` values are
  - 250 (Intellifont),
  - 1000 (Type 1) and
  - 2048 (TrueType, TrueType GX and OpenType). 
- Information stored in a font
  - [head](https://learn.microsoft.com/en-us/typography/opentype/spec/head) -
    Font Header Table
    - `unitsPerEm`: 1 `EM` is devided into a number of units per `EM`.
  - [hhea](https://learn.microsoft.com/en-us/typography/opentype/spec/hhea) -
    Horizontal Header Table
    - `ascender`: font ascent: distance from baseline (0,0) to the top of the EM box.
    - `descender`: font descent: (negative) distance from baseline (0,0) to the bottom of the EM box.
    - `lineGap`: (usually 0) determines the default line spacing combined with `ascender` and `descender` values.
  - [OS/2](https://learn.microsoft.com/en-us/typography/opentype/spec/os2) -
    OS/2 and Windows Metrics Table
    - `sTypoAscender`
    - `sTypoDescender`
    - `sTypoLineGap`
    - `sxHeight`: measured from the baseline to the top of lowercase flat glyphs such as x.
    - `sCapHeight`: height of uppercase letters, measured from baseline to the top of flat-topped glyphs like X.
- TODO: alignment-point #############################################################
- TODO: advancement (x-distance to next character) ##########################################

### Further Resources
- Lorem Ipsum generator: [loremipsum.de](https://www.loremipsum.de/)

## Business
Some remarks for starting a business -
just a view ideas and hints found in the internet...
### Setup
What to consider if you want to start an online business in Germany...  
see also internet recommendations, e.g. aufbauen-online-business.de
- business bank account ($?)  
  -> maybe additionally PayPal? (for Etsy not needed)
- address-service, e.g. anschrift.net ($) - 
  read also: [ah's experience with an "Impressum-Service"](https://www.andreashagemann.com/impressum-service#viewer-6camm).  
  It is unsure whether an address-service really complies with the regulations.  
  Think about using your middle name...
- determine an artist- / business-name and ...
  - register e-mail-address
  - register shop-name, e.g. at Etsy
  - register website / domain name ($)
- register a business (Gewerbe-Anmeldung)
- register at your professional association (Handwerkskammer, IHK, ...) ($?)
- obtain a UStID (Umsatzsteuer-ID) at your tax office
- legal texts-service, e.g. it-recht-kanzlei.de ($)
- packaging license, registration at the Central Packaging Register LUCID ($)
- inform your employer about your sideline business
- read also [some first hints regarding tax topics](https://www.alltageinesfotoproduzenten.de/2011/11/23/die-steuerlichen-aspekte-der-stockfotografie/)
- create a [Catalogue raisonné](https://en.wikipedia.org/wiki/Catalogue_raisonn%C3%A9),  
  see also [artvise.me](https://artvise.me/)

### Exhibitions
- Opportunities to show your art
  - Hotels
  - Banks
  - Hospitals
  - Restaurants
  - Bars
  - Office buildings, big companies
  - Galleries
  - Museum of Arts
  - Jewelry stores
  - Gift stores
  - Picture frame stores
  - Furniture store like IK.A
- Some locations
  - [Berufsverband Bildender Kuenstler - Niederbayern/Oberpfalz e.V.](https://www.kunst-in-ostbayern.de)
  - [Kuenstlerhaus Andreasstadel](https://www.kuenstlerhaus-andreasstadel.de)
  - [Degginger](https://www.degginger.de)
  - [Neunkubikmeter](https://www.regensburg.de/kultur/veranstaltungen-des-kulturreferats/neunkubikmeter)
  - [Kunstschaufenster](https://www.12achtzig.de)
  - [M26 - Maximilianstrasse 26](https://www.regensburg.de/m26)
---

## Next steps / check in future / ToDos / Reminders
- How to handle examples / spikes / testing / unit-test 
- Check ArjanCodes YT Channel videos


### Websites
Interesting websites I stumbled upon: check in future:
- https://openai.com/dall-e-2/ AI created images and art  
  or try: https://www.craiyon.com/  
  see also: [Dall-E Alternatives](https://alternativeto.net/software/dall-e/)
- https://midjourney.com AI created art
- https://www.jasper.ai/art AI created art
- https://www.synthesia.io/ AI Text to video
- https://repurpose.io/ Distribute your content to your social media channels
- https://pictory.ai/ Video creation made easy
- https://lumen5.com/ Create video from blog content
- https://discoveryartfair.com/how-to-sell-art/ Some infos for artists

