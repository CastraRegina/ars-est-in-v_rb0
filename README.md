# ars-est-in-verbo
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
- Folder Structure
  - See [Python Packages - Package structure](https://py-pkgs.org/04-package-structure.html#package-structure)
  - See Jason C. McDonald's [Dead Simple Python: Project Structure and Imports](https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6)
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
- Convert utf-8 file to ascii:  
  `cat input.txt | iconv -f utf-8 -t ascii//TRANSLIT > output.txt`

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
- Install python modules:  
  `python3 -m pip install pycodestyle flake8 black autopep8`  
  `python3 -m pip install lxml unicode scipy numpy`  
  `python3 -m pip install gizeh svgutils svgwrite svgpathtools svgelements cairosvg`  
  `python3 -m pip install fonttools[ufo,lxml,woff,unicode]`  
  `python3 -m pip install pillow opencv-python`  
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
  - svgwrite
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
- Helpful Shortcuts
  - `Ctrl+Shift+P` to open the Command Palette

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
- Install variable Fonts:  
  Find a font, check its license (SIL Open Font License (OFL) recommended), download it,  
  double-click on the `.ttf`-file and click *install*  in the opening *GNOME Font Viewer* window.
  - Noto Sans Mono ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL)) 
      --> see [fonts.google.com](https://fonts.google.com/noto/specimen/Noto+Sans+Mono)
  - Grandstander ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL))
      --> see [fonts.google.com](https://fonts.google.com/specimen/Grandstander)
  - Recursive ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL))
      --> see [fonts.google.com](https://fonts.google.com/specimen/Recursive)
  - Roboto Flex ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL))
      --> see [fonts.google.com](https://fonts.google.com/specimen/Roboto+Flex)
  - Roboto Mono ([Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0))
      --> see [fonts.google.com](https://fonts.google.com/specimen/Roboto+Mono)
- Alternative installation method:  
  Create folder `$HOME/.fonts` and copy the `.ttf`-files into the folder.
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
- redbubble (pwd w/o 2factor authentication)
- ... further ...
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
- artmajeur 
- singulart
- artgallery.co.uk 
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
  - Architecture
    - Big Ben, Silhouette of town, towers of town, castle, palace,
  - Flora
    - fruits: strawberry, banana, apple, pear, cherry, kiwi, pineapple, paprika/pepper, 
    - flower, daisy chain, blossom, tomato, broccoli, cabbage, pickle, pumpkin, 
    - fern, gingko leaf, maple leaf, cannabis leaf, tree, grain ear, sunflower, artichoke,
  - Fauna  
    - dragonfly (half), butterfly (half), ladybird, rabbit, hedgehog, raccoon, sloth, koala, squirrel, 
    - dear (partly), boar, donkey (partly), cow, horse, zebra, gnu, dog, dalmatian dog, greyhound, 
    - tiger, cat, lion, cheetah, panther, polar bear, fox, elephant, rhinoceros, giraffe, camel, moose, panda, kangaroo,
    - owl, eagle, pigeon, sparrow, pelican, cockatoo, swallow, black bird, robin, crow, peacock, ostrich, hummingbird,  
    - crocodile, whale, snail shell, jakobs shell, ammonite, mussel, scallop/jacobs shell, octopus, seagull, fish, seahorse,
      [nautilus shell](https://upload.wikimedia.org/wikipedia/commons/0/08/NautilusCutawayLogarithmicSpiral.jpg),
    - salamander, lizard, iguana, camelion, 
    - egg, egg-shape, dinosaur, frog, 
    - cat head with flashing eyes, 2 swans showing a heart,
    - goldfish in spherical glass, 
  - Technology
    - car, train/locomotive, tractor, airplane, lunar module, lunar roving vehicle, sailing ship, paddle steamer, scooter,
    - musical instruments: violin, guitar, hunting horn, trumpet, harp, 
    - light bulb, candle, reading lamp, clock, clock face, 
    - gun, pistol, revolver, colt, rifle,
  - Hollow Ware
    - vase, bottle, cup (of coffee), teapot, glass, wine glass, beer mug,
  - Art
    - Banksy, M. C. Escher (Penrose triangle, Penrose stairs), Triskelion, three hares, 
    - The Creation of Adam,
    - Mona Lisa,
    - Praying Hands (Duerer),
    - Golden ration - The Vitruvian Man (Leonardo da Vinci),
    - Manneken Pis,
  - Fantasy
    - unicorn, dragon, Chinese dragon, angel, monk, mermaid, medusa, Greek mythology,
      ghost, skeleton, magician, wizard, sorcerer, witch, leprechaun, troll, dwarf,
  - Portraits
    - Einstein, Goethe, Mozart, Leonardo da Vinci, Tesla, roman emperors, Yoda, 
    - black/white of portraits / face(s): eye, mouth, lips, ears, butt, breast,
  - Signs / themes
    - Luck: lucky pig, shamrock/four-leaf clover,
    - Love: heart, cupid, kiss mouth, 
    - Sports: baseball, football, soccer-ball, tennis ball, basketball, badminton, shuttlecock,
    - Peace: peace dove + human rights,
    - Mathematics: pi, infinity, delta, gradient, phi, sum, product, sine, cosine, tangent, 
      pythagorean theorem / triangles, circle+pi,
    - False Friends: freedom, this is not without,
    - Professional Category: rod of asclepius (medical), Justitia, ...
    - Food: pizza, doner, burger, hotdog, curry sausage, french fries, gummy bears, 
    - Drink: coffee, tea, water, whisky, scotch, beer, 
    - chess, checkerboard, chessmen, teddy bear, feather, viking helmet, 
    - globe/world-maps, european countries, states of Germany, islands like Sylt,
    - yin and yang, buddha, peace sign, zodiac signs, zodiac wheel,
    - collection of icons, money-signs (dollar, euro, Yen, ... money-reflection), clef / music sign,
    - traffic signs: STOP, Autobahn, traffic light man, video surveillance,
    - gender signs, cute couple as parking sign / parking garage, home of a single, ...
    - cloud, snowflake, crescent moon, sunglasses, footprint,
    - coins (maple leaf, krugerrand, old roman coins, chinese coins with holes, bitcoins, ...),
    - nude front/back, chauvinistic slogans, woman legs with seamed stockings in (red) pumps, red pumps,
    - brezel, Bavarian veal sausage, ass with ears, cigarette with smoke, no smoking,
    - knife, fork, spoon, garden gnome, hat, cap, baseball bat, condoms save...,
    - anchor, ship's propeller, aircraft's propeller,
    - Bavarian laptop and leather pants and smartphone, bavarian lion, 
    - pirate flag, country flags in general,
    - drop into water with round waves, surfer wave,
    - "folder"-drawing (=Nude drawing), toilet paper, pepper mill, 
    - silhouettes of various flying insects in a glass case,
    - masks: venetian masks, scream ghostface, Guy Fawkes mask, 

- Themes and texts like...  
  - luck and happiness: smile, be happy, don't worry be happy, home, sweet home, 
      welcome, good vibes only, do more what makes you happy,
      smiling can make you happy - just try it, 
      enjoy every moment,
  - YES, NO, maybe, 
  - big brother is watching you, adjust your crown and keep going, follow the money, 
  - don't marry be happy, body mind soul, emotional damage,
  - latin citations: errare humanum est, in dubio pro reo iudicandum est, 
    panem et circenses, ora et labora, cave canem, nomen est omen,
    veni vidi vici, alea jacta est, pecunia non olet, cui bono, 
    carpe diem et noctem, 
  - Jack Nicholson as Edward Cole - The Bucket List (2007):  
    "Three things to remember when you get older:
     never pass up a bathroom, never waste a hard-on, and never trust a fart."

- Further ideas and variations
  - Text
    - slightly skewed text
    - descriptive text (e.g. bridge, tower, window, church, river, ...)
    - black/white as filled/stroke-only text
    - black/white using only two fonts of different thickness
  - Dots
    - gray-scale as single dots in different sizes (diameters)
    - gray-scale as tiny dots of constant size (diameter), but with different density 
  - Lines
    - gray-scale as lines with different thickness
    - gray-scale as lines with different density

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


## Sizes and Resolution
- General  
  300 DPI, sRGB-colors, .png or .jpg  
  Example: 20x20cm : 1.5cm frame + 2cm margin (white space) --> picture < 16x16cm
- P.n.erest  
  ratio 2:3, 1000x1500px, 600x1260px, 600x600px, 600x900px
- P.i.tful  
  - Poster(inch): 20.32x25.40cm - 60.96x91.44cm - deviation up to 0.51 cm  
    **8x10**, **10x10**, 12x12, **11x14**, **12x16**, 12x18, 14x14, 16x16, **16x20**, 18x18, 18x24, **24x36** inch  
  - Poster(cm): 21x30cm - 70x100cm - deviation up to 0.51 cm  
    **21x30**, **30x40**, **50x70**, **61x91**, 70x100 cm
- I.EA
  - picture frame: 10x15, 13x18, 20x25, 21x30, 23x23, 30x40, 32x32, 40x50, 50x70, 61x91cm
  - see also other sizes, passepartout-sizes and more ...

## Business setup
What to consider if you want to start an online business in Germany...  
see also internet recommendations, e.g. aufbauen-online-business.de
- business bank account ($?)  
  -> maybe additionally PayPal? (for Etsy not needed)
- address-service, e.g. anschrift.net ($) - 
  read also: [ah's experience with an "Impressum-Service"](https://www.andreashagemann.com/impressum-service#viewer-6camm).  
  It is unsure whether an address-service really complies with the regulations.  
  Think about using your middle name...
- determine a artist- / business-name and ...
  - register e-mail-address
  - register shop-name, e.g. at Etsy
  - register website / domain name ($)
- register a business (Gewerbe-Anmeldung)
- register at your professional association (Handwerkskammer, IHK, ...) ($?)
- obtain a UStID (Umsatzsteuer-ID) at your tax office
- legal texts-service, e.g. it-recht-kanzlei.de ($)
- packaging license, registration at the Central Packaging Register LUCID ($)
- inform your employer about your sideline business

---

## Next steps / check in future / ToDos / Reminders
- How to handle examples / spikes / testing / unit-test 
- Check ArjanCodes YT Channel videos


### Websites
Interesting websites I stumbled upon: check in future:
- https://openai.com/dall-e-2/ AI created images and art  
  or try: https://www.craiyon.com/  
  see also: [Dall-E Alternatives](https://alternativeto.net/software/dall-e/)
- https://www.jasper.ai/art AI created art
- https://www.synthesia.io/ AI Text to video
- https://repurpose.io/ Distribute your content to your social media channels
- https://pictory.ai/ Video creation made easy
- https://lumen5.com/ Create video from blog content

