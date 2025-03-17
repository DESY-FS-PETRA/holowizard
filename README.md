 <img src="logo.png" width="200"/> 

### Info
- Helmholtz: https://helmholtz.software/software/holowizard
- Gitlab URL: https://gitlab.desy.de/fs-petra/software/holowizard/
- Gitlab ID: 15357

# Table of Contents

1. [Subprojects](#subprojects)
2. [Versioning](#versioning)
3. [Installation](#installation)
4. [Publish a new pip package](#publish-a-new-pip-package)
5. [Trouble shooting](#trouble-shooting)
6. [Further Ressources](#further-ressources)

## Subprojects

### Livereco Server
- URL: https://gitlab.desy.de/fs-petra/software/livereco_server/
- ID: 1445

### Livereco Interface
- URL: https://gitlab.desy.de/fs-petra/software/livereco_interface/
- ID: 15062

### Holo Forge
- URL: https://gitlab.desy.de/fs-petra/software/holography-data-ml/
- ID: 9756

### Holo Pipe
- URL: https://gitlab.desy.de/fs-petra/software/holopipe/
- ID: 16418


## Versioning
### Holo Wizard
The version number increases each time a subproject/subpackage is updated. Further things are not specified yet.

### Subprojects
The version numbers of the subprojects usually follow the pattern 
- {major}.{minor}.{patch}

The **minor or major** version numbers will be increased for:
- Feature extensions
- Changes in the api functions
- Bugfixes with a significant impact on the reconstruction result

The **patch** version number is reserved for 
- Minor bugfixes

There is a special rule for the **minor** version number:
- **even number**: Release version meant for production
- **odd number**: Experimental version e.g. to develop new features

## Installation
### Environment
Create a new environment
```bash
$ mamba create -p <path_to_env> python=3.11 
```

Activate enviroment
```bash
$ conda activate <path_to_env>
```

In your pip.conf file, add the lines
```bash
[global]
extra-index-url = 
	https://gitlab.desy.de/api/v4/projects/1445/packages/pypi/simple
	https://gitlab.desy.de/api/v4/projects/15062/packages/pypi/simple
	https://gitlab.desy.de/api/v4/projects/15357/packages/pypi/simple
	https://gitlab.desy.de/api/v4/projects/9756/packages/pypi/simple
	https://gitlab.desy.de/api/v4/projects/16418/packages/pypi/simple
```

Or (Linux) create new pip.conf file in 
```bash
 ~/.pip/pip.conf
```
or
```bash
~/.config/pip/pip.conf
```

### Install holowizard pip packages
```bash
$ pip install "holowizard[<package>]"
```

The following packages are available
- all
- livereco_interface
- livereco
- holoforge
- holopipe

## Publish A New Pip Package

### Set-Up Build System
1. Create personal access token under:
```bash
https://gitlab.desy.de/-/user_settings/personal_access_tokens
```

2. Create a file ~/.pypirc with content:

```bash
[distutils]
index-servers =
    <repo_name>

[<repo_name>]
repository = https://gitlab.desy.de/api/v4/projects/<id>/packages/pypi
username = <your_personal_access_token_name>
password = <your_personal_access_token>
```

3. Install required python packages

```bash
$ pip install build twine
```

### Create Package
1. Optional: Create a branch, one for each major version 1.0.x
```bash
$ git checkout -b versions/<version> # e.g. versions/1.0.0
```

2. Update pyproject.toml File. At least, increment version number!
```bash
[project]
name="<name>"
version = "<version>"
...
```

3. Update requirements.txt, e.g.:
```bash
torch
torchvision
cupy-cuda12x
...
```

4. Commit and push changes
```bash
$ git add .
$ git commit -m "<message>"
$ git push 
```

5. Build package
```bash
$ python -m build
```

6. Add and push git tag
```bash
$ git tag <version>
$ git push --tags
```

### Upload Package
1. Upload files

```bash
$ python -m twine upload --repository <repo_name> dist/<name>-<version>*
```

2. Check package
```bash
https://gitlab.desy.de/fs-petra/software/<name>/-/packages/
```

## Trouble shooting

### Location of python environment
1. beegfs as a location for the python environment breaks links of the python binaries and does not work

2. Make sure, the python binary and pip point both to your enviroment
```bash
$ which python
$ which pip
```

### Python version
Make sure the installed python version matches the given version of the installation section
```bash
$ python --version
```

### Site-packages
Somtimes, a fresh enviroment is not so fresh at it seems. Make sure that no unintentional python packages are loaded. 
The site-packages loaded are listed in
```bash
~/.local/lib/<pythonversion>/site-packages
```
Delete unknown packages or the whole <pythonversion> directory if unsure.

### Mamba in .bashrc
Make sure that the files ~/.bashrc, ~/.bash_profile, ~/.zshrc or similar shell initialization files do not initialize mamba or conda.


## Further Ressources
### ASRM Paper:
- URL: https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-7-10801&id=547807###
- DOI: 10.1364/OE.514641

```{bibtex}
@article{Dora:24,
author = {Johannes Dora and Martin M\"{o}ddel and Silja Flenner and Christian G. Schroer and Tobias Knopp and Johannes Hagemann},
journal = {Opt. Express},
keywords = {Free electron lasers; Holographic microscopy; Imaging techniques; Phase shift; X-ray imaging; Zone plates},
number = {7},
pages = {10801--10828},
publisher = {Optica Publishing Group},
title = {{Artifact-suppressing reconstruction of strongly interacting objects in X-ray near-field holography without a spatial support constraint}},
volume = {32},
month = {Mar},
year = {2024},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-32-7-10801},
doi = {10.1364/OE.514641},
abstract = {The phase problem is a well known ill-posed reconstruction problem of coherent lens-less microscopic imaging, where only the squared magnitude of a complex wavefront is measured by a detector while the phase information of the wave field is lost. To retrieve the lost information, common algorithms rely either on multiple data acquisitions under varying measurement conditions or on the application of strong constraints such as a spatial support. In X-ray near-field holography, however, these methods are rendered impractical in the setting of time sensitive in situ and operando measurements. In this paper, we will forego the spatial support constraint and propose a projected gradient descent (PGD) based reconstruction scheme in combination with proper preprocessing and regularization that significantly reduces artifacts for refractive reconstructions from only a single acquired hologram without a spatial support constraint. We demonstrate the feasibility and robustness of our approach on different data sets obtained at the nano imaging endstation of P05 at PETRA III (DESY, Hamburg) operated by Helmholtz-Zentrum Hereon.},
}
```

### Autofocus paper
- URL: https://opg.optica.org/oe/abstract.cfm?doi=10.1364/OE.544573
- DOI: 10.1364/OE.544573

```{bibtex}
@article{Dora:25,
author = {Johannes Dora and Martin M\"{o}ddel and Silja Flenner and Jan Reimers and Berit Zeller-Plumhoff and Christian G. Schroer and Tobias Knopp and Johannes Hagemann},
journal = {Opt. Express},
keywords = {Image analysis; Image metrics; Imaging systems; Phase retrieval; X-ray imaging; Zone plates},
number = {4},
pages = {6641--6657},
publisher = {Optica Publishing Group},
title = {Model-based autofocus for near-field phase retrieval},
volume = {33},
month = {Feb},
year = {2025},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-33-4-6641},
doi = {10.1364/OE.544573},
}
```

### Python Repo on Zenodo
- Version: 1.3.1
- URL: https://zenodo.org/records/14024980
- DOI: 10.5281/zenodo.14024980

```{bibtex}
@software{dora_2024_14024980,
  author       = {Dora, Johannes and
                  Flenner, Silja and
                  Lopes Marinho, André and
                  Hagemann, Johannes},
  title        = {{A Python framework for the online reconstruction
                   of X-ray near-field holography data
                  }},
  month        = nov,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {1.3.1},
  doi          = {10.5281/zenodo.14024980},
  url          = {https://doi.org/10.5281/zenodo.14024980},
}
```
