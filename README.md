# HoloWizard
<img src="https://connect.helmholtz-imaging.de/media/solution/Holowizard_logo.png" width="200"/> 

[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/DESY-FS-PETRA/holowizard)
[![LICENCE](https://img.shields.io/github/license/DESY-FS-PETRA/holowizard)](https://github.com/DESY-FS-PETRA/holowizard/blob/main/LICENSE)
![GitHub top language](https://img.shields.io/github/languages/top/DESY-FS-PETRA/holowizard)
[![PYPI](https://img.shields.io/pypi/v/holowizard)](https://pypi.org/project/holowizard/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16317646.svg)](https://doi.org/10.5281/zenodo.16275927)

HoloWizard is a collection of software tools for phase retrieval in the setting of near-field holographic microscopy and tomography.
It is developed at DESY, Hamburg in the group X-Ray Nanoscience and X-Ray Optics.
The software is designed to cover all steps in experimental/algorithm development:
- The Forge generates datasets within given variation bouds either for testing new algorithms or provide training data sets for ML applications
- The Core implements the ASRM algorithm, our phase retrieval engine. It includes also all neccessary auxillary functions to load and preprocess single holograms/diffraction patterns.
- The Pipe enables the **online** and **offline** batch processing of holo tomograms, i.e. the phase retrieval of individual projections(along with all preprocessing steps) and the computation of the tomographic volume from the retrieved projections.
- The Livereco module can show online reconstructions of single projections during an experiment i.e. for finding a suited region of interest. It can be integrated in existing experimental control software.

# Table of Contents
- [Installation](#installation)
  - [Python Environment](#python-environment)
  - [Install package](#install-package)
- [Tools](#tools)
  - [The Core](#the-core)
  - [The Workbench](#the-workbench)
    - [User Administration](#user-administration)
  - [The Forge](#the-forge)
  - [The Livereco](#the-livereco)
- [Citations](#citations)
    - [Artifact-suppressing reconstruction method:](#artifact-suppressing-reconstruction-method)
    - [Model-based autofocus:](#model-based-autofocus)
  - [Further links](#further-links)

# Installation
## Python Environment
Create a new environment with python 3.11., i.e. with mamba
```bash
$ mamba create -p <path_to_env> python=3.11 
```

Activate enviroment
```bash
$ mamba activate <path_to_env>
```

## Install package
```bash
$ pip install holowizard --no-cache
```
```
INFO: The option --no-cache should help to resolve some dependencies issues.
```

# Tools

## The Core

To create examples, open a terminal and run

```{bash}
$ holowizard_core_create_examples <directory>
```

## The Workbench

The workbench is a webinterface that provides an overview over scanned and processed datasets during and after a beamtime.
A toolbox is avalialbe for data postprocessing and 3D visualization.

### Starting an Instance

We provide a CLI command to initialize everything:

```bash
$ holowizard_workbench_start beamtime.id=YOUR_BEAMTIME_ID beamtime.year=YEAR_OF_BEAMTIME
```

This command starts the web interface. 
If the startup is successful, you’ll see output like:

```
INFO:     Uvicorn running on http://MY_IP_ADDRESS:MY_PORT (Press CTRL+C to quit)
```

Click the address to open a browser window

### User Administration

Every page of the workbench requires a login. Users and their password hashes (PBKDF2-SHA256) are stored in a JSON file,
by default

```
~/.config/holowizard/pipe_users.json
```

of the account that runs the server. Set the environment variable `HOLOPIPE_USERS_FILE` to use another file, e.g. one
shared by several instances. The file is readable only by its owner and also holds the secret used to sign the login
cookies, so keep it private.

**First start:** if the file does not exist yet, the server creates it with the user `p05user` and a random password,
which is printed once to the console:

```
========================================================================
Created login user file /home/<account>/.config/holowizard/pipe_users.json
  user:     p05user
  password: <random password>
Change it with: holowizard_workbench_users passwd p05user
========================================================================
```

**Managing users** (on the machine and with the account that runs the server):

```bash
$ holowizard_workbench_users list              # show all users
$ holowizard_workbench_users add <user>        # add a user (asks for the password)
$ holowizard_workbench_users passwd <user>     # change a password
$ holowizard_workbench_users remove <user>     # remove a user
```

- User names may contain letters, digits, `_` and `-` (at most 64 characters); passwords need at least 8 characters.
- The last remaining user cannot be removed.
- Changes take effect immediately, without restarting the server. Changing a password or removing a user signs out all
  existing logins of that user.
- A login stays valid for 7 days. *Logout* in the menu ends it earlier.

**What is stored per user:** each user has their own editable parameter set on the Phase Retrieval page
(`pipe_config/reconstruction/custom_<user>.yaml` in the beamtime's log folder, e.g.
`scratch_cc/holopipe/`), and the rotation center and tomography pages track running
jobs per user (only the user who started a reconstruction can cancel it). The user who saved a rotation center or
started a tomogram is recorded in `rotation_center.yaml` and `reconstruction.yaml`.

## The Forge

This framework can be used to generate large datasets of simulated holograms of randomly sampled objects. 

### Create New Dataset

Open a terminal and create a new config file with 

```{bash}
$ holowziard_forge_create_testconfig <args>
```

| Argument        | Description                                               | Position |
|-----------------|-----------------------------------------------------------|----------|
| `name`          | Name of the new config file (without file name extension) | 1        |
| `--output_dir`  | Output directory                                          | optional |
| `--override`    | Overrides existing configuration file                     | optional |


The config file can then be customized and used to create a new dataset with

```{bash}
$ holowizard_forge_generate_data <args>
```

| Argument      | Description                                          | Position |
|---------------|------------------------------------------------------|----------|
| `config`      | Path to the custom configuration file.               | 1        |
| `output_dir`  | Output directory where the generated data is stored. | 2        |
| `num_samples` | Number of data samples that should be generated.     | 3        |
| `--override`  | Override the output folder if it already exists.     | optional |

### Output Structure
```
output/
└── train.hdf5
└── train.json
```

The file train.hdf5 contains the training data
The file `train.json` contains the config parameters which have been used for the training data creation.

### Developer Info

#### Add new Parameters
To add a new parameter, add it to the default configuration `holowizard/forge/configs/default.json`.

## The Livereco

Start the online reconstruction server with

```{bash}
$ holowizard_livereco_server_start
```

# Citations
### Artifact-suppressing reconstruction method:
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

### Model-based autofocus:
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

## Further links
https://helmholtz.software/software/holowizard \
https://connect.helmholtz-imaging.de/solution/71 \
https://connect.helmholtz-imaging.de/blog_gallery/blogpost/10 
