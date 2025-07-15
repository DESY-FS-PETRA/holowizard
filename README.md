 <img src="https://connect.helmholtz-imaging.de/media/solution/Holowizard_logo.png" width="200"/> 


# Table of Contents
1. [General Information](#general-information)
2. [Installation](#installation)
3. [HoloWizard Core](#holowizard-core)
4. [HoloWizard Pipe](#holowizard-pipe)
5. [HoloWizard Forge](#holowizard-forge)
6. [Please Cite](#citations)

## General Information
- Repository: https://github.com/DESY-FS-PETRA/holowizard
- Zenodo: https://doi.org/10.5281/zenodo.8349364

### Further links
https://helmholtz.software/software/holowizard \
https://connect.helmholtz-imaging.de/solution/71 \
https://connect.helmholtz-imaging.de/blog_gallery/blogpost/10 

## Installation
### Python Environment
Create a new environment with python 3.11., i.e. with mamba
```bash
$ mamba create -p <path_to_env> python=3.11 
```

Activate enviroment
```bash
$ mamba activate <path_to_env>
```

### Install package
```bash
$ pip install holowizard
```

# HoloWizard Core

To create examples, open a terminal and run

```{bash}
$ holowizard_core_create_examples <directory>
```

# HoloWizard Pipe

## Getting Started (P05 DESY + Maxwell User)

Before using the pipeline, some configuration is required.

Open the `scripts/config/cluster.yaml` file and update the cluster settings accordingly — especially the list of usable partitions.

Install the package into your Python environment by running the following command from the root of the repository:

```bash
pip install .
```

---

## Setting Up an Instance

We provide a CLI command to initialize everything:

```bash
holopipe beamtime.name=YOUR_BEAMTIME_NAME beamtime.year=YEAR_OF_BEAMTIME
```

This command sets up the pipeline. You can override any other configuration value using Hydra’s override syntax:  
👉 [Hydra Override Syntax Documentation](https://hydra.cc/docs/advanced/override_grammar/basic/)

If the startup is successful, you’ll see output like:

```
INFO:     Uvicorn running on http://MY_IP_ADDRESS:MY_PORT (Press CTRL+C to quit)
```

Click the address to open a browser window showing that `holopipe` is running.  
Visit: `http://MY_IP_ADDRESS:MY_PORT/dashboard` for useful runtime information.


---

## Usage

### Add a scan with default parameters. 

You can submit scans using a simple `curl` POST request:

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{ "a0": 1.0,
           "scan-name": "nano145014_s7_1_tum",
           "holder": 220,
           "base_dir": "holopipe",
           "reconstruction": "wire",
           "find_focus": "wire",
           "energy": 17.0
         }' \
     http://MY_IP_ADDRESS:MY_PORT/api/submit_scan
```

If you are on the same machine as the server is running you can use the python script:
```bash
recontruct-scan --help # will tell you all important parameters
```

#### Required Parameters

- `name`: Folder name of the current scan  
- `holder`: Height of the holder  
- `energy`: Scan energy in keV  

#### Optional Parameters

- `a0`: Optional numeric parameter; if not provided, it will be computed automatically  
- `reconstruction`: Instruction set for reconstruction — `wire` (default) or `spider`  
- `find_focus`: Instruction set to find focus — `wire` (default) or `spider`  
- `base_dir`: Root directory for output files (default: `holopipe`)

### Parameter Optimization

To performe parameter optimization go to `http://MY_IP_ADDRESS:MY_PORT/parameter`. Here you can set all parameter for all stages. To test them click `Reconstruct`. If the parameters work well, you can save them to the current beamtime folder using the `Save as` at the lower left. 
If you want to reconstruct the whole scan you can click `Submit All` after chosing the Options. If you select `Custom` it will take the parameters from the left. 

### Other changes during beamtime

If you change the detector or anything else like removing tasks adapt parameters the full config files are located in the `beamtime/processed/holowizard_config/`folder. Changes here will reflect onto future curl requests!

# HoloWizard Forge

This framework can be used to generate large datasets of simulated holograms of randomly sampled objects. 

## Create New Dataset

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

## Output Structure
```
output/
└── train.hdf5
└── train.json
```

The file train.hdf5 contains the training data
The file `train.json` contains the config parameters which have been used for the training data creation.

## Developer Info

### Add new Parameters
To add a new parameter, add it to the default configuration `holowizard/forge/configs/default.json`.

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

### Python Repository on Zenodo
- URL: https://zenodo.org/records/14024980
- DOI: 10.5281/zenodo.8349364
