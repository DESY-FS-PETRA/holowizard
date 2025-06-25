 <img src="logo.png" width="200"/> 


# Table of Contents
1. [General Information](#general-information)
2. [Installation](#installation)
3. [Citations](#citations)

## General Information
- Repository: https://github.com/DESY-FS-PETRA/holowizard
- Zenodo: https://doi.org/10.5281/zenodo.8349364
- [Please Cite](#citations)

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

## Citations
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
