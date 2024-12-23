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
$ pip install holowizard[<package>]
```

The following packages are available
- livereco_interface
- livereco
- simulation
