# HoloWizard Pipe

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

