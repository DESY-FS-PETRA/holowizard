import subprocess
import pathlib


def start(env):
    pwd = pathlib.Path(__file__).parent.resolve()
    command = (
        "module load maxwell mamba gcc;. mamba-init ; mamba activate "
        + str(env)
        + ";lavue --configuration-path "
        + str(pwd)
        + " -i linear -t rot90 -m expert"
    )
    subprocess.call(command, shell=True)
