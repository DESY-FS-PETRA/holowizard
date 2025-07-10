import holowizard.core.utils.fileio as fileio
from pathlib import Path

REPOROOT = (Path(__file__).parent / "..").resolve()
DATAPATH = REPOROOT / "Examples/data/"

def test_load_img_data():

    dp = DATAPATH / "holograms" / "spider_hair.tiff"

    assert dp.exists()

    lt = fileio.load_img_data(dp)

    print(type(lt))

    assert lt.shape
