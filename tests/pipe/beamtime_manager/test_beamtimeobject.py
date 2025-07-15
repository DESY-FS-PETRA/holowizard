import json
import tempfile
from pathlib import Path

from holowizard.beamtime.beamtime import BeamtimeObject

def create_dummy_beamtime_structure(base_path, year, beamtime):
    root = Path(base_path) / "p05" / year / "data" / beamtime
    raw_path = root / "raw"
    processed_path = root / "processed"
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    for scan_name in ["nano_test1", "nano_test2", "other"]:
        (raw_path / scan_name).mkdir()

    metadata = {"title": "Test Beamtime", "beamtimeId": beamtime}
    metadata_path = root / f"beamtime-metadata-{beamtime}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    return str(base_path)

def test_beamtime_initialization_and_scan_filtering():
    with tempfile.TemporaryDirectory() as tmp:
        year = "2025"
        beamtime = "BT999"
        base_path = create_dummy_beamtime_structure(tmp, year, beamtime)

        bt = BeamtimeObject(beamtime, year, base_path=base_path)

        assert bt.meta_dict["beamtimeId"] == beamtime
        assert len(bt.scans_tab) == 2
        assert all("nano" in s.name for s in bt.scans_tab)

def test_beamtime_search_in_bt():
    with tempfile.TemporaryDirectory() as tmp:
        year = "2025"
        beamtime = "dummy"
        base_path = create_dummy_beamtime_structure(tmp, year, beamtime)

        bt = BeamtimeObject(beamtime, year, base_path=base_path)
        matches = bt.search_in_bt("nano")

        assert len(matches) == 2
        assert all("nano" in scan.name for scan in matches)
