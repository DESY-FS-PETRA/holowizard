import os
import json
import pytest
from unittest import mock

import holowizard.beamtime.bt_utils as bt_utils


@pytest.fixture
def mock_beamtime_structure(tmp_path):
    """
    Creates a dummy directory structure for a single beamtime in a given year.
    """
    year = "2025"
    beamtime = "BT999"
    base = tmp_path / "p05"
    path = base / year / "data" / beamtime
    raw_path = path / "raw"
    processed_path = path / "processed"
    raw_path.mkdir(parents=True)
    processed_path.mkdir()

    # Create dummy metadata file
    metadata_file = path / f"beamtime-metadata-{beamtime}.json"
    with open(metadata_file, "w") as f:
        json.dump({"beamtimeId": beamtime, "year": year}, f)

    return str(tmp_path), year, beamtime


@mock.patch("src.beamtime_manager.bt_utils.get_logged_in_user", return_value="testuser")
@mock.patch("src.beamtime_manager.bt_utils.has_access_to_folder", return_value=True)
def test_list_beamtimes(mock_access, mock_user, mock_beamtime_structure):
    tmp, year, beamtime = mock_beamtime_structure
    base_path = os.path.join(tmp, "p05")

    results = bt_utils.list_beamtimes(base_path=base_path)
    assert any(bt["name"] == beamtime for bt in results), "Expected beamtime not found in list_beamtimes."


@mock.patch("src.beamtime_manager.bt_utils.get_logged_in_user", return_value="testuser")
@mock.patch("src.beamtime_manager.bt_utils.has_access_to_folder", return_value=True)
def test_get_user_accessible_years(mock_access, mock_user, mock_beamtime_structure):
    tmp, year, _ = mock_beamtime_structure
    base_path = os.path.join(tmp, "p05")

    years = bt_utils.get_user_accessible_years(base_path=base_path)
    assert year in years, f"Year {year} should be accessible but wasn't listed."
