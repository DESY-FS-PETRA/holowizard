#!/usr/bin/env python3
import requests
import json
from argparse import ArgumentParser
from pprint import pprint
import socket

API_BASE = f"http://{socket.gethostname()}:8000/api"

def get_test_options():
    """
    Calls your UI‐router endpoints:
      GET /stage/reconstruction
      GET /stage/find_focus

    and returns a dict:
    {
      "reconstruction": [...],
      "find_focus":    [...]
    }
    """
    # strip off the "/api" to get back to the UI endpoints
    UI_BASE = API_BASE.rsplit("/api", 1)[0]

    # fetch both lists
    recon_list = requests.get(f"{UI_BASE}/stage/reconstruction")
    recon_list.raise_for_status()

    focus_list = requests.get(f"{UI_BASE}/stage/find_focus")
    focus_list.raise_for_status()

    return {
        "reconstruction": recon_list.json(),
        "find_focus":    focus_list.json(),
    }


def parse_stage_options(opt_list):
    """
    --option flags look like: --option reconstruction=wire
    """
    opts = {}
    for item in opt_list or []:
        if "=" not in item:
            raise ValueError(f"Invalid --option value '{item}', must be stage=option")
        stage, choice = item.split("=", 1)
        opts[stage] = choice
    return opts

def main():
    # 1) fetch dynamic choices
    opts = get_test_options()
    recon_choices = opts.get("reconstruction", [])
    focus_choices = opts.get("find_focus", [])

    # 2) build parser
    parser = ArgumentParser(description="Submit a scan to HoloServer")
    parser.add_argument("--scan-name",  required=True, help="Unique scan identifier")
    parser.add_argument("--holder",     required=True, type=float, help="Holder ID")
    parser.add_argument("--z01",        type=float, help="z₀₁ (optional)")
    parser.add_argument("--a0",         type=float, help="a₀ (optional)")
    parser.add_argument("--energy",     type=float, help="Beam energy")
    parser.add_argument("--stages",     nargs="+", help="Should be a subset of: flatfield, find_focus, reconstruction, tomography")
    parser.add_argument(
        "--reconstruction",
        choices=recon_choices,
        help="Preset for phase‐retrieval (Can be adapted in the holowizard config folder or under /parameter)"
    )
    parser.add_argument(
        "--find-focus",
        dest="find_focus",
        choices=focus_choices,
        help="Preset for find‐focus (Can be adapted in the holowizard config folder or under /parameter)"
    )

    args = parser.parse_args()

    # 3) assemble payload
    payload = {
        "scan_name": args.scan_name,
        "holder":    args.holder,
        "z01":       args.z01,
        "a0":        args.a0,
        "energy":    args.energy,
        "reconstruction": args.reconstruction,
        "find_focus":    args.find_focus,
    }
    if args.stages:
        payload["stages"] = args.stages
    # 4) POST it
    resp = requests.post(f"{API_BASE}/submit_scan", json=payload)
    resp.raise_for_status()
    print("Server response:")
    pprint(resp.json())

if __name__ == "__main__":
    main()