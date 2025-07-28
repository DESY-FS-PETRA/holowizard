# standard libraries
import argparse
import shutil
from pathlib import Path

# third party libraries

# local packages
from holowizard.forge.configs import TEST_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the config file without file name extension.", type=str)
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output folder where to store the config file.",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--override",
        help="Override existing custom config file, if it exists.",
        action="store_true",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{args.name}.json"
    if not args.override and output_file.exists():
        raise FileExistsError(f"The configuration already exists. Use '--override' to override the existing file.")
    shutil.copyfile(TEST_CONFIG_PATH, output_file)


if __name__ == "__main__":
    main()
