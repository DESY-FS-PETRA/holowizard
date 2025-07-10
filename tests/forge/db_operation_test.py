# standard libraries
from pathlib import Path

# third party libraries

# local libraries
from holoforge.utils.material_properties.property_db_builder import PropertyDatabaseBuilder as PropDBBuilder
from holoforge.utils.material_properties import MaterialPropertyLoader
from holoforge.utils import fileIO


__all__ = [
    'add_newly_fetches_property_to_db_test',
    'create_empty_db_test',
    'merge_db_test',
]


def create_empty_db_test():
    path = 'holoforge/tests/empty_database'
    PropDBBuilder.build_empty_database(output=path, override=True)


def merge_db_test():
    src = 'holoforge/tests/empty_database'
    target = 'database_copy'
    PropDBBuilder.merge_databases(target, src)


def add_newly_fetches_property_to_db_test():
    path = 'holoforge/tests/output/new_fetches_test'
    PropDBBuilder.build_empty_database(output=path, override=True)
    prop_loader = MaterialPropertyLoader(path)
    formula, density = 'Mg', 1.738
    delta, beta = prop_loader.get_delta_beta(formula, density, 11000)
    df = fileIO.read_csv(Path(path) / 'delta_beta' / f'{formula}-{density}.csv', header=True)
    print(df)
    print(delta, beta)
