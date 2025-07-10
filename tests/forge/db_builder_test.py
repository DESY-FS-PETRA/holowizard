# standard libraries

# third party libraries

# local libraries
from holoforge.utils.material_properties.property_db_builder import PropertyDatabaseBuilder as PropDBBuilder


__all__ = [
    'small_db_builder_test',
    'large_db_builder_test',
]


def small_db_builder_test(output: str = 'output/tests/db_builder_test_S') -> None:
    prop_db_builder = PropDBBuilder()
    prop_db_builder.build_small_database(output=output, override=True, materials=['Mg'])


def large_db_builder_test(output: str = 'output/tests/db_builder_test_L') -> None:
    prop_db_builder = PropDBBuilder()
    prop_db_builder.build_large_database(output=output, override=True, materials=['Mg'], energies=[7000])
