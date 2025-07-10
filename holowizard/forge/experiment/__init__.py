from .setup import Setup


__all__ = [
    "GLOBAL_EXP_SETUP",
    "initialize_global_exp_setup",
]


GLOBAL_EXP_SETUP: Setup = None


def initialize_global_exp_setup(setup: Setup) -> None:
    global GLOBAL_EXP_SETUP
    if GLOBAL_EXP_SETUP is None:
        GLOBAL_EXP_SETUP = setup
