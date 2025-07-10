import math


def find_next_power_of_two(x):
    return 2 ** (math.ceil(math.log(x, 2)))
