
from numpy import array, prod, ndarray
from itertools import product


def generate_bin_arrays(dims: tuple) -> tuple:
    """
    Generates all possible arrays for a binary array of shape 'dims'.

    NOT ENOUGH MEMORY TO HANDLE ALL PERUMUTATIONS OF 4+D ARRAYS!
    """
    arrays = []
    for vals in product([0, 1], repeat=prod(dims)): # type: ignore
        arrays.append(array(vals).reshape(dims))
    return tuple(arrays)


def check_repeat(actual: ndarray, hist: tuple) -> int:
    """
    Checks if an object has existed before and gives its index.
    """
    if actual in hist:
        return hist.index(actual)
    return -1