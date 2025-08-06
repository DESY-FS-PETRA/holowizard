import os


def dask_setup(worker):
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.pop("OPENBLAS_NUM_THREADS", None)
    os.environ.pop("MKL_NUM_THREADS", None)
