# Contributed from Ryan MacDonald, used in POSEIDON for more efficient memory usage when running retrievals on multiple cores.
# Ryan notes that this function may only work if all cores are on the same node.
import numpy as np
from mpi4py import MPI


def shared_memory_array(rank, comm, shape):
    """
    Creates a numpy array shared in memory across multiple cores.

    Adapted from :
    https://stackoverflow.com/questions/32485122/shared-memory-in-mpi4py

    """

    # Create a shared array of size given by product of each dimension
    size = np.prod(shape)
    itemsize = MPI.DOUBLE.Get_size()

    if rank == 0:
        nbytes = size * itemsize  # Array memory allocated for first process
    else:
        nbytes = 0  # No memory storage on other processes

    # On rank 0, create the shared block
    # On other ranks, get a handle to it (known as a window in MPI speak)
    new_comm = MPI.Comm.Split(comm)
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=new_comm)

    # Create a numpy array whose data points to the shared memory
    buf, itemsize = win.Shared_query(0)
    assert itemsize == MPI.DOUBLE.Get_size()
    array = np.ndarray(buffer=buf, dtype="d", shape=shape)

    return array, win


"""
    Example of how to use this function:
    sigma_stored, _ = shared_memory_array(node_rank, node_comm, (N_species_active, N_P_fine, N_T_fine, N_wl))     # Molecular and atomic opacities
"""
